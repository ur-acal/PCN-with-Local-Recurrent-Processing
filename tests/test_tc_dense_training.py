"""CPU checks of the opt-in TC dense path, not hardware accuracy experiments."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ode_pc import (ODEXInitFFFB, S2NoisyIYAsXZAs0, QATWrapper1State,
                    QATWrapper2State, ToggleODEXInitFFFB, ToggleQATWrapper1State)
from pc_conv import PCConvReLU6
from tc_nonidealities import TCResistanceCurves
from measured_activation import PiecewiseLinearActivation
from measured_pooling import configure_measured_pooling, MeasuredAvgPool2d

ROOT = Path(__file__).resolve().parents[1]


def make_block(two=False):
    torch.manual_seed(14)
    pc = PCConvReLU6(inp_chan=2, out_chan=2, kernel_size=1, padding=0,
                    cls=2, bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
    with torch.no_grad():
        pc.FFconv.weight.copy_(torch.tensor([.2, .1, .05, .3]).reshape_as(pc.FFconv.weight))
        pc.FBconv.weight.copy_(torch.tensor([.3, .1, .2, .1]).reshape_as(pc.FBconv.weight))
    cls = S2NoisyIYAsXZAs0 if two else ODEXInitFFFB
    return cls(pc_conv=pc, noise_level=0., method="dopri5", t_end=.3,
               tol=1e-6, sde_noise_type="addi")


def wrap(block, enabled=True, **kwargs):
    cls = QATWrapper2State if isinstance(block, S2NoisyIYAsXZAs0) else QATWrapper1State
    return cls(ode_block=block, tc_nonidealities=enabled, R=1e4, R_max=150e3,
               C=49e-15, k=1e3, v_dd=.1, state_bound=1., w_bits=5,
               thermal_noise=False, offset_eps=0., is_first=True, is_last=True, **kwargs)


def synthetic_package(block):
    levels = block._get_quant_magnitude_levels(torch.empty(0, dtype=torch.float64))
    resistances = 1e4 / levels[1:]
    grid = torch.tensor([-.1, 0., .1], dtype=torch.float64)
    return TCResistanceCurves(grid, levels, resistances, resistances,
        resistances[:, None].expand(-1, 3).clone(), torch.eye(3, dtype=torch.float64)*10,
        "synthetic", "synthetic", 1e-6)


class TCDenseTests(unittest.TestCase):
    def test_dense_matches_direct_kernel_sum_and_gradients(self):
        block = make_block()
        block.q_hi, block.weight_scale, block.v_dd = 3, 1., .1
        package = synthetic_package(block)
        block._tc_curve_package = package
        # Known R_l(x) = R_l * (1 + 2x), independently evaluable.
        curves = package.means * (1 + 2*package.v_grid)
        for transpose in (False, True):
            for groups in (1, 2):
                with self.subTest(transpose=transpose, groups=groups):
                    cls = nn.ConvTranspose2d if transpose else nn.Conv2d
                    extra = dict(output_padding=1) if transpose else {}
                    conv = cls(2, 2, 3, stride=2, padding=1, groups=groups,
                               bias=False, dtype=torch.float64, **extra)
                    with torch.no_grad():
                        codes = torch.arange(conv.weight.numel()).reshape_as(conv.weight) % 7 - 3
                        conv.weight.copy_(codes/3)
                    x = torch.linspace(-.08, .08, 32, dtype=torch.float64).reshape(1,2,4,4).requires_grad_()
                    got = block._tc_dense_conv(conv, x, curves)
                    # All codes have the same relative curve here: compare one
                    # ordinary convolution (including signed and zero weights).
                    corrected = x/(1+2*x)
                    if transpose:
                        expected = F.conv_transpose2d(corrected, conv.weight, stride=2,
                                                      padding=1, output_padding=1, groups=groups)
                    else:
                        expected = F.conv2d(corrected, conv.weight, stride=2, padding=1, groups=groups)
                    torch.testing.assert_close(got, expected)
                    gx, gw = torch.autograd.grad(got.square().sum(), (x, conv.weight), retain_graph=True)
                    ex, ew = torch.autograd.grad(expected.square().sum(), (x, conv.weight))
                    torch.testing.assert_close(gx, ex)
                    # Zero-code entries are open circuits (fixed membership).
                    torch.testing.assert_close(gw, ew*(codes != 0))

    def test_independent_code_curves_against_explicit_one_by_one_edges(self):
        block = make_block()
        block.q_hi, block.weight_scale, block.v_dd = 3, 1., .1
        package = synthetic_package(block)
        block._tc_curve_package = package
        conv = nn.Conv2d(3, 2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([1., -2/3, 0., 1/3, -1., 2/3]).reshape_as(conv.weight))
        x = torch.tensor([.02, .03, .04], dtype=torch.float64).reshape(1,3,1,1).requires_grad_()
        slopes = torch.tensor([1., 2., 3.], dtype=torch.float64)
        curves = package.means*(1+slopes[:,None]*package.v_grid)
        actual = block._tc_dense_conv(conv, x, curves)
        expected = []
        for i in range(2):
            terms = []
            for j in range(3):
                w = conv.weight[i,j,0,0]
                code = int(round(abs(w.item())*3))
                if code:
                    terms.append(w*x[:,j]/(1+slopes[code-1]*x[:,j]))
            expected.append(sum(terms))
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual.sum(), (x,conv.weight), retain_graph=True)
        b = torch.autograd.grad(expected.sum(), (x,conv.weight))
        for left,right in zip(a,b):
            torch.testing.assert_close(left,right)

    def test_disabled_equivalence_and_qat_backward_both_states(self):
        for two in (False,True):
            old, new = make_block(two), make_block(two)
            # Use the same fixed grid for numerical equivalence; adaptive
            # grids can differ when a dense sum changes float32 roundoff.
            for block in (old, new):
                block.option_aca.update(method="euler", h=.01)
            old_wrapper, new_wrapper = wrap(old,False), wrap(new)
            package = synthetic_package(new)
            new._tc_curve_package = replace(package, factor=torch.zeros_like(package.factor))
            new._tc_curve_generator = torch.Generator().manual_seed(4)
            x = torch.full((1,2,2,2), .2, requires_grad=True)
            expected, actual = old(x), new(x)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
            actual.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for module in (new.FFconv,new.FBconv):
                grad = module.parametrizations.weight.original.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_lifetimes_and_physical_fb_voltage(self):
        block = make_block()
        wrapper = wrap(block)
        block._tc_curve_package = synthetic_package(block)
        block._tc_curve_generator = torch.Generator().manual_seed(31)
        x = torch.full((1,2,2,2), .03)
        fn = block._make_ode_fn(x)
        first = block._tc_curve_samples
        y = fn(0,x)
        torch.testing.assert_close(fn(1,x),y)
        self.assertFalse(torch.equal(first['FFconv'],first['FBconv']))
        block._make_ode_fn(x)
        self.assertFalse(torch.equal(first['FFconv'],block._tc_curve_samples['FFconv']))
        torch.testing.assert_close(fn(0,x),y)  # earlier solve closure is stable
        block.eval()
        cached = block._tc_curve_samples
        block._make_ode_fn(x)
        self.assertIs(cached,block._tc_curve_samples)
        block.reset_tc_curves()
        block._make_ode_fn(x)
        self.assertIsNot(cached,block._tc_curve_samples)
        # Algebraic FB conversion happens AFTER querying the input at x.
        curves = block._tc_curve_samples
        z = block._tc_dense_conv(block.FBconv,x,curves['FBconv']) * wrapper.k/wrapper.R
        expected = block._tc_dense_conv(block.FFconv,block.act_fn(z),curves['FFconv'])/wrapper.R/wrapper.C
        torch.testing.assert_close(block._make_ode_fn(x)(0,x),expected)

    def test_real_package_and_measured_activation_qat_preservation(self):
        for two in (False,True):
            block = make_block(two)
            wrapper = wrap(block, nonlinear_R=True, nonlinear_R_train_mode="exact_curve",
                nonlinear_R_table=ROOT/'hardware_data/res_vs_vin_10k_150k.csv',
                tc_covariance_table=ROOT/'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
                nonlinear_R_curve_seed=19, enable_measured_activation=True)
            activation = block.act_fn
            self.assertIsInstance(activation,PiecewiseLinearActivation)
            self.assertEqual(block._tc_curve_package.means.shape[0],15)
            x = torch.full((1,2,2,2),.2,requires_grad=True)
            out = block(x)
            out.sum().backward()
            self.assertIs(block.act_fn,activation)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_shared_pooling_installer_preserves_final_relu_and_scales(self):
        model = nn.Module()
        model.register_parameter('dummy',nn.Parameter(torch.ones(())))
        model.max_pool2d = nn.AvgPool2d(2)
        model.relu = nn.ReLU()
        relu = model.relu
        gaussian = dict(v_grid=torch.tensor([-.1,.1]),mean=torch.ones(2),
                        factor=torch.zeros(2,2),value_scale=50e3,quantity='resistance')
        configure_measured_pooling(model,[SimpleNamespace(out_scale=1.),SimpleNamespace(out_scale=.1)],
            enable_nonideality=True,curve_gaussian=gaussian,nominal_R=50e3)
        self.assertIs(model.relu,relu)
        self.assertIsInstance(model.max_pool2d,MeasuredAvgPool2d)
        x = torch.rand(2,3,4,4)*.05
        torch.testing.assert_close(model.max_pool2d(x),F.avg_pool2d(x,2))
        torch.testing.assert_close(model.global_avg_pool2d(x),F.avg_pool2d(x,4))

    def test_toggle_cannot_enable_tc(self):
        block = make_block()
        block.__class__ = ToggleODEXInitFFFB
        with self.assertRaisesRegex(TypeError,"not toggle"):
            ToggleQATWrapper1State(ode_block=block,tc_nonidealities=True)

    def test_full_trajectory_uses_tc_curves(self):
        for two in (False,True):
            block = make_block(two)
            wrapper = wrap(block)
            block._tc_curve_package = synthetic_package(block)
            block._tc_curve_generator = torch.Generator().manual_seed(21)
            x = torch.full((1,2,2,2),.2)
            # As in the existing API, full_steps is called on physical inputs;
            # initialize QAT/time metadata first (it bypasses __call__ hooks).
            block(x)
            old = block._tc_curve_samples
            result = block.forward_full_steps(x*wrapper.q)
            self.assertIsNot(old,block._tc_curve_samples)
            def check(value):
                if isinstance(value,(tuple,list)):
                    for item in value:
                        check(item)
                elif torch.is_tensor(value):
                    self.assertTrue(torch.isfinite(value).all())
            check(result)


if __name__ == '__main__':
    unittest.main()
