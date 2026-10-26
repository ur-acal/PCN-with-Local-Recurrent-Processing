"""Literal-reference tests; CUDA tests must pass before enabling by default."""
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ode_pc import ToggleODEXInitFFFB, ToggleQATWrapper1State
from pc_conv import PCConvReLU6
from tc_shared_correction import empirical_correction, toggle_empirical_correction, triton


def block(device='cpu', dtype=torch.float32, projection=None):
    pc = PCConvReLU6(inp_chan=2, out_chan=2, kernel_size=1, padding=0,
                    cls=2, bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
    b = ToggleODEXInitFFFB(pc_conv=pc, noise_level=0., method='euler',
                         t_end=.3, tol=1e-4).to(device=device, dtype=dtype)
    grid = torch.tensor([-.7, -.5, -.2, 0., .2, .5, .7], device=device, dtype=dtype)
    resistance = torch.tensor([4e4, 4.2e4, 4.8e4, 5e4, 4.7e4, 5.1e4, 5.2e4],
                              device=device, dtype=dtype)
    b._nonlinear_R_training_pkg = dict(
        v_grid=grid.repeat(2, 1), lengths=torch.tensor([7, 7], device=device),
        R_left=resistance[:-1].repeat(2, 1),
        R_slope=((resistance[1:]-resistance[:-1])/grid.diff()).repeat(2, 1),
        R=5e4, proj_fn=projection, nonlinear_R_curve_seed=81)
    return b


def literal(b, module, source):
    """Copy of original arithmetic, independent of the optimized helper."""
    p = b._nonlinear_R_training_pkg
    i = b._sample_nonlinear_R_training_curve(source, b._nonlinear_R_training_module_key(module))
    length = int(p['lengths'][i].item())
    grid = p['v_grid'][i, :length].to(device=source.device, dtype=source.dtype)
    left = p['R_left'][i, :length-1].to(device=source.device, dtype=source.dtype)
    slope = p['R_slope'][i, :length-1].to(device=source.device, dtype=source.dtype)
    query = source
    if p.get('proj_fn') is not None:
        query = p['proj_fn'](query)
    query = query.clamp(min=grid[0], max=grid[-1])
    interval = (torch.bucketize(query.contiguous(), grid.contiguous())-1).clamp(0, length-2)
    R_eff = left[interval]+slope[interval]*(query-grid[interval])
    return source*source.new_tensor(p['R'])/R_eff


class FallbackTests(unittest.TestCase):
    def test_real_slurm_export_and_training_parser(self):
        env = dict(os.environ, TC_DRY_RUN='true', TOGGLE_MODE='odexinit',
            TASK='cifar100', IMG_TYPE='CiFAIR', SEARCH_ARCH='default',
            PCN_CHAN_0_LIST='24', PCN_NUM_LAYERS_LIST='16',
            NUM_COMB_PER_NUM_LAYER='2', COMB_SEL_SET='2',
            FUSE_TOGGLE_NONLINEAR_R_INPUT='true')
        result = subprocess.run(['bash', '-c', 'source launch_scripts/slurm_search_config.sh'],
            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FUSE_TOGGLE_NONLINEAR_R_INPUT=true', result.stdout)
        from train_ode_cifar import get_args
        with patch.dict(os.environ, {'FUSE_TOGGLE_NONLINEAR_R_INPUT': 'true'}), \
                patch('sys.argv', ['train_ode_cifar.py']):
            self.assertTrue(get_args().fuse_toggle_nonlinear_R_input)
        with patch('sys.argv', ['train_ode_cifar.py', '--fuse_toggle_nonlinear_R_input', 'false']):
            self.assertFalse(get_args().fuse_toggle_nonlinear_R_input)

    def test_cpu_literal_output_input_and_convolution_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for projection in (None, nn.Hardtanh(-.5, .5), nn.ReLU(), nn.Hardtanh(-.2, .6)):
                b = block(dtype=dtype, projection=projection)
                b._fuse_toggle_nonlinear_R_input = True
                x = torch.linspace(-1, 1, 40, dtype=dtype).reshape(2, 2, 2, 5).requires_grad_()
                expected = b.FFconv(literal(b, b.FFconv, x))
                eg = torch.autograd.grad(expected.square().sum(), (x, b.FFconv.weight))
                actual = b._apply_averaged_module(b.FFconv, x)
                ag = torch.autograd.grad(actual.square().sum(), (x, b.FFconv.weight))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for a, e in zip(ag, eg):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)
                self.assertFalse(b._toggle_nonlinear_R_prepared)

    def test_off_never_calls_optimized_helper(self):
        b = block()
        with patch('tc_shared_correction.toggle_empirical_correction', side_effect=AssertionError):
            b._averaged_nonlinear_R_source(b.FFconv, torch.ones(2))

    def test_reset_clears_only_realization_and_prepared_constants(self):
        b = block()
        b._sample_nonlinear_R_training_curve(torch.ones(2), 'FFconv')
        generator = b._nonlinear_R_training_generators['cpu']
        state = generator.get_state().clone()
        b._toggle_nonlinear_R_prepared['proof'] = 1
        b.reset_nonlinear_R_variation()
        self.assertFalse(b._nonlinear_R_training_curve_indices)
        self.assertFalse(b._toggle_nonlinear_R_prepared)
        self.assertTrue(torch.equal(state, generator.get_state()))

    def test_strict_rail_backward_differs_from_clamp_only_at_endpoints(self):
        b = block(projection=nn.Hardtanh(-.5, .5))
        x = torch.tensor([-.8, -.5, -.2, 0., .2, .5, .8], requires_grad=True)
        p = b._nonlinear_R_training_pkg
        actual = empirical_correction(x, p['v_grid'][0], p['R_left'][0],
            p['R_slope'][0], x.new_tensor(p['R']), .5, fused=False, strict_rail=True)
        expected = literal(b, b.FFconv, x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                                   torch.autograd.grad(expected.sum(), x)[0], atol=0, rtol=0)

    def test_wrapper_forwards_opt_in_without_changing_default(self):
        for enabled in (False, True):
            b = block()
            ToggleQATWrapper1State(ode_block=b, R=5e4, C=500e-15,
                v_dd=.5, state_bound=5., w_bits=5, thermal_noise=False,
                is_first=True, is_last=True, fuse_toggle_nonlinear_R_input=enabled)
            self.assertIs(b._fuse_toggle_nonlinear_R_input, enabled)


@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA/Triton required')
class CUDATests(unittest.TestCase):
    def test_actual_pinned_bank_native_grids(self):
        from utils import load_mc_res_training_curve_bank
        path = Path(__file__).resolve().parents[1]/'hardware_data/mc_45_corners/coupler_full_range'
        if not path.is_dir():
            self.skipTest('Local pinned hardware bank is not installed')
        b = block('cuda', projection=nn.Hardtanh(-.5, .5))
        b._nonlinear_R_training_pkg.update(load_mc_res_training_curve_bank(
            path, mode='exact_curve', corner_range='all', quantity='conductance',
            dtype=torch.float32, device='cuda'))
        b._fuse_toggle_nonlinear_R_input = True
        p = b._nonlinear_R_training_pkg
        for index in (0, 99, 100, 1800, 4499):
            b.reset_nonlinear_R_variation()
            b._nonlinear_R_training_curve_indices.update(FFconv=index, FBconv=(index+1)%4500)
            grid = p['v_grid'][index, :int(p['lengths'][index])]
            x = torch.cat((grid, grid[:-1]+grid.diff()/2,
                grid.new_tensor([-1., -.5, .5, 1.]))).requires_grad_()
            expected = literal(b, b.FFconv, x)
            actual = b._averaged_nonlinear_R_source(b.FFconv, x)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                torch.autograd.grad(expected.sum(), x)[0], atol=1e-5, rtol=1e-4)

    def test_knots_rails_outside_and_input_gradients(self):
        for projection in (None, nn.Hardtanh(-.5, .5)):
            b = block('cuda', projection=projection)
            b._fuse_toggle_nonlinear_R_input = True
            x = torch.tensor([-1., -.7, -.50001, -.5, -.49999, -.2, 0., .2,
                              .49999, .5, .50001, .7, 1.], device='cuda', requires_grad=True)
            for module in (b.FFconv, b.FBconv):
                expected = literal(b, module, x)
                actual = b._averaged_nonlinear_R_source(module, x)
                self.assertEqual(len(b._toggle_nonlinear_R_prepared),
                                 1 if module is b.FFconv else 2)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                a = torch.autograd.grad(actual.sum(), x)[0]
                e = torch.autograd.grad(expected.sum(), x)[0]
                self.assertTrue(bool(torch.isfinite(a).all()))
                torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-4)

    def test_convolution_weight_gradients_assignments_and_rhs_replay(self):
        torch.manual_seed(23)
        b = block('cuda', projection=nn.Hardtanh(-.5, .5))
        x = torch.randn(2, 2, 3, 3, device='cuda', requires_grad=True)
        b._fuse_toggle_nonlinear_R_input = False
        expected = b._apply_averaged_module(b.FFconv, x)
        eg = torch.autograd.grad(expected.square().sum(), (x, b.FFconv.weight))
        assignment = dict(b._nonlinear_R_training_curve_indices)
        local_rng = b._nonlinear_R_training_generators['cuda:0'].get_state().clone()
        global_rng = torch.cuda.get_rng_state().clone()
        b._fuse_toggle_nonlinear_R_input = True
        actual = checkpoint(lambda z: b._apply_averaged_module(b.FFconv, z), x,
                            use_reentrant=False)
        ag = torch.autograd.grad(actual.square().sum(), (x, b.FFconv.weight))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        for a, e in zip(ag, eg):
            torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-4)
        self.assertEqual(assignment, b._nonlinear_R_training_curve_indices)
        self.assertTrue(torch.equal(local_rng, b._nonlinear_R_training_generators['cuda:0'].get_state()))
        self.assertTrue(torch.equal(global_rng, torch.cuda.get_rng_state()))

    def test_unsupported_projection_and_dtype_fall_back(self):
        for projection, dtype in ((nn.ReLU(), torch.float32),
                                  (nn.Hardtanh(-.2, .5), torch.float32),
                                  (nn.Hardtanh(-.5, .5, inplace=True), torch.float32),
                                  (None, torch.float64), (None, torch.float16)):
            b = block('cuda', dtype, projection)
            x = torch.ones(4, device='cuda', dtype=dtype)
            self.assertIsNone(toggle_empirical_correction(x, b._nonlinear_R_training_pkg, 0, {}))

    def test_fusion_does_not_validate_curve_data_on_gpu(self):
        b = block('cuda')
        x = torch.ones(4, device='cuda')
        with patch('torch.isfinite', side_effect=AssertionError('unexpected curve validation')):
            actual = toggle_empirical_correction(x, b._nonlinear_R_training_pkg, 0, {})
        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, empirical_correction(
            x, b._nonlinear_R_training_pkg['v_grid'][0],
            b._nonlinear_R_training_pkg['R_left'][0],
            b._nonlinear_R_training_pkg['R_slope'][0],
            x.new_tensor(5e4), float('inf'), fused=False))


if __name__ == '__main__':
    unittest.main()
