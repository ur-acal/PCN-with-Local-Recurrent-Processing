from pathlib import Path
from unittest import mock

import torch
import torch.nn.utils.parametrize as parametrize

from measured_activation import CubicBSplineActivation, PiecewiseLinearActivation
from ode_pc import (SymQuantizeWeight, ToggleODEXInitFFFB,
                    TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB)
from pc_conv import PCConvReLU6


CURVE_PATH = Path(__file__).resolve().parents[1] / "hardware_data" / "relu_0p3mV.csv"


def _make_block(enable_pullback=True, physical=False, pullback_mode=None,
                scaling_mode="approx"):
    torch.manual_seed(7)
    pc_conv = PCConvReLU6(
        inp_chan=2, out_chan=2, kernel_size=1, padding=0, cls=2,
        bypass=False, tie_weights=False, tie_bp=False)
    block = ToggleODEXInitFFFB(
        pc_conv=pc_conv, noise_level=0.0, method="euler",
        t_end=1.75, t_step=0.2, tol=1e-4, toggle_n_cycles=5,
        enable_unitless_measured_pullback=enable_pullback,
        unitless_measured_pullback_mode=pullback_mode,
        odexinit_scaling_mode=scaling_mode,
        unitless_pullback_q=0.1, unitless_pullback_k=1e3,
        unitless_pullback_R=10e3)
    block.act_fn = CubicBSplineActivation(
        curve_path=CURVE_PATH, v_dd=0.1, corner="TT", num_parameters=10)
    if physical:
        block.physical = True
        block.R = 10e3
        block.C = 49e-15
        block.C_fb = block.C
        block.C_ff = block.C
        block.k = 1e3
        block.s_ff = torch.tensor(2.0)
        block.s_fb = torch.tensor(4.0)
        block.alpha = block.s_fb
        block.alpha_1state = block.s_fb * block.s_ff
        block.w_bits = 5
        block.v_dd = 0.1
    return block


def test_unitless_timing_and_qat_scale_convention():
    block = _make_block()
    t_z, t_y = block.resolve_cycle_params(torch.zeros(1, 2, 3, 3))
    assert torch.allclose(t_z, torch.tensor(1.0))
    assert torch.allclose(t_y, torch.tensor(1.75 / 5))

    s_fb, beta, beta_c = block.prospective_unitless_pullback_scales()
    qat_quantizer = SymQuantizeWeight(w_bits=5)
    qat_quantizer.compute_s(block.FBconv.weight)
    assert torch.equal(s_fb, qat_quantizer.s_w)
    assert torch.equal(beta, 0.1 * s_fb)
    assert torch.equal(beta_c, beta * 1e3 / 10e3)
    assert not s_fb.requires_grad
    assert not beta.requires_grad
    assert not beta_c.requires_grad


def test_forward_uses_exact_pullback_once_and_refreshes_after_weight_update():
    block = _make_block()
    x = torch.randn(2, 2, 3, 3)
    activation_calls = []

    def capture_activation(module, inputs, output):
        scale = module._coordinate_pullback_scale
        if scale is not None:
            activation_calls.append(
                (inputs[0].detach().clone(), output.detach().clone(), scale.detach().clone()))

    hook = block.act_fn.register_forward_hook(capture_activation)
    with mock.patch.object(
            block, "prospective_unitless_pullback_scales",
            wraps=block.prospective_unitless_pullback_scales) as scale_method:
        block(x)
        assert scale_method.call_count == 1
    first_beta_c = activation_calls[-1][2]

    for activation_input, activation_output, beta_c in activation_calls:
        expected = block.act_fn(beta_c * activation_input) / beta_c
        assert torch.allclose(activation_output, expected, rtol=1e-5, atol=1e-6)
    assert block.act_fn._coordinate_pullback_scale is None

    activation_calls.clear()
    with torch.no_grad():
        block.FBconv.weight.mul_(2.0)
    block(x)
    second_beta_c = activation_calls[-1][2]
    hook.remove()

    assert torch.allclose(second_beta_c, first_beta_c / 2)



def test_pullback_modes_select_expected_coordinate_scale():
    approx = _make_block(pullback_mode="approx")
    direct = _make_block(pullback_mode="direct")
    none = _make_block(pullback_mode="none")
    direct.load_state_dict(approx.state_dict(), strict=True)
    none.load_state_dict(approx.state_dict(), strict=True)

    _, beta, beta_c = approx.prospective_unitless_pullback_scales()
    assert approx.unitless_measured_pullback_mode == "approx"
    assert direct.unitless_measured_pullback_mode == "direct"
    assert not none.enable_unitless_measured_pullback

    x = torch.randn(2, 2, 3, 3)
    with mock.patch.object(approx.act_fn, "set_coordinate_pullback_scale",
                           wraps=approx.act_fn.set_coordinate_pullback_scale) as set_approx:
        approx(x)
    with mock.patch.object(direct.act_fn, "set_coordinate_pullback_scale",
                           wraps=direct.act_fn.set_coordinate_pullback_scale) as set_direct:
        direct(x)
    with mock.patch.object(none.act_fn, "set_coordinate_pullback_scale",
                           wraps=none.act_fn.set_coordinate_pullback_scale) as set_none:
        none(x)

    assert torch.equal(set_approx.call_args_list[0].args[0], beta_c)
    assert torch.equal(set_direct.call_args_list[0].args[0],
                       beta.new_tensor(direct.unitless_pullback_q))
    assert set_none.call_count == 0


def test_direct_physical_timing_uses_quantization_scales_without_changing_rhs():
    block = _make_block(physical=True, scaling_mode="direct")
    del block.k
    ref = torch.zeros(1, 2, 3, 3)
    expected_z = torch.tensor(10e3 * 49e-15) / block.s_fb
    expected_y = torch.tensor((1.75 * 0.2) * 10e3 * 49e-15) / block.s_ff
    source = torch.randn(1, 2, 3, 3)
    expected_ff_rhs = block.FFconv(source)
    expected_fb_rhs = block.FBconv(source)
    assert torch.allclose(block._apply_averaged_module(block.FFconv, source),
                          expected_ff_rhs)
    assert torch.allclose(block._apply_averaged_module(block.FBconv, source),
                          expected_fb_rhs)

    for cls in (ToggleODEXInitFFFB, TogglePulseODEXInitFFFB,
                TogglePulseBlkXInitFFFB):
        t_z, t_y = cls.resolve_cycle_params(block, ref)
        assert torch.allclose(torch.as_tensor(t_z), expected_z)
        assert torch.allclose(torch.as_tensor(t_y), expected_y)

    summed_rhs = torch.randn_like(source)
    pulse_block = mock.Mock()
    pulse_block.FBconv = object()
    pulse_block.FFconv = object()
    pulse_block.R = 10e3
    pulse_block.C = 49e-15
    pulse_block.C_fb = 49e-15
    pulse_block._apply_pulse_module.return_value = summed_rhs
    pulse_block._apply_spin_variation.side_effect = lambda stage, value: value
    for cls in (TogglePulseODEXInitFFFB, TogglePulseBlkXInitFFFB):
        z_rhs = cls.z_stage_rhs(pulse_block, None, None, source, None)
        y_rhs = cls.y_stage_rhs(pulse_block, None, source, None, None)
        assert torch.allclose(z_rhs, summed_rhs / (pulse_block.R * pulse_block.C_fb))
        assert torch.allclose(y_rhs, summed_rhs / (pulse_block.R * pulse_block.C))


def test_weights_stay_full_precision_and_checkpoint_keys_do_not_change():
    legacy_block = _make_block(enable_pullback=False)
    pullback_block = _make_block(enable_pullback=True)
    legacy_state = legacy_block.state_dict()

    assert set(legacy_state) == set(pullback_block.state_dict())
    pullback_block.load_state_dict(legacy_state, strict=True)
    assert not parametrize.is_parametrized(pullback_block.FFconv, "weight")
    assert not parametrize.is_parametrized(pullback_block.FBconv, "weight")

    ff_before = pullback_block.FFconv.weight.detach().clone()
    fb_before = pullback_block.FBconv.weight.detach().clone()
    pullback_block(torch.randn(2, 2, 3, 3))
    assert torch.equal(ff_before, pullback_block.FFconv.weight)
    assert torch.equal(fb_before, pullback_block.FBconv.weight)


def test_gradients_reach_input_and_full_precision_weights():
    block = _make_block()
    x = torch.randn(2, 2, 3, 3, requires_grad=True)
    block(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.FFconv.weight.grad is not None
    assert block.FBconv.weight.grad is not None
    assert torch.isfinite(block.FFconv.weight.grad).all()
    assert torch.isfinite(block.FBconv.weight.grad).all()


def test_physical_path_ignores_unitless_pullback_option():
    disabled = _make_block(enable_pullback=False, physical=True)
    enabled = _make_block(enable_pullback=True, physical=True)
    enabled.load_state_dict(disabled.state_dict(), strict=True)
    x = torch.randn(2, 2, 3, 3)

    with torch.no_grad():
        expected = disabled(x)
        actual = enabled(x)

    assert torch.equal(actual, expected)
    assert enabled.act_fn._coordinate_pullback_scale is None



def test_piecewise_linear_activation_supports_pullback_training():
    block = _make_block()
    block.act_fn = PiecewiseLinearActivation(
        curve_path=CURVE_PATH, v_dd=0.1, corner="TT")
    x = torch.randn(2, 2, 3, 3, requires_grad=True)

    block(x).square().mean().backward()

    assert block.act_fn._coordinate_pullback_scale is None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.FFconv.weight.grad is not None
    assert block.FBconv.weight.grad is not None
