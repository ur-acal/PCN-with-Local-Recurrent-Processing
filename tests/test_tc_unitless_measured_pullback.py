import copy
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from measured_activation import PiecewiseLinearActivation
from ode_pc import (ODEXInitFFFB, QATWrapper1State, QATWrapper2State,
                    S2NoisyIYAsXZAs0)
from pc_conv import PCConvReLU6
from train_ode_cifar import _measured_activation_resampling_enabled


CURVE_PATH = (
    Path(__file__).resolve().parents[1] / "hardware_data" / "relu_0p3mV.csv")


def _pc_conv():
    torch.manual_seed(19)
    return PCConvReLU6(
        inp_chan=2, out_chan=2, kernel_size=1, padding=0, cls=2,
        bypass=False, tie_weights=False, tie_bp=False)


def _measured_activation():
    return PiecewiseLinearActivation(
        curve_path=CURVE_PATH, v_dd=0.1, corner="TT",
        fuse_measured_activation=False)


def _state1(mode="approx"):
    block = ODEXInitFFFB(
        pc_conv=_pc_conv(), noise_level=0.0, method="euler",
        t_end=0.2, t_step=0.1, tol=1e-4,
        unitless_measured_pullback_mode=mode,
        unitless_pullback_q=0.1, unitless_pullback_k=1e3,
        unitless_pullback_R=10e3)
    block.act_fn = _measured_activation()
    return block


def _state2(mode="direct"):
    block = S2NoisyIYAsXZAs0(
        pc_conv=_pc_conv(), noise_level=0.0, method="euler",
        t_end=0.2, t_step=0.1, tol=1e-4,
        unitless_measured_pullback_mode=mode,
        unitless_pullback_q=0.1, unitless_pullback_k=1e3,
        unitless_pullback_R=10e3)
    block.act_fn = _measured_activation()
    return block


def _explicit_pullback(activation, value, scale):
    activation.set_coordinate_pullback_scale(scale)
    try:
        return activation(value)
    finally:
        activation.set_coordinate_pullback_scale(None)


def _put_weights_on_five_bit_grid(block):
    grid = torch.tensor(
        [1.0, 3.0 / 15.0, -2.0 / 15.0, 1.0 / 15.0],
        dtype=block.FFconv.weight.dtype).reshape_as(block.FFconv.weight)
    with torch.no_grad():
        block.FFconv.weight.copy_(grid)
        block.FBconv.weight.copy_(grid.flip(0))


def _physical_five_bit(block, state):
    wrapper_cls = QATWrapper1State if state == 1 else QATWrapper2State
    wrapper = wrapper_cls(
        ode_block=block, state_bound=1.0, R=10e3, R_max=150e3,
        C=49e-15, v_dd=0.1, w_bits=5, thermal_noise=False,
        tc_nonidealities=False, nonlinear_R=False,
        enable_spin_variation=False, enable_summing_current_noise=False,
        enable_coupler_noise=False, enable_measured_activation=True,
        activation_curve_path=CURVE_PATH, activation_corner="TT",
        activation_interpolation="piecewise_linear",
        activation_normalize_positive_endpoint=False,
        fuse_measured_activation=False, is_first=True, is_last=True)
    return wrapper.get_ode_block()


def test_pullback_mode_enables_random_curve_resampling_without_extra_flag():
    args = mock.Mock(
        enable_measured_activation=False,
        unitless_measured_pullback_mode="approx")
    assert _measured_activation_resampling_enabled(args)


def test_state1_approx_uses_beta_c_once_per_solve():
    block = _state1()
    y = torch.randn(2, 2, 3, 3)
    expected_scale = (
        0.1 * 1e3 / 10e3 / block.FBconv.weight.detach().abs().max())
    controller = block._unitless_measured_pullback

    with mock.patch.object(
            controller, "scale_for_solve",
            wraps=controller.scale_for_solve) as resolve:
        rhs = block._make_ode_fn(y)
        actual_first = rhs(torch.tensor(0.0), y)
        actual_second = rhs(torch.tensor(0.1), y)

    assert resolve.call_count == 1
    z = block.FBconv(y)
    expected = block.FFconv(
        _explicit_pullback(block.act_fn, z, expected_scale))
    assert torch.allclose(actual_first, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual_first, actual_second)
    assert block.act_fn._coordinate_pullback_scale is None


def test_state1_approx_refreshes_after_weight_update():
    block = _state1()
    x = torch.randn(1, 2, 2, 2)
    first = block._unitless_measured_pullback.scale_for_solve(block)
    with torch.no_grad():
        block.FBconv.weight.mul_(2)
    second = block._unitless_measured_pullback.scale_for_solve(block)

    assert torch.allclose(second, first / 2)
    block(x).square().mean().backward()
    assert block.FFconv.weight.grad is not None
    assert block.FBconv.weight.grad is not None
    assert torch.isfinite(block.FFconv.weight.grad).all()
    assert torch.isfinite(block.FBconv.weight.grad).all()


def test_pullback_is_reapplied_during_checkpoint_backward_replay():
    block = _state1()
    x = torch.randn(1, 2, 2, 2, requires_grad=True)
    rhs = block._make_ode_fn(x)

    output = checkpoint(
        lambda value: rhs(torch.tensor(0.0), value), x,
        use_reentrant=True)
    assert block.act_fn._coordinate_pullback_scale is None
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.act_fn._coordinate_pullback_scale is None


def test_state2_direct_uses_q_once_per_solve():
    block = _state2()
    y = torch.randn(2, 2, 3, 3)
    z = torch.randn_like(y)
    controller = block._unitless_measured_pullback

    with mock.patch.object(
            controller, "scale_for_solve",
            wraps=controller.scale_for_solve) as resolve:
        rhs = block._make_ode_fn(y)
        actual = rhs(torch.tensor(0.0), (y, z))

    expected_h = _explicit_pullback(
        block.act_fn, z, z.new_tensor(0.1))
    assert resolve.call_count == 1
    assert torch.allclose(actual[0], block.FFconv(expected_h),
                          rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[1], block.FBconv(y))
    assert block.act_fn._coordinate_pullback_scale is None


@pytest.mark.parametrize("state", (1, 2))
def test_pullback_matches_five_bit_physical_coordinates(state):
    unitless = _state1() if state == 1 else _state2()
    _put_weights_on_five_bit_grid(unitless)
    physical = _physical_five_bit(copy.deepcopy(unitless), state)
    x = torch.tensor(
        [[[[0.02, -0.01], [0.01, 0.0]],
          [[-0.01, 0.02], [0.0, 0.01]]]])

    unitless.eval()
    physical.eval()
    with torch.no_grad():
        expected = unitless(x)
        actual = physical(x)

    # The physical solve integrates sub-nanosecond float32 times, so the
    # fixed-grid arithmetic is equivalent up to its expected roundoff.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-5)


def test_disabled_modes_preserve_original_rhs_exactly():
    state1 = _state1(mode="none")
    y = torch.randn(1, 2, 3, 3)
    rhs1 = state1._make_ode_fn(y)
    assert torch.equal(
        rhs1(torch.tensor(0.0), y),
        state1.FFconv(state1.act_fn(state1.FBconv(y))))

    state2 = _state2(mode="none")
    state2.load_state_dict(state1.state_dict(), strict=False)
    z = torch.randn_like(y)
    rhs2 = state2._make_ode_fn(y)
    actual = rhs2(torch.tensor(0.0), (y, z))
    assert torch.equal(actual[0], state2.FFconv(state2.act_fn(z)))
    assert torch.equal(actual[1], state2.FBconv(y))


@pytest.mark.parametrize("marker", ("physical", "_physical_coordinate_wrapper"))
def test_physical_blocks_bypass_unitless_pullback(marker):
    state1 = _state1()
    state2 = _state2()
    y = torch.randn(1, 2, 3, 3)
    z = torch.randn_like(y)
    setattr(state1, marker, True)
    setattr(state2, marker, True)

    actual1 = state1._make_ode_fn(y)(torch.tensor(0.0), y)
    actual2 = state2._make_ode_fn(y)(torch.tensor(0.0), (y, z))

    assert torch.equal(
        actual1, state1.FFconv(state1.act_fn(state1.FBconv(y))))
    assert torch.equal(actual2[0], state2.FFconv(state2.act_fn(z)))
    assert torch.equal(actual2[1], state2.FBconv(y))


def test_tc_blocks_reject_the_wrong_coordinate_policy():
    with pytest.raises(ValueError, match="required"):
        _state1(mode="direct")
    with pytest.raises(ValueError, match="required"):
        _state2(mode="approx")


def test_checkpoint_keys_do_not_change():
    state1_none = _state1(mode="none")
    state1_approx = _state1(mode="approx")
    state2_none = _state2(mode="none")
    state2_direct = _state2(mode="direct")

    assert set(state1_none.state_dict()) == set(state1_approx.state_dict())
    assert set(state2_none.state_dict()) == set(state2_direct.state_dict())
