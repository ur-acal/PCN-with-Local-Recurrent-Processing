from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from inference_utils import replace_transpose_conv
from ode_pc import TogglePulseODEXInitFFFB, ToggleWrapper1State
from pc_conv import PCConvReLU6
from validation import MVMConv, Validator


def s2_block(cls=None, physical=False, method="rk2", **kwargs):
    from ode_pc import S2DTCFFFB, S2DTCQATWrapper2State
    torch.set_num_threads(1)
    torch.manual_seed(37)
    pc = PCConvReLU6(inp_chan=2, out_chan=4, bypass=False, layer_idx=0)
    block = (cls or S2DTCFFFB)(
        pc_conv=pc, method=method, t_end=1.75, t_step=.35,
        toggle_n_cycles=5, toggle_timing_mode="fixed", toggle_y_time=1e-8,
        z_over_y_time=1, toggle_timing_R=5e4, toggle_timing_C=5e-13, **kwargs)
    if physical:
        wrapper = S2DTCQATWrapper2State(
            ode_block=block, R=5e4, C=5e-13, v_dd=.5, state_bound=5,
            w_bits=5, weight_quant_factor_bits=1, thermal_noise=False,
            is_first=True, is_last=True)
    return block


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('noise', [False, True])
@pytest.mark.parametrize('substeps', [1, 2])
def test_s2_pulse_forward_cache_preserves_outputs_gradients_and_rng(checkpoint, noise, substeps):
    from ode_pc import S2DTCPulseFFFB

    class Uncached(dict):
        def get(self, key, default=None):
            return None

        def __setitem__(self, key, value):
            pass

    results = []
    for cached in (False, True):
        block = s2_block(S2DTCPulseFFFB, physical=True,
                         s2_dtc_substeps=substeps,
                         checkpoint_ode_rhs_training=checkpoint,
                         enable_summing_current_noise=noise, summing_noise_seed=17,
                         enable_coupler_noise=noise, coupler_noise_seed=18,
                         enable_spin_variation=noise, spin_variation_seed=19)
        if not cached:
            # Exercise the original uncached slice/noise path without changing
            # the ideal DTC waveform or any Gaussian draw.
            block.enable_dtc_nonideality = True
            block._sample_dtc_window = lambda *args: None
            block._s2_nonlinear_R_metadata = Uncached()
        x = torch.full((2, 2, 3, 3), .03, requires_grad=True)
        with patch.object(block, 'get_pulse_matrix', wraps=block.get_pulse_matrix) as masks:
            with patch.object(block, '_active_coupler_count', wraps=block._active_coupler_count) as counts:
                out = block(x)
                out.square().sum().backward()
        assert masks.call_count == (30 if cached else 150)
        assert counts.call_count == ((30 if cached else 150) if noise else 0)
        grads = [m.parametrizations.weight.original.grad.clone()
                 for m in (block.FFconv, block.FBconv)]
        rng = {key: generator.get_state() for key, generator in {
            **{('sum', k): v for k, v in block._summing_noise_generators.items()},
            **{('coupler', k): v for k, v in block._coupler_noise_generators.items()},
        }.items()}
        results.append((out.detach(), x.grad.clone(), grads, rng, torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for a, b in zip([results[0][1], *results[0][2]], [results[1][1], *results[1][2]]):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
    assert results[0][3].keys() == results[1][3].keys()
    for key in results[0][3]:
        assert torch.equal(results[0][3][key], results[1][3][key])
    assert torch.equal(results[0][4], results[1][4])


@pytest.mark.parametrize('mismatch_type', ['mul', 'add'])
def test_s2_pulse_cached_mask_gradients_survive_wrapper_cleanup(mismatch_type):
    from ode_pc import S2DTCPulseFFFB
    results = []
    for checkpoint in (False, True):
        block = s2_block(S2DTCPulseFFFB, physical=True,
                         checkpoint_ode_rhs_training=checkpoint)
        block.begin_training_pulse_mismatch(.1, mismatch_type)
        x = torch.full((1, 2, 3, 3), .02)
        out = block(x)
        block.end_training_pulse_mismatch()
        out.square().sum().backward()
        results.append((out.detach(), [m.parametrizations.weight.original.grad.clone()
                                      for m in (block.FFconv, block.FBconv)]))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


def test_s2_noise_enabled_masks_preserve_zero_skips_and_replay():
    from tc_nonidealities import TCNoiseLifecycle
    coefficients = [(torch.zeros(2), torch.ones(2)), (torch.zeros(2), torch.zeros(2))]
    outputs = []
    for active in (None, [[False, True], [False, False]]):
        torch.manual_seed(23)
        context = TCNoiseLifecycle(coefficients, {(i, s): None for i in range(2)
                                   for s in ('sum', 'coupler')}, active_sources=active)
        with patch('torch.count_nonzero', wraps=torch.count_nonzero) as checks:
            values = [context._sample(i, torch.zeros(2), coefficients[i]) for i in range(2)]
            repeated = context._sample(0, torch.zeros(2), coefficients[0])
        assert repeated is values[0]
        assert checks.call_count == (4 if active is None else 0)
        outputs.append((values, torch.get_rng_state()))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        assert torch.equal(a, b)
    assert torch.equal(outputs[0][1], outputs[1][1])


def test_s2_pulse_cache_rebuilds_after_weight_update_and_bypasses_dtc_jitter():
    from ode_pc import S2DTCPulseFFFB
    block = s2_block(S2DTCPulseFFFB, physical=True, enable_coupler_noise=True)
    x = torch.full((1, 2, 3, 3), .03)
    with patch.object(block, 'get_pulse_matrix', wraps=block.get_pulse_matrix) as masks:
        first = block(x).detach()
        with torch.no_grad():
            for m in (block.FFconv, block.FBconv):
                m.parametrizations.weight.original.neg_()
        second = block(x).detach()
    assert masks.call_count == 60
    assert not torch.equal(first, second)
    block.enable_dtc_nonideality = True
    with patch.object(block, 'get_pulse_matrix', wraps=block.get_pulse_matrix) as masks:
        with patch.object(block, '_sample_dtc_window', wraps=block._sample_dtc_window) as windows:
            assert torch.isfinite(block(x)).all()
    assert masks.call_count == 150
    assert windows.call_count == 10


def test_s2_nonlinear_R_metadata_is_cached_per_curve_and_reset_per_forward():
    block = s2_block(physical=True)
    block._nonlinear_R_training_pkg = dict(
        lengths=torch.tensor([3, 3]), v_grid=torch.tensor([[-1., 0., 1.]] * 2),
        R_left=torch.full((2, 2), 5e4), R_slope=torch.zeros(2, 2), R=5e4)
    block._nonlinear_R_training_curve_indices.update(FFconv=0, FBconv=1)
    x = torch.full((1, 2, 3, 3), .02)
    for _ in range(3):
        assert torch.equal(block._averaged_nonlinear_R_source(block.FBconv, x), x)
    assert len(block._s2_nonlinear_R_metadata) == 1
    assert all(key[1] == 1 for key in block._s2_nonlinear_R_metadata)
    block._s2_nonlinear_R_metadata[('stale',)] = None
    block(x)
    assert ('stale',) not in block._s2_nonlinear_R_metadata
    assert len(block._s2_nonlinear_R_metadata) == 2


def test_s2_dtc_all_initializations_and_no_reset():
    x = torch.rand(2, 2, 3, 3)
    for yi in ("x", "0"):
        for zi in ("x", "0"):
            b = s2_block(y_init_with=yi, z_init_with=zi)
            y, z = b._s2_dtc_initial_state(x)
            torch.testing.assert_close(y, torch.cat((x, x), 1) if yi == "x" else torch.zeros(2, 4, 3, 3))
            torch.testing.assert_close(z, x if zi == "x" else torch.zeros_like(x))
            out = b(x)
            assert torch.isfinite(out).all()
            if yi == zi == "0":
                assert torch.count_nonzero(out) == 0
            assert b.reset_z is False


@pytest.mark.parametrize('nonlinear', [False, True])
def test_s2_ordering_skew_factorization_and_full_bracket(nonlinear):
    from ode_pc import _S2DTCMixin, PulseLevelMaskImpl
    sigma, a, b = _S2DTCMixin._s2_ordering_modes(15)
    indices = torch.arange(15, dtype=torch.float64)
    expected = (indices[:, None] - indices[None, :]).sign() / 450
    actual = sum(s * (x[:, None] * y[None, :] - y[:, None] * x[None, :])
                 for s, x, y in zip(sigma, a, b))
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-12)
    assert len(sigma) == 7 and (sigma[:-1] >= sigma[1:]).all()

    block = s2_block(physical=True, s2_dtc_ordering_pairs=7).double()
    if nonlinear:
        # Smooth independent current/activation fixture exercises curvature,
        # including gradients THROUGH the directional slopes.
        block.act_fn = nn.Softplus()
        block._averaged_nonlinear_R_source = lambda module, source: source / (1 + .5 * source)
    state = (torch.rand(1, 4, 2, 2, dtype=torch.double) * .02,
             torch.rand(1, 2, 2, 2, dtype=torch.double) * .02)
    weights = block._s2_ordering_weights()
    correction = block._s2_ordering_correction(state, 1e-8, weights)
    def field(k, y, z):
        sources = (block.act_fn(block.project_state(z)), block.project_state(y))
        result = []
        for module, source, scale in zip((block.FFconv, block.FBconv), sources, (block.s_ff, block.s_fb)):
            source = block._averaged_nonlinear_R_source(module, source)
            mask = PulseLevelMaskImpl.apply(module.weight * 15, k, 15)
            conv = torch.nn.functional.conv2d if isinstance(module, nn.Conv2d) else torch.nn.functional.conv_transpose2d
            result.append(conv(source, mask, padding=module.padding) * (1e-8 / (scale * block.R * block.C)))
        return tuple(result)
    reference = tuple(torch.zeros_like(v) for v in state)
    # Independent literal ordered pair sum using JVPs, not the factorized helper.
    for k in range(15):
        for j in range(k):
            fk, fj = field(k, *state), field(j, *state)
            dkj = torch.autograd.functional.jvp(lambda y, z: field(k, y, z), state, fj, create_graph=True)[1]
            djk = torch.autograd.functional.jvp(lambda y, z: field(j, y, z), state, fk, create_graph=True)[1]
            reference = tuple(r + (x-y)/450 for r, x, y in zip(reference, dkj, djk))
    for x, y in zip(correction, reference):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-9)
    params = [m.parametrizations.weight.original for m in (block.FFconv, block.FBconv)]
    g1 = torch.autograd.grad(sum(v.square().sum() for v in correction), params)
    g2 = torch.autograd.grad(sum(v.square().sum() for v in reference), params)
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-8)


@pytest.mark.parametrize('pairs', [1, 2, 4, 7])
def test_s2_ordering_gradients_checkpoint_and_unitless_unchanged(pairs):
    x = torch.rand(2, 2, 3, 3) * .03
    results = []
    for checkpoint in (False, True):
        block = s2_block(physical=True, s2_dtc_ordering_pairs=pairs,
                         checkpoint_ode_rhs_training=checkpoint,
                         enable_summing_current_noise=True, summing_current_p=1e-12,
                         summing_noise_seed=7, enable_spin_variation=True, spin_variation_seed=8)
        output = block(x)
        output.square().sum().backward()
        grads = [module.parametrizations.weight.original.grad for module in (block.FFconv, block.FBconv)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)
        results.append((output, grads))
    torch.testing.assert_close(results[0][0], results[1][0])
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b)
    baseline = s2_block()
    configured = s2_block(s2_dtc_ordering_pairs=pairs)
    torch.testing.assert_close(baseline(x), configured(x), atol=0, rtol=0)


def test_s2_ordering_rejects_unsupported_physical_settings():
    from ode_pc import S2DTCPulseFFFB
    x = torch.rand(1, 2, 3, 3) * .01
    for kwargs in ({'method': 'euler'}, {'s2_dtc_substeps': 2}, {'cls': S2DTCPulseFFFB}):
        b = s2_block(physical=True, s2_dtc_ordering_pairs=1, **kwargs)
        with pytest.raises(ValueError, match='averaged RK2'):
            b(x)


def test_s2_ordering_interval_adds_one_h_squared_term():
    b = s2_block(physical=True, s2_dtc_ordering_pairs=7)
    state = (torch.rand(1, 4, 2, 2) * .02, torch.rand(1, 2, 2, 2) * .02)
    weights = b._s2_ordering_weights()
    full = b._s2_ordering_correction(state, 1e-8, weights)
    half = b._s2_ordering_correction(state, .5e-8, weights)
    assert any(v.abs().max() > 0 for v in full)
    baseline = b._s2_dtc_interval(state, 1e-8)
    corrected = b._s2_dtc_interval(state, 1e-8, ordering_weights=weights)
    for c, h, old, new in zip(full, half, baseline, corrected):
        torch.testing.assert_close(h, c / 4)
        torch.testing.assert_close(new, b.project_state(old + c))


@pytest.mark.parametrize('pairs', [0, 4])
@pytest.mark.parametrize('mismatch_type', ['mul', 'add'])
def test_s2_checkpoint_retains_cleared_forward_mismatch(pairs, mismatch_type):
    results = []
    for checkpoint in (False, True):
        b = s2_block(physical=True, s2_dtc_ordering_pairs=pairs,
                     checkpoint_ode_rhs_training=checkpoint)
        b.begin_training_pulse_mismatch(.1, mismatch_type)
        # Call the physical block directly, then reproduce the QAT wrapper's
        # forward-end cleanup before backward.
        y = b._s2_dtc_forward(torch.full((1, 2, 3, 3), .02))
        b.end_training_pulse_mismatch()
        y.square().sum().backward()
        assert b._training_pulse_mismatch is None
        results.append((y, [m.parametrizations.weight.original.grad for m in (b.FFconv, b.FBconv)]))
    torch.testing.assert_close(results[0][0], results[1][0])
    for a, b in zip(results[0][1], results[1][1]):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Fused kernels require CUDA')
def test_s2_ordering_fused_measured_relu_supports_backward():
    from measured_activation import PiecewiseLinearActivation
    curve = Path(__file__).resolve().parents[1] / 'hardware_data/mc_45_corners/0906_RELU_Voltage/fs_25_2.csv'
    results = []
    for fused in (False, True):
        block = s2_block(physical=True, s2_dtc_ordering_pairs=2, checkpoint_ode_rhs_training=True).cuda()
        block.act_fn = PiecewiseLinearActivation(curve, v_dd=.5, corner='MC18',
            fuse_measured_activation=fused).cuda()
        x = torch.full((2, 2, 3, 3), .03, device='cuda')
        output = block(x)
        output.square().sum().backward()
        grads = [m.parametrizations.weight.original.grad for m in (block.FFconv, block.FBconv)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert block.act_fn.fuse_measured_activation == fused
        results.append((output, grads))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=1e-4)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_s2_dtc_joint_euler_matches_explicit_reference():
    b = s2_block(method="euler")
    x = torch.rand(2, 2, 3, 3) * .05
    y, z = torch.cat((x, x), 1), torch.zeros_like(x)
    for _ in range(5):
        dy, dz = b.FFconv(b.act_fn(z)), b.FBconv(y)
        y, z = y + .4 * dy, z + .4 * dz
    torch.testing.assert_close(b(x), y)
    torch.testing.assert_close(b._last_s2_dtc_z, z)


def test_s2_dtc_fixed_timing_recipe_and_invalid_settings():
    from types import SimpleNamespace
    from train_ode_cifar import fixed_timing_recipe_scales
    args = SimpleNamespace(toggle_timing_mode='fixed', toggle_n_cycles=5,
                           t_end=1.75, R=5e4, C=5e-13, toggle_y_time=1e-8,
                           z_over_y_time=1, ode_block='S2DTCFFFB')
    assert fixed_timing_recipe_scales(args) == pytest.approx((8/7, 8/7))
    args.ode_block = 'ToggleODEXInitFFFB'
    assert fixed_timing_recipe_scales(args) == pytest.approx((8/7, .4))
    with pytest.raises(ValueError, match='positive'):
        s2_block(s2_dtc_substeps=0)
    with pytest.raises(ValueError, match='x or 0'):
        s2_block(y_init_with='invalid')


@pytest.mark.parametrize('block', ['S2DTCFFFB', 'S2DTCPulseFFFB', 'ToggleODEXInitFFFB'])
@pytest.mark.parametrize('enabled', [False, True])
def test_s2_fb_toggle_recipe_selection(block, enabled):
    from types import SimpleNamespace
    from train_ode_cifar import fixed_timing_recipe_scales
    args = SimpleNamespace(toggle_timing_mode='fixed', toggle_n_cycles=5,
                           t_end=1.75, R=67e3, C=500e-15, toggle_y_time=10e-9,
                           z_over_y_time=1, ode_block=block,
                           s2_fb_use_toggle_scale=enabled)
    ff, fb = fixed_timing_recipe_scales(args)
    expected_ff = 10e-9 / (67e3 * 500e-15 * .35)
    expected_fb = 10e-9 / (67e3 * 500e-15)
    assert ff == pytest.approx(expected_ff)
    assert fb == pytest.approx(expected_fb if enabled or block == 'ToggleODEXInitFFFB' else expected_ff)


def test_s2_fb_toggle_recipe_cli_and_environment(monkeypatch):
    import sys
    from train_ode_cifar import get_args, _backfill_recovery_defaults
    from types import SimpleNamespace
    monkeypatch.delenv('VALIDATION_MODE', raising=False)
    monkeypatch.delenv('S2_FB_USE_TOGGLE_SCALE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['train_ode_cifar.py'])
    assert get_args().s2_fb_use_toggle_scale is False
    monkeypatch.setenv('S2_FB_USE_TOGGLE_SCALE', 'true')
    assert get_args().s2_fb_use_toggle_scale is True
    monkeypatch.setattr(sys, 'argv', ['train_ode_cifar.py', '--s2_fb_use_toggle_scale', 'false',
                                     '--scale_train_recipe', '2'])
    args = get_args()
    assert args.s2_fb_use_toggle_scale is False and args.scale_train_recipe == 2.
    monkeypatch.setattr(sys, 'argv', ['train_ode_cifar.py', '--s2_fb_use_toggle_scale', 'true',
                                     '--scale_train_recipe', '2'])
    args = get_args()
    assert args.s2_fb_use_toggle_scale is True and args.scale_train_recipe == 2.
    legacy = SimpleNamespace()
    _backfill_recovery_defaults(legacy)
    assert legacy.s2_fb_use_toggle_scale is False


def test_s2_dtc_full_duty_pulses_match_averaged_on_identical_grid():
    from ode_pc import S2DTCPulseFFFB
    # Full-duty weights remove PWM switching. With identical integration grids,
    # the two implementations must agree (not an averaging-accuracy study).
    average = s2_block(physical=True, s2_dtc_substeps=30)
    pulse = s2_block(S2DTCPulseFFFB, physical=True, s2_dtc_substeps=2)
    with torch.no_grad():
        for block in (average, pulse):
            block.toggle_n_cycles = 1
            for module in (block.FFconv, block.FBconv):
                module.parametrizations.weight.original.fill_(.5)
        x = torch.full((1, 2, 3, 3), .0001)
        torch.testing.assert_close(average(x), pulse(x), rtol=2e-5, atol=1e-7)
        assert torch.all(average.FFconv.weight == 1)
        assert torch.all(pulse.FFconv.weight == 15)
        torch.testing.assert_close(average._last_s2_dtc_z, pulse._last_s2_dtc_z,
                                   rtol=2e-5, atol=1e-7)


def test_s2_dtc_unitless_physical_outputs_and_gradients_match():
    from ode_pc import S2DTCQATWrapper2State
    a, b = s2_block(), s2_block()
    with torch.no_grad():
        for block in (a, b):
            for module in (block.FFconv, block.FBconv):
                module.weight.fill_(1. / 30.)
    wrapper = S2DTCQATWrapper2State(
        ode_block=b, R=5e4, C=5e-13, v_dd=.5, state_bound=5,
        w_bits=5, weight_quant_factor_bits=1, thermal_noise=False,
        is_first=True, is_last=True)
    x = torch.full((2, 2, 3, 3), .001)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, atol=1e-7, rtol=1e-5)
    ya.square().sum().backward()
    yb.square().sum().backward()
    for name in ("FFconv", "FBconv"):
        torch.testing.assert_close(getattr(a, name).weight.grad,
                                   getattr(b, name).parametrizations.weight.original.grad,
                                   atol=1e-7, rtol=1e-4)
    assert wrapper.ode_block.physical


def test_s2_dtc_measured_pullback_matches_physical_activation():
    from measured_activation import PiecewiseLinearActivation
    from ode_pc import S2DTCQATWrapper2State
    path = Path(__file__).resolve().parents[1] / 'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv'
    a = s2_block(unitless_measured_pullback_mode='direct', unitless_pullback_q=.1)
    b = s2_block()
    a.act_fn = PiecewiseLinearActivation(curve_path=path, corner='MC18', v_dd=.5)
    with torch.no_grad():
        for block in (a, b):
            for module in (block.FFconv, block.FBconv):
                module.weight.fill_(1/30)
    S2DTCQATWrapper2State(
        ode_block=b, R=5e4, C=5e-13, v_dd=.5, state_bound=5,
        w_bits=5, weight_quant_factor_bits=1, thermal_noise=False,
        is_first=True, is_last=True, enable_measured_activation=True,
        activation_curve_path=str(path), activation_corner='MC18',
        activation_interpolation='piecewise_linear',
        activation_normalize_positive_endpoint=False, fuse_measured_activation=False)
    x = torch.full((2, 2, 3, 3), .001)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, rtol=2e-5, atol=1e-6)
    ya.sum().backward()
    yb.sum().backward()
    for name in ('FFconv', 'FBconv'):
        torch.testing.assert_close(getattr(a, name).weight.grad,
                                   getattr(b, name).parametrizations.weight.original.grad,
                                   rtol=2e-4, atol=2e-6)


def test_s2_dtc_pulse_and_average_optimizer_updates():
    from ode_pc import S2DTCFFFB, S2DTCPulseFFFB
    for cls in (S2DTCFFFB, S2DTCPulseFFFB):
        b = s2_block(cls, physical=True)
        params = [m.parametrizations.weight.original for m in (b.FFconv, b.FBconv)]
        before = [p.detach().clone() for p in params]
        opt = torch.optim.SGD(params, lr=.01)
        b(torch.rand(2, 2, 3, 3) * .1).square().sum().backward()
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        opt.step()
        assert all(not torch.equal(p, old) for p, old in zip(params, before))


def test_s2_dtc_noise_spin_lifetime_and_clipping():
    b = s2_block(physical=True, enable_spin_variation=True, spin_variation_seed=7,
                 enable_summing_current_noise=True, summing_current_p=1e-12,
                 enable_coupler_noise=True, coupler_noise_p=1e-12,
                 summing_noise_seed=3, coupler_noise_seed=5)
    x = torch.rand(2, 2, 3, 3) * .1
    b.train()
    b(x)
    first = b._spin_factor_y.clone()
    b(x)
    assert not torch.equal(first, b._spin_factor_y)
    b.eval()
    first = b._spin_factor_y.clone()
    y1, y2 = b(x), b(x)
    assert torch.equal(first, b._spin_factor_y)
    assert not torch.equal(y1, y2)
    assert b._last_s2_dtc_z.abs().max() <= .5
    assert y1.abs().max() <= 5.  # Output restored by q=.1.


def test_s2_dtc_adaptive_joint_solve_smoke():
    from ode_pc import S2DTCPulseFFFB
    for cls in (None, S2DTCPulseFFFB):
        b = s2_block(cls, physical=True, method="dopri5", enable_summing_current_noise=True,
                     summing_current_p=1e-12, summing_noise_seed=2)
        y = b(torch.rand(1, 2, 3, 3) * .01)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert torch.isfinite(b.FFconv.parametrizations.weight.original.grad).all()


def test_s2_dtc_rhs_checkpointing_preserves_outputs_and_gradients():
    from ode_pc import S2DTCPulseFFFB
    for cls, physical in ((None, False), (None, True), (S2DTCPulseFFFB, True)):
        outputs, gradients = [], []
        for checkpoint in (False, True):
            b = s2_block(cls, physical=physical, checkpoint_ode_rhs_training=checkpoint,
                         enable_summing_current_noise=physical, summing_noise_seed=7,
                         summing_current_p=1e-12)
            out = b(torch.full((2, 2, 3, 3), .03))
            out.square().sum().backward()
            outputs.append(out.detach())
            gradients.append([p.grad.clone() for p in b.parameters() if p.grad is not None])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_s2_dtc_checkpoint_metadata_and_baked_weight_encoding(tmp_path):
    import copy
    import torch.nn.utils.parametrize as P
    from inference_utils import load_and_prepare_model
    from pc_model import PCNetNoBatchNorm
    from ode_pc import (S2DTCFFFB, S2DTCPulseFFFB, S2DTCQATWrapper2State,
                        S2DTCQATTester2State, make_ode_block, wrap_ode_block,
                        SymQuantizeWeight)
    settings = dict(method="rk2", t_end=1.75, n_steps=5,
                    toggle_timing_mode="fixed", toggle_y_time=1e-8,
                    z_over_y_time=1, toggle_timing_R=5e4, toggle_timing_C=5e-13)
    hardware = dict(R=5e4, C=5e-13, v_dd=.5, one_over_q=5, w_bits=5,
                    weight_quant_factor_bits=1, thermal_noise=False)
    x = torch.full((2, 2, 3, 3), .01)
    for source, pairs in ((S2DTCFFFB, 0), (S2DTCPulseFFFB, 0), (S2DTCFFFB, 2)):
        model = PCNetNoBatchNorm(
            inp_channels=[2, 4], out_channels=[4, 4], max_pool=[False, False],
            num_classes=3, pc_conv_layer=PCConvReLU6, first_bn=False,
            avg_pooling=True, dropout=0, bypass=False, tie_weights=False)
        model.device = torch.device("cpu")
        model.cpu()
        make_ode_block(model, ode_block=source, y_init_with="0", z_init_with="x",
                       s2_dtc_substeps=1 if pairs else 2, s2_dtc_ordering_pairs=pairs, **settings)
        wrap_ode_block(model, ode_wrapper=S2DTCQATWrapper2State,
                       qat_cls=SymQuantizeWeight, **hardware)
        model.eval()
        with torch.no_grad():
            expected = model(x)
        flat = copy.deepcopy(model)
        for module in flat.modules():
            if P.is_parametrized(module, "weight"):
                P.remove_parametrizations(module, "weight", leave_parametrized=True)
        path = tmp_path / (source.__name__ + ".pth")
        torch.save(dict(net=flat.state_dict(), init_args=model.init_args,
                        net_type="PCNetNoBatchNorm"), path)
        for target in (S2DTCFFFB, S2DTCPulseFFFB):
            with torch.no_grad(), patch("torch.cuda.is_available", return_value=False):
                loaded = load_and_prepare_model(
                    str(path), "cpu", pc_conv_layer=PCConvReLU6, fuse_bn=False,
                    noise_level=0, ode_params=dict(ode_block=target, **settings),
                    ode_wrapper_params=dict(ode_wrapper=S2DTCQATTester2State, **hardware))
            loaded.eval()
            assert loaded.PcConvs[0].y_init_with == "0"
            assert loaded.PcConvs[0].z_init_with == "x"
            assert loaded.PcConvs[0].s2_dtc_substeps == (1 if pairs else 2)
            assert loaded.PcConvs[0].s2_dtc_ordering_pairs == (0 if target is S2DTCPulseFFFB else pairs)
            factor = (1 if source is target else
                      15 if target is S2DTCPulseFFFB else 1 / 15)
            for old, new in zip(flat.PcConvs, loaded.PcConvs):
                for name in ("FFconv", "FBconv"):
                    torch.testing.assert_close(getattr(new, name).weight,
                                               getattr(old, name).weight * factor)
            with torch.no_grad():
                actual = loaded(x)
                assert torch.isfinite(actual).all()
                if source is target:
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_s2_dtc_expansion_matches_dense(tmp_path):
    from ode_pc import S2DTCPulseFFFB, S2DTCWrapper2State
    model = nn.Module()
    pc = PCConvReLU6(inp_chan=2, out_chan=2, bypass=False, layer_idx=0)
    replace_transpose_conv(pc)
    block = S2DTCPulseFFFB(
        pc_conv=pc, method="rk2", t_end=1.75, t_step=.35, toggle_n_cycles=5,
        toggle_timing_mode="fixed", toggle_y_time=1e-8, z_over_y_time=1,
        toggle_timing_R=5e4, toggle_timing_C=5e-13)
    model.PcConvs = nn.ModuleList([block])
    model.forward = lambda x: model.PcConvs[0](x)
    model.wrappers = [S2DTCWrapper2State(
        ode_block=block, R=5e4, C=5e-13, v_dd=.5, state_bound=5,
        w_bits=5, weight_quant_factor_bits=1, thermal_noise=False)]
    model.eval()
    x = torch.full((2, 2, 3, 3), .01)
    with torch.no_grad():
        before = model(x)
        expand(model, tmp_path)
        assert isinstance(block.FFconv, MVMConv) and isinstance(block.FBconv, MVMConv)
        torch.testing.assert_close(model(x), before, atol=2e-6, rtol=2e-5)


class ToyPulseModel(nn.Module):
    def __init__(self, nonlinear=False, gaussian=False, joint=False):
        super().__init__()
        from ode_pc import S2DTCFFFB, S2DTCPulseFFFB, S2DTCWrapper2State
        torch.manual_seed(23)
        pc = PCConvReLU6(inp_chan=2, out_chan=2, bypass=False, layer_idx=0)
        # Represent an already quantized best checkpoint, as in the launcher.
        with torch.no_grad():
            for conv in (pc.FFconv, pc.FBconv):
                conv.weight.copy_(torch.randint(-15, 16, conv.weight.shape) / 15.)
                conv.weight.flatten()[0] = 1.
        replace_transpose_conv(pc)
        cls = (S2DTCFFFB if joint == 'average' else
               S2DTCPulseFFFB if joint else TogglePulseODEXInitFFFB)
        block = cls(
            pc_conv=pc, noise_level=0., t_end=1.75, toggle_n_cycles=5,
            method="rk2" if joint else "dopri5",
            odexinit_scaling_mode='direct', toggle_timing_mode='fixed',
            toggle_y_time=1e-8, z_over_y_time=1,
            toggle_timing_R=5e4, toggle_timing_C=5e-13)
        source = Path(__file__).resolve().parents[1] / (
            'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv'
            if gaussian else
            'hardware_data/mc_45_corners/coupler_full_range/fs_25_2.csv')
        self.wrappers = [(S2DTCWrapper2State if joint else ToggleWrapper1State)(
            ode_block=block, R=5e4, R_max=None, C=5e-13, v_dd=.5, one_over_q=5,
            w_bits=5, enob=None, weight_quant_factor_bits=1,
            nonlinear_R=nonlinear, nonlinear_R_table=str(source),
            nonlinear_R_mc_quantity=('resistance' if gaussian else 'conductance'),
            nonlinear_R_curve_sharing='per_coupler',
            nonlinear_R_curve_sampling=(
                'multivariate_gaussian' if gaussian else
                'empirical_with_replacement'),
            nonlinear_R_curve_seed=4096)]
        self.PcConvs = nn.ModuleList([block])

    def forward(self, x):
        return self.PcConvs[0](x)


def expand(model, directory, offload=False):
    x = torch.full((2, 2, 3, 3), .05)
    validator = Validator(
        model, str(directory), 'cpu',
        [(x, torch.zeros(2, dtype=torch.long))], str(directory),
        wrapper=model.wrappers,
        expanded_gaussian_curve_cpu_offload=offload)
    return x, validator


def test_both_pulse_matrices_expand_and_preserve_ideal_output(tmp_path):
    model = ToyPulseModel().eval()
    x = torch.full((2, 2, 3, 3), .05)
    with torch.no_grad():
        before = model(x)
        expand(model, tmp_path)
        block = model.PcConvs[0]
        assert isinstance(block.FFconv, MVMConv)
        assert isinstance(block.FBconv, MVMConv)
        assert not hasattr(block, '_pulse_unrolling')
        torch.testing.assert_close(model(x), before, atol=1e-6, rtol=1e-5)


def test_incomplete_cache_recovers_and_complete_cache_reuses(tmp_path):
    expand(ToyPulseModel(), tmp_path)
    cache, = tmp_path.glob('cache_*/expanded_weights_0.pth')
    data = torch.load(cache, weights_only=False)
    assert set(data) == {'FFconv', 'FBconv'}
    del data['FFconv']  # Historical buggy cache.
    torch.save(data, cache)
    model = ToyPulseModel()
    expand(model, tmp_path)
    assert set(torch.load(cache, weights_only=False)) == {'FFconv', 'FBconv'}
    with patch('validation.conv2d_to_matrix_fixed_padding', side_effect=AssertionError('must reuse cache')):
        expand(ToyPulseModel(), tmp_path)


def test_both_sides_receive_per_coupler_curves(tmp_path):
    for joint in (False, True):
        model = ToyPulseModel(nonlinear=True, joint=joint)
        x, _ = expand(model, tmp_path / str(joint))
        for conv in (model.PcConvs[0].FFconv, model.PcConvs[0].FBconv):
            assert isinstance(conv, MVMConv)
            assert conv.csv_enabled
            assert conv.nonlinear_R_curve_sharing == 'per_coupler'
            assert conv.nonlinear_R_curve_assignment.numel() == conv.mat.values().numel()
            assert conv.nonlinear_R_curve_assignment.unique().numel() > 1
        assert torch.isfinite(model(x)).all()


def test_s2_expanded_average_is_frozen_state_mean_of_pulses(tmp_path):
    model = ToyPulseModel(nonlinear=True, joint='average').eval()
    x, _ = expand(model, tmp_path)
    block = model.PcConvs[0]
    for module in (block.FFconv, block.FBconv):
        matrix = module.mat
        levels = (matrix.values().abs() * 15).round()
        expected = torch.zeros_like(x)
        for k in range(15):
            values = matrix.values().sign() * (levels > k)
            pulse = torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(),
                                           values, size=matrix.shape)
            expected += module.forward_pulse(x, pulse, nominal_R=block.R) / 15
        with patch.object(module, 'forward', side_effect=AssertionError('legacy R-code path')):
            actual = block._apply_averaged_module(module, x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        module.nonlinear_R_curve_assignment.add_(1).remainder_(100)
        changed = block._apply_averaged_module(module, x)
        assert not torch.allclose(actual, changed, rtol=1e-5, atol=1e-8)


def test_gaussian_pulse_pcn_offload_preserves_output(tmp_path):
    baseline = ToyPulseModel(nonlinear=True, gaussian=True).eval()
    baseline_x, _ = expand(baseline, tmp_path / 'baseline')
    expected = baseline(baseline_x)

    offloaded = ToyPulseModel(nonlinear=True, gaussian=True).eval()
    x, validator = expand(offloaded, tmp_path / 'offloaded', offload=True)
    try:
        for conv in (offloaded.PcConvs[0].FFconv,
                     offloaded.PcConvs[0].FBconv):
            assert not conv.training
            assert conv.nonlinear_R_curve_gaussian_R_normalized is None
            assert conv._gaussian_curve_cpu is not None
        torch.testing.assert_close(
            offloaded(x), expected, rtol=0, atol=0)
    finally:
        validator.close_curve_stager()
