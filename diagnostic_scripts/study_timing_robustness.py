#!/usr/bin/env python3
"""Endpoint stability and propagated timing error in the recorded TC models.

Diagnostic only: load the production expanded models, retain their natural
timing scopes, and replace evaluation with paired subset measurements.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_feedforward_propagated_snr import snapshot_rng, restore_rng


def rms(value):
    return value.double().square().mean().sqrt().item()


def relative(reference, value):
    return rms(value - reference) / max(rms(reference), 1e-12)


def endpoints(result, initial):
    refs = initial if isinstance(initial, tuple) else (initial,)
    values = result if isinstance(result, tuple) else (result,)
    assert len(values) == len(refs)
    out = tuple(value[-1] if value.ndim == ref.ndim + 1 else value
                for value, ref in zip(values, refs))
    assert all(a.shape == b.shape for a, b in zip(out, refs))
    return out


def time_options(options, factor):
    """Original physical-duration solve; keep Euler's step count unchanged."""
    start = options['t0']
    result = dict(options, dense_output=False)
    result['t1'] = start + factor * (options['t1'] - start)
    result['t_eval'] = [start + factor * (value - start) for value in options['t_eval']]
    if options.get('h') is not None:
        result['h'] = options['h'] * factor
    return result


@contextmanager
def current_noise(model, enabled):
    saved, seen = [], set()
    for module in model.modules():
        for config in (vars(module), getattr(module, '_tc_noise_cfg', {})):
            for key in ('enable_summing_current_noise', 'enable_coupler_noise'):
                if key in config and (id(config), key) not in seen:
                    seen.add((id(config), key))
                    saved.append((config, key, config[key]))
                    config[key] = bool(config[key]) and enabled
    try:
        yield
    finally:
        for config, key, value in saved:
            config[key] = value


def state_metrics(state, derivative, duration, project, rail):
    increment = duration * derivative
    small_step = .001 * increment
    projected = project(state + small_step)
    outward = ((state >= rail * (1 - 1e-5)) & (derivative > 0) |
               (state <= -rail * (1 - 1e-5)) & (derivative < 0))
    return dict(state_rms_V=rms(state),
        raw_motion_over_state=rms(increment) / max(rms(state), 1e-12),
        projected_motion_over_state=rms(projected - state) / (.001 * max(rms(state), 1e-12)),
        fraction_on_rail=(state.abs() >= rail * (1 - 1e-5)).float().mean().item(),
        outward_drift_energy_fraction=(derivative.double().square() * outward).sum().item() /
            max(derivative.double().square().sum().item(), 1e-30))


def linear_response(module, source, output_shape):
    """Use the actual expanded, signed 5-bit matrix without nonlinear-R."""
    if not hasattr(module, 'mat'):
        raise TypeError('Null-space probe requires the production expanded convolution.')
    matrix = module.mat
    values = getattr(module, 'clean_mat_values', matrix.values())
    clean = torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(),
        values, size=matrix.shape, device=matrix.device, dtype=matrix.dtype)
    return torch.sparse.mm(clean, source.flatten(1).T.contiguous()).T.reshape(output_shape)


def null_probe(owner, y, z, seed):
    activated = owner.act_fn(z)
    generator = torch.Generator(device=z.device).manual_seed(seed)
    clean = linear_response(owner.FFconv, activated, y.shape)
    physical = owner.FFconv(activated)
    shuffled, gaussian, physical_shuffled = [], [], []
    for _ in range(3):
        permutation = torch.randperm(activated[0].numel(), device=z.device, generator=generator)
        permuted = activated.flatten(1)[:, permutation].reshape_as(activated)
        random = torch.randn(activated.shape, device=z.device, dtype=z.dtype, generator=generator)
        axes = tuple(range(1, random.ndim))
        random *= (activated.square().mean(axes, keepdim=True).sqrt() /
                   random.square().mean(axes, keepdim=True).sqrt().clamp_min(1e-20))
        shuffled.append(rms(linear_response(owner.FFconv, permuted, y.shape)))
        gaussian.append(rms(linear_response(owner.FFconv, random, y.shape)))
        physical_shuffled.append(rms(owner.FFconv(permuted)))
    high, low = owner.act_fn(z + owner.v_dd * .001), owner.act_fn(z - owner.v_dd * .001)
    floor = owner.act_fn(torch.full_like(z, -owner.v_dd))
    return dict(activated_z_rms_V=rms(activated),
        z_negative_fraction=(z < 0).float().mean().item(),
        z_on_negative_rail_fraction=(z <= -owner.v_dd * (1-1e-5)).float().mean().item(),
        activation_floor_rms_V=rms(floor),
        activation_above_floor_rms_V=rms(activated-floor),
        activation_at_floor_fraction=((activated-floor).abs() <= 1e-9).float().mean().item(),
        activated_z_zero_fraction=(activated.abs() <= 1e-10).float().mean().item(),
        activation_locally_flat_fraction=((high - low).abs() <= 1e-9).float().mean().item(),
        linear_ff_rms=rms(clean), physical_ff_rms=rms(physical),
        linear_vs_shuffled_response=rms(clean) / max(statistics.mean(shuffled), 1e-20),
        linear_vs_gaussian_response=rms(clean) / max(statistics.mean(gaussian), 1e-20),
        physical_vs_shuffled_response=rms(physical) / max(statistics.mean(physical_shuffled), 1e-20))


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def static_signature(owners):
    digest = hashlib.sha256()
    for _, owner in owners:
        spin = getattr(owner, '_tc_spin_state', owner)
        for name in ('_spin_factor_y', '_spin_factor_z'):
            value = getattr(spin, name, None)
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def study(model, output, label, samples, local_only=False):
    from compute_time_variation import ComputeTimeVariation, configure
    from inference_utils import get_test_data, get_bn_calibration_data
    from bn_recalibration import recalibrate_batchnorm
    from TorchDiffEqPack.odesolver import odesolve

    is_pcn = label == 'pcn_s2_22l'
    loader = get_test_data(test_bs=samples, img_type='rgb', task='cifar10',
                           normalize_student_input=not is_pcn)
    inputs, labels = next(iter(loader))
    inputs = inputs.cuda()
    configure(model, SimpleNamespace())  # No timing perturbation during baseline construction.
    model.eval()
    if not is_pcn:
        stats = recalibrate_batchnorm(model, get_bn_calibration_data(
            bs=128, n_samples=5120, img_type='rgb', task='cifar10',
            normalize_student_input=True), inputs.device)
        (output/'calibration.json').write_text(json.dumps(stats, indent=2))
        print('NOMINAL_BN_CALIBRATION', json.dumps(stats), flush=True)
    owners = [(name, module) for name, module in model.named_modules()
              if getattr(module, '_tc_current_mode', False) or
              (hasattr(module, 'tc_method') and getattr(module, 'physical', False))]
    assert len(owners) == 22, (label, len(owners))
    model(inputs)  # Materialize static draws and named dynamic generators.
    fixed_before = static_signature(owners)
    rng = snapshot_rng(model)
    bn_before = {name: (m.running_mean.clone(), m.running_var.clone())
                 for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
    stack, stage_counts, handles = [], {}, []
    for name, owner in owners:
        def before(module, args, name=name):
            stack.append((name, module))
        def after(module, args, result):
            assert stack.pop()[1] is module
        handles.extend((owner.register_forward_pre_hook(before),
                        owner.register_forward_hook(after, always_call=True)))
    target = 'ode_pc.aca_ode_solve' if is_pcn else 'physical_feedforward_tc.odesolve'

    def identity():
        name, owner = stack[-1]
        stage = stage_counts.get(name, 0)
        stage_counts[name] = stage + 1
        return name + ('/coupled' if is_pcn else ('/conv1' if stage == 0 else '/conv2')), owner

    local = []
    factors = (.6, .8, .9, 1., 1.1, 1.2, 1.4)

    def local_solve(func, initial, options, **kwargs):
        key, owner = identity()
        nominal = odesolve(func, initial, dict(options, dense_output=False), **kwargs)
        reference = endpoints(nominal, initial)
        duration = float(options['t1'] - options['t0'])
        for factor in factors:
            result = nominal if factor == 1 else odesolve(func, initial, time_options(options, factor))
            states = endpoints(result, initial)
            derivatives = func(options['t0'] + duration * factor,
                               states if isinstance(initial, tuple) else states[0])
            derivatives = derivatives if isinstance(derivatives, tuple) else (derivatives,)
            for branch, (state, derivative, ref) in enumerate(zip(states, derivatives, reference)):
                initial_state = initial[branch] if isinstance(initial, tuple) else initial
                row = dict(unit=key, state='y' if branch == 0 else 'z', time_factor=factor,
                           nominal_seconds=duration, relative_endpoint_error=relative(ref, state),
                           initial_state_rms_V=rms(initial_state),
                           initial_to_endpoint_rms_ratio=rms(initial_state)/max(rms(state), 1e-12),
                           net_update_over_endpoint=rms(state-initial_state)/max(rms(state), 1e-12))
                row.update(state_metrics(state, derivative, duration,
                                        options['proj_fn'], owner.v_dd))
                if is_pcn and branch == 0 and factor in (.8, 1., 1.2):
                    row.update(null_probe(owner, *states, 991 + owner.layer_idx))
                local.append(row)
        write_rows(output/'layer_endpoints.csv', local)
        print('LAYER_ENDPOINTS', label, key, flush=True)
        return nominal

    try:
        with current_noise(model, False), patch(target, local_solve):
            restore_rng(rng)
            reference_logits = model(inputs).detach().cpu()
        print('LOCAL_SCAN_COMPLETE', label, flush=True)
        if local_only:
            assert fixed_before == static_signature(owners)
            (output/'study.json').write_text(json.dumps(dict(model=label, samples=samples,
                local_only=True, static_spin_unchanged=True), indent=2))
            print('TIMING_ROBUSTNESS_STUDY_COMPLETE', label, flush=True)
            return
        # Reconfigure factors only, never resample fixed spin/curve assignments.
        propagated, comparisons, records = [], [], []

        for stochastic in (False, True):
            captured = {}
            baseline = []
            def solve_baseline(func, initial, options, **kwargs):
                key, _ = identity()
                options = dict(options, dense_output=False)
                solver = odesolve(func, initial, options, return_solver=True)
                result, steps = solver.integrate(initial, options['t0'],
                    t_eval=options['t_eval'], return_steps=True)
                normalized = {}
                context = solver.tc_context
                if stochastic and context is not None:
                    for (index, branch), raw in context.tape.items():
                        if isinstance(branch, int):
                            a, b = context.coefficients[branch]
                            sigma = torch.hypot(a, b).clamp_min(torch.finfo(raw.dtype).tiny)
                            normalized[index, branch] = (raw / sigma).cpu()
                states = endpoints(result, initial)
                baseline.append(dict(unit=key, steps=torch.stack([
                    torch.as_tensor(v).detach().cpu() for v in steps]),
                    normals=normalized, states=tuple(s.detach().cpu() for s in states)))
                return result

            # Include before/after normalization: no recalibration between pairs.
            norm_handles = []
            for name, norm in model.named_modules():
                if isinstance(norm, torch.nn.BatchNorm2d):
                    def norm_hook(module, args, result, name=name):
                        captured[name + '/before_normalization'] = args[0].detach().cpu()
                        captured[name + '/after_normalization'] = result.detach().cpu()
                    norm_handles.append(norm.register_forward_hook(norm_hook))
            with current_noise(model, stochastic):
                configure(model, SimpleNamespace())
                restore_rng(rng)
                stage_counts.clear()
                with patch(target, solve_baseline):
                    clean = model(inputs).detach().cpu()
                captured_reference = dict(captured)
                for condition, uniform in (('repeat', 1.), ('shorter_20_percent', .8),
                                           ('longer_20_percent', 1.2), ('random_20_percent_each', None)):
                    configure(model, SimpleNamespace(compute_time_fixed_std=.2,
                        compute_time_noise_std=.2, timing_noise_scope='per_sample', data_seed=4096))
                    restore_rng(rng)
                    stage_counts.clear()
                    cursor = [0]
                    captured.clear()
                    factors_used = []

                    def force_factors(timer, unit, ref, nominal):
                        values = torch.full((ref.shape[0],), uniform, device=ref.device, dtype=ref.dtype)
                        return values

                    def solve_perturbed(func, initial, options, **kwargs):
                        key, _ = identity()
                        previous = baseline[cursor[0]]
                        cursor[0] += 1
                        assert previous['unit'] == key
                        options = dict(options, dense_output=False)
                        if stochastic:
                            solver = odesolve(func, initial, options, return_solver=True)
                            context = solver.tc_context
                            if context is not None:
                                context.normal = lambda ref, branch, ctx=context, row=previous: (
                                    row['normals'][ctx.index, branch].to(ref))
                            result = solver.integrate_predefined_grids(initial, options['t0'],
                                predefine_steps=previous['steps'].to(inputs.device), t_eval=options['t_eval'])
                        else:
                            result = odesolve(func, initial, options)
                        for index, (ref, value) in enumerate(zip(previous['states'], endpoints(result, initial))):
                            propagated.append(dict(current_noise=stochastic, condition=condition,
                                location=key + ('/y' if index == 0 else '/z'),
                                reference_rms=rms(ref), error_rms=rms(value.cpu()-ref),
                                relative_error=relative(ref, value.cpu())))
                        return result

                    with patch(target, solve_perturbed):
                        if uniform is None:
                            logits = model(inputs).detach().cpu()
                        else:
                            with patch.object(ComputeTimeVariation, 'sample_factors', force_factors):
                                logits = model(inputs).detach().cpu()
                    assert cursor[0] == len(baseline)
                    for location, ref in captured_reference.items():
                        value = captured[location]
                        propagated.append(dict(current_noise=stochastic, condition=condition,
                            location=location, reference_rms=rms(ref), error_rms=rms(value-ref),
                            relative_error=relative(ref, value)))
                    from compute_time_variation import records as timer_records
                    drawn = timer_records(model)
                    for units in drawn.values():
                        for item in units.values():
                            factors_used.extend(item['factor'])
                    if uniform is not None:
                        factors_used = [uniform] * (len(baseline) * samples)
                    factors_tensor = torch.tensor(factors_used, dtype=torch.float64)
                    row = dict(current_noise=stochastic, condition=condition, samples=samples,
                        max_abs_logit_difference=(logits-clean).abs().max().item(),
                        relative_logit_error=relative(clean, logits),
                        prediction_changes=int((logits.argmax(1) != clean.argmax(1)).sum()),
                        reference_accuracy_percent=100*clean.argmax(1).eq(labels).float().mean().item(),
                        perturbed_accuracy_percent=100*logits.argmax(1).eq(labels).float().mean().item(),
                        timing_units=len(baseline), realized_factor_mean=factors_tensor.mean().item(),
                        realized_factor_std=factors_tensor.std().item(),
                        grid='matched normalized grids' if stochastic else 'native solver grids')
                    comparisons.append(row)
                    records.append(dict(**row, reference_logits=clean.tolist(), logits=logits.tolist(),
                                        timing_draws=drawn))
                    print('PROPAGATED_TIMING', json.dumps(row), flush=True)
                    write_rows(output/'propagated_errors.csv', propagated)
                    write_rows(output/'comparisons.csv', comparisons)
            for handle in norm_handles:
                handle.remove()
            del baseline
        configure(model, SimpleNamespace())
        assert fixed_before == static_signature(owners), 'Static spin variation changed'
        for name, norm in model.named_modules():
            if name in bn_before:
                a, b = bn_before[name]
                assert torch.equal(norm.running_mean, a) and torch.equal(norm.running_var, b)
        (output/'study.json').write_text(json.dumps(dict(model=label, samples=samples,
            normalized_input=not is_pcn, static_nonidealities='all enabled and held fixed',
            cnn_bn='recalibrated at nominal timing, then frozen' if not is_pcn else 'none',
            native_timing_scope='one factor per coupled PCN block' if is_pcn else 'one factor per convolution',
            static_spin_unchanged=True, bn_unchanged=True, comparisons=records), indent=2))
        print('TIMING_ROBUSTNESS_STUDY_COMPLETE', label, flush=True)
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('pcn_s2_22l', 'cnn_resnet44'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--local-only', action='store_true', help='Only the isolated-layer endpoint probes.')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    import compute_time_variation
    production_configure = compute_time_variation.configure
    import validate_timing_noise_scope as loader
    def callback(model, _inputs, output, label):
        # PCN's loader intercepts configure() to hand the loaded model to us.
        # Restore the real function inside the callback: study() configures
        # several timing conditions and must not re-enter the model loader.
        with patch.object(compute_time_variation, 'configure', production_configure):
            study(model, output, label, args.samples, args.local_only)
    set_option = loader.set_option
    def memory_safe_option(command, key, value):
        # Curve staging changes storage only; both expanded models exceed 13 GiB
        # of static curve data. Keep the same curves, but stage them by layer.
        set_option(command, key, 'true' if key == '--expanded_gaussian_curve_cpu_offload' else value)
    # Reuse the exact production-model loader used by the validated scope test.
    sys.argv = [sys.argv[0], '--model', args.model, '--output', str(args.output)]
    with patch.object(loader, 'validate', callback), patch.object(loader, 'set_option', memory_safe_option):
        loader.main()


if __name__ == '__main__':
    main()
