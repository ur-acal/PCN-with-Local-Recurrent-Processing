#!/usr/bin/env python3
"""Compare real expanded models with per-sample and batch-one physical solves.

The stochastic comparison shares Gaussian innovations and normalized step grids;
independent adaptive SDE paths are not expected to have matching logits. The
deterministic comparison lets both solvers select their own native grids.
"""
import argparse
import copy
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ValidationComplete(Exception):
    pass


def set_option(command, key, value):
    if key in command:
        command[command.index(key) + 1] = str(value)
    else:
        command.extend((key, str(value)))


@torch.no_grad()
def validate(model, inputs, output, label):
    from compute_time_variation import ComputeTimeVariation
    from TorchDiffEqPack.odesolver import odesolve
    timers = {name: module for name, module in model.named_modules()
              if isinstance(module, ComputeTimeVariation)}
    assert timers and all(timer.scope == 'per_sample' for timer in timers.values())
    # Only disable dynamic current diffusion in the native-grid comparison.
    flags = []
    for module in model.modules():
        configs = [vars(module)]
        if hasattr(module, '_tc_noise_cfg'):
            configs.append(module._tc_noise_cfg)
        for config in configs:
            for key in ('enable_summing_current_noise', 'enable_coupler_noise'):
                if key in config:
                    flags.append((config, key, config[key]))
    target = ('physical_feedforward_tc.odesolve' if label == 'cnn_resnet44'
              else 'ode_pc.aca_ode_solve')
    results = []
    phases = [(False, 1e-6), (True, 1e-6)]
    if label == 'pcn_s2_22l':
        phases.insert(1, (False, 1e-7))
    for stochastic, tolerance in phases:
        for config, key, value in flags:
            config[key] = value if stochastic else False
        for timer in timers.values():
            timer.scope = 'per_sample'
            timer.reset()  # Same per-image factors in the tolerance comparison.
        solves = []

        def candidate_solve(func, y0, options, **kwargs):
            options = dict(options, rtol=tolerance, atol=tolerance, dense_output=False)
            solver = odesolve(func, y0, options, return_solver=True)
            result, steps = solver.integrate(y0, options['t0'],
                                             t_eval=options['t_eval'], return_steps=True)
            context = solver.tc_context
            normals = {}
            if context is not None:
                for (index, branch), raw in context.tape.items():
                    if isinstance(branch, int):
                        a, b = context.coefficients[branch]
                        total = (a.square() + b.square()).sqrt().clamp_min(
                            torch.finfo(raw.dtype).tiny)
                        normals[index, branch] = (raw / total).clone()
            solves.append(dict(steps=[torch.as_tensor(s).detach().clone() for s in steps],
                               normals=normals, nominal_end=float(options['t1'])))
            return result

        with patch(target, candidate_solve):
            actual = model(inputs).detach()
        assignments = {name: copy.deepcopy(timer.last) for name, timer in timers.items()}
        for name, timer in timers.items():
            assert all(len(values['factor']) == inputs.shape[0]
                       for values in assignments[name].values())
            timer.scope = 'per_batch'
        references = []
        for image in range(inputs.shape[0]):
            factors_by_timer = {id(timer): assignments[name] for name, timer in timers.items()}
            cursor = [0]

            def forced_factor(timer, unit):
                item = factors_by_timer[id(timer)][unit]
                value = float(item['factor'][image])
                timer.last[unit] = dict(fixed_fraction=item['fixed_fraction'],
                    noise_fraction=item['noise_fraction'][image], factor=value)
                return value

            def reference_solve(func, y0, options, **kwargs):
                options = dict(options, rtol=tolerance, atol=tolerance, dense_output=False)
                case = solves[cursor[0]]
                cursor[0] += 1
                if not stochastic:
                    return odesolve(func, y0, options, **kwargs)
                solver = odesolve(func, y0, options, return_solver=True)
                context = solver.tc_context
                if context is not None:
                    context.normal = lambda ref, branch, ctx=context, row=case: (
                        row['normals'][ctx.index, branch][image:image + 1])
                # This reference still uses the original physical-duration RHS
                # and diffusion; only the grid is paired to match Brownian paths.
                scale = float(options['t1']) / case['nominal_end']
                steps = torch.stack(case['steps']) * scale
                steps[-1] = solver.t1
                return solver.integrate_predefined_grids(
                    y0, options['t0'], predefine_steps=steps, t_eval=options['t_eval'])

            with patch.object(ComputeTimeVariation, 'factor', forced_factor), patch(target, reference_solve):
                references.append(model(inputs[image:image + 1]).detach())
            assert cursor[0] == len(solves)
        reference = torch.cat(references)
        error = actual - reference
        rms = error.square().mean().sqrt().item()
        ref_rms = reference.square().mean().sqrt().item()
        peak = reference.abs().max().item()
        row = dict(model=label, images=inputs.shape[0], fixed_std=.2, noise_std=.2,
            solver_tolerance=tolerance,
            current_noise=stochastic,
            reference_grid='paired physical grid' if stochastic else 'native adaptive/fixed grid',
            max_abs_logit_difference=error.abs().max().item(), rms_logit_difference=rms,
            relative_rms_difference=rms/max(ref_rms, 1e-12), reference_logit_peak=peak,
            identical_predictions=bool(torch.equal(actual.argmax(1), reference.argmax(1))),
            candidate_logits=actual.cpu().tolist(), reference_logits=reference.cpu().tolist())
        row['passed'] = (row['relative_rms_difference'] <= 1e-3 and
                         row['max_abs_logit_difference'] <= 1e-3*max(1., peak) and
                         row['identical_predictions'])
        # Production-tolerance independent grids characterize mesh error. The
        # tighter solve must pass the original numerical threshold, as must the
        # paired stochastic paths. Never relax the logit threshold after failure.
        row['required_for_equivalence'] = label != 'pcn_s2_22l' or stochastic or tolerance == 1e-7
        results.append(row)
        print('LOGIT_VALIDATION', json.dumps({k: v for k, v in row.items() if not k.endswith('logits')}), flush=True)
    for config, key, value in flags:
        config[key] = value
    summary = dict(passed=all(row['passed'] for row in results if row['required_for_equivalence']),
                   comparisons=results)
    (output / 'validation.json').write_text(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise AssertionError('Per-sample logits failed the predeclared tolerance; do not launch sweep.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('cnn_resnet44', 'pcn_s2_22l'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'results/tc_compute_time_sweep_20261017/manifest.json').read_text())
    command = list(next(case['command'] for case in manifest if
                       case['model'] == args.model and case['fixed_std'] == .2 and case['noise_std'] == .2))
    set_option(command, '--timing_noise_scope', 'per_sample')
    set_option(command, '--expanded_gaussian_curve_cpu_offload',
               'true' if args.model == 'pcn_s2_22l' else 'false')
    set_option(command, '--batch_size' if args.model == 'cnn_resnet44' else '--test_bs', '4')
    set_option(command, '--result_path' if args.model == 'cnn_resnet44' else '--tc_metadata_path',
               output if args.model == 'cnn_resnet44' else output/'unused_trial.jsonl')
    (output/'command.json').write_text(json.dumps(command, indent=2))
    if args.model == 'cnn_resnet44':
        import baseline.evaluate_physical_feedforward_cifar as evaluator
        def evaluate(model, loader, device, max_eval_batches=0):
            validate(model, next(iter(loader))[0][:4].to(device), output, args.model)
            raise ValidationComplete
        sys.argv = [command[2], *command[3:]]
        with patch.object(evaluator, 'evaluate', evaluate):
            try:
                evaluator.main()
            except ValidationComplete:
                pass
    else:
        import compute_time_variation
        configure = compute_time_variation.configure
        def configure_and_validate(model, runtime_args, trial=0):
            configure(model, runtime_args, trial)
            from inference_utils import get_test_data
            loader = get_test_data(test_bs=4, img_type='rgb', task='cifar10',
                                   normalize_student_input=False)
            validate(model, next(iter(loader))[0].cuda(), output, args.model)
            raise ValidationComplete
        sys.argv = [command[2], *command[3:]]
        with patch.object(compute_time_variation, 'configure', configure_and_validate):
            try:
                runpy.run_path(str(ROOT/'ode_inference.py'), run_name='__main__')
            except ValidationComplete:
                pass


if __name__ == '__main__':
    main()
