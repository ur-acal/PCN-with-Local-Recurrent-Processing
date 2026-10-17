#!/usr/bin/env python3
"""After a recorded evaluation succeeds, measure coupler-only propagated error.

Reuses the ordinary evaluator's construction and expanded cache. Only its
dataset evaluation callback is replaced, locally, with matched one-batch passes.
No production model/solver implementation is patched.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except FileNotFoundError:
        return None


def metric(reference, perturbed):
    reference = reference.double()
    error = perturbed.double() - reference
    signal_power = reference.square().mean().item()
    error_power = error.square().mean().item()
    if not math.isfinite(signal_power) or not math.isfinite(error_power):
        raise RuntimeError('Non-finite activation or error power in propagated SNR.')
    snr = (math.inf if error_power == 0 else -math.inf if signal_power == 0
           else 10 * math.log10(signal_power / error_power))
    return dict(signal_rms=math.sqrt(signal_power),
                error_rms=math.sqrt(error_power), snr_db=snr,
                max_abs_error=error.abs().max().item(),
                relative_rms_error=(math.sqrt(error_power / signal_power)
                                    if signal_power else (0. if not error_power else math.inf)))


def snapshot_rng(model):
    import numpy as np
    import torch
    generators = {}
    for module in model.modules():
        for value in vars(module).values():
            values = value.values() if isinstance(value, dict) else (value,)
            for candidate in values:
                if isinstance(candidate, torch.Generator):
                    generators[id(candidate)] = candidate
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            [(generator, generator.get_state())
             for generator in generators.values()])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])
    for generator, value in state[4]:
        generator.set_state(value)


def paired_snr(model, inputs):
    import torch
    from torch import nn
    norms = [(name, module) for name, module in model.named_modules()
             if isinstance(module, nn.BatchNorm2d)]
    owners = [module for module in model.modules()
              if hasattr(module, 'enable_coupler_noise')]
    if model.training or not norms or not owners:
        raise ValueError('Expected an eval-mode physical CNN with BatchNorm.')
    if any(module.training or not module.track_running_stats for _, module in norms):
        raise ValueError('Every BN must use frozen running statistics.')
    original_flags = [(owner, owner.enable_coupler_noise) for owner in owners]
    bn_state = {name: {key: value.detach().cpu().clone()
                       for key, value in module.state_dict().items()}
                for name, module in norms}
    reference, control, noisy = {}, {}, {}
    destination = reference
    handles = []

    def hook(name):
        def capture(module, args, output):
            destination[name] = (args[0].detach().to('cpu', copy=True),
                                 output.detach().to('cpu', copy=True))
        return capture

    try:
        with torch.no_grad():
            # Materialize static hardware draws and lazy named RNG streams at
            # the actual batch shape before snapshotting. No BN recalibration.
            for owner in owners:
                owner.enable_coupler_noise = True
            model(inputs)
            rng = snapshot_rng(model)
            for name, module in norms:
                handles.append(module.register_forward_hook(hook(name)))
            logits = []
            for enabled, target in ((False, reference), (False, control), (True, noisy)):
                restore_rng(rng)
                destination = target
                for owner in owners:
                    owner.enable_coupler_noise = enabled
                logits.append(model(inputs).detach().cpu())
                print('Captured coupler=' + str(enabled), flush=True)
            identical = torch.equal(logits[0], logits[1]) and all(
                torch.equal(a, b) for name in reference
                for a, b in zip(reference[name], control[name]))
            for name, module in norms:
                if any(not torch.equal(value.cpu(), bn_state[name][key])
                       for key, value in module.state_dict().items()):
                    raise RuntimeError('BN state changed: ' + name)
            rows = []
            floor_ratios = []

            def floor_ratio(repeat_error, coupler_error):
                return repeat_error / coupler_error if coupler_error else (
                    0. if not repeat_error else math.inf)

            for name, _ in norms:
                row = dict(layer=name)
                for index, label in enumerate(('before_bn', 'after_bn')):
                    effect = metric(reference[name][index], noisy[name][index])
                    repeat = metric(reference[name][index], control[name][index])
                    row.update({label + '_' + key: value for key, value in effect.items()})
                    row.update({label + '_repeat_' + key: value for key, value in repeat.items()})
                    ratio = floor_ratio(repeat['error_rms'], effect['error_rms'])
                    row[label + '_repeat_to_coupler_error_ratio'] = ratio
                    floor_ratios.append(ratio)
                rows.append(row)
            logits_effect = metric(logits[0], logits[2])
            logits_repeat = metric(logits[0], logits[1])
            floor_ratios.append(floor_ratio(logits_repeat['error_rms'], logits_effect['error_rms']))
            worst_ratio = max(floor_ratios)
            summary = dict(logits=metric(logits[0], logits[2]),
                           reference_repeat_logits=logits_repeat,
                           prediction_changes=int((logits[0].argmax(1) !=
                                                   logits[2].argmax(1)).sum()),
                           samples=inputs.shape[0], reference_repeat_bit_identical=identical,
                           maximum_repeat_to_coupler_error_ratio=worst_ratio,
                           numerical_floor_below_one_percent=(worst_ratio <= 0.01),
                           frozen_bn_unchanged=True)
            return rows, summary, logits[0], logits[2]
    finally:
        for handle in handles:
            handle.remove()
        for owner, enabled in original_flags:
            owner.enable_coupler_noise = enabled


def run(args):
    # Wait using PID + start time to avoid following an unrelated reused PID.
    print(f'WAITING_FOR_EULER pid={args.wait_pid}', flush=True)
    while process_start_ticks(args.wait_pid) == args.wait_start_ticks:
        time.sleep(15)
    manifest = json.loads(args.manifest.read_text())
    case = next(case for case in manifest['cases'] if case['name'] == 'euler10')
    metadata_path = ROOT / case['result_path'] / 'tc_trial_metadata.json'
    if not metadata_path.is_file():
        raise RuntimeError('Euler exited without successful evaluation metadata.')
    metadata = json.loads(metadata_path.read_text())
    if not math.isfinite(metadata['accuracy_percent']):
        raise RuntimeError('Euler returned nonfinite accuracy.')
    if not re.search(r'^trial 0: accuracy=[0-9.]+%',
                     (ROOT / case['log']).read_text(), re.M):
        raise RuntimeError('Euler did not print its successful trial completion marker.')
    print('EULER_COMPLETE accuracy=' + str(metadata['accuracy_percent']), flush=True)
    from baseline import evaluate_physical_feedforward_cifar as evaluator
    import torch
    with patch.object(sys, 'argv', ['propagated-snr', *case['command'][3:]]):
        config = evaluator.parse_args()
    if config.bn_recalibrate or config.tc_method != 'euler' or config.tc_num_steps != 10:
        raise ValueError('Expected frozen-BN Euler10 configuration.')
    if getattr(args, 'other_nonidealities', 'on') == 'off':
        for name in ('enable_spin_variation', 'enable_summing_current_noise',
                     'enable_measured_activation', 'enable_measured_pooling',
                     'enable_nonlinear_R'):
            setattr(config, name, False)
        config.expanded_gaussian_curve_cpu_offload = False
    config.result_path = str(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'configuration.json').write_text(json.dumps(vars(config), indent=2))

    @torch.no_grad()
    def measure(model, loader, device, max_eval_batches=0):
        inputs, targets = next(iter(loader))
        rows, summary, clean, noisy = paired_snr(model, inputs.to(device))
        summary.update(reference_accuracy_percent=100 * clean.argmax(1).eq(targets).float().mean().item(),
                       perturbed_accuracy_percent=100 * noisy.argmax(1).eq(targets).float().mean().item(),
                       description='One batch, accumulated coupler-noise effect; all other settings held fixed.')
        with (args.output / 'propagated_snr.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f"{'Layer':48} {'Before BN dB':>14} {'After BN dB':>14}", flush=True)
        for row in rows:
            print(f"{row['layer']:48} {row['before_bn_snr_db']:14.3f} {row['after_bn_snr_db']:14.3f}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        if not summary['numerical_floor_below_one_percent']:
            raise RuntimeError('Reference-repeat error exceeds 1% of coupler-induced error; see saved diagnostics.')
        return summary['perturbed_accuracy_percent']

    # Diagnostic callback only: construction, preprocessing, expansion, noise,
    # solver, hooks, and offload lifecycle remain the normal evaluator's code.
    with patch.object(evaluator, 'evaluate', measure):
        evaluator.evaluate_once(config)
    print('PROPAGATED_SNR_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-start-ticks', type=int, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--other-nonidealities', choices=('on', 'off'), default='on')
    run(parser.parse_args())
