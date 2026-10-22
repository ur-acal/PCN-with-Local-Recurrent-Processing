#!/usr/bin/env python3
"""Full-test-set timing sweep through the unmodified production evaluators."""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cnn_command(output):
    from baseline.evaluate_physical_feedforward_cifar import parse_args
    source = ROOT / 'results/tc_resnet44_q32_last_euler10_expanded/tc_trial_metadata.json'
    options = json.loads(source.read_text())['options']
    with patch.object(sys, 'argv', ['eval', '--model_name', 'resnet44_cifar',
                                   '--checkpoint', options['checkpoint']]):
        defaults = vars(parse_args())
    command = [sys.executable, '-u', 'baseline/evaluate_physical_feedforward_cifar.py']
    options.update(result_path=str(output), compute_time_fixed_std=0., compute_time_noise_std=0.)
    for key, value in options.items():
        if value is None and defaults[key] is None:
            continue
        command.extend(['--' + key, 'none' if value is None else
                        str(value).lower() if isinstance(value, bool) else str(value)])
    return command


def pcn_command(output):
    root = ROOT / 'saved_ckpt_runs/tc_c10_s2_22L_nonorm_rgbmid_rpf_pspin'
    checkpoints = list(root.rglob('*_full_param_last_ckpt.pth'))
    if len(checkpoints) != 1:
        raise RuntimeError('Expected exactly one final PCN checkpoint.')
    checkpoint = checkpoints[0]
    env = dict(os.environ, MODEL_NAME=checkpoint.parent.name, MODEL_DIR=str(root),
               CKPT='full_param_last', TC_STATE='2', TASK='cifar10', IMG_TYPE='rgb',
               TC_DRY_RUN='true', TEST_BS='128', N_TRIALS='1',
               TC_MAX_EVAL_BATCHES='0', EXPANDED_GAUSSIAN_CURVE_CPU_OFFLOAD='true',
               TC_METADATA_PATH=str(output/'trial.jsonl'))
    command = shlex.split(subprocess.check_output(
        ['bash', 'launch_scripts/run_tc_nonidealities.sh', 'eval'],
        cwd=ROOT, env=env, text=True))
    return [sys.executable, '-u', *command[command.index('ode_inference.py'):]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    rows = []
    # Nine matched single-trial evaluations per model, no partial dataset limits.
    for model in ('cnn_resnet44', 'pcn_s2_22l'):
        for fixed, noise in itertools.product((0., .1, .2), repeat=2):
            name = f'{model}_fixed{fixed:g}_noise{noise:g}'
            case_dir = output/name
            case_dir.mkdir()
            command = (cnn_command(case_dir) if model == 'cnn_resnet44' else
                       pcn_command(case_dir))
            command.extend(['--compute_time_fixed_std', str(fixed),
                            '--compute_time_noise_std', str(noise)])
            case = dict(model=model, fixed_std=fixed, noise_std=noise,
                        command=command, result_path=str(case_dir))
            manifest.append(case)
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
            print('START', name, shlex.join(command), flush=True)
            start = time.monotonic()
            with (case_dir/'evaluation.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed with exit {result.returncode}; see {case_dir}/evaluation.log')
            if model == 'cnn_resnet44':
                data = json.loads((case_dir/'tc_trial_metadata.json').read_text())
            else:
                trials = (case_dir/'trial.jsonl').read_text().splitlines()
                if len(trials) != 1:
                    raise RuntimeError('Expected exactly one completed PCN trial.')
                data = json.loads(trials[0])
            if not math.isfinite(data['accuracy_percent']):
                raise RuntimeError(f'{name} returned nonfinite accuracy.')
            if (fixed or noise) and not data.get('compute_time_variation'):
                raise RuntimeError(f'{name} did not record active timing variation.')
            row = dict(model=model, fixed_std=fixed, noise_std=noise,
                       accuracy_percent=data['accuracy_percent'],
                       elapsed_seconds=time.monotonic()-start)
            rows.append(row)
            with (output/'results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print('DONE', json.dumps(row), flush=True)
    print('COMPUTE_TIME_SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
