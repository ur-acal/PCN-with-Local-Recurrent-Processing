"""Wait for the Claim 1/2 queue, then recover original WRN paired BN trial data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from baseline.wrn_control_specs import SIZES, DATASETS

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {'max_additive': [n / 100 for n in range(11)],
          'rms_additive': [.25, .5, .75, 1., 1.25],
          'multiplicative': [n / 100 for n in range(0, 41, 5)]}


def plan(output):
    tasks = []
    for condition, levels in LEVELS.items():
        for dataset in DATASETS:
            for size in SIZES:
                name = f'wrn_{size}_cifar'
                run = f'custom_noresize_{dataset}_{name}'
                checkpoint = ROOT / 'checkpoint/baselines' / dataset / 'custom_noresize' / name / run / (run + '_best_ckpt.pth')
                directory = output / condition / dataset / ('WRN_' + size)
                command = [sys.executable, '-u', 'baseline/run_wrn_bn_recalibration_experiment.py',
                           '--mode', 'full', '--standalone', '--pcn_reference_policy', 'none',
                           '--datasets', dataset, '--architectures', 'WRN_' + size,
                           '--model_name_override', name, '--checkpoint_override', str(checkpoint),
                           '--output_dir', str(directory), '--data_dir', str(ROOT.parent / 'data'),
                           '--case', 'custom_noresize', '--device', 'cuda', '--batch_size', '128',
                           '--num_workers', '2', '--noisy_trials', '10', '--seed', '123',
                           '--calibration_num_samples', '5120', '--calibration_batch_size', '128',
                           '--calibration_subset_seed', '20240618', '--calibration_num_workers', '2',
                           '--mismatch_parameter_policy', 'existing',
                           '--mismatch_types', 'multiplicative' if condition == 'multiplicative' else 'additive',
                           '--additive_scale_mode', 'rms' if condition == 'rms_additive' else 'max_abs',
                           '--noise_levels', ','.join(map(str, levels))]
                tasks.append(dict(condition=condition, dataset=dataset, size=size,
                                  checkpoint=str(checkpoint), output=str(directory), command=command,
                                  status='queued'))
    return tasks


def dependency_complete(path):
    state = json.loads(path.read_text())
    tasks = [t for wave in state['waves'] for t in wave]
    if state['status'] == 'failed' or any(t['status'] == 'failed' for t in tasks):
        raise RuntimeError('Claim queue failed; baseline evaluation remains blocked')
    if state['status'] == 'complete':
        if len(tasks) != 192 or any(t['status'] != 'complete' for t in tasks):
            raise RuntimeError('Claim queue completion is inconsistent')
        return True
    os.kill(state['pid'], 0)  # Detect an exited controller instead of waiting forever.
    return False


def validate(task):
    path = Path(task['output'])
    with (path / 'full_per_trial.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(round(l, 8), t) for l in LEVELS[task['condition']] for t in range(10)}
    actual = [(round(float(r['mismatch_level']), 8), int(r['trial'])) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError(f'Incomplete trials: {path}')
    for r in rows:
        if r['dataset'] != task['dataset'] or r['architecture'] != 'WRN_' + task['size']:
            raise RuntimeError(f'Wrong model identity: {path}')
        for col in ('frozen_bn_accuracy', 'recalibrated_bn_accuracy'):
            value = float(r[col])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise RuntimeError(f'Invalid accuracy: {path}')
    with (path / 'full_aggregate.csv').open() as stream:
        if len(list(csv.DictReader(stream))) != len(LEVELS[task['condition']]):
            raise RuntimeError(f'Incomplete aggregate: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'logs/wrn_baseline_unfused_missing_trials')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output_root.resolve()
    tasks = plan(output)
    for task in tasks:
        if not Path(task['checkpoint']).is_file():
            raise FileNotFoundError(task['checkpoint'])
    if not args.dependency.is_file():
        raise FileNotFoundError(args.dependency)
    if args.dry_run:
        print(json.dumps(tasks, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='waiting_for_claims', pid=os.getpid(), dependency=str(args.dependency),
                 parallelism=8, tasks=tasks)

    def record():
        temporary = output / 'state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'state.json')
        (output / 'progress.md').write_text(
            '# Original WRN Missing Trial Data\n\n'
            f'Status: {state["status"]}; completed {sum(t["status"] == "complete" for t in tasks)}/24.\n\n'
            'Unfused frozen and recalibrated BN; 10 trials per level; eight concurrent models.\n'
            'Order: max-additive, RMS-additive, multiplicative. Raw results stay in this directory.\n')

    def run(task):
        directory = Path(task['output'])
        directory.mkdir(parents=True)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (directory / 'evaluation.log').open('x') as log:
            subprocess.run(task['command'], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        validate(task)

    record()
    try:
        while not dependency_complete(args.dependency):
            time.sleep(30)
        for condition in LEVELS:
            state['status'] = 'running_' + condition
            wave = [t for t in tasks if t['condition'] == condition]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(run, t): t for t in wave}
                for t in wave:
                    t['status'] = 'running'
                record()
                while futures:
                    for future, task in list(futures.items()):
                        if not future.done():
                            continue
                        error = future.exception()
                        task['status'] = 'failed' if error else 'complete'
                        if error:
                            task['error'] = str(error)
                        del futures[future]
                        record()
                    if futures:
                        time.sleep(5)
            if any(t['status'] == 'failed' for t in wave):
                raise RuntimeError('Evaluation wave failed; later conditions not launched')
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
