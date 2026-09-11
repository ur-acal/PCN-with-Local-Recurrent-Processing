"""Local/SLURM orchestration for the pinned WRN rows; no training logic here."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from copy import copy

from baseline.wrn_control_specs import DATASETS, ROWS, SIZES, model_name, model_options, training_override

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {
    'max_additive': ('additive', 'max_abs', '0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1'),
    'multiplicative': ('multiplicative', 'max_abs', '0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4'),
    'rms_additive': ('additive', 'rms', '0.25,0.5,0.75,1.0,1.25'),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', default='3')
    p.add_argument('--datasets', default=','.join(DATASETS))
    p.add_argument('--sizes', default=','.join(SIZES))
    p.add_argument('--stage', choices=('train', 'test', 'train-test'), default='train')
    p.add_argument('--parallelism', type=int, default=4)
    p.add_argument('--eval-parallelism', type=int, default=3)
    p.add_argument('--phased', action='store_true', help='Train group first, then evaluate models concurrently one condition at a time')
    p.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data')
    p.add_argument('--output-root', type=Path, default=ROOT / 'logs/wrn_controls_local')
    p.add_argument('--seed', type=int, default=4096)
    p.add_argument('--mismatch-seed', type=int, default=123)
    p.add_argument('--conditions', default='max_additive,multiplicative')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--simulate', action='store_true', help='Run fake subprocesses in OUTPUT_ROOT/SIMULATED only')
    p.add_argument('--simulate-failure', default='', choices=('', 'train', 'missing-checkpoint', 'evaluation', 'wrong-model', 'incomplete-evaluation'))
    args = p.parse_args(argv)
    args.rows = list(map(int, args.rows.split(',')))
    args.datasets = args.datasets.split(',')
    args.sizes = args.sizes.split(',')
    args.conditions = args.conditions.split(',')
    for selected, allowed in ((args.rows, ROWS), (args.datasets, DATASETS),
                              (args.sizes, SIZES), (args.conditions, LEVELS)):
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            p.error(f'Invalid or duplicate selection: {selected}')
    if args.parallelism < 1 or args.eval_parallelism < 1:
        p.error('parallelism must be positive')
    if args.simulate_failure and not args.simulate:
        p.error('Failure injection requires --simulate')
    args.output_root = args.output_root.resolve()
    if args.simulate:
        args.output_root = args.output_root / 'SIMULATED'
    args.data_dir = args.data_dir.resolve()
    return args


def paths(args, row, dataset, size):
    name = model_name(row, size)
    directory = args.output_root / f'row{row}' / dataset / size
    run = f'custom_noresize_{dataset}_{name}'
    checkpoint = directory / 'checkpoints' / dataset / 'custom_noresize' / name / run / f'{run}_best_ckpt.pth'
    return directory, checkpoint


def training_command(args, row, dataset, size):
    directory, _ = paths(args, row, dataset, size)
    return [sys.executable, '-u', 'baseline/train_baseline_cifar.py',
            '--model_name', model_name(row, size), '--dataset', dataset,
            '--data_dir', str(args.data_dir), '--output_dir', str(directory / 'checkpoints'),
            '--case', 'custom_noresize', '--pretrained', 'false', '--seed', str(args.seed),
            '--distill_method', 'none', '--override', training_override(row)]


def test_command(args, row, dataset, size, condition):
    directory, checkpoint = paths(args, row, dataset, size)
    kind, scale, levels = LEVELS[condition]
    bn = ROWS[row][0]
    script = 'run_wrn_bn_recalibration_experiment.py' if bn else 'run_wrn_nobn_mismatch_experiment.py'
    command = [sys.executable, '-u', 'baseline/' + script,
               '--output_dir', str(directory / 'evaluation' / condition),
               '--data_dir', str(args.data_dir), '--checkpoint_override', str(checkpoint),
               '--model_name_override', model_name(row, size), '--datasets', dataset,
               '--architectures', 'WRN_' + size, '--mismatch_types', kind,
               '--additive_scale_mode', scale, '--noise_levels', levels, '--noisy_trials', '10',
               '--seed', str(args.mismatch_seed), '--batch_size', '128', '--num_workers', '2',
               '--case', 'custom_noresize']
    if bn:
        command += ['--mode', 'full', '--standalone', '--pcn_reference_policy', 'none',
                    '--calibration_num_samples', '5120', '--calibration_batch_size', '128',
                    '--calibration_subset_seed', '20240618', '--calibration_num_workers', '2',
                    '--mismatch_parameter_policy', 'existing']
    return command


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checkpoint_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def execute(command, log):
    # Partial output is never mistaken for completion or silently overwritten.
    with log.open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f'Exit {returncode}; inspect {log}')


def execution_command(args, command):
    if not args.simulate:
        return command
    return [sys.executable, '-u', '-m', 'baseline.simulate_wrn_control',
            '--command-json', json.dumps(command), '--failure', args.simulate_failure]


def validate_results(output, command):
    """Check identity and complete level/trial coverage before claiming success."""
    def value(flag):
        return command[command.index(flag) + 1]
    levels = [float(x) for x in value('--noise_levels').split(',')]
    count = int(value('--noisy_trials'))
    expected = {(level, trial) for level in levels for trial in range(count)}
    with (output / 'full_per_trial.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {(float(r['mismatch_level']), int(r['trial'])) for r in rows} != expected:
        raise ValueError(f'Incomplete or duplicate level/trial coverage: {output}')
    bn = '--standalone' in command
    columns = ('frozen_bn_accuracy', 'recalibrated_bn_accuracy') if bn else ('accuracy',)
    for row in rows:
        for field, flag in (('dataset', '--datasets'), ('architecture', '--architectures'),
                            ('model_name', '--model_name_override'), ('checkpoint', '--checkpoint_override'),
                            ('mismatch_type', '--mismatch_types'), ('additive_scale_mode', '--additive_scale_mode')):
            if row[field] != value(flag):
                raise ValueError(f'Evaluation identity mismatch for {field}: {output}')
        seed = int(value('--seed')) + 1000 * round(float(row['mismatch_level']) * 1e6) + int(row['trial'])
        if int(row['mismatch_seed']) != seed:
            raise ValueError(f'Evaluation seed mismatch: {output}')
        if any(not math.isfinite(float(row[c])) or not 0 <= float(row[c]) <= 100 for c in columns):
            raise ValueError(f'Invalid accuracy: {output}')
    with (output / 'full_aggregate.csv').open(newline='') as handle:
        aggregate = list(csv.DictReader(handle))
    if len(aggregate) != len(levels):
        raise ValueError(f'Incomplete aggregate: {output}')
    json.loads((output / 'full_summary.json').read_text())
    return {name: checkpoint_hash(output / name) for name in
            ('full_per_trial.csv', 'full_aggregate.csv', 'full_summary.json')}


def run_evaluation(args, row, dataset, size, condition, saved):
    directory, checkpoint = paths(args, row, dataset, size)
    command = test_command(args, row, dataset, size, condition)
    output = directory / 'evaluation' / condition
    output.mkdir(parents=True, exist_ok=True)
    done = output / 'complete.json'
    expected = dict(command=command, checkpoint_sha256=saved['sha256'], simulated=args.simulate)
    if done.exists():
        record = json.loads(done.read_text())
        if record['identity'] != expected or record['artifacts'] != validate_results(output, command):
            raise ValueError(f'Evaluation configuration or results changed: {done}')
        return
    write_json(output / 'state.json', dict(status='running', started=time.time(), identity=expected))
    try:
        if checkpoint_hash(checkpoint) != saved['sha256']:
            raise ValueError(f'Checkpoint changed before evaluation: {checkpoint}')
        execute(execution_command(args, command), output / 'run.log')
        if checkpoint_hash(checkpoint) != saved['sha256']:
            raise ValueError(f'Checkpoint changed during evaluation: {checkpoint}')
        artifacts = validate_results(output, command)
        write_json(done, dict(identity=expected, artifacts=artifacts))
        write_json(output / 'state.json', dict(status='complete', finished=time.time(), identity=expected))
    except Exception as exc:
        write_json(output / 'state.json', dict(status='failed', error=str(exc), identity=expected))
        raise


def run_one(args, row, dataset, size):
    directory, checkpoint = paths(args, row, dataset, size)
    train = training_command(args, row, dataset, size)
    if args.dry_run:
        commands = ([train] if args.stage != 'test' else [])
        if args.stage != 'train':
            commands += [test_command(args, row, dataset, size, c) for c in args.conditions]
        return dict(row=row, dataset=dataset, size=size, commands=[shlex.join(c) for c in commands])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(row=row, dataset=dataset, size=size, model=model_name(row, size),
                        options=model_options(row), training_command=train)
        manifest = directory / 'manifest.json'
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if existing['identity'] != identity or existing.get('simulated', False) != args.simulate:
                raise ValueError(f'Configuration differs from existing run: {manifest}')
        else:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
            write_json(manifest, dict(identity=identity, git_commit=revision, git_status=dirty,
                                     repo_root=str(ROOT), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                     simulated=args.simulate))
        def state(status, **extra):
            write_json(directory / 'state.json', dict(status=status, updated=time.time(), **extra))
        try:
            trained = directory / 'train_complete.json'
            if not trained.exists():
                if args.stage == 'test':
                    raise RuntimeError(f'Training completion not verified: {directory}')
                state('training')
                execute(execution_command(args, train), directory / 'train.log')
                if (not checkpoint.is_file() or 'Train finished' not in (directory / 'train.log').read_text()
                        or list((directory / 'checkpoints').rglob('training_collapse.json'))):
                    raise RuntimeError(f'Training failed completion/collapse checks: {directory}')
                write_json(trained, dict(checkpoint=str(checkpoint), sha256=checkpoint_hash(checkpoint)))
            saved = json.loads(trained.read_text())
            if not checkpoint.is_file() or checkpoint_hash(checkpoint) != saved['sha256']:
                raise RuntimeError(f'Completed checkpoint is missing or changed: {checkpoint}')
            if args.stage != 'train':
                state('testing', conditions=args.conditions, eval_parallelism=args.eval_parallelism)
                errors = []
                with ThreadPoolExecutor(max_workers=args.eval_parallelism) as pool:
                    futures = {pool.submit(run_evaluation, args, row, dataset, size, c, saved): c
                               for c in args.conditions}
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            errors.append(f'{futures[future]}: {exc}')
                if errors:
                    raise RuntimeError('; '.join(errors))
            state('complete', stage=args.stage)
            return dict(row=row, dataset=dataset, size=size, status='complete')
        except Exception as exc:
            state('failed', error=str(exc))
            raise


def main():
    args = parse_args()
    jobs = [(r, d, s) for r in args.rows for d in args.datasets for s in args.sizes]
    print(f'{len(jobs)} model jobs; parallelism={args.parallelism}; stage={args.stage}', flush=True)
    if not args.dry_run:
        from baseline.wrn_control_artifacts import create_plan
        create_plan(args.output_root, args.rows, args.datasets, args.sizes, args.conditions, args.stage, args.simulate)
    failures = 0
    phases = [(args.stage, args.conditions)]
    if args.phased:
        phases = ([('train', [])] if args.stage != 'test' else [])
        if args.stage != 'train':
            phases += [('test', [condition]) for condition in args.conditions]
    failed_jobs = set()
    for phase, conditions in phases:
        phase_args = copy(args)
        phase_args.stage = phase
        phase_args.conditions = conditions
        successful = []
        workers = args.eval_parallelism if args.phased and phase == 'test' else args.parallelism
        if args.phased:
            phase_args.eval_parallelism = 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, phase_args, *job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                    successful.append(futures[future])
                except Exception as exc:
                    failures += 1
                    failed_jobs.add(futures[future])
                    print(f'FAILED {futures[future]}: {exc}', flush=True)
        if phase == 'train':
            jobs = successful
    for job in failed_jobs:
        directory, _ = paths(args, *job)
        if not args.dry_run:
            write_json(directory / 'state.json', dict(status='failed', updated=time.time(),
                       error='One or more stages failed; inspect train.log and evaluation condition states'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
