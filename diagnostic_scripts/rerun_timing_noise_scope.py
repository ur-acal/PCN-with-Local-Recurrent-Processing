#!/usr/bin/env python3
"""Replay historical timing cases bottom-up, preserving per-batch results."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))


def timing_state(model):
    from compute_time_variation import ComputeTimeVariation
    fixed, dynamic = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, ComputeTimeVariation):
            if module.scope != 'per_sample':
                raise AssertionError('Calibration/evaluation timing scope changed: ' + name)
            fixed[name] = dict(module._compute_time_fixed_offsets)
            for key, generator in module._generators.items():
                if key.endswith(':noise'):
                    dynamic[name + '/' + key] = hashlib.sha256(
                        generator.get_state().numpy().tobytes()).hexdigest()
    return fixed, dynamic


def cnn_evaluate():
    sys.argv.remove('--cnn-evaluate')
    import baseline.evaluate_physical_feedforward_cifar as evaluator
    runtime = evaluator.parse_args()
    assert runtime.timing_noise_scope == 'per_sample'
    calibrate, evaluate = evaluator.recalibrate_batchnorm, evaluator.evaluate
    audit = dict(requested_scope=runtime.timing_noise_scope, calibration=None)
    state = {}

    def checked_calibrate(model, loader, device):
        timing_state(model)
        stats = calibrate(model, loader, device)
        state['fixed'], state['dynamic'] = timing_state(model)
        audit['calibration'] = dict(scope='per_sample', **stats)
        return stats

    def checked_evaluate(model, loader, device, max_eval_batches=0):
        before_fixed, before_noise = timing_state(model)
        forwards = [0]
        def check_forward(_model, inputs, _output):
            from compute_time_variation import ComputeTimeVariation
            for module in model.modules():
                if isinstance(module, ComputeTimeVariation):
                    assert module.scope == 'per_sample'
                    assert module.last
                    for unit in module.last.values():
                        assert len(unit['factor']) == inputs[0].shape[0]
            forwards[0] += 1
        handle = model.register_forward_hook(check_forward)
        try:
            accuracy = evaluate(model, loader, device, max_eval_batches)
        finally:
            handle.remove()
        fixed, noise = timing_state(model)
        if audit['calibration'] is not None:
            assert fixed == state['fixed'], 'Fixed variation changed after calibration'
        if runtime.compute_time_noise_std:
            assert noise != before_noise, 'Dynamic timing noise was not resampled'
        audit.update(evaluation_scope='per_sample', test_forwards_verified=forwards[0],
                     fixed_assignments=fixed, fixed_unchanged_after_calibration=(
                         None if audit['calibration'] is None else fixed == state['fixed']),
                     dynamic_timing_stream_advanced=noise != before_noise,
                     accuracy_percent=accuracy)
        (Path(runtime.result_path)/'scope_audit.json').write_text(json.dumps(audit, indent=2))
        print('TIMING_SCOPE_AUDIT', json.dumps({k: v for k, v in audit.items()
                                             if k != 'fixed_assignments'}), flush=True)
        return accuracy

    evaluator.recalibrate_batchnorm = checked_calibrate
    evaluator.evaluate = checked_evaluate
    evaluator.main()


def write_csv(path, rows, fields):
    # Readers always see a complete aggregate, never a partially written CSV.
    with tempfile.NamedTemporaryFile('w', newline='', dir=path.parent,
                                     prefix='.timing-results-', delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path,
                        default=ROOT/'results/tc_compute_time_sweep_20261017/results.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', type=Path,
                        default=ROOT/'results/tc_timing_scope_validation_20261017')
    parser.add_argument('--pcn-validation', type=Path, default=None)
    args = parser.parse_args()
    for model in ('cnn', 'pcn'):
        report = (args.pcn_validation if model == 'pcn' and args.pcn_validation is not None
                  else args.validation/model/'validation.json')
        checked = json.loads(report.read_text())
        if not checked['passed'] or not all(row['passed'] for row in checked['comparisons']
                                            if row.get('required_for_equivalence', True)):
            raise RuntimeError('Full-model timing-scope validation has not passed: ' + model)
    source = args.source_results.resolve()
    with source.open() as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        historical = list(reader)
    rows = [row for row in historical if row.get('timing_noise_scope', 'per_batch') == 'per_batch']
    if len(rows) != 22:
        raise ValueError('Expected the 22 original cases, including four BN controls.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'source_results.csv', historical, fields)
    if 'timing_noise_scope' not in fields:
        fields.append('timing_noise_scope')
        for row in historical:
            row['timing_noise_scope'] = 'per_batch'
    original = json.loads((source.parent/'manifest.json').read_text())
    cases = []
    for index, row in enumerate(reversed(rows)):
        fixed, noise = float(row['fixed_std']), float(row['noise_std'])
        command = list(next(case['command'] for case in original if
            case['model'] == row['model'] and case['fixed_std'] == fixed and case['noise_std'] == noise))
        path = output/f"{index:02d}_{row['model']}_fixed{fixed:g}_noise{noise:g}_bn{int(row['bn_recalibrate']=='true')}"
        set_option(command, '--timing_noise_scope', 'per_sample')
        set_option(command, '--expanded_gaussian_curve_cpu_offload', row['expanded_gaussian_curve_cpu_offload'])
        if row['model'] == 'cnn_resnet44':
            set_option(command, '--result_path', path)
            set_option(command, '--bn_recalibrate', row['bn_recalibrate'])
            if row['bn_recalibrate'] == 'true':
                set_option(command, '--bn_calibration_samples', row['bn_calibration_samples'])
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--cnn-evaluate', *command[3:]]
        else:
            set_option(command, '--tc_metadata_path', path/'trial.jsonl')
        cases.append(dict(command=command, result_path=str(path), source_row=copy.deepcopy(row)))
    (output/'manifest.json').write_text(json.dumps(cases, indent=2))
    write_csv(source, historical, fields)
    completed = []
    for index, case in enumerate(cases):
        path = Path(case['result_path'])
        path.mkdir()
        print('START', index, path.name, flush=True)
        (output/'status.json').write_text(json.dumps(dict(status='running', index=index,
            total=len(cases), case=str(path)), indent=2))
        start = time.monotonic()
        with (path/'evaluation.log').open('x') as log:
            result = subprocess.run(case['command'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (output/'status.json').write_text(json.dumps(dict(status='failed', index=index,
                case=str(path), returncode=result.returncode), indent=2))
            raise RuntimeError(f'{path.name} failed: {result.returncode}')
        if case['source_row']['model'] == 'cnn_resnet44':
            record = json.loads((path/'tc_trial_metadata.json').read_text())
            audit = json.loads((path/'scope_audit.json').read_text())
            assert audit['evaluation_scope'] == 'per_sample'
            if case['source_row']['bn_recalibrate'] == 'true':
                assert audit['calibration']['scope'] == 'per_sample'
                assert audit['fixed_unchanged_after_calibration']
        else:
            trials = (path/'trial.jsonl').read_text().splitlines()
            assert len(trials) == 1
            record = json.loads(trials[0])
        assert record['options']['timing_noise_scope'] == 'per_sample'
        assert math.isfinite(record['accuracy_percent'])
        if float(case['source_row']['fixed_std']) or float(case['source_row']['noise_std']):
            assert record['compute_time_variation'], 'No active timing units recorded'
        for units in record['compute_time_variation'].values():
            assert all(isinstance(unit['factor'], list) for unit in units.values())
        row = dict(case['source_row'], timing_noise_scope='per_sample',
                   accuracy_percent=record['accuracy_percent'], elapsed_seconds=time.monotonic()-start)
        completed.append(row)
        write_csv(output/'results.csv', completed, fields)
        write_csv(source, historical+completed, fields)
        print('DONE', json.dumps(row), flush=True)
    (output/'status.json').write_text(json.dumps(dict(status='complete', cases=len(cases)), indent=2))
    print('PER_SAMPLE_TIMING_SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    cnn_evaluate() if '--cnn-evaluate' in sys.argv else main()
