"""Fake subprocess adapter. Never imports torch, trains a model, or reports real accuracy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(command, failure):
    def value(flag):
        return command[command.index(flag) + 1]
    training = command[2].endswith('train_baseline_cifar.py')
    output = Path(value('--output_dir'))
    if 'SIMULATED' not in output.parts:
        raise ValueError('Simulation may only write beneath a SIMULATED directory')
    output.mkdir(parents=True, exist_ok=True)
    lifecycle = dict(simulated=True, started=time.time(), command=command)
    try:
        time.sleep(0.15)
        if training:
            name, dataset, case = value('--model_name'), value('--dataset'), value('--case')
            if failure == 'train':
                raise RuntimeError('SIMULATED training failure')
            run_name = f'{case}_{dataset}_{name}'
            cfg_dir = output / dataset / case / name
            checkpoint = cfg_dir / run_name / f'{run_name}_best_ckpt.pth'
            checkpoint.parent.mkdir(parents=True)
            cfg = dict(simulated=True, model_name=name, dataset=dataset, case=case,
                       seed=value('--seed'), override=value('--override'))
            (cfg_dir / 'baseline_config.json').write_text(json.dumps(cfg, indent=2))
            if failure != 'missing-checkpoint':
                identity = dict(cfg, model_name='WRONG_MODEL' if failure == 'wrong-model' else name)
                checkpoint.write_text(json.dumps(identity))
            print('SIMULATED ONLY: no model was trained; accuracy below is a placeholder')
            print('----- Train finished, Model Name: ' + run_name + ' -----')
            print('----- Best top1: 0.0, Best top5: 0.0, Best epoch: 300 -----')
            print('----- Model path: ' + str(checkpoint) + ' -----')
            return
        name, dataset = value('--model_name_override'), value('--datasets')
        checkpoint = Path(value('--checkpoint_override'))
        identity = json.loads(checkpoint.read_text())
        if not identity.get('simulated') or identity['model_name'] != name or identity['dataset'] != dataset:
            raise ValueError('SIMULATED checkpoint/task identity mismatch')
        if failure == 'evaluation' and output.name == 'max_additive':
            raise RuntimeError('SIMULATED mismatch failure')
        bn = '--standalone' in command
        rows = []
        levels = [float(x) for x in value('--noise_levels').split(',')]
        trials = int(value('--noisy_trials'))
        for level in levels:
            for trial in range(trials):
                row = dict(dataset=dataset, architecture=value('--architectures'), model_name=name,
                           checkpoint=str(checkpoint), mismatch_type=value('--mismatch_types'),
                           additive_scale_mode=value('--additive_scale_mode'), mismatch_level=level,
                           mismatch_seed=int(value('--seed')) + 1000 * round(level * 1e6) + trial,
                           trial=trial, simulated=True)
                if bn:
                    row.update(frozen_bn_accuracy=0.0, recalibrated_bn_accuracy=0.0, paired_recovery=0.0,
                               calibration_sample_count=5120, calibration_subset_seed=20240618,
                               calibration_batch_size=128)
                else:
                    row.update(accuracy=0.0)
                rows.append(row)
        if failure == 'incomplete-evaluation' and output.name == 'max_additive':
            rows.pop()
        write_csv(output / 'full_per_trial.csv', rows)
        aggregate = []
        for level in levels:
            row = dict(dataset=dataset, architecture=value('--architectures'),
                       mismatch_type=value('--mismatch_types'), mismatch_level=level,
                       num_trials=trials, simulated=True)
            if bn:
                row.update(frozen_bn_mean_accuracy=0.0, recalibrated_bn_mean_accuracy=0.0)
            else:
                row.update(mean_accuracy=0.0)
            aggregate.append(row)
        write_csv(output / 'full_aggregate.csv', aggregate)
        if bn:
            write_csv(output / 'full_clean_sanity.csv', [dict(simulated=True, clean_original_bn_accuracy=0.0,
                                                           clean_recalibrated_bn_accuracy=0.0)])
        with checkpoint.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        (output / 'full_summary.json').write_text(json.dumps(dict(
            simulated=True, model_name=name, checkpoint_sha256=digest, command=command,
            accuracy_warning='PLACEHOLDERS ONLY; never use these values in research results'), indent=2))
        print(f'SIMULATED evaluation complete: {len(rows)} rows; BN paired={bn}')
    finally:
        lifecycle['finished'] = time.time()
        (output / 'simulation_lifecycle.json').write_text(json.dumps(lifecycle, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command-json', required=True)
    parser.add_argument('--failure', default='')
    args = parser.parse_args()
    run(json.loads(args.command_json), args.failure)


if __name__ == '__main__':
    main()
