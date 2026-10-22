"""Replay order/provenance checks; no real evaluators are launched."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class TimingScopeReplayTests(unittest.TestCase):
    def test_reverse_order_preserves_historical_rows_and_calibration(self):
        spec = importlib.util.spec_from_file_location(
            'scope_replay', ROOT/'diagnostic_scripts/rerun_timing_noise_scope.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Self-contained historical fixture: two nine-case sweeps plus four
        # CNN calibration controls. Never depend on ignored local results.
        originals, manifest = [], []
        for model in ('pcn', 'cnn_resnet44'):
            for fixed in (0., .01, .02):
                for noise in (0., .01, .02):
                    row = dict(model=model, fixed_std=str(fixed), noise_std=str(noise),
                               bn_recalibrate='false', bn_calibration_samples='5120',
                               expanded_gaussian_curve_cpu_offload='true',
                               accuracy_percent='75.0', elapsed_seconds='1.0')
                    originals.append(row)
                    command = ['python', '-u', 'evaluator.py',
                               '--compute_time_fixed_std', str(fixed),
                               '--compute_time_noise_std', str(noise)]
                    manifest.append(dict(model=model, fixed_std=fixed,
                                         noise_std=noise, command=command))
        originals.extend(dict(originals[index], bn_recalibrate='true')
                         for index in (10, 11, 12, 13))
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/'results.csv'
            fields = list(originals[0])
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(originals)
            (directory/'manifest.json').write_text(json.dumps(manifest))
            for name in ('cnn', 'pcn'):
                path = directory/'validation'/name
                path.mkdir(parents=True)
                (path/'validation.json').write_text(json.dumps(dict(
                    passed=True, comparisons=[dict(passed=True)])))

            def evaluate(command, **kwargs):
                calls.append(command)
                def option(key):
                    return command[command.index(key)+1]
                self.assertEqual(option('--timing_noise_scope'), 'per_sample')
                row = originals[-len(calls)]
                values = [command[i+1] for i, item in enumerate(command)
                          if item == '--compute_time_noise_std']
                self.assertEqual(float(values[-1]), float(row['noise_std']))
                # Timing STDs may appear twice in the historical PCN command;
                # parse the last value, as argparse does.
                values = [command[i+1] for i, item in enumerate(command)
                          if item == '--compute_time_fixed_std']
                self.assertEqual(float(values[-1]), float(row['fixed_std']))
                record = dict(options=dict(timing_noise_scope='per_sample'),
                    accuracy_percent=float(row['accuracy_percent']),
                    compute_time_variation={'layer': {'ode': {'factor': [1., 1.]}}})
                if row['model'] == 'cnn_resnet44':
                    path = Path(option('--result_path'))
                    self.assertEqual(option('--bn_recalibrate'), row['bn_recalibrate'])
                    (path/'tc_trial_metadata.json').write_text(json.dumps(record))
                    (path/'scope_audit.json').write_text(json.dumps(dict(
                        evaluation_scope='per_sample', calibration=dict(scope='per_sample'),
                        fixed_unchanged_after_calibration=True)))
                else:
                    Path(option('--tc_metadata_path')).write_text(json.dumps(record)+'\n')
                return SimpleNamespace(returncode=0)

            argv = ['replay', '--source-results', str(source), '--output', str(directory/'replay'),
                    '--validation', str(directory/'validation')]
            with patch.object(module.sys, 'argv', argv), patch.object(module.subprocess, 'run', evaluate):
                module.main()
            self.assertEqual(len(calls), 22)
            self.assertIn('--cnn-evaluate', calls[0])
            self.assertEqual(calls[0][calls[0].index('--bn_recalibrate')+1], 'true')
            with source.open() as stream:
                combined = list(csv.DictReader(stream))
            self.assertEqual(len(combined), 44)
            for before, after in zip(originals, combined[:22]):
                self.assertEqual(before, {key: after[key] for key in before})
                self.assertEqual(after['timing_noise_scope'], 'per_batch')
            self.assertTrue(all(row['timing_noise_scope']=='per_sample' for row in combined[22:]))


if __name__ == '__main__':
    unittest.main()
