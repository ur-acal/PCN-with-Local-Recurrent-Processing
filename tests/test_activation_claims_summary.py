import importlib.util
from pathlib import Path
import tempfile
import unittest
import csv
import json

SPEC = importlib.util.spec_from_file_location('claims_summary', Path(__file__).resolve().parents[1] /
                                            'baseline/generate_activation_claims_summary.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SummaryTest(unittest.TestCase):
    def test_replacement_changes_all_c2_but_not_c1(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, replacement = Path(tmp) / 'original', Path(tmp) / 'new' / 'model'
            units = [f'PcConvs.{i}' for i in range(7)]
            model = dict(spec=dict(family='pcn'), units=units, checkpoint_sha256='test')
            for directory, claims, error, acc in ((original, (1, 2), 1., 70.), (replacement, (2,), 2., 71.)):
                directory.mkdir(parents=True)
                (directory / 'model.json').write_text(json.dumps(model))
                (directory / 'state.json').write_text(json.dumps(dict(status='complete', smoke=False)))
                rows = []
                for claim in claims:
                    for trial in range(2):
                        for unit in (units if claim == 1 else units + ['final_feature', 'logits']):
                            rows.append(dict(claim=claim, trial=trial, condition='multiplicative', level=.3,
                                samples=10000, target=unit if claim == 1 else 'network', bn_policy='no_bn',
                                layer=unit, dataset_relative_l2=error, zero_reference_images=0))
                with (directory / 'per_trial.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                for trial in range(2):
                    path = directory / f'claim2/network/multiplicative/level_0.3/trial_{trial}/no_bn'
                    path.mkdir(parents=True)
                    (path / 'summary.json').write_text(json.dumps(dict(samples=10000,
                        noisy_accuracy=acc, noisy_correct=int(acc * 100), correct_to_wrong=100,
                        wrong_to_correct=20, centered_logits=dict(dataset_relative_l2=.5), margin_degradation=.2)))
            (replacement.parent / 'provenance.json').write_text(json.dumps(dict(pcn_full=True,
                deterministic_algorithms=True, original=str(original))))
            _, values, _ = M.read_run(original, 'multiplicative', .3, trials=2, pcn_c2_directory=replacement)
            self.assertEqual(values['no_bn']['mean1'], (1., 0.))
            self.assertEqual(values['no_bn']['mean2'], (2., 0.))
            self.assertEqual(values['no_bn']['accuracy'], (71., 0.))
            self.assertEqual(values['no_bn']['centered'], (.5, 0.))

    def test_stage_means_exclude_stem_and_average_per_trial(self):
        for n in (2, 4):
            units = [f'unit{i}' for i in range(1 + 3 * n)]
            metrics = {(1, 'no_bn', t, unit): float(i + 2 * t)
                       for t in range(2) for i, unit in enumerate(units)}
            metrics[1, 'no_bn', 0, units[0]] = 999.
            values = M.stage_means(metrics, units, 1, 'no_bn', 2)
            for stage in range(3):
                expected = 1 + stage * n + (n - 1) / 2 + 1
                self.assertEqual(values[f'stage{stage + 1}_c1'], (expected, 1.))
            metrics[1, 'no_bn', 0, units[1]] = None
            self.assertIsNone(M.stage_means(metrics, units, 1, 'no_bn', 2)['stage1_c1'])

    def test_summary_counts_use_per_metric_best_and_c2_accuracy(self):
        data = {}
        keys = [f'{name}{c}' for c in (1, 2) for name in ('mean', 'first')]
        keys += [f'stage{s}_c{c}' for c in (1, 2) for s in (1, 2, 3)]
        for dataset in ('cifar10', 'cifar100'):
            for size in M.SIZES:
                for condition, levels in M.LEVELS.items():
                    for level in levels:
                        for family in M.FAMILIES:
                            pcn = family == 'pcn'
                            metrics = {k: (1. if pcn else 2., 0.) for k in keys}
                            metrics['accuracy'] = (80. if pcn else 70., 0.)
                            data[dataset, size, condition, level, family,
                                 'no_bn' if pcn else 'recalibrated'] = metrics
        # A different WRN can be best for error and accuracy in the same case.
        prefix = 'cifar10', '16_2', 'max_additive', .1
        data[*prefix, 'wd1e3', 'recalibrated']['stage1_c1'] = (.5, 0.)
        data[*prefix, 'wrn', 'recalibrated']['accuracy'] = (90., 0.)
        text = '\n'.join(M.summarization_tables(data))
        self.assertIn('| First stage - 32 / 64 | 71/72 | **72/72** |', text)
        self.assertIn('| First stage - 32 / 64 | 7/8 | **8/8** |', text)
        self.assertIn('| Max-additive | 0.10 | **7/8** | C10 16-2 |', text)
        self.assertIn('**47/48**', text)
        self.assertIn('**15/16**', text)
        self.assertEqual(text.count('### PCN with lower C'), 2)
        self.assertEqual(text.count('### PCN accuracy wins in Claim 2'), 1)
        # Undefined values are labelled, not counted as normal losses or dropped.
        data[*prefix, 'wd1e3', 'recalibrated']['stage1_c1'] = None
        self.assertIn('71/72 (1 undefined)', '\n'.join(M.summarization_tables(data)))

    def test_stats_keep_undefined_and_use_population_std(self):
        self.assertEqual(M.stats([1., 3.]), (2., 1.))
        self.assertIsNone(M.stats([1., None]))
        with self.assertRaises(ValueError):
            M.stats([float('inf')])

    def test_overlapping_abnormal_types(self):
        wrn = {'last2': (2., 0.), 'logits': (2., 0.), 'accuracy': (70., 0.)}
        pcn = {'last2': (1., 0.), 'logits': (3., 0.), 'accuracy': (60., 0.)}
        self.assertEqual(M.anomalies(pcn, wrn), ['1', '3'])
        pcn['logits'] = (1., 0.)
        self.assertEqual(M.anomalies(pcn, wrn), ['2', '3'])
        pcn['last2'] = None
        self.assertEqual(M.anomalies(pcn, wrn), ['2'])
        self.assertEqual(M.anomalies(wrn, wrn), [])

    def test_incomplete_and_smoke_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for state in ('{"status":"running","smoke":false}', '{"status":"complete","smoke":true}'):
                (root / 'state.json').write_text(state)
                with self.assertRaises(ValueError):
                    M.read_run(root, 'max_additive', .1)

    def test_missing_inventory_does_not_overwrite_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.md'
            output.write_text('existing report')
            with self.assertRaises(FileNotFoundError):
                M.generate(root / 'missing', output)
            self.assertEqual(output.read_text(), 'existing report')


if __name__ == '__main__':
    unittest.main()
