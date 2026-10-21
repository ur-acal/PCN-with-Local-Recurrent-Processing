import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from baseline.activation_diagnostics import LogitDiagnostics, evaluate_cached
from baseline.run_logit_diagnostics import check_replay
from baseline.generate_activation_claims_summary import (
    FAMILIES, LEVELS, SIZES, select_top_abnormal, top_abnormal_tables, generate)


class LogitTest(unittest.TestCase):
    def test_offset_invariance_and_scaling(self):
        clean = torch.tensor([[3., 1., 0.], [0., 2., 1.]])
        labels, ids = torch.tensor([0, 1]), torch.arange(2)
        metric = LogitDiagnostics()
        metric.update(clean + 10, clean, labels, ids)
        self.assertLess(metric.result()['centered_logits']['dataset_relative_l2'], 1e-14)
        self.assertEqual(metric.result()['margin_degradation'], 0)
        metric = LogitDiagnostics()
        metric.update(clean * 2, clean, labels, ids)
        self.assertAlmostEqual(metric.result()['centered_logits']['dataset_relative_l2'], 1)

    def test_true_class_rival_and_unequal_batches(self):
        clean = torch.tensor([[3., 1., 0.], [0., 2., 1.], [1., 0., 4.]])
        noisy = torch.tensor([[0., 2., 1.], [0., 1., 2.], [5., 0., 4.]])
        labels = torch.tensor([0, 1, 0])
        metric = LogitDiagnostics()
        for sl in (slice(0, 2), slice(2, 3)):
            metric.update(noisy[sl], clean[sl], labels[sl], torch.arange(3)[sl])
        # Clean margins [2, 1, -3], noisy [-2, -1, 1].
        self.assertEqual(metric.result()['clean_margin_mean'], 0)
        self.assertAlmostEqual(metric.result()['margin_degradation'], 2 / 3)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, sl in enumerate((slice(0, 2), slice(2, 3))):
                path = Path(tmp) / f'{i}.pt'
                torch.save(dict(images=noisy[sl], logits=clean[sl], labels=labels[sl],
                                sample_ids=torch.arange(3)[sl]), path)
                paths.append(path)
            _, transition, _ = evaluate_cached(torch.nn.Identity(), 'wrn', paths, 'cpu', logits_only=True)
            self.assertAlmostEqual(transition['correct_to_wrong_percent'], 200 / 3)
            self.assertAlmostEqual(transition['wrong_to_correct_percent'], 100 / 3)
            self.assertEqual(transition['samples'], 3)

    def test_zero_centered_norm_flagged_and_nonfinite_refused(self):
        metric = LogitDiagnostics()
        clean, noisy = torch.ones(1, 3), torch.tensor([[1., 2., 3.]])
        metric.update(noisy, clean, torch.tensor([0]), torch.tensor([17]))
        result = metric.result()
        self.assertIsNone(result['centered_logits']['dataset_relative_l2'])
        self.assertEqual(result['centered_zero_references'][0]['sample_id'], 17)
        with self.assertRaises(FloatingPointError):
            metric.update(noisy * float('nan'), clean, torch.tensor([0]), torch.tensor([17]))

    def test_replay_refuses_changed_prediction_or_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = dict(samples=1, clean_correct=1, noisy_correct=0, correct_to_wrong=1, wrong_to_correct=0)
            (root / 'summary.json').write_text(json.dumps(values))
            (root / 'predictions.csv').write_text('sample_id,label,clean_prediction,noisy_prediction\n0,0,0,1\n')
            (root / 'metrics.csv').write_text('layer,error_energy,reference_energy\nlogits,2,4\n')
            check_replay(dict(error_energy=2, reference_energy=4), values, [(0, 0, 0, 1)], root)
            with self.assertRaises(ValueError):
                check_replay(dict(error_energy=3, reference_energy=4), values, [(0, 0, 0, 1)], root)
            with self.assertRaises(ValueError):
                check_replay(dict(error_energy=2, reference_energy=4), values, [(0, 0, 0, 2)], root)

    def test_two_stage_selection_uses_accuracy_comparator(self):
        data, cases = {}, []
        for dataset in ('cifar10', 'cifar100'):
            for size in SIZES:
                for condition, levels in LEVELS.items():
                    for level in levels:
                        prefix = dataset, size, condition, level
                        cases.append(prefix)
                        for family in FAMILIES:
                            values = {k: (1., 0.) for k in ('logits', 'centered', 'margin', 'last2',
                                                            'correct_to_wrong', 'wrong_to_correct')}
                            values['accuracy'] = (70. if family == 'finaldrop' else 60., 0.)
                            data[*prefix, family, 'no_bn' if family == 'pcn' else 'recalibrated'] = values
        for i, prefix in enumerate(cases[:7]):
            pcn = data[*prefix, 'pcn', 'no_bn']
            pcn.update(logits=(8. - i, 0.), accuracy=(72. + i, 0.))
        # Seventh is not eligible (zero error gap); sixth wins final accuracy ranking.
        shortlist, selected = select_top_abnormal(data)
        self.assertEqual(len(shortlist), 6)
        self.assertEqual([r['prefix'] for r in selected], list(reversed(cases[3:6])))
        self.assertTrue(all(r['family'] == 'finaldrop' for r in selected))
        text = '\n'.join(top_abnormal_tables(data))
        self.assertEqual(text.count('### Selected case'), 3)
        self.assertIn('Margin degradation', text)

        def read(directory, condition, level, logit_directory=None, pcn_c2_directory=None):
            parts = directory.name.split('_')
            dataset, size, family = parts[0], '_'.join(parts[1:3]), parts[3]
            v = dict(data[dataset, size, condition, level, family,
                          'no_bn' if family == 'pcn' else 'recalibrated'])
            if family == 'pcn':
                v['last2'] = (.5, 0.)
            v.update({k: (1., 0.) for k in ('mean1', 'mean2', 'first1', 'first2', 'last1')})
            v.update({f'stage{s}_c{c}': (1., 0.) for s in (1, 2, 3) for c in (1, 2)})
            return dict(dataset=dataset, size=size, family=family), {
                p: v for p in (('no_bn',) if family == 'pcn' else ('frozen', 'recalibrated'))}, 0

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'report.md'
            with patch('baseline.generate_activation_claims_summary.read_run', side_effect=read):
                generate(Path(tmp), output, logit_source=Path(tmp) / 'supplement')
            text = output.read_text()
            self.assertEqual(text.count('### Selected case'), 3)
            self.assertEqual(text.count('| Model | Mean C1'), 48)
            self.assertIn('Centered error PCN / WRN', text)


if __name__ == '__main__':
    unittest.main()
