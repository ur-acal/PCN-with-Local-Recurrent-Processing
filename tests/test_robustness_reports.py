import importlib.util
from pathlib import Path
import unittest
import json
import os
import subprocess
import sys
import tempfile
import csv
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location('reports', Path(__file__).resolve().parents[1] / 'baseline/generate_robustness_reports.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ReportsTest(unittest.TestCase):
    def test_wrn_markdown_and_bnfree_companion_for_all_ranking_modes(self):
        for mode in ('dataset-severe-mean', 'global-endpoint', 'per-point'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                data = M.Results(ranking_mode=mode)
                for variant in ('a', 'b', 'c', 'd'):
                    data.register(variant, variant, base=variant)
                data.register('P0', 'PCN', family='pcn')
                for dataset, architecture in data.pairs:
                    for mismatch_type in M.TYPES[:2]:
                        for level in M.GRIDS[mismatch_type]:
                            for variant in ('a', 'b', 'c', 'd', 'P0'):
                                score = 99 if variant == ('a' if level == 0 else 'd') else 65
                                data.put(variant, dataset, architecture, mismatch_type,
                                         level, score, None, 'test')
                data.require_complete(M.TYPES[:2])
                args = SimpleNamespace(output_dir=Path(directory), wrn_ranking=mode,
                                       exclude_avgpool_wrn=False, mismatch_types=M.TYPES[:2],
                                       datasets=['cifar10', 'cifar100'], controls_source=[])
                M.write_md(data, args, wrn=True)
                text = (args.output_dir / 'pcn_vs_bnfree_wrn_max_mul.md').read_text()
                self.assertTrue((args.output_dir / 'wrn_bn_bias_and_top3_comparison.md').is_file())
                if mode == 'per-point':
                    self.assertIn('## Pointwise WRN selections', text)
                    self.assertNotIn('Best fixed top-three', text)
                    self.assertIn('| cifar10 | WRN_16_2 | max_additive | 0 | a_recal (99.00)', text)
                    self.assertIn('| cifar10 | WRN_16_2 | max_additive | 0.01 | d_recal (99.00)', text)
                elif mode == 'global-endpoint':
                    self.assertIn('globally for each mismatch type by mean endpoint accuracy', text)
                    self.assertNotIn('selected separately for that dataset', text)
                    self.assertIn('| All selected | max_additive', text)
                else:
                    self.assertIn('selected separately for that dataset and mismatch type', text)
                    self.assertIn('## Fixed WRN selections', text)

    def test_top3_baseline_recal_only_and_new_data_reranking(self):
        d = M.Results()
        for i, policy, score in [('W0-R', 'recal', 40), ('a', 'recal', 70),
                                 ('b', 'recal', 60), ('c', 'recal', 50),
                                 ('frozen', 'frozen', 99), ('none', 'none', 98),
                                 ('folded', 'folded-recal', 97)]:
            d.register(i, i, policy=policy)
            for dataset, arch in M.PAIRS:
                for level in M.GRIDS['max_additive']:
                    d.put(i, dataset, arch, 'max_additive', level, score, None, 'test')
        d.require_complete(['max_additive'])
        self.assertEqual(d.top3_baseline_ids('cifar10', 'WRN_16_2', 'max_additive'),
                         ['P0', 'P3', 'a', 'b', 'c', 'W0-R'])
        self.assertEqual(d.top3_baseline_ids('cifar10', 'WRN_16_2', 'max_additive',
                                             ['P0', 'P3', 'PB-R']),
                         ['P0', 'P3', 'PB-R', 'a', 'b', 'c', 'W0-R'])
        self.assertEqual(M.top3_plot_folder(['P0', 'P3']), 'pcn_vs_top3_baseline_wrns')
        self.assertEqual(M.top3_plot_folder(['P0', 'P3', 'PB-R']),
                         'p0_p3_pb-r_vs_top3_baseline_wrns')
        self.assertEqual(M.pcn_linestyles(['P0', 'P3', 'PB-R']),
                         {'P0': '-', 'P3': '--', 'PB-R': ':'})
        self.assertEqual(M.pcn_linestyles(['PB-R', 'P0', 'PB-F']),
                         {'P0': '-', 'PB-R': '--', 'PB-F': ':'})
        d.register('new', 'new')
        for dataset, arch in M.PAIRS:
            for level in M.GRIDS['max_additive']:
                d.put('new', dataset, arch, 'max_additive', level, 90, None, 'new')
        d.require_complete(['max_additive'])
        self.assertEqual(d.top3_baseline_ids('cifar10', 'WRN_16_2', 'max_additive'),
                         ['P0', 'P3', 'new', 'a', 'b', 'W0-R'])
        del d.values['cifar100', 'WRN_28_4', 'max_additive', .01, 'new']
        d.require_complete(['max_additive'])
        self.assertNotIn('new', d.top3_baseline_ids('cifar10', 'WRN_16_2', 'max_additive'))

    def test_wrapper_argument_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'python-stub'
            log = root / 'calls.jsonl'
            stub.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                            'with open(os.environ["CALL_LOG"], "a") as f:\n'
                            '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            stub.chmod(0o755)
            script = Path(__file__).resolve().parents[1] / 'launch_scripts/generate_robustness_reports.sh'
            subprocess.run(['bash', str(script), '--output-dir', directory, '--color-mode', 'pooling',
                            '--pcn-models', 'P1,P3', '--highlight-top3', '--uncertainty',
                            '--baseline-paired-root', str(root / 'paired'),
                            '--sizes', '28_6,28_8', '--datasets', 'cifar100',
                            '--wrn-ranking', 'global-endpoint', '--top3-plus-baseline'],
                           env={**os.environ, 'PYTHON': str(stub), 'CALL_LOG': str(log)}, check=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([a[a.index('--mode')+1] for a in calls],
                             ['pcn-md', 'pcn-plots', 'wrn-md', 'wrn-plots'])
            for n, args in enumerate(calls):
                self.assertIn('--output-dir', args)
                self.assertEqual(args[args.index('--sizes') + 1], '28_6,28_8')
                self.assertIn('--datasets', args)
                self.assertIn('--baseline-paired-root', args)
                self.assertEqual(args[args.index('--wrn-ranking') + 1], 'global-endpoint')
                self.assertEqual('--uncertainty' in args, n in (1, 3))
                for option in ('--color-mode', '--pcn-models', '--highlight-top3', '--top3-plus-baseline'):
                    self.assertEqual(option in args, n == 1)

    def test_load_baseline_paired_adds_unfused_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'max_additive/cifar10/WRN_16_2/full_aggregate.csv'
            path.parent.mkdir(parents=True)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    'dataset', 'architecture', 'mismatch_type', 'mismatch_level',
                    'additive_scale_mode', 'paired_wrn_frozen_bn_mean_accuracy',
                    'paired_wrn_frozen_bn_std_accuracy',
                    'wrn_recalibrated_bn_mean_accuracy', 'wrn_recalibrated_bn_std_accuracy'])
                writer.writeheader()
                writer.writerow(dict(dataset='cifar10', architecture='WRN_16_2',
                                     mismatch_type='additive', mismatch_level=.1,
                                     additive_scale_mode='max_abs',
                                     paired_wrn_frozen_bn_mean_accuracy=41,
                                     paired_wrn_frozen_bn_std_accuracy=2,
                                     wrn_recalibrated_bn_mean_accuracy=72,
                                     wrn_recalibrated_bn_std_accuracy=1))
            data = M.Results()
            data.register('W0-R', 'WRN unfused recal-BN', base='W0', policy='recal')
            M.load_baseline_paired(data, root)
            self.assertEqual(data.meta['W0-F']['policy'], 'frozen')
            self.assertEqual(data.values['cifar10', 'WRN_16_2', 'max_additive', .1, 'W0-F'][:2], (41, 2))
            self.assertEqual(data.values['cifar10', 'WRN_16_2', 'max_additive', .1, 'W0-R'][:2], (72, 1))

    def test_separate_rankings_and_partial_oracle(self):
        d = M.Results(ranking_mode='global-endpoint')
        for i in ('a', 'b', 'c', 'partial'):
            d.register(i, i)
        for dataset, arch in M.PAIRS:
            for i, x in [('a', 80), ('b', 70), ('c', 60)]:
                d.put(i, dataset, arch, 'max_additive', .1, x, None, 'test')
                d.put(i, dataset, arch, 'multiplicative', .4, 100-x, None, 'test')
        d.put('partial', 'cifar10', 'WRN_16_2', 'max_additive', .1, 99, None, 'test')
        self.assertEqual([i for _, i in d.top('max_additive')], ['a', 'b', 'c'])
        self.assertEqual([i for _, i in d.top('multiplicative')], ['c', 'b', 'a'])
        self.assertIn('99.00 (partial_recal)', M.curve_table(d, 'cifar10', 'WRN_16_2', 'max_additive', ['a'], True))

    def test_default_dataset_severe_mean_and_pointwise_rankings(self):
        d = M.Results()
        for i in ('c10', 'c100', 'point'):
            d.register(i, i)
        for dataset, arch in M.PAIRS:
            for level in M.SEVERE_LEVELS['max_additive']:
                d.put('c10', dataset, arch, 'max_additive', level,
                      90 if dataset == 'cifar10' else 50, None, 'test')
                d.put('c100', dataset, arch, 'max_additive', level,
                      50 if dataset == 'cifar10' else 90, None, 'test')
                point_score = (99 if (dataset, arch, level) ==
                               ('cifar10', 'WRN_16_2', .1) else 60)
                d.put('point', dataset, arch, 'max_additive', level,
                      point_score, None, 'test')
        self.assertEqual(d.top('max_additive', dataset='cifar10')[0][1], 'c10')
        self.assertEqual(d.top('max_additive', dataset='cifar100')[0][1], 'c100')
        d.ranking_mode = 'per-point'
        self.assertEqual(d.top('max_additive', dataset='cifar10', architecture='WRN_16_2',
                               level=.1)[0][1], 'point')

    def test_global_completeness_and_labels(self):
        d = M.Results()
        for i in ('complete', 'partial'):
            d.register(i, i)
            for dataset, arch in M.PAIRS:
                for t in M.TYPES[:2]:
                    for level in M.GRIDS[t]:
                        d.put(i, dataset, arch, t, level, 50, None, 'test')
        del d.values['cifar100', 'WRN_28_4', 'max_additive', .01, 'partial']
        d.register('P2', 'PCNetNoBatchNorm', family='pcn')
        d.require_complete(M.TYPES[:2])
        self.assertTrue(d.eligible('complete'))
        self.assertFalse(d.eligible('partial'))
        self.assertTrue(d.eligible('P2'))
        self.assertIn('PCN_X_ConvInit', d.label('P2'))
        self.assertIn('ODEBlockPC', d.label('P2'))

    def test_filter_and_complete_matched_group(self):
        d = M.Results()
        ids = ['R4-F', 'R4-R', 'R5-N', 'R6-N']
        for row in (4, 5, 6):
            d.recipes[f'R{row}', 'cifar10', 'WRN_16_2'] = {'lr': .1}
        for i in ids:
            d.register(i, i, pooling='avg')
            d.put(i, 'cifar10', 'WRN_16_2', 'max_additive', .1, 50, None, 'test')
        self.assertEqual(d.matched('cifar10', 'WRN_16_2', 'max_additive'), ids)
        d.recipes['R6', 'cifar10', 'WRN_16_2'] = {'lr': .01}
        self.assertEqual(d.matched('cifar10', 'WRN_16_2', 'max_additive'), [])
        d.recipes['R6', 'cifar10', 'WRN_16_2'] = {'lr': .1}
        d.exclude_avg = True
        self.assertEqual(d.matched('cifar10', 'WRN_16_2', 'max_additive'), [])
        d.exclude_avg = False
        del d.values['cifar10', 'WRN_16_2', 'max_additive', .1, 'R6-N']
        self.assertEqual(d.matched('cifar10', 'WRN_16_2', 'max_additive'), [])

    def test_conflict_and_kind(self):
        d = M.Results()
        d.put('a', 'cifar10', 'WRN_16_2', 'max_additive', .1, 50, None, 'one')
        with self.assertRaises(ValueError):
            d.put('a', 'cifar10', 'WRN_16_2', 'max_additive', .1, 51, None, 'two')
        self.assertEqual(M.kind(dict(mismatch_type='additive', additive_scale_mode='max_sqrt')), 'unsupported')
        self.assertEqual(M.kind(dict(mismatch_type='additive', additive_scale_mode='rms')), 'rms_additive')

    def test_custom_pair_completeness(self):
        d = M.Results(pairs=[('cifar100', 'WRN_28_8')])
        d.register('R17-R', 'custom', base='R17')
        for level in M.GRIDS['max_additive']:
            d.put('R17-R', 'cifar100', 'WRN_28_8', 'max_additive', level, 75, None, 'test')
        d.require_complete(['max_additive'])
        self.assertTrue(d.eligible('R17-R'))
        self.assertEqual(d.top('max_additive'), [(75, 'R17-R')])

    def test_all_modes_with_new_widths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wide = root / 'wide.csv'
            wide.write_text('dataset,architecture,mismatch_type,level\n')
            extra = root / 'extra.csv'
            with extra.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['dataset', 'architecture', 'mismatch_type', 'level', 'variant', 'accuracy'])
                for size in ('28_6', '28_8'):
                    for t in M.TYPES[:2]:
                        for level in M.GRIDS[t]:
                            for i in ('R4-F', 'R4-R', 'R5-N', 'R6-N', 'P0'):
                                writer.writerow(['cifar100', 'WRN_' + size, t, level, i, 65])
            controls = root / 'controls'
            for row in (4, 5, 6):
                for size in ('28_6', '28_8'):
                    cfg = controls / f'row{row}/cifar100/{size}/baseline_config.json'
                    cfg.parent.mkdir(parents=True)
                    cfg.write_text(json.dumps({'cfg': {'model_name': f'row{row}', 'lr': .1}}))
            output = root / 'reports'
            subprocess.run(['bash', str(M.ROOT / 'launch_scripts/generate_robustness_reports.sh'),
                            '--sizes', '28_6,28_8', '--datasets', 'cifar100', '--wide-csv', str(wide),
                            '--extra-data', str(extra), '--controls-source', str(controls),
                            '--output-dir', str(output)], check=True,
                           env={**os.environ, 'PYTHON': sys.executable, 'MPLCONFIGDIR': str(root / 'mpl')},
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            manifest = json.loads((output / 'robustness_report_manifest.json').read_text())
            self.assertEqual(set(manifest), {'pcn-md', 'wrn-md', 'pcn-plots', 'wrn-plots'})
            self.assertEqual(len(list((output / 'plots').rglob('*.pdf'))), 12)
            for name in ('pcn_vs_top3_wrn_max_mul.md', 'wrn_bn_bias_and_top3_comparison.md',
                         'pcn_vs_bnfree_wrn_max_mul.md'):
                content = (output / name).read_text()
                self.assertIn('WRN_28_6', content)
                self.assertIn('WRN_28_8', content)
                self.assertNotIn('WRN_16_2', content)


if __name__ == '__main__':
    unittest.main()
