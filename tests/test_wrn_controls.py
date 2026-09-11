import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from baseline.cifar_resnet import WideResNetCIFAR
from baseline.baseline_cifar_configs import build_model, get_baseline_config
from baseline.train_baseline_cifar import parse_kv_overrides
from baseline.wrn_control_specs import ROWS, SIZES, SLURM_ROWS, model_name, model_options, training_override
from baseline.run_wrn_controls import parse_args, paths, run_one, training_command, test_command
from baseline.run_wrn_bn_recalibration_experiment import standalone_specs, parse_args as bn_args
from baseline.run_wrn_bn_recalibration_experiment import build_loaded_wrn_model
from baseline.run_baseline import load_model_weights
from types import SimpleNamespace


class WRNControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_registered_structures_and_recipe(self):
        for row in ROWS:
            for size in SIZES:
                for classes in (10, 100):
                    with self.subTest(row=row, size=size, classes=classes):
                        name = model_name(row, size)
                        cfg = get_baseline_config(name, extra_overrides=parse_kv_overrides(training_override(row)))
                        reference = get_baseline_config('wrn_' + size + '_cifar',
                            extra_overrides=parse_kv_overrides(training_override(row)))
                        self.assertEqual({k: v for k, v in cfg.items() if k != 'model_name'},
                                         {k: v for k, v in reference.items() if k != 'model_name'})
                        model = build_model(name, cfg, classes)
                        bn, bias, shortcuts, main = ROWS[row]
                        self.assertEqual(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()), bn)
                        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                        self.assertTrue(all((m.bias is not None) == bias for m in convs))
                        self.assertEqual(sum(m.kernel_size == (1, 1) for m in convs), 0 if shortcuts else 3)
                        self.assertEqual(sum(m.stride == (2, 2) for m in convs),
                                         0 if main else (2 if shortcuts else 4))
                        self.assertIsNotNone(model.fc.bias)
                        self.assertEqual(model.init_mode, 'pytorch' if row == 10 else 'wrn')
                        model.eval()
                        output = model(torch.randn(2, 3, 8, 8))
                        self.assertEqual(tuple(output.shape), (2, classes))
                        self.assertTrue(torch.isfinite(output).all())
                        if size == '16_2' and classes == 10:
                            output.sum().backward()
                            self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())

    def test_default_init_skips_only_custom_initialization(self):
        import baseline.cifar_resnet as models
        torch.manual_seed(19)
        actual = WideResNetCIFAR(depth=16, widen_factor=2, init_mode='pytorch')
        torch.manual_seed(19)
        with patch.object(models, '_init_cifar_model') as init:
            reference = WideResNetCIFAR(depth=16, widen_factor=2)
        init.assert_called_once()
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
        torch.manual_seed(19)
        old = WideResNetCIFAR(depth=16, widen_factor=2)
        self.assertFalse(torch.equal(actual.conv1.weight, old.conv1.weight))

    def test_trained_checkpoint_reconstruction_for_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            for row in ROWS:
                name = model_name(row, '16_2')
                cfg = get_baseline_config(name, extra_overrides=parse_kv_overrides(training_override(row)))
                model = build_model(name, cfg, 100)
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9,
                                            weight_decay=cfg['weight_decay'])
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(torch.randn(2, 3, 8, 8)), torch.tensor([0, 1])).backward()
                optimizer.step()
                checkpoint = Path(directory) / f'row{row}.pth'
                torch.save({'net': model.state_dict()}, checkpoint)
                if ROWS[row][0]:
                    restored, _ = build_loaded_wrn_model(SimpleNamespace(case='custom_noresize', ff_gain=1),
                                                        'cifar100', name, checkpoint, torch.device('cpu'))
                else:
                    restored = build_model(name, get_baseline_config(name), 100)
                    load_model_weights(restored, str(checkpoint), torch.device('cpu'))
                model.eval()
                restored.eval()
                x = torch.randn(2, 3, 8, 8)
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_bn_test_command_needs_no_comparison_data(self):
        args = parse_args([])
        for row in (4, 7, 10):
            for condition in ('max_additive', 'multiplicative', 'rms_additive'):
                command = test_command(args, row, 'cifar100', '28_2', condition)
                with patch.object(sys, 'argv', command[2:]):
                    parsed = bn_args()
                self.assertEqual(len(standalone_specs(parsed)), 1)
                self.assertEqual(standalone_specs(parsed)[0]['compare']['by_level'], {})
                self.assertEqual(parsed.mismatch_parameter_policy, 'existing')

    def test_slurm_schedule_eight_rows_covering_64_models_and_row3_local(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['bash', 'launch_scripts/slurm_run_wrn_controls.sh'], cwd=root,
                                env=dict(os.environ, DRY_RUN='1', ROWS=','.join(map(str, SLURM_ROWS)),
                                         DATASETS='cifar10,cifar100', SIZES=','.join(SIZES)),
                                capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 8)
        self.assertNotIn('--chdir', result.stdout)
        self.assertTrue(all('DATASETS=cifar10,cifar100 SIZES=16_2,16_4,28_2,28_4' in line for line in lines))
        self.assertFalse(any(line.startswith('ROW=3 ') or line.startswith('ROW=1 ') for line in lines))
        self.assertEqual(sum(line.startswith('ROW=10 ') for line in lines), 1)
        args = parse_args(['--dry-run'])
        self.assertEqual(args.rows, [3])
        self.assertEqual(len(args.datasets) * len(args.sizes), 8)

    def test_evaluators_run_on_synthetic_data_without_legacy_csvs(self):
        import baseline.run_wrn_bn_recalibration_experiment as bn_eval
        import baseline.run_wrn_nobn_mismatch_experiment as nobn_eval
        from torch.utils.data import DataLoader, TensorDataset
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--output-root', directory])
            x = torch.randn(2, 3, 8, 8)
            test_loader = DataLoader(TensorDataset(x, torch.tensor([0, 1])), batch_size=2)
            calibration = DataLoader(x + 1, batch_size=2)
            for row, evaluator in ((4, bn_eval), (6, nobn_eval)):
                name = model_name(row, '16_2')
                model = build_model(name, get_baseline_config(name), 100)
                run, checkpoint = paths(args, row, 'cifar100', '16_2')
                checkpoint.parent.mkdir(parents=True)
                torch.save({'net': model.state_dict()}, checkpoint)
                command = test_command(args, row, 'cifar100', '16_2', 'max_additive')
                command[command.index('--noisy_trials') + 1] = '1'
                command[command.index('--noise_levels') + 1] = '0,0.01'
                command += ['--device', 'cpu']
                with patch.object(sys, 'argv', command[2:]), \
                     patch.object(evaluator, 'build_test_loader', return_value=test_loader):
                    if row == 4:
                        with patch.object(bn_eval, 'build_bn_calibration_loader', return_value=calibration), \
                             patch.object(bn_eval, 'discover_compare_specs', side_effect=AssertionError('Legacy CSV read')):
                            evaluator.main()
                    else:
                        evaluator.run(evaluator.parse_args())
                self.assertTrue((run / 'evaluation/max_additive/full_aggregate.csv').is_file())

    def test_completion_and_failure_are_not_confused(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--output-root', directory, '--rows', '3', '--sizes', '16_2',
                               '--datasets', 'cifar100'])
            path, checkpoint = paths(args, 3, 'cifar100', '16_2')
            def fake_training(command, log):
                log.write_text('Train finished\n')
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b'test checkpoint')
            with patch('baseline.run_wrn_controls.execute', side_effect=fake_training) as execute:
                run_one(args, 3, 'cifar100', '16_2')
                run_one(args, 3, 'cifar100', '16_2')
                self.assertEqual(execute.call_count, 1)
            self.assertTrue((path / 'train_complete.json').exists())
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                run_one(args, 3, 'cifar100', '16_2')
            self.assertEqual(json.loads((path / 'state.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
