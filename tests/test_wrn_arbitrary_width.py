import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from baseline.wrn_control_specs import parse_size, model_name, SIZES
from baseline.baseline_cifar_configs import get_baseline_config, build_model
from baseline.run_wrn_controls import parse_args
from baseline.run_baseline_missing_mismatch_queue import plan, dependency_complete
from baseline import run_baseline_missing_mismatch_queue as queue


class WidthTest(unittest.TestCase):
    def test_sizes_and_defaults(self):
        self.assertEqual(SIZES, ('16_2', '16_4', '28_2', '28_4'))
        for size in ('16_1', '28_6', '28_8', '28_12'):
            self.assertEqual(parse_size(size), tuple(map(int, size.split('_'))))
            self.assertIn(size, model_name(17, size))
        for size in ('28_0', '28_-2', '28_2.5', '28_06', '27_8', '4_8', ''):
            with self.assertRaises(ValueError):
                parse_size(size)
        self.assertEqual(parse_args(['--sizes', '28_6,28_8']).sizes, ['28_6', '28_8'])

    def test_fresh_process_registration_and_forward(self):
        # A clean subprocess must resolve a model by name, without launcher registration.
        code = '''
import torch
from baseline.baseline_cifar_configs import get_baseline_config, build_model
torch.set_num_threads(1)
for name in ('wrn_28_6_cifar', 'wrn_28_8_cifar_control_r17',
             'wrn_28_6_cifar_control_r18', 'wrn_16_3_cifar_nobn_no_bias'):
    cfg = get_baseline_config(name)
    assert cfg['case'] == 'custom_noresize'
    model = build_model(name, cfg, 100).eval()
    assert model.conv1.out_channels == 16
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 100)
'''
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_existing_construction_unchanged(self):
        torch.manual_seed(7)
        from baseline.cifar_resnet import wrn_28_2_cifar
        old = wrn_28_2_cifar(num_classes=10)
        torch.manual_seed(7)
        new = build_model('wrn_28_2_cifar', get_baseline_config('wrn_28_2_cifar'), 10)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, new.state_dict()[k]), k)

    def test_wait_gate_and_plan(self):
        tasks = plan(Path('/tmp/test-baseline'))
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({t['output'] for t in tasks}), 24)
        for task in tasks:
            cmd = task['command']
            self.assertIn('--standalone', cmd)
            self.assertEqual(cmd[cmd.index('--mismatch_parameter_policy') + 1], 'existing')
            self.assertEqual(cmd[cmd.index('--noisy_trials') + 1], '10')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            state = dict(status='running', pid=123, waves=[[dict(status='queued') for _ in range(192)]])
            path.write_text(json.dumps(state))
            with patch('os.kill'):
                self.assertFalse(dependency_complete(path))
            state['status'] = 'complete'
            path.write_text(json.dumps(state))
            with self.assertRaises(RuntimeError):
                dependency_complete(path)
            for t in state['waves'][0]:
                t['status'] = 'complete'
            path.write_text(json.dumps(state))
            self.assertTrue(dependency_complete(path))

    def test_fake_queue_to_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint.pth'
            checkpoint.touch()
            dependency = root / 'claims.json'
            dependency.write_text('{}')
            output = root / 'results'
            tasks = plan(output)
            for task in tasks:
                task['checkpoint'] = str(checkpoint)

            def fake_eval(command, **kwargs):
                dest = Path(command[command.index('--output_dir') + 1])
                task = next(t for t in tasks if t['output'] == str(dest))
                with (dest / 'full_per_trial.csv').open('w') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['dataset', 'architecture', 'mismatch_level', 'trial',
                                     'frozen_bn_accuracy', 'recalibrated_bn_accuracy'])
                    for level in queue.LEVELS[task['condition']]:
                        for trial in range(10):
                            writer.writerow([task['dataset'], 'WRN_' + task['size'], level, trial, 40, 60])
                with (dest / 'full_aggregate.csv').open('w') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['mismatch_level'])
                    writer.writerows([[l] for l in queue.LEVELS[task['condition']]])

            with patch.object(sys, 'argv', ['queue', '--dependency', str(dependency), '--output-root', str(output)]), \
                 patch.object(queue, 'plan', return_value=tasks), \
                 patch.object(queue, 'dependency_complete', side_effect=[False, True]) as gate, \
                 patch.object(queue.time, 'sleep'), patch.object(queue.subprocess, 'run', side_effect=fake_eval):
                queue.main()
            self.assertEqual(gate.call_count, 2)
            state = json.loads((output / 'state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertTrue(all(t['status'] == 'complete' for t in state['tasks']))
            # A truncated trial file must never count as completed.
            trial_file = Path(tasks[0]['output']) / 'full_per_trial.csv'
            trial_file.write_text(trial_file.read_text().splitlines()[0] + '\n')
            with self.assertRaises(RuntimeError):
                queue.validate(tasks[0])


if __name__ == '__main__':
    unittest.main()
