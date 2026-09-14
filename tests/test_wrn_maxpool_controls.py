import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from baseline.cifar_resnet import WideBasicBlock, ChannelZeroPad, MaxPoolChannelPad
from baseline.cifar_wrn_pool_after_add import WideBasicBlockPoolAfterAdd
from baseline.wrn_control_specs import NEW_POOL_ROWS, model_options, training_override
from baseline.wrn_control_artifacts import pack


class MaxPoolControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_new_rows_preserve_parent_recipe_and_parameter_layout(self):
        for row in range(11, 17):
            parent = row - 7
            self.assertEqual(training_override(row), training_override(parent))
            new = model_options(row)
            self.assertEqual({k: v for k, v in new.items()
                              if k not in ('pool_type', 'pool_after_add')}, model_options(parent))
            self.assertEqual(new['pool_type'], 'max')
        for row in (17, 18):
            self.assertEqual(training_override(row), training_override(10))
            self.assertEqual(model_options(row)['init_mode'], 'pytorch')

    def test_maxpool_is_after_sum_and_prepool_is_exposed(self):
        # Opposite maxima distinguish pooling each branch from pooling the sum.
        for pool_type in ('avg', 'max'):
            block = WideBasicBlockPoolAfterAdd(1, 1, 0, stride=2, pool_type=pool_type)
            block.bn1 = block.bn2 = block.relu1 = block.relu2 = nn.Identity()
            block.conv1 = nn.Identity()
            class Negative(nn.Module):
                def forward(self, x):
                    return -x
            block.conv2 = Negative()
            seen = []
            hook = block.pre_pool.register_forward_hook(lambda m, i, o: seen.append(o.clone()))
            x = torch.tensor([[[[1., 2.], [3., 4.]]]])
            output = block(x)
            torch.testing.assert_close(output, torch.zeros_like(output))
            self.assertEqual(seen[0].shape, x.shape)
            self.assertIsInstance(block.shortcut, ChannelZeroPad)
            self.assertIsInstance(block.main_downsample, nn.MaxPool2d if pool_type == 'max' else nn.AvgPool2d)
            hook.remove()
        separate = nn.MaxPool2d(2)(x) + nn.MaxPool2d(2)(-x)
        self.assertEqual(separate.item(), 3.)

    def test_shortcut_only_maxpool_removes_all_learned_projections(self):
        for stride in (1, 2):
            block = WideBasicBlock(4, 8, 0, stride=stride,
                                   avgpool_downsample_shortcut=True, pool_type='max')
            self.assertIsInstance(block.shortcut, ChannelZeroPad if stride == 1 else MaxPoolChannelPad)
            self.assertEqual(block.conv2.stride, (stride, stride))
            self.assertIsInstance(block.main_downsample, nn.Identity)
        with self.assertRaises(ValueError):
            WideBasicBlockPoolAfterAdd(4, 8, 0, pool_type='invalid')

    def test_all_64_simulated_train_test_and_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, SIMULATE='1', DRY_RUN='0', ROWS=','.join(map(str, NEW_POOL_ROWS)),
                       DATASETS='cifar10,cifar100', SIZES='16_2,16_4,28_2,28_4',
                       STAGE='train-test', PARALLELISM='4', EVAL_PARALLELISM='8',
                       CONDITIONS='max_additive,multiplicative,rms_additive',
                       OUTPUT_ROOT=tmp, REPO_ROOT=str(Path(__file__).resolve().parents[1]),
                       PYTHON_BIN=sys.executable, SIMULATE_FAILURE='',
                       PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
            run = subprocess.run(['bash', '-c', 'source launch_scripts/slurm_run_wrn_controls.sh'],
                                 cwd=env['REPO_ROOT'], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            root = Path(tmp) / 'SIMULATED'
            inventory = pack(root, Path(tmp) / 'results.tar.gz')
            self.assertEqual(inventory['complete_tasks'], 64)
            self.assertEqual(len(list(root.glob('row*/*/*/evaluation/*/complete.json'))), 192)
            for path in root.glob('row*/*/*/manifest.json'):
                identity = json.loads(path.read_text())['identity']
                self.assertEqual(identity['options'], model_options(identity['row']))
                self.assertIn(training_override(identity['row']), identity['training_command'])
                directory = path.parent
                training = json.loads((directory / 'checkpoints/simulation_lifecycle.json').read_text())
                previous = training['finished']
                for condition in env['CONDITIONS'].split(','):
                    life = json.loads((directory / f'evaluation/{condition}/simulation_lifecycle.json').read_text())
                    self.assertGreaterEqual(life['started'], previous)
                    previous = life['finished']


if __name__ == '__main__':
    unittest.main()
