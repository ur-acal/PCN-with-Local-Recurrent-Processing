import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import pcn_bn_evaluation as evaluation
from ode_inference import parse_args
from pc_model import PCNetBoundaryBN


class PairedBNTests(unittest.TestCase):
    def test_worker_forwards_paired_policy_for_all_conditions(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'PCNetBoundaryBN_C100_1.75TEnd'
            (root / name).mkdir()
            (root / name / f'{name}_best_ckpt.pth').touch()
            capture = root / 'commands'
            env = dict(os.environ, REPO_ROOT=str(repo), MODEL_DIR=str(root),
                       OUTPUT_ROOT=str(root / 'results'), IS_SLURM='0',
                       MODEL_NAME=name, MODEL_INDEX='6', ARCHITECTURE='WRN_28_2',
                       ODE_BLOCK='ODEXInitFFFB', PC_CONV='PCConvNoisy',
                       BN_EVAL_MODE='paired', NOISE_TO_BN='false', DATASETS='all',
                       EVAL_MODE='mismatch', CONDITIONS='multiplicative,max_additive,rms_additive',
                       CAPTURE=str(capture))
            # Stub only the Python process; execute the actual worker's argument construction.
            result = subprocess.run(['bash', '-c',
                'python() { printf "%s\\0" "$@" >> "$CAPTURE"; printf "\\n" >> "$CAPTURE"; }; '
                'source ./launch_scripts/run_rgb_ode_mismatch_eval.sh'],
                cwd=repo, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = capture.read_bytes().splitlines()
            self.assertEqual(len(commands), 3)
            for command in commands:
                tokens = command.decode().rstrip('\0').split('\0')
                with patch.object(sys, 'argv', tokens[1:]):
                    args = parse_args()
                self.assertEqual(args.bn_eval_mode, 'paired')
                self.assertFalse(args.noise_to_bn)
                self.assertFalse(args.noise_to_conv_bias)
                self.assertEqual(args.task, 'cifar100')
                self.assertEqual(args.calibration_num_samples, 5120)
                self.assertEqual(args.calibration_subset_seed, 20240618)
                self.assertEqual(args.calibration_batch_size, 128)
                self.assertTrue(args.output_pickle.endswith('paired_complete.json'))

    def test_real_checkpoint_noise_calibration_and_outputs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'test_1.75TEnd'
            torch.manual_seed(12)
            net = PCNetBoundaryBN(inp_channels=[3, 4], out_channels=[4, 4],
                                  max_pool=[0, 1], num_classes=10, dropout=0.25,
                                  first_bn=False, bias=False, cls=0, bypass=False)
            checkpoint = root / name / f'{name}_best_ckpt.pth'
            checkpoint.parent.mkdir()
            torch.save(dict(net_type='PCNetBoundaryBN', init_args=net.init_args,
                            net=net.state_dict()), checkpoint)
            clean = {n: v.clone() for n, v in net.state_dict().items()}
            inputs = torch.randn(4, 3, 4, 4)
            test_loader = DataLoader(TensorDataset(inputs, torch.arange(4)), batch_size=2)
            calibration = DataLoader(inputs + 2, batch_size=2)
            argv = ['test', '--model_dir', str(root), '--model_name', name,
                    '--bn_eval_mode', 'paired', '--pc_conv', 'PCConvNoisy',
                    '--ode_block', 'ODEXInitFFFB', '--thermal_noise', 'false',
                    '--noise_to_conv_bias', 'false', '--seed', '123',
                    '--noise_level_list', '0,0.01', '--noisy_trials', '2',
                    '--calibration_num_workers', '0', '--calibration_num_samples', '4',
                    '--ts_scale', '1', '--tol', '0.0001',
                    '--output_pickle', str(root / 'results' / 'result.pkl')]
            observed = []
            real_evaluate_pair = evaluation.evaluate_pair

            def checked_pair(model, *args):
                # Older init_args omit dropout; exercise the functional dropout path explicitly.
                model.dropout = 0.25
                # A fresh checkpoint is used even after the preceding trial recalibrates BN.
                for n, v in clean.items():
                    if n.startswith(('BNs.', 'BNend.')):
                        self.assertTrue(torch.equal(model.state_dict()[n], v), n)
                observed.append({n: p.detach().clone() for n, p in model.named_parameters()})
                if args[-2] > 0:
                    for n, p in model.named_parameters():
                        if n.endswith(('FFconv.weight', 'FBconv.weight')):
                            self.assertFalse(torch.equal(p, clean[n]), n)
                    self.assertFalse(torch.equal(model.linear.weight, clean['linear.weight']))
                    self.assertFalse(torch.equal(model.linear.bias, clean['linear.bias']))
                events = []
                from torch.nn import functional as F
                original_dropout = F.dropout

                def dropout(input, p=0.5, training=True, inplace=False):
                    events.append(training)
                    return original_dropout(input, p, training, inplace)

                with patch('pc_model.F.dropout', side_effect=dropout):
                    result = real_evaluate_pair(model, *args)
                self.assertTrue(events)
                self.assertFalse(any(events))
                self.assertEqual(result[2].num_calibration_samples, 4)
                self.assertEqual(result[2].num_bn_modules, 2)
                self.assertTrue(result[2].changed_bn_buffers)
                return result

            for mismatch, scale in [('mul', 'max_abs'), ('add', 'max_abs'), ('add', 'rms')]:
                with self.subTest(mismatch=mismatch, scale=scale):
                    with patch.object(sys, 'argv', argv + ['--mismatch_type', mismatch,
                                                         '--additive_scale_mode', scale]):
                        args = parse_args()
                    with patch.object(evaluation, 'get_test_data', return_value=test_loader), \
                         patch.object(evaluation, 'build_bn_calibration_loader', return_value=calibration), \
                         patch.object(evaluation, 'evaluate_pair', side_effect=checked_pair), \
                         patch('torch.cuda.is_available', return_value=False):
                        observed.clear()
                        evaluation.run_paired_bn_evaluation(args)
                        first = observed[:]
                        observed.clear()
                        evaluation.run_paired_bn_evaluation(args)
                    self.assertEqual(len(observed), 3)
                    for old, new in zip(first, observed):
                        for n in old:
                            self.assertTrue(torch.equal(old[n], new[n]), n)
                    output = root / 'results'
                    self.assertEqual(json.loads((output / 'paired_complete.json').read_text())['status'], 'complete')
                    for mode in ('frozen', 'recalibrated'):
                        with (output / mode / 'result.pkl').open('rb') as handle:
                            values = pickle.load(handle)[1.75]['noise_acc_spec']['Johnson']
                        self.assertEqual(len(values[0]), 1)
                        self.assertEqual(len(values[0.01]), 2)
                    self.assertEqual(len(json.loads((output / 'trials.json').read_text())), 3)


if __name__ == '__main__':
    unittest.main()
