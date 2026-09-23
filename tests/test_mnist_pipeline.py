"""MNIST assembly, recipe, real trainer/checkpoint round trips and shell wiring."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_train_eval.mnist_config import parse_args, model_name, ROOT
from mnist_train_eval.mnist_data import DEFAULT_DATA_DIR
from mnist_train_eval.mnist_train import build_cnn, build_pcn
from mnist_train_eval.mnist_evaluate import build_pcn_inference, cpu_unroll_convolution
from mnist_train_eval.mnist_trainer import MNISTTrainer


class MNISTPipelineTests(unittest.TestCase):
    def test_launcher_modes_preserve_stage_commands(self):
        # Capture actual shell-built argv; never invoke training or CUDA.
        script = '''python() { printf 'CAPTURE'; printf ' <%s>' "$@"; printf '\\n'; }
source ./launch_scripts/run_mnist_pipeline.sh
'''
        with tempfile.TemporaryDirectory() as directory:
            for family, name in [('pcn', 'mnist_pcn3_state1'), ('cnn', 'mnist_cnn5_avgpool')]:
                output = Path(directory) / family
                checkpoint = output / (name + '_pretrain') / (name + '_pretrain_last_ckpt.pth')
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b'fixture')
                env = dict(PATH=os.environ['PATH'], HOME=os.environ.get('HOME', '/tmp'),
                           FAMILY=family, VARIANT='deep', OUTPUT_DIR=str(output))
                def run(**extra):
                    return subprocess.run(['bash', '-c', script], cwd=ROOT,
                        env=dict(env, **extra), capture_output=True, text=True)
                def commands(result):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return [line for line in result.stdout.splitlines() if line.startswith('CAPTURE')]
                baseline = commands(run())
                self.assertEqual(len(baseline), 3)
                for mode, indices in [('default', [0, 1, 2]), ('pretrain_only', [0]),
                                      ('ft_only', [1]), ('ft_and_eval', [1, 2])]:
                    self.assertEqual(commands(run(mode=mode)), [baseline[i] for i in indices])
                for stage, index in [('pretrain', 0), ('ft', 1), ('eval', 2)]:
                    self.assertEqual(commands(run(STAGE=stage)), [baseline[index]])
                self.assertNotEqual(run(mode='unknown').returncode, 0)
                self.assertNotEqual(run(mode='ft_and_eval', MODEL_CKPT=str(checkpoint)).returncode, 0)
                failed_ft = script.replace("printf 'CAPTURE'", "return 7; printf 'CAPTURE'")
                result = subprocess.run(['bash', '-c', failed_ft], cwd=ROOT,
                    env=dict(env, mode='ft_and_eval'), capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertNotIn('mnist_evaluate', result.stdout)
                checkpoint.unlink()
                for mode in ('ft_only', 'ft_and_eval'):
                    result = run(mode=mode)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Missing or empty pretrained checkpoint', result.stderr)
                    self.assertNotIn('CAPTURE', result.stdout)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recipe_and_external_data(self):
        with patch.dict(os.environ, {}, clear=True):
            pre = parse_args([])
            ft = parse_args(['--stage', 'ft', '--checkpoint', 'unused'])
            ev = parse_args(['--checkpoint', 'unused'], evaluation=True)
        self.assertEqual(DEFAULT_DATA_DIR, ROOT.parent / 'data')
        self.assertEqual((pre.epochs, pre.batch_size, pre.optimizer, pre.lr, pre.gamma),
                         (14, 64, 'adadelta', 1., .7))
        self.assertEqual(ft.lr, .01)
        self.assertEqual(ft.tol, 1e-6)
        self.assertEqual(ev.n_trials, 10)
        self.assertTrue(ft.activation_curve_path.endswith('/tt_25_1.csv'))
        self.assertTrue(ev.activation_curve_path.endswith('/0906_RELU_Voltage'))
        from trainer import TrainerCiFar
        self.assertEqual(TrainerCiFar.normalize_dataset_name('cifar100'), 'cifar100')
        with self.assertRaises(ValueError):
            TrainerCiFar.normalize_dataset_name('mnist')

    def test_evaluation_releases_previous_trial_before_next_build(self):
        from mnist_train_eval.mnist_evaluate import main

        class TinyModel(nn.Module):
            def forward(self, x):
                return x.new_zeros((x.shape[0], 10))

        references = []
        def build(*args, **kwargs):
            if references:
                self.assertIsNone(references[-1](), 'Previous hardware model is still retained')
            model = TinyModel()
            references.append(weakref.ref(model))
            return model, [SimpleNamespace(block=model)]

        loader = DataLoader(TensorDataset(torch.zeros(2, 1, 28, 28),
                                         torch.zeros(2, dtype=torch.long)), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pth'
            torch.save({'mnist_config': dict(family='cnn', variant='small', tc_state=1,
                       dropout=.25, t_end=1.75, stage='ft')}, checkpoint)
            with patch('mnist_train_eval.mnist_evaluate.device_for', return_value=torch.device('cpu')), \
                 patch('mnist_train_eval.mnist_evaluate.mnist_loader', return_value=loader), \
                 patch('mnist_train_eval.mnist_evaluate.build_cnn', new=build), \
                 patch('feedforward_validation.FeedForwardCNNValidator',
                       new=lambda model, *a, **k: SimpleNamespace(model=model)):
                records = main(['--checkpoint', str(checkpoint), '--output_dir', directory,
                                '--n_trials', '2', '--num_workers', '0'])
            self.assertEqual(len(records), 2)
            self.assertTrue(all(ref() is None for ref in references))

    def test_registered_cnn_shapes_and_biases(self):
        import timm
        for variant, count in [('small', 3), ('deep', 5)]:
            a = parse_args(['--variant', variant])
            model = timm.create_model(model_name(a), in_chans=1, num_classes=10)
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            self.assertEqual(len(convs), count)
            self.assertTrue(all(m.bias is None for m in convs))
            self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
            self.assertIsNotNone(model.fc.bias)
            self.assertEqual(model(torch.randn(2, 1, 28, 28)).shape, (2, 10))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_qat_installation_order(self):
        if torch.cuda.mem_get_info()[0] < 4 * 2**30:
            self.skipTest('Insufficient free GPU memory for the bounded smoke test')
        args = parse_args(['--stage', 'ft', '--checkpoint', 'fixture'])
        model, _ = build_cnn(args, torch.device('cuda'))
        output = model(torch.rand(2, 1, 28, 28, device='cuda'))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        from physical_feedforward_tc import TCPhysicalBasicBlock
        kernel = torch.randn(2, 1, 3, 3, device='cuda')
        expected, _, _ = TCPhysicalBasicBlock.unroll_convolution((1, 4, 4), kernel, padding=1)
        actual, _, _ = cpu_unroll_convolution((1, 4, 4), kernel, padding=1)
        self.assertTrue(torch.equal(actual.crow_indices(), expected.crow_indices()))
        self.assertTrue(torch.equal(actual.col_indices(), expected.col_indices()))
        self.assertTrue(torch.equal(actual.values(), expected.values()))

    def test_actual_training_ft_and_checkpoint_reload(self):
        # Real forward/backward, health evaluation, both serializers and physical
        # reloads; only the small dataset and CPU selection are test fixtures.
        loader = DataLoader(TensorDataset(torch.rand(2, 1, 28, 28),
                                         torch.tensor([0, 1])), batch_size=2)
        def make_loader(*args, **kwargs):
            return DataLoader(loader.dataset, batch_size=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('mnist_train_eval.mnist_trainer.mnist_loader', side_effect=make_loader):
            for family, state, variant in ((f, s, v) for f, s in
                    [('cnn', 1), ('pcn', 1), ('pcn', 2)] for v in ('small', 'deep')):
                common = ['--family', family, '--tc_state', str(state), '--variant', variant, '--epochs', '1',
                          '--output_dir', directory, '--num_workers', '0',
                          '--health_check_epochs', '1']
                pre = parse_args(common)
                build = build_cnn if family == 'cnn' else build_pcn
                model, _ = build(pre, torch.device('cpu'))
                trainer = MNISTTrainer(model, pre, model_name(pre) + '_pretrain')
                self.assertIsInstance(trainer.optimizer, torch.optim.Adadelta)
                with patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluate:
                    result = trainer.train()
                    self.assertEqual(evaluate.call_count, 2)  # train health + final test
                    self.assertEqual(sum(c.args[0] is trainer.val_dataloader for c in evaluate.call_args_list), 1)
                self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], .7)
                checkpoint = torch.load(result['checkpoint'], weights_only=False)
                ft = parse_args(common + ['--stage', 'ft', '--checkpoint', result['checkpoint']])
                model, wrappers = build(ft, torch.device('cpu'), checkpoint=checkpoint)
                self.assertTrue(wrappers)
                trainer = MNISTTrainer(model, ft, model_name(ft) + '_ft')
                result = trainer.train()
                full = torch.load(result['checkpoint'], weights_only=False)
                self.assertEqual(full['checkpoint_weight_format'], 'full_param')
                baked_path = result['checkpoint'].replace('_full_param_last', '_last')
                baked = torch.load(baked_path, weights_only=False)
                self.assertEqual(baked['checkpoint_weight_format'], 'flattened_quantized')
                ev = parse_args(common + ['--stage', 'eval', '--checkpoint', baked_path], evaluation=True)
                if family == 'cnn':
                    restored, _ = build_cnn(ev, torch.device('cpu'), inference=True, checkpoint=baked)
                else:
                    restored, _ = build_pcn_inference(ev, torch.device('cpu'), 4096)
                restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(restored(next(iter(loader))[0])).all())

    def test_shell_resolves_stages_and_preserves_failure(self):
        # Shell function records argv without executing training/submitting jobs.
        script = '''
python() { printf '%s\\n' "$*"; [[ "$*" != *"--stage $FAIL_STAGE "* ]]; }
source ./launch_scripts/run_mnist_pipeline.sh
'''
        base = dict(PATH=os.environ['PATH'], HOME=os.environ['HOME'], REPO_ROOT=str(ROOT),
                    FAMILY='pcn', TC_STATE='2', OUTPUT_DIR='/tmp/mnist-shell-test')
        result = subprocess.run(['bash', '-c', script], cwd=ROOT, env=base,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith('-u -m')]
        self.assertEqual(len(lines), 3)
        self.assertIn('--lr 1.0', lines[0])
        self.assertIn('--lr 0.01', lines[1])
        self.assertIn('mnist_pcn2_state2_ft_last_ckpt.pth', lines[2])
        failed = subprocess.run(['bash', '-c', script], cwd=ROOT,
                                env=dict(base, FAIL_STAGE='ft'), capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn('--stage eval', failed.stdout)

    def test_slurm_inherits_environment_without_submitting(self):
        script = '''
sbatch() { printf '%s\\n' "$*" "$FAMILY/$VARIANT/$TC_STATE"; }
source ./launch_scripts/slurm_run_mnist.sh
'''
        env = dict(os.environ, REPO_ROOT=str(ROOT), FAMILY='pcn', VARIANT='deep', TC_STATE='2')
        env.pop('DRY_RUN', None)
        result = subprocess.run(['bash', '-c', script], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--parsable --gres=gpu:1 --export=ALL', result.stdout)
        self.assertIn('pcn/deep/2', result.stdout)
        worker = (ROOT / 'launch_scripts/mnist_pipeline.sbatch').read_text()
        established = (ROOT / 'launch_scripts/run_kdcrd_then_ft.sbatch').read_text()
        for directive in ('#SBATCH -p ising', '#SBATCH -t 90:10:00',
                          '#SBATCH --nodelist=bhgrb4x0081,bhgrb4x0082'):
            self.assertIn(directive, established)
            self.assertIn(directive, worker)
        self.assertIn('source activate base\nconda activate scanbase', worker)
        self.assertNotIn('module swap', worker)


if __name__ == '__main__':
    unittest.main()
