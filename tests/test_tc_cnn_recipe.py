"""Exercise real CNN construction, sampling, preprocessing, and checkpoint inheritance."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline import train_baseline_cifar as training
from baseline import evaluate_physical_feedforward_cifar as evaluation
from baseline.cifar_resnet import CIFARResNet
from input_preprocessing import student_preprocessing_metadata
from measured_activation import PiecewiseLinearActivation


class TCCNNRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def construct(self, directory, *extra):
        captured = {}
        original = training.build_trainer_kwargs

        def collect(args, cfg, model, teacher):
            kwargs = original(args, cfg, model, teacher)
            captured.update(model=model, kwargs=kwargs, args=args)
            return kwargs

        arguments = ['train', '--model_name', 'resnet44_cifar', '--dataset', 'cifar100',
                     '--output_dir', str(directory), '--print_only', 'true',
                     '--physical_feedforward', 'true', '--tc_feedforward', 'true',
                     '--R', '10000', '--C', '49e-15', '--v_dd', '0.1',
                     '--one_over_q', '32', '--enob', 'none', '--distill_method', 'none',
                     *extra]
        with patch.object(sys, 'argv', arguments), \
                patch.object(training, 'build_model', side_effect=lambda *args:
                             CIFARResNet(depth=8, base_width=2, num_classes=100)), \
                patch.object(training, 'build_trainer_kwargs', side_effect=collect), \
                patch('torch.cuda.is_available', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            training.main()
        return captured

    def test_raw_pretrain_to_random_per_spin_ft_and_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre = self.construct(root / 'pre', '--physical_pretraining', 'true',
                                 '--normalize_student_input', 'false')
            self.assertFalse(pre['kwargs']['normalize_student_input'])
            pretrained = root / 'pretrained.pth'
            torch.save(dict(net=pre['model'].state_dict(),
                            student_preprocessing=student_preprocessing_metadata(
                                SimpleNamespace(**pre['kwargs'])),
                            measured_activation_scope=pre['model'].measured_activation_scope), pretrained)
            ft = self.construct(
                root / 'ft', '--resume_checkpoint', str(pretrained),
                '--timm_aug_level', 'rgb_mid', '--enable_measured_activation', 'true',
                '--activation_curve_path', './hardware_data/mc_45_corners/0906_RELU_Voltage',
                '--activation_corner', 'TT_25_1_MC18',
                '--activation_corner_mode', 'random_per_forward',
                '--activation_random_curve_sharing', 'per_spin')
            self.assertFalse(ft['args'].normalize_student_input)
            self.assertFalse(ft['kwargs']['normalize_student_input'])
            model = ft['model'].train()
            measured = [module for module in model.modules()
                        if isinstance(module, PiecewiseLinearActivation)]
            self.assertGreater(len(measured), 1)
            self.assertEqual(model._measured_activation_random_curve_sharing, 'per_spin')
            self.assertTrue(all(len(module.corner_names) > 100 for module in measured))
            self.assertEqual(len({module.curve_seed for module in measured}), len(measured))
            inputs = torch.rand(2, 3, 8, 8)
            output = model(inputs)
            output.square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
            self.assertTrue(any(grad.abs().sum() > 0 for grad in grads))
            first = [module._sampled_curve_indices.clone() for module in measured]
            model(inputs)
            self.assertTrue(any(not torch.equal(old, module._sampled_curve_indices)
                                for old, module in zip(first, measured)))
            self.assertTrue(all(module._sampled_curve_indices.shape[0] == 1 for module in measured))

            checkpoint = root / 'full_param_last.pth'
            torch.save(dict(net=model.state_dict(),
                            student_preprocessing=student_preprocessing_metadata(
                                SimpleNamespace(**ft['kwargs'])),
                            measured_activation_scope=model.measured_activation_scope), checkpoint)
            arguments = ['eval', '--model_name', 'resnet44_cifar', '--checkpoint', str(checkpoint),
                         '--dataset', 'cifar100', '--img_type', 'rgb', '--batch_size', '2',
                         '--tc_feedforward', 'true', '--physical_level', '2',
                         '--R', '10000', '--C', '49e-15', '--one_over_q', '32', '--enob', 'none',
                         '--weight_quant_factor_bits', '-1',
                         '--expanded_weight_dir', str(root / 'expanded'),
                         '--result_path', str(root / 'evaluation'),
                         '--enable_spin_variation', 'false',
                         '--enable_summing_current_noise', 'false', '--enable_coupler_noise', 'false',
                         '--enable_dtc_nonideality', 'false',
                         '--enable_nonlinear_R', 'false', '--enable_measured_pooling', 'false',
                         '--activation_curve_path', './hardware_data/mc_45_corners/0906_RELU_Voltage',
                         '--activation_corner', 'TT_25_1_MC18', '--activation_curve_sharing', 'per_spin']
            loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1])), batch_size=2)
            with patch.object(sys, 'argv', arguments):
                config = evaluation.parse_args()
            with patch.object(evaluation, 'build_model', side_effect=lambda *args:
                              CIFARResNet(depth=8, base_width=2, num_classes=100)), \
                    patch.object(evaluation, 'get_test_data', return_value=loader) as data, \
                    patch('torch.cuda.is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                accuracy = evaluation.evaluate_once(config)
            self.assertTrue(0 <= accuracy <= 100)
            self.assertFalse(data.call_args.kwargs['normalize_student_input'])
            self.assertFalse(config.normalize_student_input)

    def test_default_normalization_remains_on(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.construct(directory, '--physical_pretraining', 'true')
            self.assertTrue(result['kwargs']['normalize_student_input'])

    def test_recovery_rejects_changed_random_sampling(self):
        from training_recovery import restore_latest
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'latest.pth'
            torch.save(dict(training_recovery=dict(version=1, config=dict(
                activation_random_curve_sharing='per_spin'))), checkpoint)
            trainer = SimpleNamespace(recovery_checkpoint=str(checkpoint),
                                      recovery_config=dict(activation_random_curve_sharing='per_layer'))
            with self.assertRaisesRegex(ValueError, 'unchanged activation_random_curve_sharing'):
                restore_latest(trainer)

    def test_bn_calibration_keeps_the_student_normalization(self):
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        from data_utils import _CIFAR_STATS
        from inference_utils import get_bn_calibration_data

        class Images:
            def __init__(self, root, train, download, transform):
                self.transform = transform

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return self.transform(Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))), index

        for dataset in ('cifar10', 'cifar100'):
            with patch('torchvision.datasets.CIFAR10', Images), \
                    patch('torchvision.datasets.CIFAR100', Images):
                raw = get_bn_calibration_data(task=dataset, normalize_student_input=False).dataset[0][0]
                normalized = get_bn_calibration_data(task=dataset).dataset[0][0]
                self.assertTrue(torch.equal(raw, torch.ones_like(raw)))
                self.assertTrue(torch.equal(normalized, transforms.Normalize(*_CIFAR_STATS[dataset])(raw)))


if __name__ == '__main__':
    unittest.main()
