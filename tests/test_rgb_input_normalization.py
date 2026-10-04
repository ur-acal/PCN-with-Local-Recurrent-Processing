"""Normalization-only controls: no data downloads or scheduler submissions."""
import contextlib
import copy
import io
import os
import random
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from data_utils import _CIFAR_STATS
from input_preprocessing import resolve_student_normalization
from rgb_teacher_preprocessing import (
    prepare_rgb_teacher_input, rgb_teacher_metadata, rgb_teacher_transforms,
    resolve_teacher_training_normalization)
from trainer import TrainerCiFar
from trainer_timm import TrainerCiFarTimmStyle

ROOT = Path(__file__).resolve().parents[1]


class RGBNormalizationTests(unittest.TestCase):
    def test_teacher_views_all_four_combinations_and_legacy(self):
        raw = torch.rand(4, 3, 32, 32)
        for dataset in ('cifar10', 'cifar100'):
            mean, std = _CIFAR_STATS[dataset]
            normalized = transforms.Normalize(mean, std)(raw)
            for student_norm in (False, True):
                for teacher_norm in (False, True):
                    teacher = nn.Identity()
                    teacher.rgb_teacher_input_size = 40
                    teacher.rgb_teacher_preprocessing = rgb_teacher_metadata(
                        dataset, 40, normalize=teacher_norm)
                    source = normalized if student_norm else raw
                    original = source.clone()
                    got = prepare_rgb_teacher_input(source, teacher, dataset, student_norm)
                    expected = torch.nn.functional.interpolate(
                        normalized if teacher_norm else raw, size=(40, 40),
                        mode='bilinear', align_corners=False)
                    torch.testing.assert_close(got, expected, rtol=1e-6, atol=3e-7)
                    self.assertTrue(torch.equal(source, original))
                    if student_norm == teacher_norm:
                        self.assertTrue(torch.equal(got, expected))
            # Metadata-free teacher retains its old normalized input unchanged.
            self.assertIs(prepare_rgb_teacher_input(normalized, nn.Identity(), dataset), normalized)

    def test_teacher_only_normalization_changes_no_augmentation(self):
        image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3))
        for dataset in ('cifar10', 'cifar100'):
            on = rgb_teacher_transforms(dataset, 40, 40)
            off = rgb_teacher_transforms(dataset, 40, 40, normalize=False)
            for normal_transform, raw_transform in zip(on, off):
                torch.manual_seed(17)
                normalized = normal_transform(image)
                torch.manual_seed(17)
                raw = raw_transform(image)
                self.assertGreaterEqual(raw.min().item(), 0)
                self.assertLessEqual(raw.max().item(), 1)
                expected = transforms.Normalize(*_CIFAR_STATS[dataset])(raw)
                torch.testing.assert_close(normalized, expected, rtol=1e-6, atol=4e-7)

    def test_student_timm_augs_unchanged_except_normalization(self):
        from baseline.baseline_cifar_configs import CASE_DEFAULTS
        cfg = CASE_DEFAULTS['custom_noresize']
        trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
        trainer._get_timm_data_config = lambda dataset: ((3, 32, 32), *_CIFAR_STATS[dataset], 'bicubic')
        trainer.timm_train_scale = cfg['timm_train_scale']
        trainer.timm_train_ratio = cfg['timm_train_ratio']
        trainer.hflip = cfg['hflip']; trainer.vflip = 0
        trainer.auto_augment = cfg['auto_augment']; trainer.color_jitter = cfg['color_jitter']
        trainer.re_prob = 0.; trainer.re_mode = 'pixel'; trainer.re_count = 1
        for enabled in (True, False):
            trainer.normalize_student_input = enabled
            mock_transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize(*_CIFAR_STATS['cifar100'])])
            with patch('trainer_timm.create_transform', return_value=mock_transform) as create:
                _, test = trainer._build_rgb_timm_transforms('cifar100')
            kw = create.call_args.kwargs
            self.assertEqual(kw['auto_augment'], cfg['auto_augment'])
            self.assertEqual(kw['hflip'], cfg['hflip'])
            self.assertEqual(kw['re_prob'], 0.)
            self.assertEqual(kw['mean'], _CIFAR_STATS['cifar100'][0])
            image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
            expected = transforms.ToTensor()(image)
            if enabled:
                expected = transforms.Normalize(*_CIFAR_STATS['cifar100'])(expected)
            self.assertTrue(torch.equal(test(image), expected))

    def test_shared_mixup_teacher_and_student_logits_and_gradients(self):
        # Affine normalization commutes with the shared Mixup operation.
        torch.manual_seed(3)
        raw = torch.rand(4, 3, 8, 8)
        mean, std = _CIFAR_STATS['cifar100']
        normalized = transforms.Normalize(mean, std)(raw)
        mix = lambda x: .7*x + .3*x.flip(0)
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 5))
        teacher.rgb_teacher_preprocessing = rgb_teacher_metadata('cifar100', 8)
        teacher.rgb_teacher_input_size = 8
        got = prepare_rgb_teacher_input(mix(raw), teacher, 'cifar100', False)
        torch.testing.assert_close(teacher(got), teacher(mix(normalized)), rtol=1e-5, atol=2e-7)
        # Default-enabled teacher conversion is exactly the old path, including gradients.
        student = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 5))
        reference = copy.deepcopy(student)
        a = teacher(prepare_rgb_teacher_input(normalized, teacher, 'cifar100')).detach()
        b = teacher(normalized).detach()
        ya = student(normalized); yb = reference(normalized)
        (ya-a).square().mean().backward(); (yb-b).square().mean().backward()
        self.assertTrue(torch.equal(ya, yb))
        for p, q in zip(student.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_real_randaugment_preserves_fill_and_samples(self):
        from baseline.baseline_cifar_configs import CASE_DEFAULTS
        cfg = CASE_DEFAULTS['custom_noresize']
        trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
        trainer._get_timm_data_config = lambda dataset: ((3, 32, 32), *_CIFAR_STATS[dataset], 'bicubic')
        for name, value in dict(timm_train_scale=cfg['timm_train_scale'],
                timm_train_ratio=cfg['timm_train_ratio'], hflip=.5, vflip=0,
                auto_augment=cfg['auto_augment'], color_jitter=cfg['color_jitter'],
                re_prob=0., re_mode='pixel', re_count=1).items():
            setattr(trainer, name, value)
        trainer.normalize_student_input = True
        normal, _ = trainer._build_rgb_timm_transforms('cifar100')
        trainer.normalize_student_input = False
        raw, _ = trainer._build_rgb_timm_transforms('cifar100')
        image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3))
        for seed in range(20):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            expected = normal(image)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            got = transforms.Normalize(*_CIFAR_STATS['cifar100'])(raw(image))
            self.assertTrue(torch.equal(got, expected))

    def test_checkpoint_inheritance_overrides_and_exact_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'student.pth'
            for enabled in (False, True):
                torch.save({'student_preprocessing': {'normalize_student_input': enabled}}, p)
                self.assertIs(resolve_student_normalization(p), enabled)
                self.assertIs(resolve_student_normalization(p, not enabled), not enabled)
                with self.assertRaisesRegex(ValueError, 'Exact recovery'):
                    resolve_student_normalization(p, not enabled, exact=True)
            torch.save({'net': {}}, p)
            self.assertTrue(resolve_student_normalization(p))
            self.assertTrue(resolve_student_normalization())

    def test_teacher_checkpoint_save_and_test_only_inheritance(self):
        from train_teacher import save_teacher_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'teacher.pth'
            args = SimpleNamespace(img_type='rgb', dataset='cifar100', test_size=40,
                normalize_input=False, mismatch_type='mul', mismatch_ramp_start=None,
                mismatch_ramp_epochs=0)
            save_teacher_checkpoint(p, nn.Linear(2, 2), args, 1, [])
            meta = torch.load(p, weights_only=False)['teacher_preprocessing']
            self.assertFalse(meta['normalize'])
            self.assertEqual(meta['kind'], 'rgb_cifar_resize')
            args.test_only = True; args.checkpoint = str(p); args.normalize_input = None
            self.assertFalse(resolve_teacher_training_normalization(args))
            args.normalize_input = True
            with self.assertRaisesRegex(ValueError, 'must match'):
                resolve_teacher_training_normalization(args)

    def test_distillation_teacher_loader_uses_checkpoint_normalization(self):
        from train_ode_cifar import build_teacher_model
        def model_factory(weights=None):
            model = nn.Module()
            model.features = nn.Sequential(nn.Conv2d(3, 4, 1))
            model.classifier = nn.Sequential(nn.Linear(4, 100))
            return model
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'teacher.pth'
            args = SimpleNamespace(distill_method='srrl', distill_alpha=.3,
                teacher_ckpt=str(p), teacher_arch='efficientnet_v2_l',
                teacher_arch_source='torchvision', num_classes=100,
                img_type='rgb', dataset='cifar100')
            for normalize in (True, False):
                meta = rgb_teacher_metadata('cifar100', 40, normalize)
                torch.save(dict(net=model_factory().state_dict(), teacher_preprocessing=meta), p)
                with patch('torchvision.models.efficientnet_v2_l', model_factory), \
                        patch('torch.cuda.is_available', return_value=False):
                    teacher = build_teacher_model(args, student_in_channels=3)
                self.assertEqual(teacher.rgb_teacher_preprocessing, meta)
                self.assertIs(teacher.rgb_teacher_preprocessing['normalize'], normalize)
                self.assertEqual(teacher.rgb_teacher_input_size, 40)
                trainer = TrainerCiFar.__new__(TrainerCiFar)
                trainer.teacher_model = teacher; trainer.img_type = 'rgb'
                trainer.dataset_name = 'cifar100'; trainer.normalize_student_input = False
                raw = torch.rand(2, 3, 32, 32)
                expected = transforms.Normalize(*_CIFAR_STATS['cifar100'])(raw) if normalize else raw
                expected = torch.nn.functional.interpolate(expected, (40, 40), mode='bilinear', align_corners=False)
                self.assertTrue(torch.equal(trainer._prepare_rgb_teacher_inputs(raw), expected))

    def test_recovery_saves_raw_setting_and_rejects_changed_recipe(self):
        from training_recovery import save_latest, restore_latest, latest_path, HISTORY
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(model=nn.Linear(2, 2), device='cpu',
                save_path=directory, model_name='tiny', normalize_student_input=False,
                recovery_config={'normalize_student_input': False},
                dataset_name='cifar100', img_type='rgb')
            history = {key: None for key in HISTORY}
            save_latest(trainer, 3, history)
            path = latest_path(trainer)
            self.assertFalse(resolve_student_normalization(path, exact=True))
            trainer.recovery_checkpoint = path
            trainer.recovery_config['normalize_student_input'] = True
            with self.assertRaisesRegex(ValueError, 'unchanged normalize_student_input'):
                restore_latest(trainer)

    def test_real_tc_one_and_two_state_forward_backward(self):
        from ode_pc import ODEXInitFFFB, S2NoisyIYAsXZAs0, QATWrapper1State, QATWrapper2State
        from pc_conv import PCConvReLU6
        for block_cls, wrapper_cls in ((ODEXInitFFFB, QATWrapper1State),
                                       (S2NoisyIYAsXZAs0, QATWrapper2State)):
            for normalize in (True, False):
                torch.manual_seed(31)
                pc = PCConvReLU6(inp_chan=3, out_chan=3, kernel_size=1, padding=0,
                    cls=2, bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
                block = block_cls(pc_conv=pc, noise_level=0., method='dopri5',
                                 t_end=.03, tol=1e-4, sde_noise_type='addi')
                wrapper = wrapper_cls(ode_block=block, tc_nonidealities=True,
                    R=1e4, R_max=150e3, C=49e-15, k=1e3, v_dd=.1, state_bound=1.,
                    w_bits=5, thermal_noise=False, offset_eps=0., is_first=True, is_last=True)
                raw = torch.rand(2, 3, 2, 2)
                inputs = transforms.Normalize(*_CIFAR_STATS['cifar100'])(raw) if normalize else raw
                for _ in range(3):
                    block.zero_grad()
                    outputs = block(inputs)
                    outputs.square().mean().backward()
                    self.assertTrue(torch.isfinite(outputs).all())
                    gradients = [p.grad for p in block.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                if not normalize:
                    # No information lost at the initial hardware input projection.
                    self.assertTrue(torch.equal(wrapper.wrap_input(raw), raw*.1))

    def test_local_tc_launch_forwarding_and_defaults(self):
        from train_ode_cifar import get_args
        from ode_inference import parse_args
        for stage, parser in [('ft', get_args), ('eval', parse_args)]:
            for value in ('true', 'false', ''):
                env = dict(os.environ, TC_DRY_RUN='true', MODEL_NAME='TIMMPCNet_C100',
                           IMG_TYPE='rgb', NORMALIZE_STUDENT_INPUT=value)
                out = subprocess.check_output(['bash', 'launch_scripts/run_tc_nonidealities.sh', stage],
                                              cwd=ROOT, env=env, text=True)
                with patch.dict(os.environ, env), patch.object(sys, 'argv', shlex.split(out)[2:]), \
                        contextlib.redirect_stderr(io.StringIO()):
                    args = parser()
                self.assertIs(args.normalize_student_input, None if value == '' else value == 'true')
                if stage == 'ft':
                    self.assertEqual(args.timm_re_prob, 0)

    def test_teacher_launch_forwarding_and_distinct_output(self):
        import train_teacher, train_teacher_timm
        for recipe, module in [('legacy', train_teacher), ('timm_oldaugs', train_teacher_timm)]:
            env = dict(os.environ, TEACHER_DRY_RUN='true', REPO_ROOT=str(ROOT),
                       RGB_TEACHER_RECIPE=recipe, NORMALIZE_TEACHER_INPUT='false')
            out = subprocess.check_output(['bash', 'launch_scripts/run_rgb_teacher.sbatch'],
                                          cwd=ROOT, env=env, text=True)
            lines = [shlex.split(line)[2:] for line in out.splitlines() if line.startswith('python ')]
            self.assertEqual(len(lines), 2)
            for argv in lines:
                with patch.object(sys, 'argv', argv):
                    args = module.parse_args()
                self.assertFalse(args.normalize_input)
                self.assertIn('_NoNorm', args.checkpoint)

    def test_slurm_student_export_and_real_worker_stages(self):
        from train_ode_cifar import get_args
        from ode_inference import parse_args
        env = dict(os.environ, TC_DRY_RUN='true', TC_NONIDEALITIES='true',
                   TC_STATE='1', TOGGLE_MODE='none', SWITCH_INF='false',
                   TASK='cifar100', IMG_TYPE='rgb', NORMALIZE_STUDENT_INPUT='false',
                   PCN_CHAN_0_LIST='16', PCN_NUM_LAYERS_LIST='22',
                   NUM_COMB_PER_NUM_LAYER='3', COMB_SEL_SET='2')
        out = subprocess.check_output(['bash', 'launch_scripts/slurm_search_config.sh'],
                                      cwd=ROOT, env=env, text=True)
        submissions = [line for line in out.splitlines() if line.startswith('sbatch ')]
        self.assertTrue(submissions)
        self.assertTrue(all('NORMALIZE_STUDENT_INPUT=false' in line for line in submissions))
        worker = (ROOT/'launch_scripts/run_kdcrd_then_ft.sbatch').read_text().split('# PHASE 1:')[0]
        worker = worker.replace('source activate base', ':').replace('conda activate scanbase', ':')
        with tempfile.TemporaryDirectory() as directory:
            worker = '\n'.join('LOGDIR='+shlex.quote(directory) if line.startswith('LOGDIR=')
                               else line for line in worker.splitlines())
            commands = worker + '''
python() { printf 'CAPTURE '; printf '%q ' "$@"; printf '\\n'; }
run_one_combo $'inspect\\t0\\t3 4\\t4 4\\t0 0' "$ODE_BLOCK"
finetune_one_combo_model inspect "$MODEL_NAME" "$ODE_BLOCK"
eval_one_combo_model inspect "$MODEL_NAME" post_ft "$ODE_BLOCK"
'''
            env.update(OUTPUT_SAVE_PATH=directory, COMB_LIST='inspect',
                       MODEL_NAME='PCNetNoBatchNorm_PCConvReLU6_0.0eps_ODEXInitFFFB_1.75TEnd_16Layers')
            out = subprocess.check_output(['bash', '-c', commands], cwd=ROOT, env=env, text=True)
            captured = [shlex.split(line)[2:] for line in out.splitlines() if line.startswith('CAPTURE ')]
            self.assertEqual(len(captured), 3)
            for argv, parser in zip(captured, (get_args, get_args, parse_args)):
                with patch.dict(os.environ, env), patch.object(sys, 'argv', argv), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertFalse(parser().normalize_student_input)

    def test_evaluation_loader_and_student_checkpoint_metadata(self):
        from inference_utils import get_test_data
        from torch.utils.data import Dataset
        class Data(Dataset):
            def __init__(self, root, train, download, transform):
                self.transform = transform
            def __len__(self): return 2
            def __getitem__(self, index):
                return self.transform(Image.fromarray(np.full((32,32,3), 255, dtype=np.uint8))), index
        with patch('torchvision.datasets.CIFAR100', Data), \
                patch.dict(os.environ, {'DATALOADER_NUM_WORKERS': '0'}):
            raw, _ = next(iter(get_test_data(test_bs=2, task='cifar100', normalize_student_input=False)))
            self.assertTrue(torch.equal(raw, torch.ones_like(raw)))
            normal, _ = next(iter(get_test_data(test_bs=2, task='cifar100')))
            self.assertTrue(torch.equal(normal, transforms.Normalize(*_CIFAR_STATS['cifar100'])(raw)))
        with tempfile.TemporaryDirectory() as directory:
            trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
            trainer.is_timm_model = True; trainer.save_path = directory; trainer.model_name = 'tiny'
            trainer.model = nn.Linear(2, 2); trainer.normalize_student_input = False
            trainer.dataset_name = 'cifar100'; trainer.img_type = 'rgb'
            trainer.save_flattened_and_full_param = False
            for name in ('timm_model_name', 'pretrained', 'use_model_data_config',
                         'timm_input_size', 'timm_mean', 'timm_std', 'interpolation', 'validation_split_metadata'):
                setattr(trainer, name, None)
            path = trainer._save_model_ckpt(0., 1, '_last_ckpt.pth')
            self.assertFalse(resolve_student_normalization(path))

    def test_calibration_normalization_and_cache_separation(self):
        import cross_sim_inference as calibration
        from torch.utils.data import Dataset, DataLoader

        class Data(Dataset):
            def __init__(self, root, train, download, transform):
                self.transform = transform
            def __len__(self): return 2
            def __getitem__(self, index):
                image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
                return self.transform(image), index

        def loader(*args, **kwargs):
            kwargs['num_workers'] = 0
            return DataLoader(*args, **kwargs)

        for dataset in ('cifar10', 'cifar100'):
            with tempfile.TemporaryDirectory() as directory, \
                    patch('torchvision.datasets.CIFAR10', Data), \
                    patch('torchvision.datasets.CIFAR100', Data), \
                    patch('torch.utils.data.DataLoader', side_effect=loader):
                model = nn.Sequential(nn.Conv2d(3, 1, 1))
                results = {}
                for enabled in (True, False):
                    torch.manual_seed(123)
                    results[enabled] = calibration.calibrate_input(
                        model, 'cpu', 'tiny', calib_bs=2, save_to=directory,
                        dataset_name=dataset, normalize_student_input=enabled)
                self.assertEqual(results[False]['min_max'][0, 1], 1.)
                self.assertGreater(results[True]['min_max'][0, 1], 1.)
                self.assertEqual(len(list(Path(directory).glob('*.pkl'))), 2)
                with patch.object(calibration, 'get_calib_loader', side_effect=AssertionError('cache miss')):
                    for enabled in (True, False):
                        cached = calibration.calibrate_input(
                            model, 'cpu', 'tiny', calib_bs=2, save_to=directory,
                            dataset_name=dataset, normalize_student_input=enabled)
                        np.testing.assert_array_equal(cached['min_max'], results[enabled]['min_max'])

    def test_test_only_forwards_normalization_to_calibration(self):
        import ode_inference as inference
        with patch.object(sys, 'argv', ['ode_inference.py', '--model_dir', '.',
                '--model_name', 'tiny_1.75TEnd_1Layers', '--ode_wrapper', 'none',
                '--ode_block', 'ODEXInitFFFB']):
            args = inference.parse_args()
        model = nn.Sequential(nn.Conv2d(3, 1, 1))
        model.ics = [3]; model.ocs = [1]; model.max_pool = [False]
        for enabled in (False, True):
            args.normalize_student_input = enabled
            with patch.object(inference, 'load_and_prepare_model', return_value=model), \
                    patch.object(inference, 'test_once'), \
                    patch.object(inference, 'calibrate_input', return_value={}) as calibrate, \
                    contextlib.redirect_stdout(io.StringIO()):
                inference.run_test_only(args, [], 'unused.pth', None, 'cpu')
            self.assertIs(calibrate.call_args.kwargs['normalize_student_input'], enabled)

    def test_nonrgb_teacher_preprocessing_unchanged(self):
        trainer = TrainerCiFar.__new__(TrainerCiFar)
        trainer.teacher_input_size = 4; trainer.adapt_PIL_teacher = False
        trainer.teacher_center_crop = False; trainer.img_type = 'CiFAIR'
        trainer.normalize_student_input = False
        raw = torch.rand(2, 4, 4, 4)
        self.assertTrue(torch.equal(trainer._prepare_teacher_inputs(raw), (raw-.5)/.5))
        self.assertIs(trainer._prepare_rgb_teacher_inputs(raw), raw)


if __name__ == '__main__':
    unittest.main()
