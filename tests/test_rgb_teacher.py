"""RGB teacher migration: no downloads, real training, or Slurm submissions."""
import contextlib
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np

from rgb_teacher_preprocessing import rgb_teacher_transforms, rgb_teacher_metadata
from train_teacher import build_transforms, create_datasets, load_efficientnet_v2_4ch
from train_teacher_timm import RGBResizeFineTuneTrainer
from trainer import TrainerCiFar
from trainer_timm import TrainerCiFarTimmStyleFeatureKD

ROOT = Path(__file__).resolve().parents[1]


class RGBTeacherTests(unittest.TestCase):
    def test_transforms_match_distillation_and_preserve_legacy(self):
        image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32,32,3))
        for dataset in ('cifar10', 'cifar100'):
            train, test = rgb_teacher_transforms(dataset)
            self.assertIsInstance(train.transforms[0], transforms.RandomCrop)
            self.assertEqual(train.transforms[0].size, (32,32))
            meta = rgb_teacher_metadata(dataset)
            native = transforms.Normalize(meta['mean'], meta['std'])(transforms.ToTensor()(image))[None]
            trainer = TrainerCiFar.__new__(TrainerCiFar)
            trainer.img_type='rgb'; trainer.input_quant_bits=None
            trainer.teacher_model=nn.Identity()
            self.assertIs(trainer._prepare_rgb_teacher_inputs(native), native)
            trainer.teacher_model.rgb_teacher_input_size=224
            expected = test(image)[None]
            torch.testing.assert_close(trainer._prepare_rgb_teacher_inputs(native), expected, rtol=0, atol=0)
            actual = TrainerCiFarTimmStyleFeatureKD._prepare_feature_kd_teacher_inputs(trainer,native)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Base/logit KD also goes through the same resize boundary.
            trainer._teacher_feature_module=None
            logits,_=trainer._teacher_forward(native)
            torch.testing.assert_close(logits,expected,rtol=0,atol=0)
            self.assertEqual(train(image).shape, (3,224,224))
        args=SimpleNamespace(img_type='CiFAIR',input_quant_bits=None,match_distill_preprocess=True,
            train_size=224,test_size=224,train_transform='cifar',test_center_crop=True)
        old,_=build_transforms(args)
        self.assertEqual(type(old.transforms[0]).__name__,'DirectTensorResize')

    def test_rgb_dataset_root_and_no_sensor_path(self):
        args=SimpleNamespace(img_type='rgb',dataset='cifar100',root='/explicit/data',
            input_quant_bits=None,train_size=224,test_size=224)
        with patch('torchvision.datasets.CIFAR100') as dataset, patch('train_teacher.load_noise_config') as noise:
            create_datasets(args)
            self.assertEqual(dataset.call_count,2)
            for call in dataset.call_args_list:
                self.assertEqual(call.kwargs['root'],'/explicit/data')
                self.assertFalse(call.kwargs['download'])
            noise.assert_not_called()

    def test_model_keeps_three_channel_pretrained_stem(self):
        model=nn.Module(); model.features=nn.Sequential(nn.Conv2d(3,4,3))
        model.classifier=nn.Sequential(nn.Linear(4,1000))
        stem=model.features[0]
        with patch('train_teacher._get_efficientnet_v2_builder',return_value=lambda **kw:model), \
             patch('train_teacher._get_efficientnet_v2_weights',return_value='cached'):
            out=load_efficientnet_v2_4ch(100,'efficientnet_v2_l',in_channels=3)
        self.assertIs(out.features[0],stem)
        self.assertEqual(out.classifier[-1].out_features,100)

    def test_timm_dataset_loader_uses_same_transform(self):
        trainer=RGBResizeFineTuneTrainer.__new__(RGBResizeFineTuneTrainer)
        trainer.rgb_data_root='/rgb';trainer.timm_input_size=(3,224,224)
        trainer.batch_size=32;trainer.test_batch_size=64;trainer.num_workers=0
        trainer.pin_memory=False;trainer.persistent_workers=False
        with patch('torchvision.datasets.CIFAR10',return_value=[0]*64) as dataset:
            trainer._prepare_cifar('rgb','cifar10')
        self.assertEqual(dataset.call_args_list[0].args[0],'/rgb')
        self.assertIsInstance(dataset.call_args_list[0].kwargs['transform'].transforms[0],transforms.RandomCrop)
        self.assertTrue(trainer.train_dataloader.drop_last)

    def test_worker_resolves_both_recipes_for_both_datasets(self):
        import train_teacher, train_teacher_timm
        for recipe, module in [('legacy',train_teacher),('timm_oldaugs',train_teacher_timm)]:
            env=dict(os.environ,REPO_ROOT=str(ROOT),TEACHER_DRY_RUN='true',
                RGB_TEACHER_RECIPE=recipe,RGB_DATA_ROOT='/rgb-data',TEACHER_OUTPUT_DIR='/output',
                TEACHER_LOG_DIR='/logs')
            out=subprocess.check_output(['bash','launch_scripts/run_rgb_teacher.sbatch'],cwd=ROOT,env=env,text=True)
            lines=[shlex.split(line) for line in out.splitlines() if line.startswith('python ')]
            self.assertEqual(len(lines),2)
            for dataset, tokens in zip(('cifar10','cifar100'),lines):
                with patch.object(sys,'argv',tokens[2:]):
                    args=module.parse_args()
                self.assertEqual(args.dataset,dataset);self.assertEqual(args.img_type,'rgb')
                self.assertEqual(args.arch_source,'torchvision');self.assertEqual(args.lr,.002)
                self.assertIn(dataset,args.checkpoint)
                if recipe=='legacy':
                    self.assertEqual(args.root,'/rgb-data');self.assertEqual(args.ne,100)
                    self.assertTrue(args.match_distill_preprocess)
                else:
                    self.assertEqual(args.data_root,'/rgb-data');self.assertEqual(args.epochs,100)
                    self.assertTrue(args.use_old_augs_for_timm)
                    self.assertEqual(args.lr_reduce_on,'10,20,30,40,50,60,70,80,90')
                    self.assertEqual(args.first_eval_epoch,10);self.assertEqual(args.eval_every,5)
                    self.assertEqual(args.warmup_epochs,0);self.assertEqual(args.weight_decay,1e-6)

    def test_submitter_preserves_environment_and_one_submission(self):
        env=dict(os.environ,REPO_ROOT=str(ROOT),TEACHER_DRY_RUN='true',
                 RGB_DATA_ROOT='/rgb-data',RGB_TEACHER_RECIPE='timm_oldaugs')
        out=subprocess.check_output(['bash','-c','source launch_scripts/slurm_run_rgb_teacher.sh'],cwd=ROOT,env=env,text=True)
        lines=out.splitlines();self.assertEqual(len(lines),1)
        tokens=shlex.split(lines[0])
        self.assertEqual(tokens[0],'sbatch')
        self.assertTrue(any(x.startswith('--export=ALL,') for x in tokens))

    def test_worker_failure_stops_before_cifar100(self):
        source=(ROOT/'launch_scripts/run_rgb_teacher.sbatch').read_text()
        source=source.replace('source activate base', ':').replace('conda activate scanbase', ':')
        # Stub preflight/training only; retain the actual loop and pipefail.
        stubs='python() { :; }\nbash() { echo "TRAIN_ATTEMPT:$DATASET_NAME"; return 7; }\n'
        for recipe in ('legacy','timm_oldaugs'):
            with tempfile.TemporaryDirectory() as d:
                env=dict(os.environ,TEACHER_DRY_RUN='false',REPO_ROOT=str(ROOT),
                    RGB_TEACHER_RECIPE=recipe,TEACHER_LOG_DIR=d,TEACHER_OUTPUT_DIR=d)
                result=subprocess.run(['bash','-c',stubs+source],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,7,result.stderr)
                self.assertIn('TRAIN_ATTEMPT:cifar10',result.stdout)
                self.assertNotIn('TRAIN_ATTEMPT:cifar100',result.stdout)

    def test_real_timm_training_step_and_checkpoint(self):
        import train_teacher_timm
        class Data(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.transform=transform
            def __len__(self): return 2
            def __getitem__(self, i):
                return self.transform(Image.fromarray(np.full((32,32,3),80+i,dtype=np.uint8))),i
        def factory(**kwargs):
            self.assertEqual(kwargs['in_channels'],3)
            return nn.Sequential(nn.Conv2d(3,2,1),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,10))
        def one_step(trainer):
            x,y=next(iter(trainer.train_dataloader))
            self.assertEqual(x.shape,(2,3,224,224))
            x,y=x.to(trainer.device),y.to(trainer.device)
            trainer.optimizer.zero_grad()
            loss=nn.functional.cross_entropy(trainer.model(x),y)
            loss.backward();trainer.optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(trainer.mixup_alpha,0)
            self.assertEqual(trainer.cutmix_alpha,0)
            path=trainer._save_model_ckpt(.5,0,'_best_ckpt.pth')
            saved=torch.load(path,weights_only=False,map_location='cpu')
            self.assertEqual(saved['teacher_preprocessing']['dataset'],'cifar10')
            self.assertEqual(saved['training_recipe'],'rgb_native_old_augs')
        with tempfile.TemporaryDirectory() as d, \
             patch('torchvision.datasets.CIFAR10',Data), \
             patch.object(train_teacher_timm,'load_efficientnet_v2_4ch',side_effect=factory), \
             patch.object(RGBResizeFineTuneTrainer,'train',one_step), \
             patch.object(sys,'argv',['teacher','--img_type','rgb','--dataset','cifar10',
                '--checkpoint',d+'/teacher.pth','--arch_source','torchvision',
                '--use_old_augs_for_timm','true','--batch_size','2','--num_workers','0',
                '--timm_opt','sgd','--timm_sched','multistep','--warmup_epochs','0']), \
             contextlib.redirect_stdout(io.StringIO()):
            train_teacher_timm.main()


if __name__=='__main__': unittest.main()
