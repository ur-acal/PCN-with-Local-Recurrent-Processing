"""Replay the successful CIFAR-100 FT recipe from its saved configuration."""
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import train_ode_cifar as training

root = Path('saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study')
path, = root.glob('TIMMQAT*22Layers6l7l6*/*_latest_ckpt.pth')
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
config = checkpoint['training_recovery']['config'].copy()
del checkpoint
# Start FT from the original pretrained `last`, not the finished QAT weights.
# Isolate output, and retain the desktop's no-worker-process policy.
config.update(output_save_path='saved_ckpt_runs/local_tc_cifar100_ft_memory_check',
              num_workers=0)
print('Source FT configuration:', path, flush=True)
print('Replay configuration:', config, flush=True)
training.get_args = lambda: Namespace(**config)
training.main()
