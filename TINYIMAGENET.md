# Tiny ImageNet Support

## Dataset

Expected repository-relative location:

```text
../data/tiny-imagenet-200/
  train/<wnid>/images/*.JPEG
  val/images/*.JPEG
  val/val_annotations.txt
```

This resolves to `/scratch/rzeng7/repos/data/tiny-imagenet-200` on the SLURM
server and `../data/tiny-imagenet-200` locally. The loader also accepts a
validation split reorganized as `val/<wnid>/*.JPEG`.

The implementation validates 200 classes, 100,000 training images, 10,000
validation images, and a shared train/validation class mapping. It uses mean
`(0.4802, 0.4481, 0.3975)` and standard deviation
`(0.2302, 0.2265, 0.2262)`.

## Architecture

Tiny ImageNet uses the current CIFAR-style WRN and corresponding
`PCNetNoBatchNorm` channel layouts. The 3x3 stride-1 stem and two stage
reductions are unchanged. A 64x64 input therefore reaches the final global
average pool at 16x16; no extra pooling stage is added.

Supported comparison sizes are WRN-16-2, WRN-16-4, WRN-28-2, WRN-28-4,
WRN-16-8, and WRN-40-2. Matching BN-free WRN factories are registered as
`*_cifar_nobn`.

## Starting Recipe

| Setting | PCNetNoBatchNorm | WRN |
|---|---:|---:|
| Epochs | 300 | 300 |
| Train / validation batch | 128 / 512 | 128 / 512 |
| Optimizer | SGD, momentum 0.9 | SGD, momentum 0.9 |
| Initial learning rate | 0.01 | 0.1 |
| Weight decay | 0.001 | 0.001 |
| Scheduler | cosine, minimum 1e-6 | cosine, minimum 1e-6 |
| Warmup | 5 epochs from 1e-5 | 5 epochs from 1e-5 |
| Label smoothing | 0.1 | 0.1 |
| Mixup / CutMix | 0.2 / 1.0 | 0.2 / 1.0 |
| RandAugment | rand-m9-mstd0.5-inc1 | rand-m9-mstd0.5-inc1 |
| Color jitter / random erasing | 0.1 / 0.25 | 0.1 / 0.25 |
| Random-resized-crop scale | (0.75, 1.0) | (0.75, 1.0) |
| Final-feature dropout | 0.25 | 0.25 |
| WRN block dropout | N/A | 0.0 |
| Seed | 4096 | 4096 |
| Earliest best checkpoint | epoch 75 | epoch 75 |

The PCN launcher defaults to the established comparison dynamics:
`ODEXInitFFFB`, `t_end=1.75`, `PCConv`, and `PCNetNoBatchNorm`.

BN-free WRNs are supported, but should first be piloted with the saved
CIFAR-100 searched recipe via `TRAIN_OVERRIDE`; they are intentionally not in
the default six-model SLURM submission list.

## Launchers

Run one model locally from an activated `scanbase` environment:

```bash
MODEL_ARCH=wrn_28_4 bash launch_scripts/run_tinyimagenet_pcn_train.sh
MODEL_NAME=wrn_28_4_cifar bash launch_scripts/run_tinyimagenet_wrn_train.sh
```

Submit the six PCN or WRN sizes on SLURM:

```bash
bash launch_scripts/slurm_run_tinyimagenet_pcn_train.sh
bash launch_scripts/slurm_run_tinyimagenet_wrn_train.sh
```

Tiny ImageNet mismatch evaluation is available through the existing entry
points by passing `--task tinyimagenet --data_dir ../data/tiny-imagenet-200`
to `ode_inference.py`, or `--dataset tinyimagenet --data_dir
../data/tiny-imagenet-200` to `baseline/run_baseline.py`.
