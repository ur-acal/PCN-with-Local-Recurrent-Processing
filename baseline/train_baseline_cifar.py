#!/usr/bin/env python3
"""
Config-driven CIFAR baseline launcher using existing TrainerCiFarTimmStyle.

This script does NOT implement a training loop. It only:
    1. chooses the correct CIFAR baseline config,
    2. builds the model,
    3. applies CIFAR adaptation when needed,
    4. passes hyperparameters to TrainerCiFarTimmStyle,
    5. calls trainer.train().
"""

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import timm
import torch.nn as nn

# This file is intended to live inside baseline/ while trainer.py lives one level up:
#   project/
#     trainer.py
#     baseline/
#       train_baseline_cifar_with_trainer.py
#       baseline_cifar_configs.py
#       cifar_resnet.py
#       adapt_model_cifar.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Required local baseline utilities.
# Make sure baseline/__init__.py exists.
import baseline.cifar_resnet  # registers custom CIFAR models into timm
from baseline.baseline_cifar_configs import get_baseline_config, build_model

from trainer_timm import TrainerCiFarTimmStyle


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def parse_kv_overrides(s: str) -> dict:
    """
    Parse simple key=value,key2=value2 overrides.

    Values are parsed as bool/int/float/string in that order.
    Example:
        --override "lr=0.05,num_epochs=300,batch_size=256"
    """
    if not s:
        return {}

    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()

        if val.lower() in {"true", "false"}:
            out[key] = val.lower() == "true"
            continue
        try:
            out[key] = int(val)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(val)
            continue
        except ValueError:
            pass
        out[key] = val

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR baseline using TrainerCiFarTimmStyle.")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument('--noise_level', default=None, type=float,
                        help='noise level in noise inject training. None means normal training without noise injection')
    parser.add_argument('--noise_type', default='mul', type=str, choices=['mul', 'add'],
                        help='Multiplicative or additive noise')

    # auto means:
    #   custom CIFAR model -> 1a custom_noresize
    #   supported adapted timm model -> 1b adapt_noresize
    #   otherwise -> 2 resize
    parser.add_argument(
        "--case",
        type=str,
        default="auto",
        choices=[
            "auto",
            "custom_noresize",
            "adapt_noresize_scratch",
            "adapt_noresize_finetune",
            "resize_scratch",
            "resize_finetune",
        ],
    )
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument(
        "--prefer_resize",
        type=str2bool,
        default=False,
        help="For supported timm models, choose resize case instead of adapt_noresize when case=auto.",
    )

    parser.add_argument("--output_dir", type=str, default="../checkpoint/baselines/")

    # Simple config override interface. Use this only for experiment-level overrides.
    # Example: --override "lr=0.05,num_epochs=300,batch_size=256"
    parser.add_argument("--override", type=str, default="")
    parser.add_argument("--print_only", type=str2bool, default=False)

    return parser.parse_args()


def infer_num_classes(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    raise ValueError(dataset_name)


def build_trainer_kwargs(args, cfg: dict, model: nn.Module) -> dict:
    """
    Build kwargs for TrainerCiFarTimmStyle.

    The CLI only chooses the baseline/model/config. TrainerCiFar init arguments are
    filled here from the config so the launcher stays simple.
    """
    save_path = os.path.join(args.output_dir, args.dataset, cfg["case"], args.model_name)
    model_save_name = f"{cfg['case']}_{args.dataset}_{args.model_name}"

    trainer_kwargs = dict(
        # Parent TrainerCiFar args.
        model=model,
        model_name=model_save_name,
        save_path=save_path,
        batch_size=cfg["batch_size"],
        optim_type="sgd",  # ignored by TrainerCiFarTimmStyle._get_optimizer
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["lr"],
        num_epochs=cfg["num_epochs"],
        warmup_epoch=cfg["warmup_epoch"],
        lr_reduce_on=cfg.get("lr_reduce_on", "80,122,150,225,262"),
        test_bs=cfg["test_batch_size"],
        max_norm=cfg.get("max_norm", None),
        aug=False,  # transforms are handled by TrainerCiFarTimmStyle
        eval_every=cfg.get("eval_every", 5),
        img_type="rgb",
        dataset_name=args.dataset,

        # Keep these disabled for plain baseline training.
        noise_level=cfg.get("noise_level", None),
        noise_type=cfg.get("noise_type", None),
        mismatch_levels=None,
        distill_alpha=0.0,
        teacher_model=None,
        orig_t_inp=False,

        # TrainerCiFarTimmStyle-specific args.
        timm_opt=cfg["timm_opt"],
        momentum=cfg["momentum"],
        timm_sched=cfg["timm_sched"],
        min_lr=cfg["min_lr"],
        warmup_lr=cfg["warmup_lr"],
        timm_aug=True,
        timm_input_size=cfg["timm_input_size"],
        timm_train_scale=cfg["timm_train_scale"],
        timm_train_ratio=cfg["timm_train_ratio"],
        hflip=cfg["hflip"],
        color_jitter=cfg["color_jitter"],
        auto_augment=cfg["auto_augment"],
        re_prob=cfg["re_prob"],
        label_smoothing=cfg["label_smoothing"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],

        skip_eval_epochs=cfg["skip_eval_epochs"],
    )

    return trainer_kwargs


def main():
    args = parse_args()
    num_classes = infer_num_classes(args.dataset)

    extra_overrides = parse_kv_overrides(args.override)
    cfg = get_baseline_config(
        model_name=args.model_name,
        pretrained=args.pretrained,
        case=args.case,
        prefer_resize=args.prefer_resize,
        extra_overrides=extra_overrides,
    )

    model = build_model(args.model_name, cfg, num_classes)
    trainer_kwargs = build_trainer_kwargs(args, cfg, model)

    os.makedirs(trainer_kwargs["save_path"], exist_ok=True)
    with open(os.path.join(trainer_kwargs["save_path"], "baseline_config.json"), "w") as f:
        json.dump(
            {
                "cfg": cfg,
                "trainer_kwargs_keys": list(trainer_kwargs.keys()),
            },
            f,
            indent=2,
            default=str,
        )

    print("\nResolved baseline config:")
    pprint(cfg)
    print("\nTrainer kwargs:")
    pprint({k: v for k, v in trainer_kwargs.items() if k != "model"})

    if args.print_only:
        return

    trainer = TrainerCiFarTimmStyle(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()