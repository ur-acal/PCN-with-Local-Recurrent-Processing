#!/usr/bin/env python3
"""
Config-driven CIFAR baseline launcher using the existing timm-style trainers.

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
from baseline.baseline_cifar_configs import (
    RGGB_DEFAULTS,
    RGGB_MID_AUG,
    RGGB_MILD_AUG,
    RGGB_NO_AUG,
    RGGB_TO_RGB_EXTRAS,
    build_model,
    get_baseline_config,
)
from train_ode_cifar import build_teacher_model, evaluate_teacher, _get_feature_kd_trainer



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

    parser.add_argument("--img_type", default="rgb",
                        help="Input data type: rgb, scanGFI, raw, _raw, or CiFAIR.")
    parser.add_argument("--rggb_to_rgb", type=str2bool, default=False)
    parser.add_argument("--timm_aug_level", default="none",
                        choices=("none", "no_aug", "mild", "mid"))
    parser.add_argument("--timm_re_prob", type=float, default=None)

    parser.add_argument("--teacher_ckpt", default=None)
    parser.add_argument("--teacher_arch", default=None)
    parser.add_argument("--teacher_arch_source", default="auto",
                        choices=("auto", "torchvision", "hankyul2"))
    parser.add_argument("--teacher_input_size", type=int, default=224)
    parser.add_argument("--teacher_center_crop", type=str2bool, default=True)
    parser.add_argument("--adapt_PIL_teacher", "--adapt_pil_teacher",
                        dest="adapt_PIL_teacher", type=str2bool, default=False)
    parser.add_argument("--orig_t_inp", type=str2bool, default=False)
    parser.add_argument("--distill_method", default="none",
                        choices=("none", "kd", "srrl"))
    parser.add_argument("--distill_alpha", type=float, default=0.3)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--srrl_weight", type=float, default=1.0)

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


def build_trainer_kwargs(args, cfg: dict, model: nn.Module, teacher_model=None) -> dict:
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
        img_type=args.img_type,
        dataset_name=args.dataset,

        # Plain baseline training keeps distill_method=none and teacher_model=None.
        noise_level=cfg.get("noise_level", None),
        noise_type=cfg.get("noise_type", None),
        mismatch_levels=None,
        distill_method=args.distill_method,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        teacher_model=teacher_model,
        orig_t_inp=args.orig_t_inp,
        teacher_input_size=args.teacher_input_size,
        teacher_center_crop=args.teacher_center_crop,

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

        convert_non_rgb_to_rgb=cfg.get("convert_non_rgb_to_rgb", False),
        non_rgb_spatial_aug=cfg.get("non_rgb_spatial_aug", True),
        non_rgb_crop_padding=cfg.get("non_rgb_crop_padding", 2),
        non_rgb_affine_degrees=cfg.get("non_rgb_affine_degrees", 0),
        non_rgb_affine_translate=cfg.get("non_rgb_affine_translate", None),
        non_rgb_affine_shear=cfg.get("non_rgb_affine_shear", None),

        skip_eval_epochs=cfg["skip_eval_epochs"],
    )

    return trainer_kwargs


def main():
    args = parse_args()
    img_type_lower = args.img_type.lower()
    if img_type_lower == "cifair":
        args.img_type = "CiFAIR"
    elif img_type_lower in {"scangfi", "raw", "_raw"}:
        args.img_type = "scanGFI"
    elif img_type_lower == "rgb":
        args.img_type = "rgb"
    else:
        raise ValueError("img_type must be rgb, scanGFI/raw, or CiFAIR")

    num_classes = infer_num_classes(args.dataset)
    args.num_classes = num_classes

    extra_overrides = parse_kv_overrides(args.override)
    cfg = get_baseline_config(
        model_name=args.model_name,
        pretrained=args.pretrained,
        case=args.case,
        prefer_resize=args.prefer_resize,
        extra_overrides=extra_overrides,
    )

    if args.img_type != "rgb" and args.rggb_to_rgb:
        cfg.update(RGGB_TO_RGB_EXTRAS)
    elif args.img_type != "rgb":
        cfg.update(RGGB_DEFAULTS)

    if args.img_type != "rgb":
        if args.timm_aug_level == "no_aug":
            cfg.update(RGGB_NO_AUG)
        elif args.timm_aug_level == "mild":
            cfg.update(RGGB_MILD_AUG)
        elif args.timm_aug_level == "mid":
            cfg.update(RGGB_MID_AUG)

        # Match the current PCN pretraining default.
        if args.timm_re_prob is None:
            cfg["re_prob"] = 0.0

    if args.timm_re_prob is not None:
        if not 0.0 <= args.timm_re_prob <= 1.0:
            raise ValueError("--timm_re_prob must be between 0 and 1")
        cfg["re_prob"] = args.timm_re_prob

    distill_enabled = args.distill_method != "none"
    if distill_enabled and not args.teacher_ckpt:
        if args.img_type == "CiFAIR":
            args.teacher_ckpt = (
                f"checkpoint/efficientnet_v2_l_{args.dataset}_"
                "CiFAIR_OldNoTimm_MatchDistill.pth"
            )
        elif args.dataset == "cifar10":
            args.teacher_ckpt = "checkpoint/b4.pth"
        else:
            args.teacher_ckpt = "checkpoint/b4_100.pth"
    if distill_enabled and not args.teacher_arch:
        args.teacher_arch = (
            "efficientnet-b4"
            if args.img_type != "CiFAIR" and args.dataset == "cifar10"
            else "efficientnet_v2_l"
        )
    if distill_enabled:
        # Match the PCN pretraining loader sizing and avoid evaluating the
        # 224x224 EfficientNet teacher with the baseline-only 1024 batch.
        cfg["test_batch_size"] = cfg["batch_size"]

    model = build_model(args.model_name, cfg, num_classes)
    teacher_model = None
    if distill_enabled and not args.print_only:
        teacher_model = build_teacher_model(
            args,
            student_in_channels=cfg.get("in_chans", 3),
            orig_t_inp=args.orig_t_inp,
        )

    trainer_kwargs = build_trainer_kwargs(args, cfg, model, teacher_model)
    trainer_cls = _get_feature_kd_trainer(args)
    if args.distill_method == "srrl":
        if args.srrl_weight < 0.0:
            raise ValueError("--srrl_weight must be non-negative")
        trainer_kwargs["srrl_beta"] = args.srrl_weight

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
    pprint({k: v for k, v in trainer_kwargs.items() if k not in {"model", "teacher_model"}})
    if distill_enabled:
        print("\nResolved distillation config:")
        pprint({
            "teacher_ckpt": args.teacher_ckpt,
            "teacher_arch": args.teacher_arch,
            "teacher_arch_source": args.teacher_arch_source,
            "teacher_input_size": args.teacher_input_size,
            "teacher_center_crop": args.teacher_center_crop,
            "adapt_PIL_teacher": args.adapt_PIL_teacher,
            "orig_t_inp": args.orig_t_inp,
            "distill_method": args.distill_method,
            "distill_alpha": args.distill_alpha,
            "distill_temperature": args.distill_temperature,
            "srrl_weight": args.srrl_weight,
        })

    if args.print_only:
        return

    trainer = trainer_cls(**trainer_kwargs)
    if teacher_model is not None:
        evaluate_teacher(teacher_model, trainer)
    trainer.train()


if __name__ == "__main__":
    main()