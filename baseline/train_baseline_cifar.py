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
import logging
import os
import random
import sys
from pathlib import Path
from pprint import pprint

import timm
import numpy as np
import torch
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
from measured_activation import (
    configure_feedforward_activation_pullback,
    configure_feedforward_measured_activation,
    configure_measured_activation_corner_mode,
    feedforward_measured_activation_factory,
)
from measured_pooling import configure_feedforward_measured_pooling
from input_preprocessing import (
    append_preprocessing_suffix,
    resolve_preprocessing,
    write_run_config,
)
from train_ode_cifar import build_teacher_model, evaluate_teacher, _get_feature_kd_trainer
from trainer_timm import TrainerCiFarTimmStyleReviewKD



def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def optional_int(v):
    return None if str(v).strip().lower() in {"", "none", "auto"} else int(v)


def optional_bool(v):
    if str(v).strip().lower() in {"", "none", "auto"}:
        return None
    return str2bool(v)


def scale_train_recipe_value(v):
    text = str(v).strip().lower()
    if text in {"", "none", "false", "f", "no", "off", "0"}:
        return 0.0
    if text in {"true", "t", "yes", "on"}:
        return 1.0
    value = float(text)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "scale_train_recipe must be nonnegative.")
    return value


class TrainerCiFarTimmStyleFeedForwardReviewKD(
        TrainerCiFarTimmStyleReviewKD):
    """ReviewKD feature capture for CIFAR ResNet/WRN students."""

    def _student_forward_feature_kd(self, inputs):
        inputs = self._prepare_student_inputs(inputs)
        required = ("layer1", "layer2", "layer3", "global_pool")
        if not all(hasattr(self.model, name) for name in required):
            raise ValueError(
                "Feedforward ReviewKD requires layer1/layer2/layer3/global_pool.")

        stage_features = []
        pooled_features = []
        handles = []

        def stage_hook(_module, _inputs, output):
            stage_features.append(self._require_feature_tensor(
                output, "ReviewKD feedforward stage"))

        def pool_hook(_module, _inputs, output):
            pooled_features.append(self._require_feature_tensor(
                output, "ReviewKD feedforward pooling"))

        for name in ("layer1", "layer2", "layer3"):
            handles.append(getattr(self.model, name).register_forward_hook(
                stage_hook))
        handles.append(self.model.global_pool.register_forward_hook(pool_hook))
        try:
            outputs = (
                self.noisy_model(inputs)
                if self.noisy_model is not None and self.model.training
                else self.model(inputs))
        finally:
            for handle in handles:
                handle.remove()

        features = stage_features + pooled_features
        if len(features) < self.reviewkd_num_stages:
            raise RuntimeError(
                "ReviewKD requested {} stages, but feedforward capture produced {}."
                .format(self.reviewkd_num_stages, len(features)))
        return outputs, features[-self.reviewkd_num_stages:]


def get_feedforward_kd_trainer(args):
    if "reviewkd" in str(args.distill_method).lower():
        return TrainerCiFarTimmStyleFeedForwardReviewKD
    return _get_feature_kd_trainer(args)


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
    parser.add_argument("--wrn_depth", type=int, default=None)
    parser.add_argument("--wrn_first_stage_channels", type=int, default=None)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4096)
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
                        choices=("none", "kd", "srrl", "mgd", "reviewkd"))
    parser.add_argument("--distill_alpha", type=float, default=0.3)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--srrl_weight", type=float, default=1.0)
    parser.add_argument("--mgd_alpha", type=float, default=7e-5)
    parser.add_argument("--mgd_lambda", type=float, default=0.5)
    parser.add_argument("--mgd_mask_mode", default="channel")
    parser.add_argument("--reviewkd_weight", type=float, default=1.0)
    parser.add_argument("--reviewkd_warmup_epochs", type=float, default=20.0)
    parser.add_argument("--reviewkd_num_stages", type=int, default=4)

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

    # Separate feedforward physical-finetuning route.  These options are inert
    # for ordinary baseline training.
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--physical_feedforward", type=str2bool, default=False)
    parser.add_argument("--physical_pretraining", type=str2bool, default=False)
    parser.add_argument("--physical_level", type=int, choices=(2, 3), default=2)
    parser.add_argument("--R", type=float, default=67e3)
    parser.add_argument("--C", type=float, default=282e-15)
    parser.add_argument("--v_dd", type=float, default=0.1)
    parser.add_argument("--one_over_q", type=float, default=1.0)
    parser.add_argument("--w_bits", type=int, default=5)
    parser.add_argument("--weight_quant_factor_bits", type=int, default=-1)
    parser.add_argument("--pulse_mismatch_training_mode",
                        default="post_quant_amplitude",
                        choices=("post_quant_amplitude", "pre_quant_weight"))
    parser.add_argument("--enob", type=optional_int, default=8)
    parser.add_argument("--toggle_timing_mode", choices=("derived", "fixed"), default="derived")
    parser.add_argument("--toggle_y_time", type=float, default=5e-9)
    parser.add_argument("--z_over_y_time", type=float, default=1.0)
    parser.add_argument("--input_quant_bits", type=optional_int, default=None)
    parser.add_argument("--center_student_input", type=optional_bool, default=None)
    parser.add_argument("--enable_spin_variation", type=str2bool, default=False)
    parser.add_argument("--sigma_spin", type=float, default=0.10)
    parser.add_argument("--spin_variation_mean", type=float, default=1.0)
    parser.add_argument("--spin_variation_seed", type=optional_int, default=None)
    parser.add_argument("--enable_summing_current_noise", type=str2bool, default=False)
    parser.add_argument("--summing_current_p", type=float, default=0.6e-12)
    parser.add_argument("--summing_noise_seed", type=optional_int, default=None)
    parser.add_argument("--enable_coupler_noise", type=str2bool, default=False)
    parser.add_argument("--coupler_noise_p", type=float, default=0.6e-12)
    parser.add_argument("--coupler_noise_seed", type=optional_int, default=None)
    parser.add_argument("--enable_slow_summing_current", type=str2bool, default=False)
    parser.add_argument("--slow_summing_current", type=float, default=2.47e-9)
    parser.add_argument("--enable_slow_coupler_noise", type=str2bool, default=False)
    parser.add_argument("--slow_coupler_noise", type=float, default=2.47e-9)
    parser.add_argument("--enable_dtc_nonideality", type=str2bool, default=False)
    parser.add_argument("--dtc_leading_edge_variation_std", type=float, default=0.0)
    parser.add_argument("--dtc_width_variation_mean", type=float, default=0.0)
    parser.add_argument("--dtc_width_variation_std", type=float, default=0.018)
    parser.add_argument("--dtc_leading_edge_jitter_std", type=float, default=0.005)
    parser.add_argument("--dtc_falling_edge_jitter_std", type=float, default=0.005)
    parser.add_argument("--dtc_timing_seed", type=optional_int, default=None)
    parser.add_argument("--enable_measured_activation", type=str2bool, default=False)
    parser.add_argument("--enable_pretrain_measured_activation",
                        type=str2bool, default=False)
    parser.add_argument("--activation_curve_path", default=str(
        PROJECT_ROOT / "hardware_data" / "relu_current_0p2uA_finer.csv"))
    parser.add_argument("--activation_corner", default="TT")
    parser.add_argument("--activation_corner_mode", default="fixed",
                        choices=("fixed", "random_per_forward"))
    parser.add_argument("--activation_curve_sharing", default="per_model",
                        choices=("per_model", "per_layer", "per_spin"))
    parser.add_argument("--activation_curve_seed", type=int, default=None)
    parser.add_argument("--activation_interpolation", default="piecewise_linear",
                        choices=("piecewise_linear", "cubic_bspline"))
    parser.add_argument("--activation_spline_parameters", type=int, default=10)
    parser.add_argument("--activation_fit_constraint", default="auto",
                        choices=("none", "nonnegative", "auto"))
    parser.add_argument("--activation_normalize_positive_endpoint",
                        type=str2bool, default=False)
    parser.add_argument("--compile_measured_activation", type=str2bool, default=False)
    parser.add_argument("--unitless_measured_pullback_mode", default="none",
                        choices=("none", "direct"))
    parser.add_argument("--unitless_pullback_q", type=float, default=None)
    parser.add_argument("--nonlinear_R_train_mode", default="none",
                        choices=("none", "exact_curve", "mean"))
    parser.add_argument("--nonlinear_R_table", default=str(
        PROJECT_ROOT / "hardware_data" / "mc_45_corners" / "coupler_monte"))
    parser.add_argument("--nonlinear_R_mc_quantity", default="conductance",
                        choices=("conductance", "resistance"))
    parser.add_argument("--nonlinear_R_corner_range", default="all")
    parser.add_argument("--nonlinear_R_curve_seed", type=optional_int, default=None)
    parser.add_argument("--nonlinear_R_curve_sharing", default="shared",
                        choices=("shared", "per_coupler", "per_input",
                                 "per_input_output"))
    parser.add_argument("--enable_measured_pooling", type=str2bool, default=False)
    parser.add_argument("--scale_train_recipe", type=scale_train_recipe_value,
                        default=0.0)
    parser.add_argument("--ff_train_scale", type=float, default=1.0)
    parser.add_argument("--fb_train_scale", type=float, default=1.0)

    return parser.parse_args()


def infer_num_classes(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    raise ValueError(dataset_name)


def feedforward_fixed_timing_scales(args):
    """Unitless gains of the two physical feedforward stages."""
    if args.toggle_timing_mode != "fixed":
        return 1.0, 1.0
    rc = float(args.R) * float(args.C)
    # conv1 uses the base stage time; conv2 uses z_over_y_time.
    fb_scale = float(args.toggle_y_time) / rc
    ff_scale = float(args.toggle_y_time) * float(args.z_over_y_time) / rc
    return ff_scale, fb_scale


@torch.no_grad()
def scale_feedforward_initial_weights(model, ff_scale, fb_scale):
    from physical_feedforward import iter_physical_blocks
    for block in iter_physical_blocks(model):
        block.conv1.weight.div_(fb_scale)
        if block.conv1.bias is not None:
            block.conv1.bias.div_(fb_scale)
        if block.conv2 is not None:
            block.conv2.weight.div_(ff_scale)
            if block.conv2.bias is not None:
                block.conv2.bias.div_(ff_scale)


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("net", checkpoint)


def is_physical_feedforward_state(state_dict):
    return any(key.endswith(".scale1") or key == "conv1.ode_block.scale1"
               for key in state_dict)


def is_parametrized_weight_state(state_dict):
    return any(".parametrizations.weight.original" in key
               for key in state_dict)


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
        bias_lr_multiplier=cfg.get("bias_lr_multiplier", 1.0),
        bias_weight_decay=cfg.get("bias_weight_decay", None),
        aug=False,  # transforms are handled by TrainerCiFarTimmStyle
        eval_every=(
            args.eval_every if args.eval_every is not None else
            (2 if args.physical_feedforward and
             not args.physical_pretraining else cfg.get("eval_every", 5))),
        img_type=args.img_type,
        dataset_name=args.dataset,

        # Plain baseline training keeps distill_method=none and teacher_model=None.
        noise_level=(args.noise_level if args.noise_level is not None
                     else cfg.get("noise_level", None)),
        noise_type=(args.noise_type if args.noise_level is not None
                    else cfg.get("noise_type", None)),
        mismatch_levels=None,
        distill_method=args.distill_method,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        teacher_model=teacher_model,
        orig_t_inp=args.orig_t_inp,
        teacher_input_size=args.teacher_input_size,
        teacher_center_crop=args.teacher_center_crop,
        adapt_PIL_teacher=args.adapt_PIL_teacher,
        input_quant_bits=args.input_quant_bits,
        center_student_input=args.center_student_input,

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
        scale_train_recipe=bool(args.scale_train_recipe),
        ff_train_scale=args.ff_train_scale,
        fb_train_scale=args.fb_train_scale,
        pulse_mismatch_training_mode=args.pulse_mismatch_training_mode,
        save_flattened_and_full_param=(
            args.physical_feedforward and not args.physical_pretraining),
    )

    return trainer_kwargs


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.physical_feedforward:
        inference_path = args.resume_checkpoint or args.output_dir
        args.input_quant_bits, args.center_student_input = resolve_preprocessing(
            inference_path, args.input_quant_bits,
            args.center_student_input)
        args.output_dir = append_preprocessing_suffix(
            args.output_dir, args.input_quant_bits,
            args.center_student_input)
        write_run_config(
            args.output_dir, args.input_quant_bits,
            args.center_student_input)
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
    cfg["wrn_depth"] = args.wrn_depth
    cfg["wrn_first_stage_channels"] = args.wrn_first_stage_channels

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

    if (args.physical_pretraining and
            str(cfg.get("timm_sched", "")).lower() == "cosine"):
        cfg["skip_eval_epochs"] = min(
            max(cfg["skip_eval_epochs"], 0.5 * cfg["num_epochs"]),
            0.8 * cfg["num_epochs"])

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
    checkpoint = (
        torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False)
        if args.resume_checkpoint else None)
    state_dict = (
        checkpoint.get("net", checkpoint)
        if checkpoint is not None else None)
    checkpoint_weight_format = (
        checkpoint.get("checkpoint_weight_format")
        if isinstance(checkpoint, dict) else None)

    if args.physical_feedforward:
        from physical_feedforward import (
            convert_wide_resnet_to_physical,
            iter_physical_blocks,
            iter_physical_wrappers,
        )
        qf_bits = (
            None if args.weight_quant_factor_bits < 0
            else args.weight_quant_factor_bits)
        if args.scale_train_recipe and args.toggle_timing_mode != "fixed":
            raise ValueError(
                "scale_train_recipe requires fixed feedforward timing.")
        if (args.nonlinear_R_train_mode != "none" and
                args.nonlinear_R_curve_sharing != "shared"):
            raise ValueError(
                "Level-2 nonlinear-R training requires "
                "--nonlinear_R_curve_sharing shared.")
        ff_scale, fb_scale = feedforward_fixed_timing_scales(args)
        if args.scale_train_recipe:
            ff_scale *= args.scale_train_recipe
            fb_scale *= args.scale_train_recipe
            args.ff_train_scale = ff_scale
            args.fb_train_scale = fb_scale

        checkpoint_is_physical = (
            state_dict is not None and
            is_physical_feedforward_state(state_dict))
        checkpoint_is_qat = (
            state_dict is not None and
            is_parametrized_weight_state(state_dict))
        checkpoint_is_unitless_physical = (
            checkpoint_is_physical and not checkpoint_is_qat and
            checkpoint_weight_format not in {
                "flattened_quantized", "full_param"})
        if state_dict is not None and not checkpoint_is_physical:
            model.load_state_dict(state_dict, strict=True)
        model = convert_wide_resnet_to_physical(
            model,
            activation_factory=None,
            physical_level=args.physical_level,
            physical=not args.physical_pretraining,
            qat=False,
            R=args.R,
            C=args.C,
            v_dd=args.v_dd,
            one_over_q=args.one_over_q,
            w_bits=args.w_bits,
            weight_quant_factor_bits=qf_bits,
            noise_level=(0.0 if args.noise_level is None else args.noise_level),
            mismatch_type=args.noise_type,
            toggle_timing_mode=args.toggle_timing_mode,
            toggle_y_time=args.toggle_y_time,
            z_over_y_time=args.z_over_y_time,
            enob=None if args.physical_pretraining else args.enob,
            enable_spin_variation=(
                False if args.physical_pretraining
                else args.enable_spin_variation),
            sigma_spin=args.sigma_spin,
            spin_variation_mean=args.spin_variation_mean,
            spin_variation_seed=args.spin_variation_seed,
            enable_summing_current_noise=(
                False if args.physical_pretraining
                else args.enable_summing_current_noise),
            summing_current_p=args.summing_current_p,
            summing_noise_seed=args.summing_noise_seed,
            enable_coupler_noise=(
                False if args.physical_pretraining
                else args.enable_coupler_noise),
            coupler_noise_p=args.coupler_noise_p,
            coupler_noise_seed=args.coupler_noise_seed,
            enable_slow_summing_current=(
                False if args.physical_pretraining
                else args.enable_slow_summing_current),
            slow_summing_current=args.slow_summing_current,
            enable_slow_coupler_noise=(
                False if args.physical_pretraining
                else args.enable_slow_coupler_noise),
            slow_coupler_noise=args.slow_coupler_noise,
            enable_dtc_nonideality=(
                False if args.physical_pretraining
                else args.enable_dtc_nonideality),
            dtc_leading_edge_variation_std=args.dtc_leading_edge_variation_std,
            dtc_width_variation_mean=args.dtc_width_variation_mean,
            dtc_width_variation_std=args.dtc_width_variation_std,
            dtc_leading_edge_jitter_std=args.dtc_leading_edge_jitter_std,
            dtc_falling_edge_jitter_std=args.dtc_falling_edge_jitter_std,
            dtc_timing_seed=args.dtc_timing_seed,
        )

        if checkpoint_is_qat:
            for wrapper in iter_physical_wrappers(model):
                wrapper.enable_qat_()
        if checkpoint_is_physical:
            model.load_state_dict(state_dict, strict=True)
            if (not args.physical_pretraining and
                    checkpoint_is_unitless_physical):
                # Conversion already scaled BatchNorm eps, which is not part
                # of state_dict. The unitless checkpoint overwrote only the
                # affine parameters and running statistics, so rescale those.
                from physical_feedforward import (
                    scale_batchnorm_to_physical_domain)
                scale_batchnorm_to_physical_domain(
                    model, args.v_dd / args.one_over_q,
                    scale_eps=False)
        if (args.physical_pretraining and args.scale_train_recipe and
                not checkpoint_is_physical):
            scale_feedforward_initial_weights(model, ff_scale, fb_scale)
        if not args.physical_pretraining and not checkpoint_is_qat:
            for wrapper in iter_physical_wrappers(model):
                wrapper.enable_qat_()

        activation_enabled = (
            args.enable_pretrain_measured_activation
            if args.physical_pretraining else
            args.enable_measured_activation)
        if activation_enabled:
            if (args.activation_corner_mode == "random_per_forward" and
                    args.activation_curve_sharing != "per_model"):
                raise ValueError(
                    "random_per_forward matches PCN training semantics and "
                    "requires activation_curve_sharing=per_model.")
            activation_factory = feedforward_measured_activation_factory(
                args.activation_curve_path, args.v_dd,
                corner=args.activation_corner,
                curve_sharing=args.activation_curve_sharing,
                curve_seed=args.activation_curve_seed,
                normalize_positive_endpoint=(
                    args.activation_normalize_positive_endpoint),
                interpolation=args.activation_interpolation,
                spline_parameters=args.activation_spline_parameters,
                fit_constraint=args.activation_fit_constraint,
                compile_evaluator=args.compile_measured_activation)
            configure_feedforward_measured_activation(
                model, activation_factory)
            if args.physical_pretraining:
                pullback_q = (
                    args.unitless_pullback_q
                    if args.unitless_pullback_q is not None else
                    args.v_dd / args.one_over_q)
                configure_feedforward_activation_pullback(
                    model, mode=args.unitless_measured_pullback_mode,
                    q=pullback_q)
            configure_measured_activation_corner_mode(
                model, mode=args.activation_corner_mode,
                sharing=args.activation_curve_sharing)
        if (args.nonlinear_R_train_mode != "none" and
                not args.physical_pretraining):
            wrappers = list(iter_physical_wrappers(model))
            package = wrappers[0].configure_nonlinear_R_training(
                args.nonlinear_R_table,
                mode=args.nonlinear_R_train_mode,
                corner_range=args.nonlinear_R_corner_range,
                quantity=args.nonlinear_R_mc_quantity,
                curve_seed=args.nonlinear_R_curve_seed)
            for wrapper in wrappers[1:]:
                wrapper.install_nonlinear_R_training_package(package)
        if args.enable_measured_pooling and not args.physical_pretraining:
            configure_feedforward_measured_pooling(
                model, enable_nonideality=True,
                curve_path=args.nonlinear_R_table,
                quantity=args.nonlinear_R_mc_quantity,
                nominal_R=args.R,
                seed=args.nonlinear_R_curve_seed,
                training_curve_mode=(
                    args.nonlinear_R_train_mode
                    if args.nonlinear_R_train_mode != "none"
                    else "exact_curve"),
                corner_range=args.nonlinear_R_corner_range)
    elif state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    teacher_model = None
    if distill_enabled and not args.print_only:
        teacher_model = build_teacher_model(
            args,
            student_in_channels=cfg.get("in_chans", 3),
            orig_t_inp=args.orig_t_inp,
        )

    trainer_kwargs = build_trainer_kwargs(args, cfg, model, teacher_model)
    trainer_cls = get_feedforward_kd_trainer(args)
    if args.distill_method == "srrl":
        if args.srrl_weight < 0.0:
            raise ValueError("--srrl_weight must be non-negative")
        trainer_kwargs["srrl_beta"] = args.srrl_weight
    elif args.distill_method == "mgd":
        trainer_kwargs.update(
            mgd_alpha=args.mgd_alpha,
            mgd_lambda=args.mgd_lambda,
            mgd_mask_mode=args.mgd_mask_mode,
        )
    elif args.distill_method == "reviewkd":
        trainer_kwargs.update(
            reviewkd_weight=args.reviewkd_weight,
            reviewkd_warmup_epochs=args.reviewkd_warmup_epochs,
            reviewkd_num_stages=args.reviewkd_num_stages,
        )

    os.makedirs(trainer_kwargs["save_path"], exist_ok=True)
    with open(os.path.join(trainer_kwargs["save_path"], "baseline_config.json"), "w") as f:
        json.dump(
            {
                "cfg": cfg,
                "trainer_kwargs_keys": list(trainer_kwargs.keys()),
                "physical_feedforward": (
                    None if not args.physical_feedforward else {
                        "physical_level": args.physical_level,
                        "physical_pretraining": args.physical_pretraining,
                        "R": args.R,
                        "C": args.C,
                        "v_dd": args.v_dd,
                        "one_over_q": args.one_over_q,
                        "w_bits": args.w_bits,
                        "weight_quant_factor_bits": args.weight_quant_factor_bits,
                        "toggle_timing_mode": args.toggle_timing_mode,
                        "toggle_y_time": args.toggle_y_time,
                        "z_over_y_time": args.z_over_y_time,
                        "scale_train_recipe": args.scale_train_recipe,
                        "input_quant_bits": args.input_quant_bits,
                        "center_student_input": args.center_student_input,
                        "nonlinear_R_train_mode": args.nonlinear_R_train_mode,
                        "nonlinear_R_table": args.nonlinear_R_table,
                        "enable_measured_activation": args.enable_measured_activation,
                        "enable_pretrain_measured_activation": (
                            args.enable_pretrain_measured_activation),
                        "activation_corner_mode": args.activation_corner_mode,
                        "unitless_measured_pullback_mode": (
                            args.unitless_measured_pullback_mode),
                        "unitless_pullback_q": args.unitless_pullback_q,
                        "enable_measured_pooling": args.enable_measured_pooling,
                    }),
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
            "mgd_alpha": args.mgd_alpha,
            "mgd_lambda": args.mgd_lambda,
            "mgd_mask_mode": args.mgd_mask_mode,
            "reviewkd_weight": args.reviewkd_weight,
            "reviewkd_warmup_epochs": args.reviewkd_warmup_epochs,
            "reviewkd_num_stages": args.reviewkd_num_stages,
        })

    if args.print_only:
        return

    trainer = trainer_cls(**trainer_kwargs)
    if (args.resume_checkpoint is not None and
            hasattr(trainer, "load_feature_kd_from_ckpt")):
        logging.warning(
            "Loading distillation method's auxiliary module for FT.")
        trainer.load_feature_kd_from_ckpt(
            ckpt_path=args.resume_checkpoint)
    if teacher_model is not None:
        evaluate_teacher(teacher_model, trainer)
    trainer.train()


if __name__ == "__main__":
    main()
