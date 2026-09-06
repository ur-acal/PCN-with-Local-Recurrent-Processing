#!/usr/bin/env python3
"""Compare original and fixed-timing-rescaled feedforward checkpoints."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline.cifar_resnet  # noqa: F401 -- register CIFAR models
from baseline.baseline_cifar_configs import RGGB_DEFAULTS, build_model, get_baseline_config
from inference_utils import get_test_data
from physical_feedforward import (
    convert_wide_resnet_to_physical,
    scale_batchnorm_to_physical_domain,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--rescaled", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--img-type", default="CiFAIR")
    parser.add_argument("--wrn-depth", type=int, default=None)
    parser.add_argument("--wrn-first-stage-channels", type=int, default=None)
    parser.add_argument("--original-time", type=float, default=5e-9)
    parser.add_argument("--rescaled-time", type=float, default=15e-9)
    parser.add_argument("--R", type=float, default=50e3)
    parser.add_argument("--C", type=float, default=500e-15)
    parser.add_argument("--v-dd", type=float, default=0.5)
    parser.add_argument("--one-over-q", type=float, default=5.0)
    parser.add_argument("--input-quant-bits", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4096)
    return parser.parse_args()


def build_physical_model(args, checkpoint_path, duration, device):
    cfg = get_baseline_config(args.model_name, case="custom_noresize")
    if args.img_type.lower() != "rgb":
        cfg.update(RGGB_DEFAULTS)
    cfg["wrn_depth"] = args.wrn_depth
    cfg["wrn_first_stage_channels"] = args.wrn_first_stage_channels
    classes = 100 if args.dataset == "cifar100" else 10
    model = build_model(args.model_name, cfg, classes)
    model = convert_wide_resnet_to_physical(
        model,
        physical_level=2,
        physical=True,
        qat=False,
        quantize_weights=False,
        R=args.R,
        C=args.C,
        v_dd=args.v_dd,
        one_over_q=args.one_over_q,
        weight_quant_factor_bits=None,
        toggle_timing_mode="fixed",
        toggle_y_time=duration,
        z_over_y_time=1.0,
        enob=None,
        enable_spin_variation=False,
        enable_summing_current_noise=False,
        enable_coupler_noise=False,
        enable_slow_summing_current=False,
        enable_slow_coupler_noise=False,
        enable_dtc_nonideality=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("net", checkpoint), strict=True)
    # Conversion scaled BN eps before loading. The checkpoint then restored
    # unitless affine/running state, so transform that state exactly once.
    scale_batchnorm_to_physical_domain(
        model, args.v_dd / args.one_over_q, scale_eps=False)
    return model.to(device).eval()


@torch.no_grad()
def compare(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original = build_physical_model(
        args, args.original, args.original_time, device)
    rescaled = build_physical_model(
        args, args.rescaled, args.rescaled_time, device)
    loader = get_test_data(
        test_bs=args.batch_size,
        img_type=args.img_type,
        task=args.dataset,
        shuffle=False,
        input_quant_bits=args.input_quant_bits,
        center_student_input=False,
    )

    total = correct_original = correct_rescaled = 0
    max_output_error = 0.0
    output_error_sum = 0.0
    output_elements = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        output_original = original(inputs)
        output_rescaled = rescaled(inputs)
        correct_original += output_original.argmax(1).eq(targets).sum().item()
        correct_rescaled += output_rescaled.argmax(1).eq(targets).sum().item()
        difference = (output_original - output_rescaled).abs()
        max_output_error = max(max_output_error, difference.max().item())
        output_error_sum += difference.sum().item()
        output_elements += difference.numel()
        total += targets.numel()
        print(
            "samples={} original={:.4f}% rescaled={:.4f}%".format(
                total,
                100.0 * correct_original / total,
                100.0 * correct_rescaled / total,
            ),
            flush=True,
        )

    print("original_accuracy={:.4f}%".format(100.0 * correct_original / total))
    print("rescaled_accuracy={:.4f}%".format(100.0 * correct_rescaled / total))
    print("max_output_difference={:.9g}".format(max_output_error))
    print("mean_output_difference={:.9g}".format(
        output_error_sum / output_elements))


if __name__ == "__main__":
    compare(parse_args())
