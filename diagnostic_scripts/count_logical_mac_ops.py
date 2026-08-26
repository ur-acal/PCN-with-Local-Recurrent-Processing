#!/usr/bin/env python3
"""Count logical PCN inference MACs without simulation-only expansion.

This is an analytical counter.  It reads the architecture saved in a PCN
checkpoint and counts one logical FF convolution and one logical FB
convolution per toggle cycle.  Pulse slices, expanded MVM matrices,
non-idealities, fake quantization, and numerical ODE-solver work are not
executed or counted.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def _numel(shape):
    return math.prod(int(value) for value in shape)


def _format_count(value):
    return f"{int(value):,}"


def _format_result(value):
    value = int(value)
    if abs(value) >= 1_000_000_000_000:
        scale, unit = 1_000_000_000_000, "T"
    elif abs(value) >= 1_000_000_000:
        scale, unit = 1_000_000_000, "G"
    else:
        scale, unit = 1_000_000, "M"
    return f"{value / scale:,.6f} {unit} ({_format_count(value)})"


def _checkpoint_state(checkpoint):
    for key in ("net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain a net/model/state_dict mapping.")


def _strip_module_prefix(state):
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _saved_model_args(checkpoint):
    init_args = checkpoint.get("init_args", {})
    model_args = init_args.get("model_args", {})
    kwargs = init_args.get("kwargs", {})
    required = ("inp_channels", "out_channels", "max_pool")
    missing = [name for name in required if name not in model_args]
    if missing:
        raise ValueError(
            "Checkpoint init_args.model_args is missing: " + ", ".join(missing))
    return model_args, kwargs


def _conv_output_size(size, kernel, stride, padding, dilation=1):
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _infer_weight_bits(path):
    match = re.search(r"QAT(\d+)b", str(path))
    return int(match.group(1)) if match else None


def build_report(checkpoint_path, toggle_cycles, input_height, input_width,
                 weight_bits=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint.")
    state = _strip_module_prefix(_checkpoint_state(checkpoint))
    model_args, kwargs = _saved_model_args(checkpoint)

    inp_channels = [int(value) for value in model_args["inp_channels"]]
    out_channels = [int(value) for value in model_args["out_channels"]]
    pool_after = [bool(value) for value in model_args["max_pool"]]
    strides = model_args.get("stride", [1] * len(inp_channels))
    kernels = model_args.get("kernel_size", [3] * len(inp_channels))
    padding = int(kwargs.get("padding", 1))

    n_layers = len(inp_channels)
    if not (len(out_channels) == len(pool_after) == len(strides) == len(kernels) == n_layers):
        raise ValueError("Saved channel, pooling, stride, and kernel lists differ in length.")
    if toggle_cycles <= 0:
        raise ValueError("toggle_cycles must be positive.")
    if kwargs.get("use_pc", True) is not True:
        raise ValueError("Logical toggle counting requires use_pc=True.")
    if kwargs.get("bypass", False):
        raise ValueError("This counter currently requires bypass=False.")

    height, width = int(input_height), int(input_width)
    layer_rows = []
    ff_macs_per_cycle = 0
    fb_macs_per_cycle = 0
    pcn_weight_params = 0
    pcn_aux_params = 0
    state_elements_per_cycle = 0
    activation_elements_per_cycle = 0
    pool_adds = 0
    pool_scales = 0

    for index, (cin, cout, stride, kernel, do_pool) in enumerate(zip(
            inp_channels, out_channels, strides, kernels, pool_after)):
        ff_key = f"PcConvs.{index}.FFconv.weight"
        fb_key = f"PcConvs.{index}.FBconv.weight"
        if ff_key not in state or fb_key not in state:
            raise ValueError(f"Checkpoint is missing {ff_key} or {fb_key}.")

        ff_shape = tuple(int(value) for value in state[ff_key].shape)
        fb_shape = tuple(int(value) for value in state[fb_key].shape)
        if len(ff_shape) != 4 or len(fb_shape) != 4:
            raise ValueError("Only 2-D PC convolutions are supported.")
        if ff_shape[:2] != (cout, cin):
            raise ValueError(f"{ff_key} shape {ff_shape} disagrees with saved channels.")
        if fb_shape[:2] != (cout, cin):
            raise ValueError(f"{fb_key} shape {fb_shape} disagrees with saved channels.")

        kernel_h, kernel_w = ff_shape[2:]
        stride = int(stride)
        out_h = _conv_output_size(height, kernel_h, stride, padding)
        out_w = _conv_output_size(width, kernel_w, stride, padding)

        # Standard dense-convolution convention: padded kernel positions count.
        # ConvTranspose consumes the FF output and has the same logical weighted
        # operation count for this paired kernel.
        ff_macs = out_h * out_w * cout * cin * kernel_h * kernel_w
        fb_macs = out_h * out_w * cout * cin * kernel_h * kernel_w
        ff_macs_per_cycle += ff_macs
        fb_macs_per_cycle += fb_macs
        pcn_weight_params += _numel(ff_shape) + _numel(fb_shape)

        b0_key = f"PcConvs.{index}.b0.0"
        if b0_key in state:
            pcn_aux_params += int(state[b0_key].numel())

        z_elements = cin * height * width
        y_elements = cout * out_h * out_w
        state_elements_per_cycle += z_elements + y_elements
        activation_elements_per_cycle += z_elements

        layer_rows.append({
            "index": index + 1,
            "shape": f"{cin}x{height}x{width} -> {cout}x{out_h}x{out_w}",
            "ff": ff_macs,
            "fb": fb_macs,
            "cycles": toggle_cycles * (ff_macs + fb_macs),
            "params": _numel(ff_shape) + _numel(fb_shape),
            "pool": do_pool,
        })

        height, width = out_h, out_w
        if do_pool:
            pooled_h, pooled_w = height // 2, width // 2
            outputs = cout * pooled_h * pooled_w
            pool_adds += outputs * 3       # 2x2 average: three additions.
            pool_scales += outputs         # One divide/multiply by four.
            height, width = pooled_h, pooled_w

    linear_weight = state.get("linear.weight")
    if linear_weight is None or linear_weight.ndim != 2:
        raise ValueError("Checkpoint is missing a 2-D linear.weight classifier.")
    classifier_macs = int(linear_weight.shape[0] * linear_weight.shape[1])
    classifier_weight_params = int(linear_weight.numel())
    classifier_bias_params = int(state.get("linear.bias", torch.empty(0)).numel())
    classifier_bias_adds = classifier_bias_params

    global_pool_elements = int(linear_weight.shape[1]) * height * width
    global_pool_outputs = int(linear_weight.shape[1])
    global_pool_adds = global_pool_elements - global_pool_outputs
    global_pool_scales = global_pool_outputs
    final_relu_elements = global_pool_elements

    pcn_macs = toggle_cycles * (ff_macs_per_cycle + fb_macs_per_cycle)
    all_macs = pcn_macs + classifier_macs

    # Each logical z/y stage multiplies its convolution result by the step size
    # and accumulates it into the state. RC/hardware scaling is intentionally
    # outside this logical neural-operation count.
    state_elements = toggle_cycles * state_elements_per_cycle
    step_size_multiply_ops = state_elements
    state_add_ops = state_elements

    relu6_elements = toggle_cycles * activation_elements_per_cycle
    relu6_compare_ops = 2 * relu6_elements
    final_relu_compare_ops = final_relu_elements

    pool_add_ops = pool_adds + global_pool_adds
    pool_scale_ops = pool_scales + global_pool_scales
    pcn_conv_ops = 2 * pcn_macs
    pcn_core_ops = (
        pcn_conv_ops + step_size_multiply_ops + state_add_ops +
        relu6_compare_ops)
    pooling_ops = pool_add_ops + pool_scale_ops
    classifier_ops = 2 * classifier_macs + classifier_bias_adds
    other_ops = pooling_ops + final_relu_compare_ops + classifier_ops
    all_ops = pcn_core_ops + other_ops

    if weight_bits is None:
        weight_bits = _infer_weight_bits(checkpoint_path)

    lines = []
    lines.append("Logical MAC/op report")
    lines.append("=====================")
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Input: 1 x {inp_channels[0]} x {input_height} x {input_width}")
    lines.append(f"Toggle cycles: {toggle_cycles}")
    lines.append(f"Stored weight precision: {weight_bits if weight_bits is not None else 'unknown'} bits")

    lines.append("")
    lines.append("PCN core")
    lines.append("--------")
    lines.append(f"FF/FB convolution MACs:            {_format_result(pcn_macs)}")
    lines.append(f"FF/FB convolution operations:      {_format_result(pcn_conv_ops)}")
    lines.append(f"Step-size multiplications:         {_format_result(step_size_multiply_ops)}")
    lines.append(f"State accumulation additions:      {_format_result(state_add_ops)}")
    lines.append(f"ReLU6 operations:                  {_format_result(relu6_compare_ops)}")
    lines.append(f"PCN core logical operations:       {_format_result(pcn_core_ops)}")

    lines.append("")
    lines.append("Other model layers")
    lines.append("------------------")
    lines.append(f"Pooling operations:                {_format_result(pooling_ops)}")
    lines.append(f"Final ReLU operations:             {_format_result(final_relu_compare_ops)}")
    lines.append(f"Classifier MACs:                   {_format_result(classifier_macs)}")
    lines.append(f"Classifier operations:             {_format_result(classifier_ops)}")

    lines.append("")
    lines.append("Model totals")
    lines.append("------------")
    lines.append(f"All-layer logical MACs:            {_format_result(all_macs)}")
    lines.append(f"All-layer logical operations:      {_format_result(all_ops)}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Quantized *_best_ckpt.pth to inspect.")
    parser.add_argument("--toggle-cycles", type=int, default=5,
                        help="Logical FF/FB cycles per PC layer (default: 5).")
    parser.add_argument("--input-height", type=int, default=16)
    parser.add_argument("--input-width", type=int, default=16)
    parser.add_argument("--weight-bits", type=int, default=None,
                        help="Display metadata; inferred from QAT<N>b when omitted.")
    parser.add_argument("--output", default=None,
                        help="Optional text report path; stdout is always printed.")
    args = parser.parse_args()

    report = build_report(
        checkpoint_path=args.checkpoint,
        toggle_cycles=args.toggle_cycles,
        input_height=args.input_height,
        input_width=args.input_width,
        weight_bits=args.weight_bits)
    print(report, end="")
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)


if __name__ == "__main__":
    main()
