#!/usr/bin/env python3
"""Reparameterize a feedforward checkpoint for a longer fixed MVM time.

For fixed timing, multiplying every physical convolution's duration by
``factor`` and dividing that convolution's weight and bias by ``factor``
preserves its ideal affine result.  BatchNorm, classifier, and feature-KD
state are copied without modification.
"""

import argparse
import copy
import os
import tempfile
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source checkpoint")
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--source-time", type=float, default=5e-9)
    parser.add_argument("--target-time", type=float, default=15e-9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convolution_prefix(key, value):
    if not torch.is_tensor(value) or value.ndim != 4:
        return None
    if key.endswith(".parametrizations.weight.original"):
        return key[: -len(".parametrizations.weight.original")]
    if key.endswith(".weight"):
        return key[: -len(".weight")]
    return None


def rescale_state_dict(state_dict, factor):
    result = copy.deepcopy(state_dict)
    conv_prefixes = []
    scaled_keys = []
    for key, value in state_dict.items():
        prefix = convolution_prefix(key, value)
        if prefix is None:
            continue
        result[key] = value / factor
        conv_prefixes.append(prefix)
        scaled_keys.append(key)

    for prefix in conv_prefixes:
        bias_key = prefix + ".bias"
        if bias_key in state_dict and torch.is_tensor(state_dict[bias_key]):
            result[bias_key] = state_dict[bias_key] / factor
            scaled_keys.append(bias_key)
    return result, scaled_keys


def main():
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise ValueError("Input and output checkpoints must be different.")
    if args.factor <= 0 or args.source_time <= 0 or args.target_time <= 0:
        raise ValueError("Factor and times must be positive.")
    expected_target = args.source_time * args.factor
    if not abs(args.target_time - expected_target) <= 1e-9 * expected_target:
        raise ValueError(
            "target-time must equal source-time * factor; got {} versus {}."
            .format(args.target_time, expected_target))
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace it: {}"
            .format(output))

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint.")
    state_key = "net" if "net" in checkpoint else None
    state_dict = checkpoint[state_key] if state_key else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model state must be a dictionary.")

    scaled_state, scaled_keys = rescale_state_dict(state_dict, args.factor)
    if not scaled_keys:
        raise ValueError("No convolution parameters were found.")
    result = copy.deepcopy(checkpoint)
    if state_key:
        result[state_key] = scaled_state
    else:
        # Wrap a bare state_dict so provenance does not become an unexpected
        # model-state key. Existing loaders already accept a top-level `net`.
        result = {"net": scaled_state}
    result["fixed_timing_reparameterization"] = {
        "source_checkpoint": str(source),
        "weight_and_bias_divisor": args.factor,
        "source_toggle_y_time": args.source_time,
        "target_toggle_y_time": args.target_time,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=output.name + ".", suffix=".tmp",
            delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(result, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    conv_count = sum(
        convolution_prefix(key, value) is not None
        for key, value in state_dict.items())
    bias_count = len(scaled_keys) - conv_count
    print("Wrote {}".format(output))
    print("Scaled {} convolution weights and {} convolution biases by 1/{}"
          .format(conv_count, bias_count, args.factor))


if __name__ == "__main__":
    main()
