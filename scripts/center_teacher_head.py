#!/usr/bin/env python3
"""Center a teacher's final classifier across classes.

For a linear classifier ``z = W h + b``, this writes

    W' = W - mean_class(W)
    b' = b - mean_class(b)

so that ``z' = z - mean_class(z)``.  Softmax probabilities and predictions
are unchanged, while class-common raw-logit offsets are removed.
"""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from pathlib import Path

import torch


STATE_DICT_KEYS = ("net", "state_dict", "model", "model_state_dict")


def _state_dict(checkpoint) -> MutableMapping[str, torch.Tensor]:
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("Expected the checkpoint to be a mapping.")
    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, MutableMapping):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(
        "Could not find a state dict under any of: " + ", ".join(STATE_DICT_KEYS)
    )


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_head_centered{input_path.suffix}")


def center_head(
    input_path: Path,
    output_path: Path,
    weight_key: str,
    bias_key: str | None,
) -> None:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state = _state_dict(checkpoint)

    if weight_key not in state:
        raise KeyError(f"Classifier weight {weight_key!r} is absent from {input_path}.")
    weight = state[weight_key]
    if weight.ndim != 2:
        raise ValueError(f"{weight_key} must be 2D, got shape {tuple(weight.shape)}.")

    weight_mean = weight.mean(dim=0, keepdim=True)
    state[weight_key] = weight - weight_mean

    bias_mean = None
    if bias_key is not None:
        if bias_key not in state:
            raise KeyError(f"Classifier bias {bias_key!r} is absent from {input_path}.")
        bias = state[bias_key]
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"{bias_key} must have shape ({weight.shape[0]},), got {tuple(bias.shape)}."
            )
        bias_mean = bias.mean()
        state[bias_key] = bias - bias_mean

    if isinstance(checkpoint, MutableMapping):
        checkpoint["head_centering"] = {
            "method": "subtract_class_mean_from_linear_weight_and_bias",
            "source_checkpoint": str(input_path),
            "weight_key": weight_key,
            "bias_key": bias_key,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    saved = torch.load(output_path, map_location="cpu", weights_only=False)
    saved_state = _state_dict(saved)
    max_weight_mean = saved_state[weight_key].mean(dim=0).abs().max().item()
    max_bias_mean = (
        abs(saved_state[bias_key].mean().item()) if bias_key is not None else 0.0
    )

    # Verify the defining identity on synthetic features without using any data.
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(8, weight.shape[1], generator=generator, dtype=weight.dtype)
    original_bias = state[bias_key] + bias_mean if bias_key is not None else None
    original_weight = state[weight_key] + weight_mean
    original_logits = torch.nn.functional.linear(features, original_weight, original_bias)
    centered_logits = torch.nn.functional.linear(
        features,
        saved_state[weight_key],
        saved_state[bias_key] if bias_key is not None else None,
    )
    expected = original_logits - original_logits.mean(dim=1, keepdim=True)
    max_logit_error = (centered_logits - expected).abs().max().item()
    max_softmax_error = (
        original_logits.softmax(dim=1) - centered_logits.softmax(dim=1)
    ).abs().max().item()
    predictions_equal = torch.equal(
        original_logits.argmax(dim=1), centered_logits.argmax(dim=1)
    )

    print(f"source: {input_path}")
    print(f"output: {output_path}")
    print(f"classifier: {weight_key} {tuple(weight.shape)}")
    print(f"max |mean_class(centered weight)|: {max_weight_mean:.3e}")
    if bias_key is not None:
        print(f"|mean_class(centered bias)|: {max_bias_mean:.3e}")
    print(f"max |centered logits - expected|: {max_logit_error:.3e}")
    print(f"max softmax probability difference: {max_softmax_error:.3e}")
    print(f"synthetic predictions identical: {predictions_equal}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source teacher checkpoint")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weight-key", default="head.classifier.weight")
    parser.add_argument("--bias-key", default="head.classifier.bias")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )
    center_head(input_path, output_path, args.weight_key, args.bias_key)


if __name__ == "__main__":
    main()
