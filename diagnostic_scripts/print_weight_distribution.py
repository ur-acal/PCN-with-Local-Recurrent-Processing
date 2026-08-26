#!/usr/bin/env python3
"""Print post-quantization FF/FB weight-level counts."""

import argparse

import torch

from diagnostic_config import (
    W_BITS, add_model_arguments, layer_indices, load_model_state)


def level_counts(weight, q_hi):
    levels = (weight.detach().abs() * q_hi).round().to(torch.int64)
    return torch.bincount(levels.reshape(-1), minlength=q_hi + 1)


def print_counts(title, ff_counts, fb_counts, q_hi):
    total = ff_counts + fb_counts
    print(title)
    print("{:>7} {:>12} {:>12} {:>12} {:>10}".format(
        "|w|", "FF count", "FB count", "Total", "Total %"))
    denominator = int(total.sum())
    for level in range(q_hi + 1):
        if level == 0:
            label = "0"
        elif level == q_hi:
            label = "1"
        else:
            label = "{}/{}".format(level, q_hi)
        print("{:>7} {:>12d} {:>12d} {:>12d} {:>9.4f}%".format(
            label, int(ff_counts[level]), int(fb_counts[level]),
            int(total[level]), 100.0 * int(total[level]) / denominator))
    print("{:>7} {:>12d} {:>12d} {:>12d} {:>9.4f}%".format(
        "All", int(ff_counts.sum()), int(fb_counts.sum()), denominator, 100.0))
    print()


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    args = parser.parse_args()

    state, paths = load_model_state(
        model_name=args.model_name, model_root=args.model_dir,
        full_param=False)
    q_hi = (1 << (W_BITS - 1)) - 1
    overall_ff = torch.zeros(q_hi + 1, dtype=torch.int64)
    overall_fb = torch.zeros(q_hi + 1, dtype=torch.int64)

    print("Task: {}".format(paths["task"]))
    print("Model: {}".format(paths["model_name"]))
    print("Checkpoint: {}".format(paths["quantized"]))
    print("Stored post-quantization magnitude levels: 0, 1/{0}, ..., {0}/{0}".format(q_hi))
    print()

    for layer in layer_indices(state):
        ff = level_counts(state["PcConvs.{}.FFconv.weight".format(layer)], q_hi)
        fb = level_counts(state["PcConvs.{}.FBconv.weight".format(layer)], q_hi)
        overall_ff += ff
        overall_fb += fb
        print_counts("Layer {}".format(layer + 1), ff, fb, q_hi)
    print_counts("Overall model", overall_ff, overall_fb, q_hi)


if __name__ == "__main__":
    main()

