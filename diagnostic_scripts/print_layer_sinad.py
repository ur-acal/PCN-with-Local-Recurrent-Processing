#!/usr/bin/env python3
"""Print propagated per-layer SINAD and equivalent ENOB."""

import argparse
import csv
import gc
import math
from pathlib import Path

import torch

from diagnostic_config import REPO_ROOT, add_model_arguments
from diagnostic_runtime import DEFAULT_CORNER, DEFAULT_SEED, build_runtime_trial
from print_layer_snr import _capture_layer_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--corner", default=DEFAULT_CORNER)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def _sinad_row(layer, clean, all_on):
    clean = clean.double()
    error = all_on.double() - clean
    clean_power = clean.square().sum().item()
    error_power = error.square().sum().item()
    clean_rms = math.sqrt(clean_power / clean.numel())
    error_rms = math.sqrt(error_power / clean.numel())
    sinad_db = math.inf if error_power == 0 else 10.0 * math.log10(
        clean_power / error_power)
    enob = math.inf if math.isinf(sinad_db) else (sinad_db - 1.76) / 6.02
    return {
        "layer": layer,
        "enob_bits": enob,
        "sinad_db": sinad_db,
        "clean_rms": clean_rms,
        "noise_distortion_rms": error_rms,
    }


def _display_model_dir(path):
    path = Path(path).resolve()
    saved_root = (REPO_ROOT / "saved_ckpt_runs").resolve()
    try:
        return Path("saved_ckpt_runs") / path.relative_to(saved_root)
    except ValueError:
        return path


def main():
    args = parse_args()
    common = {
        "model_name": args.model_name,
        "model_root": args.model_dir,
        "corner": args.corner,
        "batch_size": args.batch_size,
        "trial_index": args.trial,
        "seed": args.seed,
        "device": args.device,
    }

    clean_trial = build_runtime_trial(
        **common, nonideality_profile="clean")
    inputs, _ = clean_trial.first_batch()
    inputs_cpu = inputs.detach().cpu()
    clean, clean_logits = _capture_layer_outputs(
        clean_trial.model, inputs, output_device="cpu")
    model_name = clean_trial.args.model_name
    model_dir = _display_model_dir(clean_trial.args.model_dir)
    task = clean_trial.args.task
    del inputs, clean_trial
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_on_trial = build_runtime_trial(
        **common, nonideality_profile="all_on")
    inputs = inputs_cpu.to(all_on_trial.device)
    all_on, all_on_logits = _capture_layer_outputs(
        all_on_trial.model, inputs, output_device="cpu")

    rows = [_sinad_row(layer, clean[layer], all_on[layer])
            for layer in sorted(clean)]
    rows.append(_sinad_row("logits", clean_logits, all_on_logits))

    print("Model: {}".format(model_name))
    print("Model directory: {}".format(model_dir))
    print("Task: {} | corner: {} | trial: {} | batch: {}".format(
        task, all_on_trial.corner, args.trial, inputs.shape[0]))
    print("Clean: all modeled hardware nonidealities disabled.")
    print("All-on: the corner-specific runtime nonidealities enabled.")
    print("SINAD = 10 log10(sum(clean^2) / sum((all-on - clean)^2)).")
    print("ENOB = (SINAD - 1.76) / 6.02.")
    print()
    print("{:<8} {:>14} {:>14} {:>14} {:>14}".format(
        "Layer", "ENOB (bits)", "SINAD (dB)", "Clean RMS", "N+D RMS"))
    for row in rows:
        print("{:<8} {:>14.3f} {:>14.3f} {:>14.6e} {:>14.6e}".format(
            row["layer"], row["enob_bits"], row["sinad_db"],
            row["clean_rms"], row["noise_distortion_rms"]))

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("\nCSV: {}".format(path.resolve()))


if __name__ == "__main__":
    main()
