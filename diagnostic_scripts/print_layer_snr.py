#!/usr/bin/env python3
"""Print propagated per-layer SNR from paired one-batch forward passes."""

import argparse
import csv
import math
from pathlib import Path

import torch

from diagnostic_config import add_model_arguments
from diagnostic_runtime import (
    DEFAULT_CORNER,
    DEFAULT_SEED,
    build_runtime_trial,
    restore_rng_state,
    set_current_noise,
    snapshot_rng_state,
)


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


def _capture_layer_outputs(model, inputs, output_device=None):
    outputs = {}
    handles = []
    for index, block in enumerate(model.PcConvs, start=1):
        def hook(_module, _inputs, output, layer=index):
            value = output.detach()
            if output_device is not None:
                value = value.to(output_device)
            outputs[layer] = value.clone()
        handles.append(block.register_forward_hook(hook))
    with torch.no_grad():
        logits = model(inputs)
    for handle in handles:
        handle.remove()
    logits = logits.detach()
    if output_device is not None:
        logits = logits.to(output_device)
    return outputs, logits.clone()


def _snr_row(layer, reference, noisy):
    reference = reference.double()
    error = noisy.double() - reference
    signal_power = reference.square().sum().item()
    noise_power = error.square().sum().item()
    signal_rms = math.sqrt(signal_power / reference.numel())
    noise_rms = math.sqrt(noise_power / reference.numel())
    snr_db = math.inf if noise_power == 0 else 10.0 * math.log10(
        signal_power / noise_power)
    return {
        "layer": layer,
        "shape": "x".join(str(value) for value in reference.shape),
        "signal_rms": signal_rms,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
    }


def main():
    args = parse_args()
    trial = build_runtime_trial(
        model_name=args.model_name, model_root=args.model_dir,
        corner=args.corner, batch_size=args.batch_size,
        trial_index=args.trial, seed=args.seed, device=args.device)
    inputs, _ = trial.first_batch()

    # Snapshot after model construction so fixed spin factors, curve assignments,
    # and expanded matrices are identical. Restoring this snapshot also makes the
    # DTC timing realization identical in the two forwards.
    state = snapshot_rng_state(trial.model)
    set_current_noise(trial.model, False)
    reference, reference_logits = _capture_layer_outputs(
        trial.model, inputs)

    restore_rng_state(state)
    set_current_noise(trial.model, True)
    noisy, noisy_logits = _capture_layer_outputs(trial.model, inputs)

    rows = [_snr_row(layer, reference[layer], noisy[layer])
            for layer in sorted(reference)]
    rows.append(_snr_row("logits", reference_logits, noisy_logits))

    print("Task: {} | corner: {} | trial: {} | batch: {}".format(
        trial.args.task, trial.corner, args.trial, inputs.shape[0]))
    print("Only summing-current and per-coupler current noise differ between runs.")
    print("SNR = 10 log10(sum(reference^2) / sum((noisy-reference)^2)).")
    print()
    print("{:<8} {:<22} {:>14} {:>14} {:>12}".format(
        "Layer", "Output shape", "Signal RMS", "Noise RMS", "SNR (dB)"))
    for row in rows:
        print("{:<8} {:<22} {:>14.6e} {:>14.6e} {:>12.3f}".format(
            row["layer"], row["shape"], row["signal_rms"],
            row["noise_rms"], row["snr_db"]))

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
