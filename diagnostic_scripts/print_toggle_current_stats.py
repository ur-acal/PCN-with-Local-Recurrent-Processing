#!/usr/bin/env python3
"""Print exact Level-3 slice-current statistics for one runtime batch."""

import argparse
import csv
import math
from pathlib import Path
import types

import torch

from diagnostic_config import add_model_arguments
from diagnostic_runtime import (
    DEFAULT_CORNER,
    DEFAULT_SEED,
    build_runtime_trial,
    set_current_noise,
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


class Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.max_abs = 0.0

    def update(self, values):
        values = values.detach().double()
        self.count += values.numel()
        self.total += values.sum().item()
        self.square_total += values.square().sum().item()
        self.max_abs = max(self.max_abs, values.abs().max().item())

    def values(self):
        mean = self.total / self.count
        rms = math.sqrt(self.square_total / self.count)
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return mean, math.sqrt(variance), rms, self.max_abs


class CurrentRecord:
    def __init__(self):
        self.det = Moments()
        self.total = Moments()
        self.summing = Moments()
        self.coupler = Moments()
        self.noise = Moments()

    def update(self, deterministic, summing, coupler):
        noise = summing + coupler
        self.det.update(deterministic)
        self.total.update(deterministic + noise)
        self.summing.update(summing)
        self.coupler.update(coupler)
        self.noise.update(noise)

    def row(self, layer, cycle, stage):
        det_mean, det_std, det_rms, det_max = self.det.values()
        total_mean, total_std, total_rms, total_max = self.total.values()
        _, _, summing_rms, _ = self.summing.values()
        _, _, coupler_rms, _ = self.coupler.values()
        _, _, noise_rms, _ = self.noise.values()
        snr_db = math.inf if self.noise.square_total == 0 else (
            10.0 * math.log10(
                self.det.square_total / self.noise.square_total))
        return {
            "layer": layer,
            "cycle": cycle,
            "stage": stage,
            "det_mean_A": det_mean,
            "det_std_A": det_std,
            "det_rms_A": det_rms,
            "det_max_abs_A": det_max,
            "total_mean_A": total_mean,
            "total_std_A": total_std,
            "total_rms_A": total_rms,
            "total_max_abs_A": total_max,
            "summing_noise_rms_A": summing_rms,
            "coupler_noise_rms_A": coupler_rms,
            "combined_noise_rms_A": noise_rms,
            "current_snr_db": snr_db,
        }


class RuntimeCurrentRecorder:
    """Wrap exact runtime methods without changing ode_pc.py."""

    def __init__(self, model):
        self.model = model
        self.records = {}
        self.layer_records = {}
        self.original = []
        self.current_cycle = {}
        self.pending = {}

    def _update(self, layer, cycle, stage, deterministic, summing, coupler):
        key = (layer, cycle, stage)
        self.records.setdefault(key, CurrentRecord()).update(
            deterministic, summing, coupler)
        layer_key = (layer, stage)
        self.layer_records.setdefault(layer_key, CurrentRecord()).update(
            deterministic, summing, coupler)

    def attach(self):
        for layer, block in enumerate(self.model.PcConvs, start=1):
            self.current_cycle[layer] = 0
            self.pending[layer] = {"summing": None, "coupler": None}
            methods = {
                "run_z_stage": block.run_z_stage,
                "_brownian_increment": block._brownian_increment,
                "_coupler_brownian_increment": (
                    block._coupler_brownian_increment),
                "integrate_pulse_slice": block.integrate_pulse_slice,
            }
            self.original.append((block, methods))

            original_z = methods["run_z_stage"]
            original_summing = methods["_brownian_increment"]
            original_coupler = methods["_coupler_brownian_increment"]
            original_integrate = methods["integrate_pulse_slice"]

            def run_z(this, y_hold, z, T_z, layer=layer,
                      original=original_z):
                self.current_cycle[layer] += 1
                return original(y_hold, z, T_z)

            def summing_increment(this, state, duration, stage, layer=layer,
                                  original=original_summing):
                value = original(state, duration, stage)
                self.pending[layer]["summing"] = value
                return value

            def coupler_increment(this, state, duration, stage,
                                  active_coupler_count, layer=layer,
                                  original=original_coupler):
                value = original(
                    state, duration, stage, active_coupler_count)
                self.pending[layer]["coupler"] = value
                return value

            def integrate(this, state, duration, rhs_fn, stage, slice_idx,
                          constant_rhs=None, active_coupler_count=None,
                          layer=layer, original=original_integrate):
                self.pending[layer]["summing"] = None
                self.pending[layer]["coupler"] = None
                output = original(
                    state, duration, rhs_fn, stage, slice_idx,
                    constant_rhs=constant_rhs,
                    active_coupler_count=active_coupler_count)
                if constant_rhs is None:
                    raise RuntimeError(
                        "Current diagnostics require toggle_fast_path=true.")
                capacitance = float(this._stage_capacitance(stage))
                duration_value = torch.as_tensor(
                    duration, device=state.device, dtype=state.dtype)
                deterministic = constant_rhs * capacitance
                summing_delta = self.pending[layer]["summing"]
                coupler_delta = self.pending[layer]["coupler"]
                summing = (torch.zeros_like(deterministic)
                            if summing_delta is None else
                            summing_delta * capacitance / duration_value)
                coupler = (torch.zeros_like(deterministic)
                            if coupler_delta is None else
                            coupler_delta * capacitance / duration_value)
                self._update(
                    layer, self.current_cycle[layer], stage,
                    deterministic, summing, coupler)
                return output

            block.run_z_stage = types.MethodType(run_z, block)
            block._brownian_increment = types.MethodType(
                summing_increment, block)
            block._coupler_brownian_increment = types.MethodType(
                coupler_increment, block)
            block.integrate_pulse_slice = types.MethodType(integrate, block)

    def detach(self):
        for block, methods in self.original:
            for name, method in methods.items():
                setattr(block, name, method)


def _print_rows(title, rows):
    scale = 1e6
    print("\n{} (currents in uA)".format(title))
    print("{:<5} {:<5} {:<5} {:>10} {:>10} {:>10} {:>10} "
          "{:>10} {:>10} {:>10} {:>10}".format(
              "L", "Step", "Spin", "I_det mu", "I_det sd",
              "I_det rms", "max|I_det|", "I_tot mu", "I_tot sd",
              "I_tot rms", "max|I_tot|"))
    for row in rows:
        print("{:<5} {:<5} {:<5} {:>10.4f} {:>10.4f} {:>10.4f} "
              "{:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                  row["layer"], row["cycle"], row["stage"],
                  row["det_mean_A"] * scale, row["det_std_A"] * scale,
                  row["det_rms_A"] * scale,
                  row["det_max_abs_A"] * scale,
                  row["total_mean_A"] * scale,
                  row["total_std_A"] * scale,
                  row["total_rms_A"] * scale,
                  row["total_max_abs_A"] * scale))
    print("\n{:<5} {:<5} {:<5} {:>14} {:>14} {:>14} {:>12}".format(
        "L", "Step", "Spin", "Sum noise rms", "Cpl noise rms",
        "All noise rms", "SNR (dB)"))
    for row in rows:
        print("{:<5} {:<5} {:<5} {:>14.6f} {:>14.6f} {:>14.6f} "
              "{:>12.3f}".format(
                  row["layer"], row["cycle"], row["stage"],
                  row["summing_noise_rms_A"] * scale,
                  row["coupler_noise_rms_A"] * scale,
                  row["combined_noise_rms_A"] * scale,
                  row["current_snr_db"]))


def main():
    args = parse_args()
    trial = build_runtime_trial(
        model_name=args.model_name, model_root=args.model_dir,
        corner=args.corner, batch_size=args.batch_size,
        trial_index=args.trial, seed=args.seed, device=args.device)
    inputs, _ = trial.first_batch()
    set_current_noise(trial.model, True)
    recorder = RuntimeCurrentRecorder(trial.model)
    recorder.attach()
    try:
        with torch.no_grad():
            trial.model(inputs)
    finally:
        recorder.detach()

    order = lambda key: (key[0], key[1], 0 if key[2] == "z" else 1)
    rows = [recorder.records[key].row(*key)
            for key in sorted(recorder.records, key=order)]
    aggregate_rows = [
        recorder.layer_records[key].row(key[0], "all", key[1])
        for key in sorted(
            recorder.layer_records,
            key=lambda key: (key[0], 0 if key[1] == "z" else 1))]
    print("Task: {} | corner: {} | trial: {} | batch: {}".format(
        trial.args.task, trial.corner, args.trial, inputs.shape[0]))
    print("I_det includes nonlinear-R, sampled curve variation, DTC, and "
          "spin variation.")
    print("I_total_eff = I_det + C*deltaV_summing/dt + "
          "C*deltaV_coupler/dt, before state projection/clamping.")
    print("SNR = 10 log10(sum(I_det^2) / sum((I_total_eff-I_det)^2)).")
    _print_rows("Per-toggle-step statistics", rows)
    _print_rows("Per-layer stage aggregate", aggregate_rows)

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
