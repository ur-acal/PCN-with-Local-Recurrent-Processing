#!/usr/bin/env python3
"""Plot exact Level-3 z/y pulse-slice trajectories for sampled spins."""

import argparse
import csv
from pathlib import Path
import types

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from diagnostic_config import REPO_ROOT, add_model_arguments
from diagnostic_runtime import (
    DEFAULT_CORNER,
    DEFAULT_SEED,
    build_runtime_trial,
    restore_rng_state,
    snapshot_rng_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--corner", default=DEFAULT_CORNER)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Optional one-based ODE layer index; omit to record every layer.")
    parser.add_argument("--toggle_step", type=int, default=1,
                        help="One-based toggle step shown in the z plot.")
    parser.add_argument("--n_spins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "results" / "diagnostic_runtime" /
                    "spin_trajectories"))
    parser.add_argument(
        "--full_accuracy", action="store_true",
        help="Also run the complete test set after the equivalence check.")
    return parser.parse_args()


class SpinTrajectoryRecorder:
    """Observe exact returned slice states without changing their computation."""

    def __init__(self, block, layer, n_spins, toggle_step, seed):
        self.block = block
        self.layer = layer
        self.n_spins = int(n_spins)
        self.toggle_step = int(toggle_step)
        self.seed = int(seed)
        self.z_flat_indices = None
        self.y_flat_indices = None
        self.current_cycle = 0
        self.absolute_time = 0.0
        self.stage_start = 0.0
        self.y_hold = None
        self.z_times = []
        self.z_values = []
        self.y_times = []
        self.y_values = []
        self.y_update_intervals = []
        self.original = None

    @staticmethod
    def _seconds(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _select_spins(self, state, stage):
        name = "{}_flat_indices".format(stage)
        if getattr(self, name) is not None:
            return
        n_available = state[0].numel()
        generator = torch.Generator(device="cpu")
        stage_offset = 0 if stage == "z" else 1
        generator.manual_seed(self.seed + 2 * self.layer + stage_offset)
        indices = torch.randperm(
            n_available, generator=generator)[:min(
                self.n_spins, n_available)]
        setattr(self, name, indices)

    def _sample(self, state, stage):
        self._select_spins(state, stage)
        indices = getattr(self, "{}_flat_indices".format(stage)).to(
            state.device)
        return state[0].reshape(-1).index_select(
            0, indices).detach().cpu()

    def _append_z(self, time_s, state):
        self.z_times.append(float(time_s))
        self.z_values.append(self._sample(state, "z"))

    def _append_y(self, time_s, state):
        self.y_times.append(float(time_s))
        self.y_values.append(self._sample(state, "y"))

    def attach(self):
        original_z = self.block.run_z_stage
        original_y = self.block.run_y_stage
        original_integrate = self.block.integrate_pulse_slice
        self.original = {
            "run_z_stage": original_z,
            "run_y_stage": original_y,
            "integrate_pulse_slice": original_integrate,
        }

        def run_z(this, y_hold, z, T_z):
            self.current_cycle += 1
            self.stage_start = self.absolute_time
            self.y_hold = y_hold
            if self.current_cycle == 1:
                self._append_y(self.absolute_time, y_hold)
            if self.current_cycle == self.toggle_step:
                # Start at the already-reset z state; no reset transition is
                # added to the plot.
                self._append_z(0.0, z)
            output = original_z(y_hold, z, T_z)
            self.absolute_time += self._seconds(T_z)
            return output

        def run_y(this, y, h_hold, T_y):
            self.stage_start = self.absolute_time
            self.y_update_intervals.append((
                self.stage_start,
                self.stage_start + self._seconds(T_y)))
            output = original_y(y, h_hold, T_y)
            self.absolute_time += self._seconds(T_y)
            return output

        def integrate(this, state, duration, rhs_fn, stage, slice_idx,
                      constant_rhs=None, active_coupler_count=None):
            output = original_integrate(
                state, duration, rhs_fn, stage, slice_idx,
                constant_rhs=constant_rhs,
                active_coupler_count=active_coupler_count)
            time_s = self.stage_start + (
                slice_idx + 1) * self._seconds(duration)
            if stage == "z":
                # y is held while z charges; retain its actual pulse-grid
                # samples in the complete y timeline.
                self._append_y(time_s, self.y_hold)
                if self.current_cycle == self.toggle_step:
                    self._append_z(time_s - self.stage_start, output)
            else:
                self._append_y(time_s, output)
            return output

        self.block.run_z_stage = types.MethodType(run_z, self.block)
        self.block.run_y_stage = types.MethodType(run_y, self.block)
        self.block.integrate_pulse_slice = types.MethodType(
            integrate, self.block)

    def detach(self):
        if self.original is None:
            return
        for name, method in self.original.items():
            setattr(self.block, name, method)

    def tensors(self):
        return (
            torch.tensor(self.z_times, dtype=torch.float64),
            torch.stack(self.z_values, dim=0),
            torch.tensor(self.y_times, dtype=torch.float64),
            torch.stack(self.y_values, dim=0),
        )


def _plot(times_s, values, title, output_path, update_intervals=None):
    times_ns = times_s.numpy() * 1e9
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    if update_intervals:
        for index, (start_s, end_s) in enumerate(update_intervals):
            start_ns = start_s * 1e9
            end_ns = end_s * 1e9
            axis.axvspan(
                start_ns, end_ns, color="tab:blue", alpha=0.10)
            axis.axvline(
                start_ns, color="tab:blue", linestyle="--",
                linewidth=0.8, alpha=0.65)
            axis.axvline(
                end_ns, color="tab:blue", linestyle="--",
                linewidth=0.8, alpha=0.65)
    for spin in range(values.shape[1]):
        axis.plot(times_ns, values[:, spin].numpy(), linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Spin value")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _write_trajectory_csv(path, times_s, values, indices):
    fieldnames = ["time_s"] + [
        "spin_{}_flat_index_{}".format(number + 1, int(index))
        for number, index in enumerate(indices)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row, time_s in zip(values.tolist(), times_s.tolist()):
            writer.writerow([time_s] + row)


def _full_accuracy(trial, initial_rng, layers, n_spins, toggle_step, seed):
    restore_rng_state(initial_rng)
    trial.reset_data_rng()
    total = 0
    correct = 0
    for batch_index, (inputs, targets) in tqdm(
            enumerate(trial.dataloader), total=len(trial.dataloader)):
        inputs = inputs.to(trial.device)
        targets = targets.to(trial.device)
        recorders = []
        if batch_index == 0:
            # Instrument the first full-accuracy batch, then restore the exact
            # original methods for all remaining batches.
            recorders = [
                SpinTrajectoryRecorder(
                    trial.model.PcConvs[layer - 1], layer, n_spins,
                    toggle_step, seed)
                for layer in layers
            ]
            for recorder in recorders:
                recorder.attach()
        try:
            with torch.no_grad():
                logits = trial.model(inputs)
        finally:
            for recorder in reversed(recorders):
                recorder.detach()
        total += targets.numel()
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return 100.0 * correct / total


def main():
    args = parse_args()
    trial = build_runtime_trial(
        model_name=args.model_name, model_root=args.model_dir,
        corner=args.corner, batch_size=args.batch_size,
        trial_index=args.trial, seed=args.seed, device=args.device)
    inputs, _ = trial.first_batch()
    if args.layer is None:
        layers = list(range(1, len(trial.model.PcConvs) + 1))
    else:
        if args.layer < 1 or args.layer > len(trial.model.PcConvs):
            raise ValueError("layer must be between 1 and {}".format(
                len(trial.model.PcConvs)))
        layers = [args.layer]
    if args.toggle_step < 1 or args.toggle_step > 5:
        raise ValueError("toggle_step must be between 1 and 5")

    initial_rng = snapshot_rng_state(trial.model)
    with torch.no_grad():
        reference_logits = trial.model(inputs)

    restore_rng_state(initial_rng)
    with torch.no_grad():
        repeated_reference_logits = trial.model(inputs)

    restore_rng_state(initial_rng)
    recorders = [
        SpinTrajectoryRecorder(
            trial.model.PcConvs[layer - 1], layer,
            args.n_spins, args.toggle_step, args.seed)
        for layer in layers
    ]
    for recorder in recorders:
        recorder.attach()
    try:
        with torch.no_grad():
            recorded_logits = trial.model(inputs)
    finally:
        for recorder in reversed(recorders):
            recorder.detach()

    baseline_difference = (
        repeated_reference_logits - reference_logits).abs()
    recorded_difference = (recorded_logits - reference_logits).abs()
    predictions_match = torch.equal(
        recorded_logits.argmax(dim=1), reference_logits.argmax(dim=1))
    allclose = torch.allclose(
        recorded_logits, reference_logits, rtol=1e-3, atol=5e-4)
    print("Uninstrumented repeat: max_abs={:.6e} mean_abs={:.6e}".format(
        baseline_difference.max().item(), baseline_difference.mean().item()))
    print("Recorder equivalence: allclose={} prediction_match={} "
          "max_abs={:.6e} mean_abs={:.6e}".format(
              allclose, predictions_match, recorded_difference.max().item(),
              recorded_difference.mean().item()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Recorded {} layer(s) in one forward pass.".format(len(recorders)))
    for layer, recorder in zip(layers, recorders):
        z_times, z_values, y_times, y_values = recorder.tensors()
        print("Layer {} samples: z={} (one 15-slice step), y={} "
              "(five complete z/y steps)".format(
                  layer, z_times.numel(), y_times.numel()))
        print("Layer {} minimum time gaps: z={:.6f} ns, y={:.6f} ns".format(
            layer, torch.diff(z_times).min().item() * 1e9,
            torch.diff(y_times).min().item() * 1e9))
        prefix = "{}_{}_trial{}_layer{}".format(
            trial.args.task, trial.corner, args.trial, layer)
        z_pdf = output_dir / (prefix + "_z_step{}_trajectories.pdf".format(
            args.toggle_step))
        y_pdf = output_dir / (prefix + "_y_full_trajectories.pdf")
        _plot(
            z_times, z_values,
            "z-spin trajectories — layer {}, toggle step {}".format(
                layer, args.toggle_step), z_pdf)
        _plot(
            y_times, y_values,
            "y-spin trajectories — layer {}, full inference".format(layer),
            y_pdf, update_intervals=recorder.y_update_intervals)
        _write_trajectory_csv(
            z_pdf.with_suffix(".csv"), z_times, z_values,
            recorder.z_flat_indices)
        _write_trajectory_csv(
            y_pdf.with_suffix(".csv"), y_times, y_values,
            recorder.y_flat_indices)
        print("z trajectory: {}".format(z_pdf.resolve()))
        print("y trajectory: {}".format(y_pdf.resolve()))

    if args.full_accuracy:
        accuracy = _full_accuracy(
            trial, initial_rng, layers, args.n_spins,
            args.toggle_step, args.seed)
        print("Full test accuracy with first-batch trajectory instrumentation: "
              "{:.4f}%".format(accuracy))


if __name__ == "__main__":
    main()
