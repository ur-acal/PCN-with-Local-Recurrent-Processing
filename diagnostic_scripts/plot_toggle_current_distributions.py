#!/usr/bin/env python3
"""Plot pooled Level-3 summing-current distributions on an evaluation split."""

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
from statistics import NormalDist
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnostic_config import REPO_ROOT, add_model_arguments, model_paths
from diagnostic_runtime import (
    DEFAULT_SEED,
    _parse_ablation_args,
    build_runtime_trial,
)
from data_utils import MC45CornerData


def parse_args():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--corners", default="FS_V2_T1",
                        help="Comma-separated corner IDs, or 'all'.")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_bins", type=int, default=50)
    parser.add_argument("--max_bins", type=int, default=200)
    parser.add_argument("--minimum_train_accuracy", type=float, default=0.10)
    parser.add_argument("--dataset_split", choices=("train", "test"),
                        default="train", help=argparse.SUPPRESS)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Diagnostic-only bound; omit for the full split.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--granularity", choices=("layer", "stage"), default="layer",
                        help="Pool after layer N ends its stage at layer N; classifier stays separate.")
    parser.add_argument("--plot_bound_percentile", type=float, default=99.0,
                        help="Central fitted-Gaussian interval drawn on plots.")
    parser.add_argument("--summary_bound_percentiles", default="95,99",
                        help="Comma-separated fitted-Gaussian intervals in summaries.")
    parser.add_argument("--plot_lower_sigma", type=float, default=4.0,
                        help="Plot lower limit as mean minus this many fitted standard deviations.")
    parser.add_argument("--plot_upper_sigma", type=float, default=4.0,
                        help="Plot upper limit as mean plus this many fitted standard deviations.")
    parser.add_argument("--replot_existing", default=None,
                        help="Regenerate plots/tables from an existing result directory without inference.")
    return parser.parse_args()


def parse_bound_percentiles(value):
    values = [float(item.strip()) for item in str(value).split(",")
              if item.strip()]
    if not values or any(not math.isfinite(item) or item <= 0.0 or item >= 100.0
                         for item in values):
        raise ValueError("Bound percentiles must be between 0 and 100.")
    return list(dict.fromkeys(values))


def gaussian_bounds(mean, std, percentile):
    z_score = NormalDist().inv_cdf(0.5 + float(percentile) / 200.0)
    return mean - z_score * std, mean + z_score * std


def resolve_model_arguments(args):
    """Replace optional CLI model arguments with their effective defaults."""
    paths = model_paths(model_name=args.model_name, model_root=args.model_dir)
    args.model_name = paths["model_name"]
    args.model_dir = str(paths["model_root"])
    args.task = paths["task"]
    args.img_type = paths["img_type"]
    return args


def dataset_description(args):
    classes = "100" if args.task == "cifar100" else "10"
    if args.img_type.lower() == "cifair":
        name = "CiFAIR-{}".format(classes)
    elif args.img_type.lower() == "scangfi":
        name = "scanGFI CIFAR-{}".format(classes)
    else:
        name = "{} CIFAR-{}".format(args.img_type, classes)
    return "{} {} split with evaluation preprocessing".format(
        name, args.dataset_split)


class BatchMoments:
    def __init__(self):
        self.count = 0
        self.total = None
        self.square_total = None
        self.minimum = None
        self.maximum = None

    @staticmethod
    def reduce(values):
        values = values.detach()
        return (
            values.numel(),
            values.sum(dtype=torch.float64),
            values.square().sum(dtype=torch.float64),
            values.min().double(),
            values.max().double(),
        )

    def merge(self, reduced):
        count, total, square_total, minimum, maximum = reduced
        self.count += count
        if self.total is None:
            self.total = total.clone()
            self.square_total = square_total.clone()
            self.minimum = minimum.clone()
            self.maximum = maximum.clone()
        else:
            self.total.add_(total)
            self.square_total.add_(square_total)
            self.minimum.copy_(torch.minimum(self.minimum, minimum))
            self.maximum.copy_(torch.maximum(self.maximum, maximum))

    def result(self):
        if not self.count:
            raise RuntimeError("A requested current distribution is empty.")
        total = self.total.item()
        square_total = self.square_total.item()
        mean = total / self.count
        variance = max(square_total / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "sum": total,
            "square_sum": square_total,
            "mean_A": mean,
            "std_A": math.sqrt(variance),
            "min_A": self.minimum.item(),
            "max_A": self.maximum.item(),
        }


def distribution_keys(layer, branch):
    return (
        ("separate", layer, branch),
        ("combined", layer, "combined"),
    )


class MomentSink:
    def __init__(self):
        self.values = {"deterministic": {}, "total": {}}

    def update(self, layer, branch, deterministic, total):
        reduced = {
            "deterministic": BatchMoments.reduce(deterministic),
            "total": BatchMoments.reduce(total),
        }
        for kind in ("deterministic", "total"):
            for key in distribution_keys(layer, branch):
                self.values[kind].setdefault(key, BatchMoments()).merge(
                    reduced[kind])

    def results(self):
        return {
            kind: {key: value.result() for key, value in entries.items()}
            for kind, entries in self.values.items()
        }


class HistogramSink:
    def __init__(self, specifications):
        self.specifications = specifications
        self.counts = {
            kind: {key: None for key in entries}
            for kind, entries in specifications.items()
        }
        self.moments = MomentSink()

    def update(self, layer, branch, deterministic, total):
        self.moments.update(layer, branch, deterministic, total)
        tensors = {"deterministic": deterministic, "total": total}
        for kind, tensor in tensors.items():
            for key in distribution_keys(layer, branch):
                spec = self.specifications[kind][key]
                histogram = torch.histc(
                    tensor.detach().float(), bins=spec["bins"],
                    min=spec["edge_min_A"], max=spec["edge_max_A"]).double()
                if self.counts[kind][key] is None:
                    self.counts[kind][key] = histogram
                else:
                    self.counts[kind][key].add_(histogram)

    def cpu_counts(self):
        return {
            kind: {key: value.cpu().numpy().astype(np.int64)
                   for key, value in entries.items()}
            for kind, entries in self.counts.items()
        }


class CurrentRecorder:
    """Observe existing Level-3 updates without changing production dynamics."""

    def __init__(self, model, sink, layer_groups=None):
        self.model = model
        self.sink = sink
        self.original = []
        self.pending = {}
        self.layer_groups = layer_groups or {}

    def _blocks(self):
        blocks = [
            (block, "layer_{:02d}".format(index), None)
            for index, block in enumerate(self.model.PcConvs, start=1)
        ]
        from final_linear import AnalogLinear
        head = getattr(self.model, "linear", None)
        if isinstance(head, AnalogLinear):
            if head._circuit is None:
                raise RuntimeError(
                    "Analog classifier circuit was not built by Level-3 preparation.")
            blocks.append((head._circuit, "final_linear", "FF"))
        return blocks

    def attach(self):
        for block, layer, forced_branch in self._blocks():
            layer = self.layer_groups.get(layer, layer)
            identity = id(block)
            self.pending[identity] = {"summing": None, "coupler": None}
            methods = {
                "_brownian_increment": block._brownian_increment,
                "_coupler_brownian_increment": block._coupler_brownian_increment,
                "integrate_pulse_slice": block.integrate_pulse_slice,
            }
            self.original.append((block, methods))
            original_summing = methods["_brownian_increment"]
            original_coupler = methods["_coupler_brownian_increment"]
            original_integrate = methods["integrate_pulse_slice"]

            def summing_increment(this, state, duration, stage,
                                  ident=identity, original=original_summing):
                value = original(state, duration, stage)
                self.pending[ident]["summing"] = value
                return value

            def coupler_increment(this, state, duration, stage,
                                  active_coupler_count, ident=identity,
                                  original=original_coupler):
                value = original(
                    state, duration, stage, active_coupler_count)
                self.pending[ident]["coupler"] = value
                return value

            def integrate(this, state, duration, rhs_fn, stage, slice_idx,
                          constant_rhs=None, active_coupler_count=None,
                          ident=identity, layer_name=layer,
                          branch_override=forced_branch,
                          original=original_integrate):
                self.pending[ident]["summing"] = None
                self.pending[ident]["coupler"] = None
                output = original(
                    state, duration, rhs_fn, stage, slice_idx,
                    constant_rhs=constant_rhs,
                    active_coupler_count=active_coupler_count)
                if constant_rhs is None:
                    raise RuntimeError(
                        "Current distributions require toggle_fast_path=true.")
                capacitance = float(this._stage_capacitance(stage))
                duration_tensor = torch.as_tensor(
                    duration, device=state.device, dtype=state.dtype)
                deterministic = constant_rhs * capacitance
                summing_delta = self.pending[ident]["summing"]
                coupler_delta = self.pending[ident]["coupler"]
                summing = (torch.zeros_like(deterministic)
                            if summing_delta is None else
                            summing_delta * capacitance / duration_tensor)
                coupler = (torch.zeros_like(deterministic)
                            if coupler_delta is None else
                            coupler_delta * capacitance / duration_tensor)
                branch = branch_override or ("FB" if stage == "z" else "FF")
                self.sink.update(
                    layer_name, branch, deterministic,
                    deterministic + summing + coupler)
                return output

            block._brownian_increment = types.MethodType(
                summing_increment, block)
            block._coupler_brownian_increment = types.MethodType(
                coupler_increment, block)
            block.integrate_pulse_slice = types.MethodType(integrate, block)

    def detach(self):
        for block, methods in self.original:
            for name, method in methods.items():
                setattr(block, name, method)


class Accuracy:
    def __init__(self):
        self.correct1 = 0
        self.correct5 = 0
        self.count = 0

    def update(self, logits, targets):
        k = min(5, logits.shape[1])
        predictions = logits.topk(k, dim=1).indices
        self.correct1 += predictions[:, 0].eq(targets).sum().item()
        self.correct5 += predictions.eq(targets[:, None]).any(dim=1).sum().item()
        self.count += targets.numel()

    def result(self):
        return {
            "samples": self.count,
            "top1": self.correct1 / self.count,
            "top5": self.correct5 / self.count,
        }


def canonical_corners(args):
    config = _parse_ablation_args(
        args.model_name, args.model_dir, args.batch_size, 0, args.seed)
    catalog = MC45CornerData(
        config.mc_45_corner_dir,
        spin_variation_source=config.mc_spin_variation_source,
        dtc_pulse_width_variation_source=(
            config.mc_dtc_pulse_width_variation_source),
        relu_monte_carlo_source=config.mc_relu_monte_carlo_source,
        coupler_nonlinear_variation_source=(
            config.mc_coupler_nonlinear_variation_source),
        coupler_nonlinear_variation_quantity=(
            config.mc_coupler_nonlinear_variation_quantity),
        coupler_nominal_R=config.mc_coupler_nominal_R)
    available = [entry["id"] for entry in catalog.corners]
    requested = args.corners.strip()
    if requested.lower() == "all":
        return available
    selected = [value.strip().upper() for value in requested.split(",")
                if value.strip()]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError("Unknown corners: {}".format(", ".join(unknown)))
    if not selected:
        raise ValueError("At least one corner is required.")
    return selected


def run_pass(args, corners, sink, pass_number):
    rows = []
    pooled = Accuracy()
    for corner in corners:
        for trial_index in range(args.n_trials):
            print("Pass {} | corner {} | trial {}/{}".format(
                pass_number, corner, trial_index + 1, args.n_trials), flush=True)
            runtime = build_runtime_trial(
                model_name=args.model_name, model_root=args.model_dir,
                corner=corner, batch_size=args.batch_size,
                trial_index=trial_index, seed=args.seed, device=args.device,
                dataset_split=args.dataset_split)
            runtime.reset_data_rng()
            from current_stages import stage_members
            members = stage_members(len(runtime.model.PcConvs), [
                i + 1 for i, pooled_layer in enumerate(runtime.model.max_pool) if pooled_layer])
            if hasattr(args, 'stage_members') and args.stage_members != members:
                raise RuntimeError('Stage layout changed between trials/passes.')
            args.stage_members = members
            groups = ({layer: group for group, layers in members.items() for layer in layers}
                      if getattr(args, 'granularity', 'layer') == 'stage' else {})
            recorder = CurrentRecorder(runtime.model, sink, groups)
            recorder.attach()
            accuracy = Accuracy()
            try:
                with torch.no_grad():
                    for batch_index, (inputs, targets) in enumerate(runtime.dataloader):
                        if (args.max_batches is not None and
                                batch_index >= args.max_batches):
                            break
                        inputs = inputs.to(runtime.device)
                        targets = targets.to(runtime.device)
                        logits = runtime.model(inputs)
                        if isinstance(logits, tuple):
                            logits = logits[-1]
                        accuracy.update(logits, targets)
                        pooled.update(logits, targets)
                        if (batch_index + 1) % 25 == 0:
                            current = accuracy.result()
                            print("  batch {} | {} top1 {:.4%}".format(
                                batch_index + 1, args.dataset_split,
                                current["top1"]), flush=True)
            finally:
                recorder.detach()
            result = accuracy.result()
            result.update(corner=corner, trial=trial_index)
            rows.append(result)
            print("  completed | {} top1 {:.4%} | top5 {:.4%}".format(
                args.dataset_split, result["top1"], result["top5"]),
                flush=True)
            del recorder, runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, pooled.result()


def make_specifications(results, min_bins, max_bins):
    specifications = {"deterministic": {}, "total": {}}
    for kind, entries in results.items():
        for key, stats in entries.items():
            lower, upper = stats["min_A"], stats["max_A"]
            if upper == lower:
                padding = max(abs(lower) * 1e-6, 1e-15)
                edge_min, edge_max, bins = lower - padding, upper + padding, 1
            else:
                width = 3.5 * stats["std_A"] * stats["count"] ** (-1.0 / 3.0)
                proposed = (math.ceil((upper - lower) / width)
                            if width > 0 and math.isfinite(width) else min_bins)
                bins = max(min_bins, min(max_bins, proposed))
                # A replay can differ by a few float32 ULPs even with identical
                # seeds. Keep those endpoint samples inside the exact histogram.
                padding = max((upper - lower) * 1e-6, 1e-15)
                edge_min, edge_max = lower - padding, upper + padding
            specifications[kind][key] = {
                "bins": int(bins),
                "edge_min_A": float(edge_min),
                "edge_max_A": float(edge_max),
            }
    return specifications


def assert_replay(first, second):
    for kind in first:
        if set(first[kind]) != set(second[kind]):
            raise RuntimeError("Passes produced different distribution keys.")
        for key in first[kind]:
            a, b = first[kind][key], second[kind][key]
            if a["count"] != b["count"]:
                raise RuntimeError("Pass count mismatch for {} {}".format(kind, key))
            for name in ("mean_A", "std_A", "min_A", "max_A"):
                if not math.isclose(a[name], b[name], rel_tol=1e-5, abs_tol=1e-13):
                    raise RuntimeError(
                        "Pass replay mismatch for {} {} {}: {} versus {}".format(
                            kind, key, name, a[name], b[name]))


def select_unit(results):
    maximum = max(abs(stats[name]) for entries in results.values()
                  for stats in entries.values() for name in ("min_A", "max_A"))
    return (1e6, "µA") if maximum >= 1e-6 else (1e9, "nA")


def normal_pdf(x, mean, std):
    if std == 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * math.sqrt(2.0 * math.pi))


def key_name(key):
    mode, layer, branch = key
    return "{}__{}__{}".format(mode, layer, branch)


def write_outputs(output_dir, args, corners, first, specifications, counts,
                  accuracy_rows, pooled_accuracy, preserve_run_files=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    scale, unit = select_unit(first)
    summary_percentiles = parse_bound_percentiles(
        args.summary_bound_percentiles)
    csv_rows = []
    npz = {}
    colors = {"deterministic": "#4C78A8", "total": "#4C78A8"}
    for kind, entries in first.items():
        for key in sorted(entries):
            mode, layer, branch = key
            stats = entries[key]
            spec = specifications[kind][key]
            edges_A = np.linspace(
                spec["edge_min_A"], spec["edge_max_A"], spec["bins"] + 1)
            histogram = counts[kind][key]
            if int(histogram.sum()) != stats["count"]:
                raise RuntimeError(
                    "Histogram count mismatch for {} {}: {} versus {}".format(
                        kind, key, histogram.sum(), stats["count"]))
            edges = edges_A * scale
            widths = np.diff(edges)
            density = histogram / (stats["count"] * widths)
            mean = stats["mean_A"] * scale
            std = stats["std_A"] * scale
            low, high = gaussian_bounds(
                mean, std, args.plot_bound_percentile)
            x_low = mean - args.plot_lower_sigma * std
            x_high = mean + args.plot_upper_sigma * std
            if x_high == x_low:
                x_low, x_high = x_low - 1.0, x_high + 1.0
            x = np.linspace(x_low, x_high, 600)
            pdf = normal_pdf(x, mean, std)

            leaf = output_dir / kind / mode
            leaf.mkdir(parents=True, exist_ok=True)
            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            axis.bar(edges[:-1], density, width=widths, align="edge",
                     alpha=0.28, color=colors[kind], edgecolor=colors[kind],
                     linewidth=0.6)
            axis.plot(x, pdf, color=colors[kind], linewidth=2.2,
                      label="Gaussian fit: μ={:.4g}, σ={:.4g} {}".format(
                          mean, std, unit))
            axis.axvline(low, color="#444444", linestyle="--", linewidth=1.3,
                         label="Fitted {:g}% interval".format(
                             args.plot_bound_percentile))
            axis.axvline(high, color="#444444", linestyle="--", linewidth=1.3)
            axis.set_xlim(x_low, x_high)
            label = "{:.4g} {}"
            axis.annotate(label.format(low, unit), xy=(low, 0),
                          xycoords=axis.get_xaxis_transform(), xytext=(-4, 5),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=8.5, color="#444444")
            axis.annotate(label.format(high, unit), xy=(high, 0),
                          xycoords=axis.get_xaxis_transform(), xytext=(4, 5),
                          textcoords="offset points", ha="left", va="bottom",
                          fontsize=8.5, color="#444444")
            axis.set_title("{} {} summing-current distribution".format(
                layer.replace("_", " ").title(), branch))
            axis.set_xlabel("Summing current ({})".format(unit))
            axis.set_ylabel("Probability density")
            axis.grid(axis="y", alpha=0.22)
            axis.legend(fontsize=8.5)
            figure.tight_layout()
            stem = "{}_{}".format(layer, branch.lower())
            figure.savefig(leaf / (stem + ".pdf"), bbox_inches="tight")
            figure.savefig(leaf / (stem + ".png"), dpi=180,
                           bbox_inches="tight")
            plt.close(figure)

            row = {
                "kind": kind,
                "mode": mode,
                "layer": layer,
                "branch": branch,
                "samples": stats["count"],
                "mean_A": stats["mean_A"],
                "std_A": stats["std_A"],
                "minimum_A": stats["min_A"],
                "maximum_A": stats["max_A"],
                "bins": spec["bins"],
            }
            for percentile in summary_percentiles:
                label = "{:g}".format(percentile)
                bound_low, bound_high = gaussian_bounds(
                    stats["mean_A"], stats["std_A"], percentile)
                row["fitted_{}_low_A".format(label)] = bound_low
                row["fitted_{}_high_A".format(label)] = bound_high
            csv_rows.append(row)
            prefix = "{}__{}".format(kind, key_name(key))
            npz[prefix + "__edges_A"] = edges_A
            npz[prefix + "__counts"] = histogram
            npz[prefix + "__fit_x_A"] = x / scale
            npz[prefix + "__fit_density_per_A"] = pdf * scale

    fields = list(csv_rows[0])
    with (output_dir / "statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(output_dir / "histogram_data.npz", **npz)

    for kind in ("deterministic", "total"):
        for mode in ("separate", "combined"):
            rows = [row for row in csv_rows
                    if row["kind"] == kind and row["mode"] == mode]
            leaf = output_dir / kind / mode
            with (leaf / "summary.md").open("w") as handle:
                handle.write("# {} {} summing-current distributions\n\n".format(
                    kind.title(), mode))
                handle.write("All currents are signed. Bounds are central intervals "
                             "from the fitted Gaussian.\n\n")
                columns = ["Layer", "Branch", "Samples", "Mean ({})".format(unit)]
                for percentile in summary_percentiles:
                    columns.extend(("Lower {:g}% ({})".format(percentile, unit),
                                    "Upper {:g}% ({})".format(percentile, unit)))
                columns.extend(("Observed min ({})".format(unit),
                                "Observed max ({})".format(unit)))
                handle.write("| " + " | ".join(columns) + " |\n")
                handle.write("|---|---:|---:" + "|---:" * (len(columns) - 3) + "|\n")
                for row in rows:
                    values = [row["layer"], row["branch"], "{:,}".format(row["samples"]),
                              "{:.6g}".format(row["mean_A"] * scale)]
                    for percentile in summary_percentiles:
                        label = "{:g}".format(percentile)
                        values.extend((
                            "{:.6g}".format(row["fitted_{}_low_A".format(label)] * scale),
                            "{:.6g}".format(row["fitted_{}_high_A".format(label)] * scale)))
                    values.extend(("{:.6g}".format(row["minimum_A"] * scale),
                                   "{:.6g}".format(row["maximum_A"] * scale)))
                    handle.write("| " + " | ".join(values) + " |\n")

    if preserve_run_files:
        config_path = output_dir / "run_config.json"
        if config_path.exists():
            with config_path.open() as handle:
                metadata = json.load(handle)
            metadata["plot_bound_percentile"] = args.plot_bound_percentile
            metadata["summary_bound_percentiles"] = summary_percentiles
            metadata["plot_lower_sigma"] = args.plot_lower_sigma
            metadata["plot_upper_sigma"] = args.plot_upper_sigma
            with config_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)
                handle.write("\n")
        return

    accuracy_filename = "{}_accuracy.md".format(args.dataset_split)
    with (output_dir / accuracy_filename).open("w") as handle:
        handle.write("# {}-split accuracy sanity check\n\n".format(
            args.dataset_split.title()))
        handle.write("| Corner | Trial | Samples | Top-1 | Top-5 |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in accuracy_rows:
            handle.write("| {} | {} | {:,} | {:.4%} | {:.4%} |\n".format(
                row["corner"], row["trial"], row["samples"],
                row["top1"], row["top5"]))
        handle.write("| **Pooled** | — | {:,} | **{:.4%}** | **{:.4%}** |\n".format(
            pooled_accuracy["samples"], pooled_accuracy["top1"],
            pooled_accuracy["top5"]))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=True).stdout.strip()
    metadata = {
        "model_name": args.model_name,
        "granularity": getattr(args, 'granularity', 'layer'),
        "stage_members": getattr(args, 'stage_members', None),
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset": dataset_description(args),
        "dataset_split": args.dataset_split,
        "corners": corners,
        "n_trials": args.n_trials,
        "base_seed": args.seed,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "git_commit": commit,
        "accuracy": pooled_accuracy,
        "bin_rule": "Scott, clipped to [{}, {}]".format(
            args.min_bins, args.max_bins),
        "plot_bound_percentile": args.plot_bound_percentile,
        "summary_bound_percentiles": summary_percentiles,
        "plot_lower_sigma": args.plot_lower_sigma,
        "plot_upper_sigma": args.plot_upper_sigma,
    }
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    with (output_dir / "README.md").open("w") as handle:
        handle.write("# Toggle summing-current distributions\n\n")
        handle.write("- {} accuracy: [{}]({})\n".format(
            args.dataset_split.title(), accuracy_filename, accuracy_filename))
        handle.write("- Exact histogram arrays: `histogram_data.npz`\n")
        handle.write("- Complete statistics: `statistics.csv`\n")
        handle.write("- Deterministic plots: `deterministic/` "
                     "([combined table](deterministic/combined/summary.md), "
                     "[separate FF/FB table](deterministic/separate/summary.md))\n")
        handle.write("- Total-current plots: `total/` "
                     "([combined table](total/combined/summary.md), "
                     "[separate FF/FB table](total/separate/summary.md))\n")


def replot_existing(output_dir, args):
    """Regenerate plots and tables from saved histograms without inference."""
    output_dir = Path(output_dir).resolve()
    statistics_path = output_dir / "statistics.csv"
    histogram_path = output_dir / "histogram_data.npz"
    if not statistics_path.is_file() or not histogram_path.is_file():
        raise FileNotFoundError(
            "Existing results require statistics.csv and histogram_data.npz")

    first = {"deterministic": {}, "total": {}}
    specifications = {"deterministic": {}, "total": {}}
    counts = {"deterministic": {}, "total": {}}
    with statistics_path.open(newline="") as handle, np.load(histogram_path) as data:
        for row in csv.DictReader(handle):
            kind = row["kind"]
            key = (row["mode"], row["layer"], row["branch"])
            prefix = "{}__{}".format(kind, key_name(key))
            edges = data[prefix + "__edges_A"]
            histogram = data[prefix + "__counts"].astype(np.int64)
            first[kind][key] = {
                "count": int(row["samples"]),
                "mean_A": float(row["mean_A"]),
                "std_A": float(row["std_A"]),
                "min_A": float(row["minimum_A"]),
                "max_A": float(row["maximum_A"]),
            }
            specifications[kind][key] = {
                "bins": len(histogram),
                "edge_min_A": float(edges[0]),
                "edge_max_A": float(edges[-1]),
            }
            counts[kind][key] = histogram
    write_outputs(
        output_dir, args, [], first, specifications, counts, [], {},
        preserve_run_files=True)


def default_output(args, corners):
    if len(corners) == 45:
        corner_tag = "all45"
    elif len(corners) == 1:
        corner_tag = corners[0]
    else:
        digest = hashlib.sha256(",".join(corners).encode()).hexdigest()[:8]
        corner_tag = "{}corners_{}".format(len(corners), digest)
    experiment = Path(args.model_dir).name
    split_tag = "" if args.dataset_split == "train" else "_{}".format(
        args.dataset_split)
    if getattr(args, 'granularity', 'layer') == 'stage':
        split_tag += '_stage'
    return (REPO_ROOT / "results" / "toggle_summing_current_distribution" /
            "{}{}_{}_{}trials".format(
                experiment, split_tag, corner_tag, args.n_trials))


def main():
    args = parse_args()
    parse_bound_percentiles(args.summary_bound_percentiles)
    if not 0.0 < args.plot_bound_percentile < 100.0:
        raise ValueError("plot_bound_percentile must be between 0 and 100.")
    if (not math.isfinite(args.plot_lower_sigma) or args.plot_lower_sigma <= 0.0 or
            not math.isfinite(args.plot_upper_sigma) or args.plot_upper_sigma <= 0.0):
        raise ValueError("plot_lower_sigma and plot_upper_sigma must be finite and positive.")
    if args.replot_existing:
        replot_existing(args.replot_existing, args)
        print("Plots and summaries regenerated: {}".format(
            Path(args.replot_existing).resolve()), flush=True)
        return
    args = resolve_model_arguments(args)
    if args.n_trials <= 0 or args.batch_size <= 0:
        raise ValueError("n_trials and batch_size must be positive.")
    if args.min_bins <= 0 or args.max_bins < args.min_bins:
        raise ValueError("Invalid histogram bin limits.")
    corners = canonical_corners(args)
    output_dir = Path(args.output_dir) if args.output_dir else default_output(args, corners)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Output directory is not empty; use --overwrite: {}".format(output_dir))

    print("Output: {}".format(output_dir.resolve()), flush=True)
    print("Corners: {} | trials per corner: {}".format(
        ", ".join(corners), args.n_trials), flush=True)
    first_sink = MomentSink()
    accuracy_rows, pooled_accuracy = run_pass(
        args, corners, first_sink, pass_number=1)
    first = first_sink.results()
    print("Pass 1 pooled {} accuracy: top1 {:.4%}, top5 {:.4%}".format(
        args.dataset_split, pooled_accuracy["top1"],
        pooled_accuracy["top5"]), flush=True)
    if pooled_accuracy["top1"] < args.minimum_train_accuracy:
        raise RuntimeError(
            "{} accuracy {:.4%} is below sanity threshold {:.4%}; "
            "refusing to generate potentially invalid distributions.".format(
                args.dataset_split.title(), pooled_accuracy["top1"],
                args.minimum_train_accuracy))

    specifications = make_specifications(
        first, args.min_bins, args.max_bins)
    second_sink = HistogramSink(specifications)
    second_accuracy_rows, second_pooled = run_pass(
        args, corners, second_sink, pass_number=2)
    second = second_sink.moments.results()
    assert_replay(first, second)
    for first_row, second_row in zip(accuracy_rows, second_accuracy_rows):
        if (first_row["corner"], first_row["trial"], first_row["samples"],
                first_row["top1"], first_row["top5"]) != (
                second_row["corner"], second_row["trial"], second_row["samples"],
                second_row["top1"], second_row["top5"]):
            raise RuntimeError("Accuracy changed between replay passes.")
    if pooled_accuracy != second_pooled:
        raise RuntimeError("Pooled accuracy changed between replay passes.")

    write_outputs(
        output_dir, args, corners, first, specifications,
        second_sink.cpu_counts(), accuracy_rows, pooled_accuracy)
    print("{} top1: {:.4%} | top5: {:.4%}".format(
        args.dataset_split.title(), pooled_accuracy["top1"],
        pooled_accuracy["top5"]), flush=True)
    print("Plots: {}".format(output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()
