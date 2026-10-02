"""Observe one unchanged MC45 inference trial using its saved COMMAND log.

Save one batch as NumPy arrays; retain full-test logits only in memory for
five summary statistics and two PDF plots. Save margins for plot-only reruns.
No centering or softmax is used.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_saved_data(output_dir, drop_pp):
    with np.load(output_dir / "margin_data.npz") as data:
        margin = data["margin"].astype(np.float64)
        correct = data["correct"]
    accuracy = 100 * correct.mean()
    if not 0 < drop_pp <= accuracy:
        raise ValueError(f"drop_pp must be positive and at most the accuracy ({accuracy:g})")
    sorted_correct = np.sort(margin[correct])
    target_count = int(np.ceil(drop_pp * len(margin) / 100))
    # Strict margin < tau: move just above the selected margin, including ties.
    tau_star = np.nextafter(sorted_correct[target_count - 1], np.inf)
    actual_count = int(np.searchsorted(sorted_correct, tau_star, side="left"))
    actual_drop = 100 * actual_count / len(margin)
    remaining = accuracy - actual_drop
    marker = dict(requested_drop_pp=drop_pp, actual_drop_pp=actual_drop,
                  threshold=float(tau_star), current_accuracy=accuracy,
                  potential_remaining_accuracy=remaining,
                  vulnerable_correct_samples=actual_count,
                  fraction_of_correct_percent=100 * actual_count / correct.sum())
    (output_dir / "threshold_marker.json").write_text(json.dumps(marker, indent=2) + "\n")
    (output_dir / f"threshold_marker_drop{drop_pp:g}pp.json").write_text(json.dumps(marker, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margin, bins=60, weights=np.full(len(margin), 100 / len(margin)))
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="True-class margin (raw logit units)", ylabel="Test samples (%)",
           title="Final linear layer margin distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "signed_margin_histogram.pdf")
    plt.close(fig)

    tau = np.unique(np.r_[np.linspace(0, np.nextafter(float(np.abs(margin).max()), np.inf), 501), tau_star])
    near_correct = 100 * np.searchsorted(sorted_correct, tau, side="left") / len(margin)
    near_all = 100 * np.searchsorted(np.sort(np.abs(margin)), tau, side="left") / len(margin)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(tau, near_correct, label="Correct prediction and margin < τ")
    ax.plot(tau, near_all, label="Absolute margin < τ (all predictions)")
    ax.axhline(accuracy, color="0.4", linewidth=1, label="Current accuracy")
    ax.plot(tau_star, actual_drop, "o", color="tab:red", zorder=5)
    ax.vlines(tau_star, 0, actual_drop, colors="tab:red", linestyles="--", linewidth=1)
    ax.hlines(actual_drop, 0, tau_star, colors="tab:red", linestyles="--", linewidth=1)
    ax.set(xlabel="Threshold τ (raw logit units)", ylabel="Entire test set (%)",
           title="Final linear layer threshold sweep", xlim=(0, tau[-1]), ylim=(0, 100))
    for value, color, offset in ((accuracy, "0.35", 0), (actual_drop, "tab:red", 0)):
        ax.annotate(f"{value:.2f}%", xy=(0, value), xycoords=("axes fraction", "data"),
                    xytext=(-12, offset), textcoords="offset points", ha="right", va="center",
                    color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.8), annotation_clip=False)
    ax.annotate(f"τ* = {tau_star:.6g}", xy=(tau_star, 0), xytext=(0, -28),
                textcoords="offset points", ha="center", color="tab:red", annotation_clip=False)
    ax.set_yticks([v for v in ax.get_yticks() if abs(v-accuracy) > 3 and abs(v-actual_drop) > 3 and 0 <= v <= 100])
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.labelpad = 34
    ax.yaxis.labelpad = 60
    fig.subplots_adjust(left=0.22, bottom=0.22, top=0.90, right=0.98)
    fig.savefig(output_dir / "threshold_sweep.pdf")
    fig.savefig(output_dir / f"threshold_sweep_drop{drop_pp:g}pp.pdf")
    plt.close(fig)
    print("Threshold marker:", marker, flush=True)


class LinearStudy:
    def __init__(self, output_dir, batch_index, drop_pp=1.0):
        self.output_dir = output_dir
        self.batch_index = batch_index
        self.logits = []
        self.labels = []
        self.seen = 0
        self.saved = False
        self.drop_pp = drop_pp

    def start(self, model, args, trial_index):
        self.args = args
        self.trial_index = trial_index
        self.handle = model.linear.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.current_logits = output.detach().cpu().numpy().copy()
        if len(self.logits) == self.batch_index:
            self.batch = dict(
                linear_input=inputs[0].detach().cpu().numpy().copy(),
                linear_output=self.current_logits.copy(),
                weight=module.weight.detach().cpu().numpy().copy(),
                bias=(module.bias.detach().cpu().numpy().copy() * getattr(module, "physical_bias_scale", 1.0)
                      if module.bias is not None else np.zeros(output.shape[-1], dtype=self.current_logits.dtype)),
                bias_scale=np.asarray(getattr(module, "physical_bias_scale", 1.0)),
            )

    def record(self, batch_idx, targets, output_tensor):
        labels = targets.detach().cpu().numpy().copy()
        np.testing.assert_array_equal(
            self.current_logits, output_tensor.detach().cpu().numpy())
        if batch_idx == self.batch_index:
            reconstructed = self.batch["linear_input"] @ self.batch["weight"].T + self.batch["bias"]
            self.reconstruction_error = float(np.max(np.abs(reconstructed - self.batch["linear_output"])))
            np.testing.assert_allclose(reconstructed, self.batch["linear_output"], rtol=1e-4, atol=1e-5)
            self.batch.update(labels=labels, sample_indices=np.arange(self.seen, self.seen + len(labels)))
            np.savez_compressed(self.output_dir / "linear_batch.npz", **self.batch)
            self.saved = True
        self.logits.append(self.current_logits)
        self.labels.append(labels)
        self.seen += len(labels)

    def finish(self, accuracy):
        self.handle.remove()
        if not self.saved:
            raise ValueError("Requested batch_index is outside the test loader")
        logits = np.concatenate(self.logits)
        labels = np.concatenate(self.labels)
        correct = logits.argmax(axis=1) == labels
        other = logits.copy()
        other[np.arange(len(labels)), labels] = -np.inf
        margin = logits[np.arange(len(labels)), labels] - other.max(axis=1)
        np.savez_compressed(self.output_dir / "margin_data.npz", margin=margin, correct=correct,
                            mean_absolute_logit=np.abs(logits).mean())
        rows = [
            ("Test accuracy (%)", accuracy),
            ("Mean absolute raw logit", np.abs(logits).mean()),
            ("Mean signed margin (all samples)", margin.mean()),
            ("Mean margin (correct predictions)", margin[correct].mean() if correct.any() else float("nan")),
            ("Mean margin (incorrect predictions)", margin[~correct].mean() if (~correct).any() else float("nan")),
        ]
        summary = ["# Final linear layer study", "",
                   f"Corner: {self.args.ablation_case_name}; trial: {self.trial_index}; samples: {len(labels)}.", "",
                   "Raw logits; no softmax or centering. Margin = true-class logit minus highest incorrect-class logit.", "",
                   "| Statistic | Value |", "|---|---:|"]
        summary += [f"| {name} | {value:.6f} |" for name, value in rows]
        summary += ["", f"Saved batch: {self.batch_index} ({len(self.batch['labels'])} samples).",
                    f"Maximum absolute NumPy linear reconstruction error: {self.reconstruction_error:.8g}.", "",
                    "Both threshold curves use the entire test set as denominator: correct predictions with margin < threshold, and all predictions with absolute margin < threshold. The marker denotes potential accuracy loss if every vulnerable correct prediction flips, not an expected noise-induced loss.",
                    "Exact ties use argmax for correctness. See linear_batch.npz, signed_margin_histogram.pdf and threshold_sweep.pdf."]
        (self.output_dir / "summary.md").write_text("\n".join(summary) + "\n")
        metadata = dict(config=vars(self.args), trial_index=self.trial_index,
                        batch_index=self.batch_index, samples=len(labels),
                        reconstruction_max_abs_error=self.reconstruction_error)
        (self.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

        plot_saved_data(self.output_dir, self.drop_pp)
        print("\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_log", type=Path,
                        help="MC45 corner log containing the actual COMMAND: ode_inference.py invocation")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--drop_pp", type=float, default=1.0, help="Potential accuracy drop in percentage points")
    parser.add_argument("--plot_only", action="store_true", help="Regenerate PDFs from saved margins without inference")
    args = parser.parse_args()
    if args.plot_only:
        plot_saved_data(args.output_dir, args.drop_pp)
        return
    command_line = next(line for line in args.reference_log.open() if line.startswith("COMMAND: "))
    command = shlex.split(command_line[len("COMMAND: "):])
    inference_args = command[command.index("ode_inference.py") + 1:]
    inference_args[inference_args.index("--noisy_trials") + 1] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_command.txt").write_text(shlex.join(command[:command.index("ode_inference.py") + 1] + inference_args) + "\n")
    import ode_inference
    sys.argv = ["ode_inference.py"] + inference_args
    ode_inference.run_ode_inference(linear_study=LinearStudy(args.output_dir, args.batch_index, args.drop_pp))


if __name__ == "__main__":
    main()
