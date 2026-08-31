#!/usr/bin/env python3
"""Evaluate trained BN-free WRN counterparts under fixed weight mismatch."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.run_baseline import (  # noqa: E402
    FixedMismatchHelper, build_model, build_test_loader, combined_hash,
    convolution_bias_parameter_names, evaluate_once, get_baseline_config,
    infer_num_classes, load_model_weights,
)
from baseline.run_wrn_bn_recalibration_experiment import mismatch_seed  # noqa: E402

ARCHITECTURES = ("WRN_16_2", "WRN_16_4", "WRN_28_2", "WRN_28_4")
LEVELS = {
    "multiplicative": (0, .05, .10, .15, .20, .25, .30, .35, .40),
    "additive": (0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="logs/wrn_nobn_mismatch")
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--checkpoint_root", default="checkpoint/baselines_nobn")
    p.add_argument("--checkpoint_override", default="")
    p.add_argument("--model_name_override", default="")
    p.add_argument("--datasets", default="cifar10,cifar100")
    p.add_argument("--architectures", default=",".join(ARCHITECTURES))
    p.add_argument("--mismatch_types", default="additive,multiplicative")
    p.add_argument("--additive_scale_mode", choices=["max_abs", "rms"], default="max_abs")
    p.add_argument("--noise_levels", default="default")
    p.add_argument("--noisy_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--model_index_offset", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--case", default="custom_noresize")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--max_eval_batches", type=int, default=None)
    return p.parse_args()


def split_csv(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def model_name(architecture):
    return f"wrn_{architecture.removeprefix('WRN_').lower()}_cifar_nobn"


def checkpoint_path(root, dataset, case, name):
    run = f"{case}_{dataset}_{name}"
    return root / dataset / case / name / run / f"{run}_best_ckpt.pth"


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["architecture"], row["mismatch_type"], row["mismatch_level"])
        groups[key].append(row)
    output = []
    for (dataset, architecture, kind, level), items in sorted(groups.items()):
        acc = np.asarray([item["accuracy"] for item in items])
        output.append({
            "dataset": dataset, "architecture": architecture,
            "model_name": items[0]["model_name"], "mismatch_type": kind,
            "mismatch_level": level, "num_trials": len(items),
            "additive_scale_mode": items[0]["additive_scale_mode"],
            "wrn_nobn_mean_accuracy": float(acc.mean()),
            "wrn_nobn_std_accuracy": float(acc.std()),
            "parameter_count": items[0]["parameter_count"],
            "checkpoint": items[0]["checkpoint"],
            "mismatch_parameter_policy": "all_non_norm_params_except_conv_bias",
            "excluded_conv_bias_count": items[0]["excluded_conv_bias_count"],
        })
    return output


def validate_model(model):
    bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if bn:
        raise AssertionError(f"BN-free WRN contains {len(bn)} BatchNorm modules")
    excluded = convolution_bias_parameter_names(model)
    if not excluded:
        raise AssertionError("BN-free WRN has no convolution biases to protect")
    return excluded


def run(args):
    datasets, architectures = split_csv(args.datasets), split_csv(args.architectures)
    kinds = split_csv(args.mismatch_types)
    if set(architectures) - set(ARCHITECTURES):
        raise ValueError("Unsupported architecture selection")
    if set(kinds) - set(LEVELS):
        raise ValueError("Unsupported mismatch type selection")
    if (args.checkpoint_override or args.model_name_override) and (
        len(datasets) != 1 or len(architectures) != 1
    ):
        raise ValueError("Checkpoint/model overrides require exactly one dataset and architecture")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows, model_index = [], args.model_index_offset

    for dataset in datasets:
        for architecture in architectures:
            name = args.model_name_override or model_name(architecture)
            cfg = get_baseline_config(name, False, args.case, False, None)
            model = build_model(name, cfg, infer_num_classes(dataset, None)).to(device)
            checkpoint = (
                Path(args.checkpoint_override)
                if args.checkpoint_override
                else checkpoint_path(Path(args.checkpoint_root), dataset, args.case, name)
            )
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            load_model_weights(model, str(checkpoint), device)
            excluded = validate_model(model)
            parameter_count = sum(p.numel() for p in model.parameters())
            loader_args = argparse.Namespace(
                dataset=dataset, data_dir=args.data_dir, batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=args.pin_memory,
            )
            loader = build_test_loader(model, loader_args, cfg)

            for kind in kinds:
                levels = LEVELS[kind] if args.noise_levels == "default" else tuple(map(float, split_csv(args.noise_levels)))
                helper = FixedMismatchHelper(
                    model, 0, kind, False, True, exclude_param_names=excluded,
                    additive_scale_mode=args.additive_scale_mode,
                )
                helper.snapshot_clean_state()
                for level in levels:
                    for trial in range(args.noisy_trials):
                        helper.restore_clean_state()
                        helper.noise_sigma = level
                        helper.seed = mismatch_seed(args.seed, model_index, level, trial)
                        if level > 0:
                            summary = helper.add_noise()
                            applied = {r.name for r in summary.applied_records}
                            skipped = {r.name for r in summary.skipped_records}
                            if applied & excluded or not excluded <= skipped:
                                raise AssertionError("Convolution bias mismatch exclusion failed")
                        noisy_hash = combined_hash(model.named_parameters())
                        accuracy = evaluate_once(
                            model, loader, device, args.use_amp, args.max_eval_batches
                        )
                        if combined_hash(model.named_parameters()) != noisy_hash:
                            raise AssertionError("Evaluation changed fixed noisy parameters")
                        rows.append({
                            "dataset": dataset, "architecture": architecture,
                            "model_name": name, "checkpoint": str(checkpoint),
                            "mismatch_type": kind, "mismatch_level": level,
                            "additive_scale_mode": args.additive_scale_mode,
                            "mismatch_seed": helper.seed, "trial": trial,
                            "accuracy": accuracy, "parameter_count": parameter_count,
                            "excluded_conv_bias_count": len(excluded),
                            "excluded_conv_bias_names": ";".join(sorted(excluded)),
                            "fixed_noisy_parameter_hash": noisy_hash,
                        })
                        print(f"{dataset} {architecture} {kind} level={level:g} trial={trial} acc={accuracy:.2f}")
                helper.restore_clean_state()
            model_index += 1

    output_dir = Path(args.output_dir)
    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "full_per_trial.csv", rows)
    write_csv(output_dir / "full_aggregate.csv", aggregate)
    summary = {
        "datasets": datasets, "architectures": architectures,
        "mismatch_types": kinds, "noisy_trials": args.noisy_trials,
        "additive_scale_mode": args.additive_scale_mode,
        "mismatch_parameter_policy": "all_non_norm_params_except_conv_bias",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return rows, aggregate


if __name__ == "__main__":
    run(parse_args())
