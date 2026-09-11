#!/usr/bin/env python3
"""Paired WRN frozen-BN vs post-mismatch BN-stat recalibration experiment.

This script intentionally leaves the existing WRN/PCN comparison CSVs untouched.
It reuses baseline.run_baseline model loading, mismatch, deterministic transforms,
and BN-stat recalibration utilities.
"""
import argparse
import copy
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mismatch_utils import ADDITIVE_SCALE_MODES, apply_wrn_ff_gain  # noqa: E402

from baseline.run_baseline import (  # noqa: E402
    BNRecalibrationConfig,
    FixedMismatchHelper,
    build_bn_calibration_loader,
    build_model,
    build_test_loader,
    combined_hash,
    conv_linear_classifier_tensors,
    evaluate_once,
    get_baseline_config,
    infer_num_classes,
    load_model_weights,
    maybe_fold_norms,
    parse_checkpoint_map,
    recalibrate_batchnorm_statistics,
)

COMPARE_RE = re.compile(r"wrn_compare_(cifar10|cifar100)_(additive|multiplicative)_WRN_(\d+_\d+)\.csv$")


def parse_args():
    p = argparse.ArgumentParser(description="Run paired WRN BN recalibration mismatch experiment.")
    p.add_argument("--compare_dir", default="logs/wrn_like_compare_target_models_by_arch")
    p.add_argument("--output_dir", default="logs/wrn_bn_recalibration")
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--checkpoint_root", default="checkpoint/baselines")
    p.add_argument("--checkpoint_override", default="")
    p.add_argument("--model_name_override", default="")
    p.add_argument("--mode", choices=["preflight", "full"], default="preflight")
    p.add_argument("--datasets", default="all", help="Comma list or all.")
    p.add_argument("--architectures", default="all", help="Comma list like WRN_16_2,WRN_28_4 or all.")
    p.add_argument("--mismatch_types", default="all", help="Comma list of additive,multiplicative or all.")
    p.add_argument("--additive_scale_mode", choices=ADDITIVE_SCALE_MODES, default="max_abs")
    p.add_argument("--ff_gain", type=float, default=1.0)
    p.add_argument(
        "--pcn_reference_policy",
        choices=["condition_matched_legacy", "none"],
        default="condition_matched_legacy",
        help=(
            "Controls optional PCN comparison fields only. Legacy references are "
            "used for multiplicative and max-absolute additive runs, never RMS additive runs."
        ),
    )
    p.add_argument("--noise_levels", default="csv", help="csv for all levels from files, or comma list.")
    p.add_argument("--standalone", action="store_true",
                   help="Use explicit dataset/architecture/mismatch/levels without historical comparison CSVs.")
    p.add_argument("--noisy_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--max_eval_batches", type=lambda s: None if str(s).lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--calibration_num_samples", default="5000", help="Integer or all.")
    p.add_argument("--calibration_batch_size", type=int, default=128)
    p.add_argument("--calibration_subset_seed", type=int, default=20240618)
    p.add_argument("--calibration_num_workers", type=int, default=4)
    p.add_argument("--case", default="custom_noresize")
    p.add_argument("--preflight_architecture", default="WRN_16_2")
    p.add_argument("--preflight_dataset", default="cifar10")
    p.add_argument("--preflight_additive_level", type=float, default=0.01)
    p.add_argument("--preflight_multiplicative_level", type=float, default=0.05)
    p.add_argument("--preflight_trials", type=int, default=2)
    p.add_argument("--preflight_repro_tolerance", type=float, default=15.0, help="Percentage points vs old frozen-BN mean.")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument(
        "--mismatch_parameter_policy",
        choices=["existing", "bn_fold_no_mismatch_bias"],
        default="existing",
        help=(
            "WRN parameter treatment before mismatch. 'existing' preserves the original evaluator. "
            "'bn_fold_no_mismatch_bias' folds supported WRN BatchNorms and excludes only induced folded Conv bias params from mismatch."
        ),
    )
    p.add_argument(
        "--include_all_bn_recal_before_fold_column",
        action="store_true",
        help=(
            "Also evaluate a companion WRN column with the same mismatch realization applied to the original unfused model, "
            "all original BatchNorm statistics recalibrated, then supported WRN BN pairs folded before test evaluation."
        ),
    )
    return p.parse_args()


def split_filter(raw: str, values: Iterable[str]) -> List[str]:
    values = sorted(set(values))
    if raw == "all":
        return values
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    return [x for x in values if x in wanted]


def wrn_label_to_model_name(label: str) -> str:
    return "wrn_" + label.replace("WRN_", "").lower() + "_cifar"


def model_name_to_checkpoint(root: Path, dataset: str, case: str, model_name: str) -> Path:
    run_name = f"{case}_{dataset}_{model_name}"
    return root / dataset / case / model_name / run_name / f"{run_name}_best_ckpt.pth"


def read_compare_csv(path: Path) -> Dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    levels = [float(row["noise_level"]) for row in rows]
    by_level = {}
    for row in rows:
        level = float(row["noise_level"])
        pcn_entries = []
        wrn_frozen_entries = []
        for key, val in row.items():
            if key == "noise_level" or not val:
                continue
            mean, std = parse_acc(val)
            item = {"column": key, "mean": mean, "std": std}
            if key.startswith("WRN-"):
                wrn_frozen_entries.append(item)
            else:
                pcn_entries.append(item)
        selected_pcn_entries = [
            item for item in pcn_entries if "PCNetNoBatchNorm" in item["column"]
        ]
        if len(selected_pcn_entries) != 1:
            raise ValueError(
                f"Expected exactly one PCNetNoBatchNorm column in {path} at "
                f"noise level {level}, found {len(selected_pcn_entries)}: "
                f"{[item['column'] for item in selected_pcn_entries]}"
            )
        # Keep the legacy key for output compatibility. This is a fixed model
        # selection, not a per-level maximum over different PCN variants.
        best_pcn = selected_pcn_entries[0]
        by_level[level] = {
            "pcn_entries": pcn_entries,
            "best_pcn": best_pcn,
            "old_wrn_entries": wrn_frozen_entries,
        }
    return {"levels": levels, "by_level": by_level}


def parse_acc(raw: str) -> tuple[float, float]:
    raw = raw.strip().replace("%", "")
    if "±" in raw:
        mean, std = raw.split("±", 1)
        return float(mean), float(std)
    return float(raw), 0.0


def discover_compare_specs(compare_dir: Path) -> List[Dict]:
    specs = []
    for path in sorted(compare_dir.glob("wrn_compare_*.csv")):
        m = COMPARE_RE.match(path.name)
        if not m:
            continue
        dataset, mismatch_type, arch_tail = m.groups()
        arch = f"WRN_{arch_tail}"
        specs.append({
            "dataset": dataset,
            "mismatch_type": mismatch_type,
            "architecture": arch,
            "path": path,
            "compare": read_compare_csv(path),
        })
    return specs


def select_specs(args, specs: List[Dict]) -> List[Dict]:
    datasets = split_filter(args.datasets, [s["dataset"] for s in specs])
    archs = split_filter(args.architectures, [s["architecture"] for s in specs])
    mismatch_types = split_filter(args.mismatch_types, [s["mismatch_type"] for s in specs])
    out = [s for s in specs if s["dataset"] in datasets and s["architecture"] in archs and s["mismatch_type"] in mismatch_types]
    if args.mode == "preflight":
        wanted = {
            (args.preflight_dataset, args.preflight_architecture, "additive"),
            (args.preflight_dataset, args.preflight_architecture, "multiplicative"),
        }
        out = [s for s in out if (s["dataset"], s["architecture"], s["mismatch_type"]) in wanted]
    return out


def standalone_specs(args):
    if args.mode != 'full' or args.pcn_reference_policy != 'none' or args.noise_levels == 'csv':
        raise ValueError('Standalone evaluation requires full mode, explicit levels, and PCN references disabled')
    if args.datasets not in {'cifar10', 'cifar100'} or args.architectures not in {
        'WRN_16_2', 'WRN_16_4', 'WRN_28_2', 'WRN_28_4'
    } or args.mismatch_types not in {'additive', 'multiplicative'}:
        raise ValueError('Standalone evaluation requires one explicit CIFAR dataset, WRN size, and mismatch type')
    return [dict(dataset=args.datasets, architecture=args.architectures,
                 mismatch_type=args.mismatch_types, compare=dict(levels=[], by_level={}))]


def levels_for_spec(args, spec: Dict) -> List[float]:
    if args.mode == "preflight":
        return [args.preflight_additive_level] if spec["mismatch_type"] == "additive" else [args.preflight_multiplicative_level]
    if args.noise_levels == "csv":
        return list(spec["compare"]["levels"])
    wanted = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    return wanted


def mismatch_seed(base_seed: int, model_idx: int, noise_level: float, trial: int) -> int:
    return base_seed + 100000 * model_idx + 1000 * int(round(noise_level * 1e6)) + trial


def evaluate_with_predictions(model, loader, device, use_amp=False, max_batches=None):
    model.eval()
    total = 0
    correct = 0
    preds_all = []
    targets_all = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp and device.type in ["cuda", "cpu"]):
                outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())
    acc = 100.0 * correct / max(total, 1)
    preds = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long)
    targets = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.long)
    reconstructed = 100.0 * preds.eq(targets).sum().item() / max(int(targets.numel()), 1)
    if not np.isclose(acc, reconstructed):
        raise AssertionError(f"Prediction-reconstructed accuracy mismatch: eval={acc}, pred={reconstructed}")
    return acc, preds, targets


def save_predictions(path: Path, preds: torch.Tensor, targets: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"preds": preds, "targets": targets}, path)


def run_clean_sanity(model, calibration_loader, test_loader, device, recal_cfg, args, metadata: Dict) -> Dict:
    original_acc = evaluate_once(model, test_loader, device=device, use_amp=args.use_amp, max_batches=args.max_eval_batches)
    t0 = time.time()
    report = recalibrate_batchnorm_statistics(model, calibration_loader, device, recal_cfg, metadata["model_name"], 0.0, -1)
    recal_time = time.time() - t0
    recal_acc = evaluate_once(model, test_loader, device=device, use_amp=args.use_amp, max_batches=args.max_eval_batches)
    return {
        **metadata,
        "clean_original_bn_accuracy": original_acc,
        "clean_recalibrated_bn_accuracy": recal_acc,
        "clean_recalibration_delta": recal_acc - original_acc,
        "calibration_runtime_sec": recal_time,
        "calibration_sample_count": report.num_calibration_samples,
    }


def build_loaded_wrn_model(args, dataset: str, model_name: str, checkpoint: Path, device: torch.device):
    num_classes = infer_num_classes(dataset, None)
    cfg = get_baseline_config(model_name=model_name, pretrained=False, case=args.case, prefer_resize=False, extra_overrides=None)
    model = build_model(model_name, cfg, num_classes=num_classes).to(device)
    load_model_weights(model, str(checkpoint), device)
    gain_records = apply_wrn_ff_gain(model, args.ff_gain)
    print(
        f"Applied WRN FF gain {args.ff_gain:g} to {len(gain_records)} tensors "
        f"({sum(record[2] for record in gain_records)} elements)."
    )
    return model, cfg


def evaluate_all_bn_recal_before_fold_trial(
    *,
    source_model: nn.Module,
    calibration_loader,
    test_loader,
    device: torch.device,
    recal_cfg: BNRecalibrationConfig,
    args,
    model_name: str,
    mismatch_type: str,
    noise_level: float,
    mismatch_seed_value: int,
    trial: int,
) -> Dict[str, object]:
    model = copy.deepcopy(source_model).to(device)
    helper = FixedMismatchHelper(
        model,
        noise_sigma=noise_level,
        noise_type=mismatch_type,
        noise_to_norm=False,
        include_buffers=True,
        additive_scale_mode=args.additive_scale_mode,
    )
    helper.seed = mismatch_seed_value
    helper.snapshot_clean_state()
    if noise_level > 0:
        helper.add_noise()

    noisy_hash_before_recal = combined_hash(conv_linear_classifier_tensors(model))
    report = recalibrate_batchnorm_statistics(
        model=model,
        calibration_loader=calibration_loader,
        device=device,
        recal_cfg=recal_cfg,
        model_arch=f"{model_name}_all_bn_before_fold",
        noise_level=noise_level,
        trial=trial,
    )
    noisy_hash_after_recal = combined_hash(conv_linear_classifier_tensors(model))
    if noisy_hash_before_recal != noisy_hash_after_recal:
        raise AssertionError("Noisy weight hash changed during all-BN recalibration before folding.")

    model = maybe_fold_norms(model, mode="wrn_preact_no_mismatch_bias").to(device)
    acc, _, _ = evaluate_with_predictions(
        model, test_loader, device, use_amp=args.use_amp, max_batches=args.max_eval_batches
    )
    return {
        "accuracy": acc,
        "changed_bn_buffer_count": len(report.changed_bn_buffers),
        "calibration_sample_count": report.num_calibration_samples,
        "pre_fold_noisy_weight_hash": noisy_hash_before_recal,
        "pre_fold_noisy_weight_hash_unchanged": noisy_hash_before_recal == noisy_hash_after_recal,
        "folded_excluded_mismatch_params": ";".join(sorted(getattr(model, "_mismatch_excluded_param_names", set()))),
    }


def run_one_spec(args, spec: Dict, rows: List[Dict], clean_rows: List[Dict], preflight_checks: List[Dict]):
    dataset = spec["dataset"]
    mismatch_type = spec["mismatch_type"]
    architecture = spec["architecture"]
    model_name = args.model_name_override or wrn_label_to_model_name(architecture)
    checkpoint = (
        Path(args.checkpoint_override)
        if args.checkpoint_override
        else model_name_to_checkpoint(Path(args.checkpoint_root), dataset, args.case, model_name)
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg = build_loaded_wrn_model(args, dataset, model_name, checkpoint, device)
    all_bn_source_model = None
    if args.mismatch_parameter_policy == "bn_fold_no_mismatch_bias":
        if args.include_all_bn_recal_before_fold_column:
            all_bn_source_model, _ = build_loaded_wrn_model(args, dataset, model_name, checkpoint, device)
        model = maybe_fold_norms(model, mode="wrn_preact_no_mismatch_bias").to(device)
    elif args.include_all_bn_recal_before_fold_column:
        raise ValueError("--include_all_bn_recal_before_fold_column requires --mismatch_parameter_policy bn_fold_no_mismatch_bias.")

    loader_args = argparse.Namespace(
        dataset=dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = build_test_loader(model, loader_args, cfg)
    recal_cfg = BNRecalibrationConfig(
        enabled=True,
        num_samples=str(args.calibration_num_samples),
        batch_size=args.calibration_batch_size,
        subset_seed=args.calibration_subset_seed,
        num_workers=args.calibration_num_workers,
        diagnostics_dir=str(Path(args.output_dir) / "diagnostics" / dataset / architecture / mismatch_type),
    )
    calibration_loader = build_bn_calibration_loader(model, loader_args, cfg, recal_cfg)
    all_bn_before_fold_recal_cfg = BNRecalibrationConfig(
        enabled=True,
        num_samples=str(args.calibration_num_samples),
        batch_size=args.calibration_batch_size,
        subset_seed=args.calibration_subset_seed,
        num_workers=args.calibration_num_workers,
        diagnostics_dir=str(Path(args.output_dir) / "diagnostics_all_bn_before_fold" / dataset / architecture / mismatch_type),
    )

    helper = FixedMismatchHelper(
        model,
        noise_sigma=0.0,
        noise_type=mismatch_type,
        noise_to_norm=False,
        include_buffers=True,
        exclude_param_names=getattr(model, "_mismatch_excluded_param_names", set()),
        additive_scale_mode=args.additive_scale_mode,
    )
    helper.snapshot_clean_state()

    clean_metadata = {
        "dataset": dataset,
        "architecture": architecture,
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "mismatch_parameter_policy": args.mismatch_parameter_policy,
        "ff_gain": args.ff_gain,
        "excluded_mismatch_params": ";".join(sorted(getattr(model, "_mismatch_excluded_param_names", set()))),
    }
    helper.restore_clean_state()
    clean_rows.append(run_clean_sanity(model, calibration_loader, test_loader, device, recal_cfg, args, clean_metadata))

    levels = levels_for_spec(args, spec)
    trials = args.preflight_trials if args.mode == "preflight" else args.noisy_trials
    if args.mode == "preflight":
        print(f"Preflight {dataset} {architecture} {mismatch_type}: levels={levels}, trials={trials}")
    else:
        print(f"Full {dataset} {architecture} {mismatch_type}: {len(levels)} levels, trials={trials}")

    for noise_level in levels:
        old_info = spec["compare"]["by_level"].get(float(noise_level), {})
        legacy_condition_matches = (
            mismatch_type == "multiplicative" or args.additive_scale_mode == "max_abs"
        )
        old_wrn = old_info.get("old_wrn_entries", []) if legacy_condition_matches else []
        old_frozen_mean = old_wrn[0]["mean"] if old_wrn else None
        best_pcn = (
            old_info.get("best_pcn")
            if legacy_condition_matches and args.pcn_reference_policy == "condition_matched_legacy"
            else None
        )
        for trial in range(trials):
            helper.restore_clean_state()
            helper.noise_sigma = noise_level
            helper.seed = mismatch_seed(args.seed, 0, noise_level, trial)
            if noise_level > 0:
                helper.add_noise()

            noisy_hash_before = combined_hash(conv_linear_classifier_tensors(model))
            frozen_acc, frozen_preds, targets = evaluate_with_predictions(
                model, test_loader, device, use_amp=args.use_amp, max_batches=args.max_eval_batches
            )
            noisy_hash_after_frozen = combined_hash(conv_linear_classifier_tensors(model))
            if noisy_hash_before != noisy_hash_after_frozen:
                raise AssertionError("Noisy weight hash changed during frozen-BN evaluation.")

            t0 = time.time()
            report = recalibrate_batchnorm_statistics(
                model=model,
                calibration_loader=calibration_loader,
                device=device,
                recal_cfg=recal_cfg,
                model_arch=model_name,
                noise_level=noise_level,
                trial=trial,
            )
            runtime = time.time() - t0
            noisy_hash_after_recal = combined_hash(conv_linear_classifier_tensors(model))
            if noisy_hash_before != noisy_hash_after_recal:
                raise AssertionError("Noisy weight hash changed during BN recalibration.")

            recal_acc, recal_preds, recal_targets = evaluate_with_predictions(
                model, test_loader, device, use_amp=args.use_amp, max_batches=args.max_eval_batches
            )
            if not torch.equal(targets, recal_targets):
                raise AssertionError("Frozen/recalibrated evaluations used different target ordering.")

            all_bn_before_fold = None
            if all_bn_source_model is not None:
                all_bn_before_fold = evaluate_all_bn_recal_before_fold_trial(
                    source_model=all_bn_source_model,
                    calibration_loader=calibration_loader,
                    test_loader=test_loader,
                    device=device,
                    recal_cfg=all_bn_before_fold_recal_cfg,
                    args=args,
                    model_name=model_name,
                    mismatch_type=mismatch_type,
                    noise_level=noise_level,
                    mismatch_seed_value=helper.seed,
                    trial=trial,
                )

            pred_path = ""
            if args.save_predictions or args.mode == "preflight":
                pred_path_obj = Path(args.output_dir) / "predictions" / dataset / architecture / mismatch_type / f"level_{noise_level:g}_trial_{trial}.pt"
                save_predictions(pred_path_obj, frozen_preds, targets)
                save_predictions(pred_path_obj.with_name(pred_path_obj.stem + "_recal.pt"), recal_preds, recal_targets)
                pred_path = str(pred_path_obj)

            row = {
                "dataset": dataset,
                "architecture": architecture,
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "mismatch_type": mismatch_type,
                "additive_scale_mode": args.additive_scale_mode,
                "ff_gain": args.ff_gain,
                "mismatch_level": noise_level,
                "mismatch_seed": helper.seed,
                "trial": trial,
                "calibration_sample_count": report.num_calibration_samples,
                "calibration_subset_seed": args.calibration_subset_seed,
                "calibration_batch_size": args.calibration_batch_size,
                "frozen_bn_accuracy": frozen_acc,
                "recalibrated_bn_accuracy": recal_acc,
                "paired_recovery": recal_acc - frozen_acc,
                "noisy_weight_hash": noisy_hash_before,
                "calibration_runtime_sec": runtime,
                "changed_bn_buffer_count": len(report.changed_bn_buffers),
                "bn_affine_hash": report.bn_affine_hash,
                "fixed_noisy_param_hash": report.fixed_noisy_param_hash,
                "old_wrn_frozen_mean_accuracy": old_frozen_mean,
                "pcn_node_best_column": best_pcn["column"] if best_pcn else "",
                "pcn_node_best_mean_accuracy": best_pcn["mean"] if best_pcn else "",
                "pcn_node_best_std_accuracy": best_pcn["std"] if best_pcn else "",
                "prediction_path": pred_path,
                "mismatch_parameter_policy": args.mismatch_parameter_policy,
                "excluded_mismatch_params": ";".join(sorted(getattr(model, "_mismatch_excluded_param_names", set()))),
                "old_seed_logic_reconstructed": True,
            }
            if all_bn_before_fold is not None:
                row.update(
                    {
                        "all_bn_recal_before_fold_accuracy": all_bn_before_fold["accuracy"],
                        "all_bn_recal_before_fold_recovery": all_bn_before_fold["accuracy"] - frozen_acc,
                        "all_bn_recal_before_fold_changed_bn_buffer_count": all_bn_before_fold["changed_bn_buffer_count"],
                        "all_bn_recal_before_fold_calibration_sample_count": all_bn_before_fold["calibration_sample_count"],
                        "all_bn_recal_before_fold_pre_fold_noisy_weight_hash": all_bn_before_fold["pre_fold_noisy_weight_hash"],
                        "all_bn_recal_before_fold_pre_fold_noisy_weight_hash_unchanged": all_bn_before_fold["pre_fold_noisy_weight_hash_unchanged"],
                        "all_bn_recal_before_fold_excluded_mismatch_params_after_fold": all_bn_before_fold["folded_excluded_mismatch_params"],
                    }
                )
            rows.append(row)

            if args.mode == "preflight" and old_frozen_mean is not None:
                diff = abs(frozen_acc - old_frozen_mean)
                preflight_checks.append({
                    "dataset": dataset,
                    "architecture": architecture,
                    "mismatch_type": mismatch_type,
                    "mismatch_level": noise_level,
                    "trial": trial,
                    "new_frozen_accuracy": frozen_acc,
                    "old_frozen_mean_accuracy": old_frozen_mean,
                    "abs_diff_pct_points": diff,
                    "passed_tolerance": diff <= args.preflight_repro_tolerance,
                    "noisy_weight_hash_unchanged": noisy_hash_before == noisy_hash_after_recal,
                    "prediction_accuracy_reconstructed": True,
                    "changed_bn_buffer_count": len(report.changed_bn_buffers),
                })


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["architecture"],
            row["mismatch_type"],
            float(row["mismatch_level"]),
            float(row.get("ff_gain", 1.0)),
        )
        groups[key].append(row)
    out = []
    for (dataset, arch, mismatch_type, level, ff_gain), items in sorted(groups.items()):
        frozen = np.array([float(x["frozen_bn_accuracy"]) for x in items])
        recal = np.array([float(x["recalibrated_bn_accuracy"]) for x in items])
        all_bn_before_fold_vals = [
            float(x["all_bn_recal_before_fold_accuracy"])
            for x in items
            if x.get("all_bn_recal_before_fold_accuracy", "") != ""
        ]
        all_bn_before_fold = np.array(all_bn_before_fold_vals) if all_bn_before_fold_vals else None
        recovery = recal - frozen
        pcn_vals = [x for x in items if x["pcn_node_best_mean_accuracy"] != ""]
        pcn_mean = float(pcn_vals[0]["pcn_node_best_mean_accuracy"]) if pcn_vals else None
        pcn_std = float(pcn_vals[0]["pcn_node_best_std_accuracy"]) if pcn_vals else None
        row = {
            "dataset": dataset,
            "architecture": arch,
            "mismatch_type": mismatch_type,
            "mismatch_level": level,
            "additive_scale_mode": items[0].get("additive_scale_mode", "max_abs"),
            "ff_gain": ff_gain,
            "num_trials": len(items),
            "pcn_node_column": pcn_vals[0]["pcn_node_best_column"] if pcn_vals else "",
            "pcn_node_mean_accuracy": pcn_mean if pcn_mean is not None else "",
            "pcn_node_std_accuracy": pcn_std if pcn_std is not None else "",
            "paired_wrn_frozen_bn_mean_accuracy": float(frozen.mean()),
            "paired_wrn_frozen_bn_std_accuracy": float(frozen.std()),
            "wrn_recalibrated_bn_mean_accuracy": float(recal.mean()),
            "wrn_recalibrated_bn_std_accuracy": float(recal.std()),
            "mean_paired_bn_recovery": float(recovery.mean()),
            "std_paired_bn_recovery": float(recovery.std()),
            "remaining_pcn_gap": float(pcn_mean - recal.mean()) if pcn_mean is not None else "",
            "mismatch_parameter_policy": items[0].get("mismatch_parameter_policy", "existing"),
        }
        if all_bn_before_fold is not None:
            all_bn_recovery = all_bn_before_fold - frozen
            row.update(
                {
                    "wrn_all_bn_recal_before_fold_mean_accuracy": float(all_bn_before_fold.mean()),
                    "wrn_all_bn_recal_before_fold_std_accuracy": float(all_bn_before_fold.std()),
                    "mean_paired_all_bn_recal_before_fold_recovery": float(all_bn_recovery.mean()),
                    "std_paired_all_bn_recal_before_fold_recovery": float(all_bn_recovery.std()),
                    "remaining_pcn_gap_all_bn_recal_before_fold": (
                        float(pcn_mean - all_bn_before_fold.mean()) if pcn_mean is not None else ""
                    ),
                }
            )
        out.append(row)
    return out


def interpretation_label(aggregate: List[Dict]) -> str:
    valid = [r for r in aggregate if r["remaining_pcn_gap"] != ""]
    if not valid:
        return "PCN comparison unavailable"
    mean_recovery = float(np.mean([r["mean_paired_bn_recovery"] for r in valid]))
    mean_gap = float(np.mean([r["remaining_pcn_gap"] for r in valid]))
    if mean_recovery < 1.0:
        return "negligible BN recovery"
    if mean_gap <= 0:
        return "recalibrated WRN matches or exceeds PCN"
    if mean_gap < 1.0:
        return "most of the PCN advantage explained by BN-statistic recalibration"
    return "partial BN recovery with substantial PCN advantage remaining"


def make_plots(output_dir: Path, aggregate: List[Dict], rows: List[Dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created = []
    groups = defaultdict(list)
    for row in aggregate:
        groups[(row["dataset"], row["architecture"], row["mismatch_type"])].append(row)
    for key, items in groups.items():
        dataset, arch, mismatch_type = key
        items = sorted(items, key=lambda r: float(r["mismatch_level"]))
        levels = [float(r["mismatch_level"]) for r in items]
        pcn = [float(r["pcn_node_mean_accuracy"]) if r["pcn_node_mean_accuracy"] != "" else np.nan for r in items]
        frozen = [float(r["paired_wrn_frozen_bn_mean_accuracy"]) for r in items]
        recal = [float(r["wrn_recalibrated_bn_mean_accuracy"]) for r in items]
        all_bn_before_fold = [
            float(r["wrn_all_bn_recal_before_fold_mean_accuracy"])
            if r.get("wrn_all_bn_recal_before_fold_mean_accuracy", "") != ""
            else np.nan
            for r in items
        ]
        recovery = [float(r["mean_paired_bn_recovery"]) for r in items]
        gap = [float(r["remaining_pcn_gap"]) if r["remaining_pcn_gap"] != "" else np.nan for r in items]

        base = f"{dataset}_{arch}_{mismatch_type}"
        fig, ax = plt.subplots()
        if not np.all(np.isnan(pcn)):
            ax.plot(levels, pcn, marker="o", label="PCN/NODE")
        ax.plot(levels, frozen, marker="o", label="WRN frozen BN")
        ax.plot(levels, recal, marker="o", label="WRN recalibrated BN")
        if not np.all(np.isnan(all_bn_before_fold)):
            ax.plot(levels, all_bn_before_fold, marker="o", label="WRN all-BN recal before fold")
        ax.set_xlabel("Mismatch level")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        path = plot_dir / f"{base}_accuracy.png"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        created.append(str(path))

        fig, ax = plt.subplots()
        ax.plot(levels, recovery, marker="o")
        ax.set_xlabel("Mismatch level")
        ax.set_ylabel("BN recovery (pp)")
        path = plot_dir / f"{base}_bn_recovery.png"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        created.append(str(path))

        if not np.all(np.isnan(gap)):
            fig, ax = plt.subplots()
            ax.plot(levels, gap, marker="o")
            ax.set_xlabel("Mismatch level")
            ax.set_ylabel("Remaining PCN gap (pp)")
            path = plot_dir / f"{base}_remaining_gap.png"
            fig.savefig(path, bbox_inches="tight", dpi=180)
            plt.close(fig)
            created.append(str(path))

        seed_rows = [r for r in rows if (r["dataset"], r["architecture"], r["mismatch_type"]) == key]
        fig, ax = plt.subplots()
        for level in sorted({float(r["mismatch_level"]) for r in seed_rows}):
            vals = [r for r in seed_rows if float(r["mismatch_level"]) == level]
            xs = [float(r["frozen_bn_accuracy"]) for r in vals]
            ys = [float(r["recalibrated_bn_accuracy"]) for r in vals]
            ax.plot(xs, ys, marker="o", linestyle="none", label=f"{level:g}")
        ax.set_xlabel("Frozen BN accuracy (%)")
        ax.set_ylabel("Recalibrated BN accuracy (%)")
        ax.legend(fontsize="small")
        path = plot_dir / f"{base}_paired_scatter.png"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        created.append(str(path))
    return created


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = standalone_specs(args) if args.standalone else discover_compare_specs(Path(args.compare_dir))
    specs = select_specs(args, specs)
    if not specs:
        raise SystemExit("No comparison CSV specs selected.")
    if (args.checkpoint_override or args.model_name_override) and len(specs) != 1:
        raise ValueError("Checkpoint/model overrides require exactly one selected comparison spec.")

    rows = []
    clean_rows = []
    preflight_checks = []
    for spec in specs:
        run_one_spec(args, spec, rows, clean_rows, preflight_checks)

    per_trial_path = output_dir / f"{args.mode}_per_trial.csv"
    clean_path = output_dir / f"{args.mode}_clean_sanity.csv"
    aggregate = aggregate_rows(rows)
    aggregate_path = output_dir / f"{args.mode}_aggregate.csv"
    write_csv(per_trial_path, rows)
    write_csv(clean_path, clean_rows)
    write_csv(aggregate_path, aggregate)

    plots = []
    if args.mode == "full":
        plots = make_plots(output_dir, aggregate, rows)

    summary = {
        "mode": args.mode,
        "per_trial_csv": str(per_trial_path),
        "aggregate_csv": str(aggregate_path),
        "clean_sanity_csv": str(clean_path),
        "plots": plots,
        "calibration_num_samples": args.calibration_num_samples,
        "calibration_subset_seed": args.calibration_subset_seed,
        "calibration_batch_size": args.calibration_batch_size,
        "old_mismatch_seed_logic_reconstructed": True,
        "mismatch_parameter_policy": args.mismatch_parameter_policy,
        "interpretation_label": interpretation_label(aggregate),
        "preflight_checks": preflight_checks,
    }
    summary_path = output_dir / f"{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if args.mode == "preflight":
        failed = [x for x in preflight_checks if not x["passed_tolerance"] or not x["noisy_weight_hash_unchanged"] or not x["prediction_accuracy_reconstructed"]]
        print(json.dumps(summary, indent=2, sort_keys=True))
        if failed:
            raise SystemExit(f"Preflight failed {len(failed)} checks; not safe to launch full sweep.")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
