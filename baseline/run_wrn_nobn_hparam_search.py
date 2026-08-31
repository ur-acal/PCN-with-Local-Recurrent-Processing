#!/usr/bin/env python3
"""Durable, bounded hyperparameter search for BN-free WRN-16-2 studies."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "logs" / "wrn_nobn_hparam_search"
STUDIES = {
    "projection": "wrn_16_2_cifar_nobn",
    "maxpool_shortcut": "wrn_16_2_cifar_nobn_maxpool_shortcut",
}
AUGMENTATIONS = {
    "wrn_default": {
        "auto_augment": "rand-m9-mstd0.5-inc1", "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0, "label_smoothing": 0.1,
        "re_prob": 0.25, "color_jitter": 0.1,
    },
    "moderate": {
        "auto_augment": "rand-m7-mstd0.5-inc1", "mixup_alpha": 0.1,
        "cutmix_alpha": 0.5, "label_smoothing": 0.05,
        "re_prob": 0.1, "color_jitter": 0.1,
    },
    "mild": {
        "auto_augment": "rand-m5-mstd0.5-inc1", "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0, "label_smoothing": 0.05,
        "re_prob": 0.05, "color_jitter": 0.1,
    },
    "randaugment_only": {
        "auto_augment": "rand-m9-mstd0.5-inc1", "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0, "label_smoothing": 0.1,
        "re_prob": 0.1, "color_jitter": 0.1,
    },
    "mixing_no_randaugment": {
        "auto_augment": None, "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0, "label_smoothing": 0.1,
        "re_prob": 0.25, "color_jitter": 0.1,
    },
    "plain_timm": {
        "auto_augment": None, "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0, "label_smoothing": 0.0,
        "re_prob": 0.0, "color_jitter": 0.0,
    },
}
OPTIMIZERS = {
    "no_clip": {
        "lr": 0.1, "weight_decay": 5e-4, "max_norm": None,
        "bias_lr_multiplier": 1.0, "bias_weight_decay": None,
        "dropout_rate": 0.0, "warmup_epoch": 5,
    },
    "clip1": {
        "lr": 0.1, "weight_decay": 5e-4, "max_norm": 1.0,
        "bias_lr_multiplier": 1.0, "bias_weight_decay": None,
        "dropout_rate": 0.0, "warmup_epoch": 5,
    },
    "low_lr_old_bias": {
        "lr": 0.05, "weight_decay": 1e-4, "max_norm": 1.0,
        "bias_lr_multiplier": 0.2, "bias_weight_decay": 0.0,
        "dropout_rate": 0.0, "warmup_epoch": 0,
    },
    "low_lr": {
        "lr": 0.05, "weight_decay": 5e-4, "max_norm": 1.0,
        "bias_lr_multiplier": 1.0, "bias_weight_decay": 0.0,
        "dropout_rate": 0.0, "warmup_epoch": 5,
    },
    "high_lr": {
        "lr": 0.15, "weight_decay": 5e-4, "max_norm": 1.0,
        "bias_lr_multiplier": 1.0, "bias_weight_decay": None,
        "dropout_rate": 0.0, "warmup_epoch": 5,
    },
    "light_decay_dropout": {
        "lr": 0.1, "weight_decay": 1e-4, "max_norm": 2.0,
        "bias_lr_multiplier": 0.5, "bias_weight_decay": 0.0,
        "dropout_rate": 0.1, "warmup_epoch": 5,
    },
}
BASE_OPTIMIZER = OPTIMIZERS["clip1"]
FIELDNAMES = [
    "trial_id", "study", "phase", "augmentation", "optimizer", "epochs", "seed",
    "status", "attempt", "best_accuracy", "best_epoch", "started_at", "finished_at",
    "duration_seconds", "checkpoint", "log", "error",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_records(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["trial_id"]: row for row in csv.DictReader(handle)}


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    temp.replace(path)


def encode_override(config):
    values = {
        "num_epochs": config["epochs"],
        "eval_every": 10 if config["epochs"] < 300 else 5,
        "skip_eval_epochs": 80 if config["epochs"] < 300 else 70,
        "test_batch_size": 512,
        **config["augmentation_values"],
        **config["optimizer_values"],
    }
    return ",".join(
        f"{key}={'none' if value is None else str(value).lower() if isinstance(value, bool) else value}"
        for key, value in values.items()
    )


def make_trial(study, phase, aug_name, opt_name, epochs, seed, suffix=""):
    trial_id = f"{study}__{phase}__{aug_name}__{opt_name}__s{seed}{suffix}"
    return {
        "trial_id": trial_id,
        "study": study,
        "phase": phase,
        "augmentation": aug_name,
        "optimizer": opt_name,
        "epochs": epochs,
        "seed": seed,
        "augmentation_values": AUGMENTATIONS[aug_name],
        "optimizer_values": BASE_OPTIMIZER if opt_name == "screen_base" else OPTIMIZERS[opt_name],
    }


def successful(records, study, phase):
    return [
        row for row in records.values()
        if row["study"] == study and row["phase"] == phase and row["status"] == "completed"
    ]


def top_names(rows, field, count):
    ordered = sorted(rows, key=lambda row: float(row["best_accuracy"]), reverse=True)
    output = []
    for row in ordered:
        value = row[field]
        if value not in output:
            output.append(value)
        if len(output) == count:
            break
    return output


def build_pending(records):
    pending = []
    for study in STUDIES:
        for aug_name in AUGMENTATIONS:
            trial = make_trial(study, "augmentation", aug_name, "screen_base", 150, 4096)
            if records.get(trial["trial_id"], {}).get("status") != "completed":
                pending.append(trial)
    if pending:
        return pending

    for study in STUDIES:
        best_augs = top_names(successful(records, study, "augmentation"), "augmentation", 2)
        for aug_name in best_augs:
            for opt_name in OPTIMIZERS:
                trial = make_trial(study, "joint", aug_name, opt_name, 150, 4096)
                if records.get(trial["trial_id"], {}).get("status") != "completed":
                    pending.append(trial)
    if pending:
        return pending

    for study in STUDIES:
        candidates = successful(records, study, "augmentation") + successful(records, study, "joint")
        ordered = sorted(candidates, key=lambda row: float(row["best_accuracy"]), reverse=True)
        seen = set()
        finalists = []
        for row in ordered:
            key = (row["augmentation"], row["optimizer"])
            if key in seen:
                continue
            seen.add(key)
            finalists.append(key)
            if len(finalists) == 5:
                break
        for rank, (aug_name, opt_name) in enumerate(finalists, 1):
            trial = make_trial(study, "full", aug_name, opt_name, 300, 4096, f"__r{rank}")
            if records.get(trial["trial_id"], {}).get("status") != "completed":
                pending.append(trial)
    if pending:
        return pending

    for study in STUDIES:
        winner = max(successful(records, study, "full"), key=lambda row: float(row["best_accuracy"]))
        for seed in (4097, 4098):
            trial = make_trial(
                study, "confirmation", winner["augmentation"], winner["optimizer"], 300, seed
            )
            if records.get(trial["trial_id"], {}).get("status") != "completed":
                pending.append(trial)
    return pending


def checkpoint_path(trial_root, model_name, dataset):
    run_name = f"custom_noresize_{dataset}_{model_name}"
    return (
        trial_root / "checkpoints" / dataset / "custom_noresize" / model_name /
        run_name / f"{run_name}_best_ckpt.pth"
    )


def run_trial(args, search_root, records, trial):
    trial_root = search_root / "trials" / trial["study"] / trial["trial_id"]
    trial_root.mkdir(parents=True, exist_ok=True)
    model_name = STUDIES[trial["study"]]
    log_path = trial_root / "train.log"
    ckpt = checkpoint_path(trial_root, model_name, args.dataset)
    previous = records.get(trial["trial_id"], {})
    attempt = int(previous.get("attempt") or 0) + 1
    row = {
        "trial_id": trial["trial_id"], "study": trial["study"], "phase": trial["phase"],
        "augmentation": trial["augmentation"], "optimizer": trial["optimizer"],
        "epochs": str(trial["epochs"]), "seed": str(trial["seed"]), "status": "running",
        "attempt": str(attempt), "best_accuracy": "", "best_epoch": "",
        "started_at": now(), "finished_at": "", "duration_seconds": "",
        "checkpoint": str(ckpt), "log": str(log_path), "error": "",
    }
    records[trial["trial_id"]] = row
    write_records(search_root / "trials.csv", records)
    atomic_json(search_root / "current_trial.json", {**trial, "command_override": encode_override(trial)})

    command = [
        sys.executable, "baseline/train_baseline_cifar.py",
        "--model_name", model_name,
        "--dataset", args.dataset,
        "--data_dir", str(ROOT.parent / "data"),
        "--output_dir", str(trial_root / "checkpoints"),
        "--case", "custom_noresize",
        "--pretrained", "false",
        "--seed", str(trial["seed"]),
        "--validation_samples", str(args.validation_samples),
        "--validation_seed", str(args.validation_seed),
        "--override", encode_override(trial),
    ]
    start = time.time()
    with log_path.open("a") as log:
        log.write(f"\n===== attempt={attempt} started={row['started_at']} =====\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while process.poll() is None:
            atomic_json(search_root / "heartbeat.json", {
                "time": now(), "controller_pid": os.getpid(), "trial_id": trial["trial_id"],
                "trial_pid": process.pid, "phase": trial["phase"], "study": trial["study"],
            })
            time.sleep(30)
        return_code = process.returncode

    row["finished_at"] = now()
    row["duration_seconds"] = f"{time.time() - start:.1f}"
    if return_code == 0 and ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        row["status"] = "completed"
        row["best_accuracy"] = str(float(payload["acc"]))
        row["best_epoch"] = str(int(payload["epoch"]))
    else:
        row["status"] = "failed"
        row["error"] = f"return_code={return_code}; checkpoint_exists={ckpt.exists()}"
    records[trial["trial_id"]] = row
    write_records(search_root / "trials.csv", records)
    return row


def summarize(search_root, records, target_accuracy):
    summary = {
        "status": "completed",
        "finished_at": now(),
        "target_normal_wrn_accuracy": target_accuracy,
        "studies": {},
    }
    for study in STUDIES:
        full = successful(records, study, "full")
        winner = max(full, key=lambda row: float(row["best_accuracy"]))
        confirmations = [
            row for row in successful(records, study, "confirmation")
            if row["augmentation"] == winner["augmentation"] and row["optimizer"] == winner["optimizer"]
        ]
        accuracies = [float(winner["best_accuracy"])] + [
            float(row["best_accuracy"]) for row in confirmations
        ]
        summary["studies"][study] = {
            "winner": winner,
            "confirmation_trials": confirmations,
            "validation_accuracies": accuracies,
            "mean_validation_accuracy": sum(accuracies) / len(accuracies),
            "normal_wrn_target_gap": target_accuracy - sum(accuracies) / len(accuracies),
        }
    atomic_json(search_root / "result_summary.json", summary)
    atomic_json(search_root / "state.json", summary)


def main():
    global STUDIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--validation_samples", type=int, default=5000)
    parser.add_argument("--validation_seed", type=int, default=20240826)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--target_accuracy", type=float, default=None)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument(
        "--studies", nargs="+", choices=tuple(STUDIES), default=list(STUDIES),
        help="Run only the selected architecture studies while preserving prior records.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    STUDIES = {name: STUDIES[name] for name in args.studies}
    search_root = args.search_root.resolve()
    target_accuracy = args.target_accuracy
    if target_accuracy is None:
        target_accuracy = 0.9482 if args.dataset == "cifar10" else 0.7568
    search_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_or_resumed_at": now(),
        "controller_pid": os.getpid(),
        "root": str(ROOT),
        "search_root": str(search_root),
        "dataset": args.dataset,
        "target_normal_wrn_accuracy": target_accuracy,
        "studies": STUDIES,
        "augmentations": AUGMENTATIONS,
        "optimizers": OPTIMIZERS,
        "screen_epochs": 150,
        "full_epochs": 300,
        "augmentation_trials_per_study": 6,
        "joint_trials_per_study": 12,
        "full_trials_per_study": 5,
        "confirmation_trials_per_study": 2,
        "maximum_trials_total": 25 * len(STUDIES),
        "validation_samples": args.validation_samples,
        "validation_seed": args.validation_seed,
        "finite_budget": True,
    }
    atomic_json(search_root / "manifest.json", manifest)
    records = read_records(search_root / "trials.csv")
    for row in records.values():
        if row["status"] == "running":
            row["status"] = "interrupted"
    write_records(search_root / "trials.csv", records)

    if args.dry_run:
        pending = build_pending(records)
        print(json.dumps({"next_trials": pending, "manifest": manifest}, indent=2))
        return

    atomic_json(search_root / "state.json", {
        "status": "running", "started_or_resumed_at": now(), "controller_pid": os.getpid()
    })
    while True:
        pending = build_pending(records)
        if not pending:
            summarize(search_root, records, target_accuracy)
            return
        trial = pending[0]
        row = run_trial(args, search_root, records, trial)
        if row["status"] == "failed" and int(row["attempt"]) >= args.max_attempts:
            atomic_json(search_root / "state.json", {
                "status": "failed", "failed_trial": row, "time": now(),
            })
            raise RuntimeError(f"Trial failed after {row['attempt']} attempts: {row['trial_id']}")



if __name__ == "__main__":
    main()
