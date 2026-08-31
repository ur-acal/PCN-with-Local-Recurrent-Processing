#!/usr/bin/env python3
"""Durable, bounded augmentation-only search for Tiny ImageNet WRN-28-4."""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = ROOT / "logs" / "tinyimagenet_wrn28_4_aug_search"
MODEL_NAME = "wrn_28_4_cifar"
DATASET = "tinyimagenet"
ALLOWED_TUNED_FIELDS = {
    "auto_augment",
    "timm_train_scale",
    "color_jitter",
    "re_prob",
    "mixup_alpha",
    "cutmix_alpha",
}
FIXED_RECIPE = {
    "num_epochs": 150,
    "eval_every": 5,
    "skip_eval_epochs": 75,
    "test_batch_size": 512,
    "lr": 0.1,
    "weight_decay": 1e-3,
    "timm_opt": "sgd",
    "momentum": 0.9,
    "timm_sched": "cosine",
    "min_lr": 1e-6,
    "warmup_epoch": 5,
    "warmup_lr": 1e-5,
    "label_smoothing": 0.1,
    "timm_train_ratio": (0.75, 4.0 / 3.0),
    "hflip": 0.5,
    "batch_size": 128,
    "final_dropout_rate": 0.25,
    "dropout_rate": 0.0,
}
ANCHOR = {
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "timm_train_scale": (0.75, 1.0),
    "color_jitter": 0.1,
    "re_prob": 0.25,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
}
CANDIDATES = {
    "anchor": {},
    "no_randaugment": {"auto_augment": None},
    "randaugment_m5": {"auto_augment": "rand-m5-mstd0.5-inc1"},
    "randaugment_m7": {"auto_augment": "rand-m7-mstd0.5-inc1"},
    "crop_scale_050": {"timm_train_scale": (0.5, 1.0)},
    "crop_scale_090": {"timm_train_scale": (0.9, 1.0)},
    "no_mixup_cutmix": {"mixup_alpha": 0.0, "cutmix_alpha": 0.0},
    "moderate_mixup_cutmix": {"mixup_alpha": 0.1, "cutmix_alpha": 0.5},
    "no_random_erasing": {"re_prob": 0.0},
    "random_erasing_010": {"re_prob": 0.1},
    "no_color_jitter": {"color_jitter": 0.0},
    "color_jitter_040": {"color_jitter": 0.4},
}
FIELDNAMES = [
    "trial_id", "status", "attempt", "best_accuracy", "best_epoch",
    "started_at", "finished_at", "duration_seconds", "checkpoint", "log", "error",
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
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    temp.replace(path)


def candidate_config(name):
    changed = CANDIDATES[name]
    unexpected = set(changed) - ALLOWED_TUNED_FIELDS
    if unexpected:
        raise ValueError(f"Candidate {name} changes forbidden fields: {sorted(unexpected)}")
    return {**ANCHOR, **changed}


def encode_value(value):
    if value is None:
        return "none"
    if isinstance(value, tuple):
        return ":".join(str(item) for item in value)
    return str(value).lower() if isinstance(value, bool) else str(value)


def encode_override(name):
    values = {**FIXED_RECIPE, **candidate_config(name)}
    return ",".join(f"{key}={encode_value(value)}" for key, value in values.items())


def checkpoint_path(trial_root):
    run_name = f"custom_noresize_{DATASET}_{MODEL_NAME}"
    return (
        trial_root / "checkpoints" / DATASET / "custom_noresize" / MODEL_NAME /
        run_name / f"{run_name}_best_ckpt.pth"
    )


def run_trial(args, search_root, trial_id, previous, active, lock):
    trial_root = search_root / "trials" / trial_id
    trial_root.mkdir(parents=True, exist_ok=True)
    log_path = trial_root / "train.log"
    checkpoint = checkpoint_path(trial_root)
    attempt = int(previous.get("attempt") or 0) + 1
    started_at = now()
    command = [
        sys.executable,
        "baseline/train_baseline_cifar.py",
        "--model_name", MODEL_NAME,
        "--dataset", DATASET,
        "--data_dir", str(args.data_root),
        "--output_dir", str(trial_root / "checkpoints"),
        "--case", "custom_noresize",
        "--pretrained", "false",
        "--seed", str(args.seed),
        "--override", encode_override(trial_id),
    ]
    with log_path.open("a") as log:
        log.write(f"\n===== attempt={attempt} started={started_at} =====\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with lock:
            active[trial_id] = process.pid
        return_code = process.wait()
        with lock:
            active.pop(trial_id, None)

    row = {
        "trial_id": trial_id,
        "status": "failed",
        "attempt": str(attempt),
        "best_accuracy": "",
        "best_epoch": "",
        "started_at": started_at,
        "finished_at": now(),
        "duration_seconds": "",
        "checkpoint": str(checkpoint),
        "log": str(log_path),
        "error": "",
    }
    start_timestamp = datetime.fromisoformat(started_at).timestamp()
    row["duration_seconds"] = f"{time.time() - start_timestamp:.1f}"
    if return_code == 0 and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        row["status"] = "completed"
        row["best_accuracy"] = str(float(payload["acc"]))
        row["best_epoch"] = str(int(payload["epoch"]))
    else:
        row["error"] = f"return_code={return_code}; checkpoint_exists={checkpoint.exists()}"
    return row


def write_summary(search_root, records):
    completed = [row for row in records.values() if row["status"] == "completed"]
    ranking = sorted(completed, key=lambda row: float(row["best_accuracy"]), reverse=True)
    atomic_json(search_root / "result_summary.json", {
        "status": "completed" if len(completed) == len(CANDIDATES) else "incomplete",
        "finished_at": now(),
        "ranking": ranking,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--data_root", type=Path, default=ROOT.parent / "data" / "tiny-imagenet-200")
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    search_root = args.search_root.resolve()
    search_root.mkdir(parents=True, exist_ok=True)

    resolved_candidates = {name: candidate_config(name) for name in CANDIDATES}
    manifest = {
        "created_or_resumed_at": now(),
        "controller_pid": os.getpid(),
        "model": MODEL_NAME,
        "dataset": DATASET,
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "parallelism": args.parallelism,
        "finite_budget": True,
        "trial_count": len(CANDIDATES),
        "allowed_tuned_fields": sorted(ALLOWED_TUNED_FIELDS),
        "fixed_recipe": FIXED_RECIPE,
        "anchor_augmentation": ANCHOR,
        "candidates": resolved_candidates,
    }
    atomic_json(search_root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    records_path = search_root / "trials.csv"
    records = read_records(records_path)
    for row in records.values():
        if row["status"] == "running":
            row["status"] = "interrupted"
    write_records(records_path, records)

    pending = [
        trial_id for trial_id in CANDIDATES
        if records.get(trial_id, {}).get("status") != "completed"
        and int(records.get(trial_id, {}).get("attempt") or 0) < args.max_attempts
    ]
    active = {}
    lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.wait(30):
            with lock:
                current = dict(active)
            atomic_json(search_root / "state.json", {
                "status": "running",
                "controller_pid": os.getpid(),
                "active_trials": current,
                "completed": sum(row["status"] == "completed" for row in records.values()),
                "total": len(CANDIDATES),
                "updated_at": now(),
            })

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    atomic_json(search_root / "state.json", {
        "status": "running", "controller_pid": os.getpid(),
        "active_trials": {}, "completed": len(CANDIDATES) - len(pending),
        "total": len(CANDIDATES), "updated_at": now(),
    })

    try:
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {}
            for trial_id in pending:
                previous = records.get(trial_id, {})
                records[trial_id] = {
                    "trial_id": trial_id, "status": "queued",
                    "attempt": str(int(previous.get("attempt") or 0) + 1),
                    "best_accuracy": "", "best_epoch": "", "started_at": now(),
                    "finished_at": "", "duration_seconds": "",
                    "checkpoint": str(checkpoint_path(search_root / "trials" / trial_id)),
                    "log": str(search_root / "trials" / trial_id / "train.log"), "error": "",
                }
                write_records(records_path, records)
                future = executor.submit(
                    run_trial, args, search_root, trial_id, previous, active, lock
                )
                futures[future] = trial_id
            for future in as_completed(futures):
                row = future.result()
                records[row["trial_id"]] = row
                write_records(records_path, records)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)

    write_summary(search_root, records)
    failures = [row for row in records.values() if row["status"] != "completed"]
    atomic_json(search_root / "state.json", {
        "status": "completed" if not failures else "failed",
        "controller_pid": os.getpid(),
        "active_trials": {},
        "completed": len(CANDIDATES) - len(failures),
        "total": len(CANDIDATES),
        "failures": failures,
        "updated_at": now(),
    })
    if failures:
        raise RuntimeError(f"{len(failures)} augmentation trials did not complete")


if __name__ == "__main__":
    main()
