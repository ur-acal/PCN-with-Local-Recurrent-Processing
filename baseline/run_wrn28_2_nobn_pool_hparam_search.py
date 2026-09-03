#!/usr/bin/env python3
"""Bounded, stability-first tuning for parameter-free-shortcut BN-free WRN-28-2."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "logs" / "wrn28_2_nobn_pool_hparam_search_v2"
DATASET_ORDER = ("cifar100", "cifar10")
STUDIES = {
    "avgpool_main": "wrn_28_2_cifar_nobn_avgpool",
    "stride2_main": "wrn_28_2_cifar_nobn_avgpool_shortcut",
}
ARCHITECTURE = "WRN_28_2"
FULL_SEEDS = (4096, 4097, 4098, 4099, 4100)
MISMATCH_SEED = 123
TERMINAL_STATUSES = {"completed", "collapsed", "not_learned"}
MIN_LEARNED_ACCURACY = {"cifar10": 0.20, "cifar100": 0.05}

STUDY_PLANS = {
    ("cifar100", "avgpool_main"): {
        "priority": "high", "candidate_count": 16,
        "screen_seeds": (4096, 4097, 4098), "promotion_count": 3,
    },
    ("cifar10", "stride2_main"): {
        "priority": "high", "candidate_count": 16,
        "screen_seeds": (4096, 4097, 4098), "promotion_count": 3,
    },
    ("cifar10", "avgpool_main"): {
        "priority": "lower", "candidate_count": 8,
        "screen_seeds": (4096, 4097), "promotion_count": 2,
    },
    ("cifar100", "stride2_main"): {
        "priority": "lower", "candidate_count": 8,
        "screen_seeds": (4096, 4097), "promotion_count": 2,
    },
}

REFERENCE_ACCURACIES = {
    "cifar10": {"avgpool_main_current": 0.9397, "stride2_main_current": 0.9379},
    "cifar100": {"avgpool_main_current": 0.6773, "stride2_main_current": 0.7648},
}

AUGMENTATIONS = {
    "mild": {
        "auto_augment": "rand-m5-mstd0.5-inc1", "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0, "label_smoothing": 0.05,
        "re_prob": 0.05, "color_jitter": 0.1,
    },
    "moderate": {
        "auto_augment": "rand-m7-mstd0.5-inc1", "mixup_alpha": 0.1,
        "cutmix_alpha": 0.5, "label_smoothing": 0.05,
        "re_prob": 0.1, "color_jitter": 0.1,
    },
    "randaugment_only": {
        "auto_augment": "rand-m9-mstd0.5-inc1", "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0, "label_smoothing": 0.1,
        "re_prob": 0.1, "color_jitter": 0.1,
    },
}

# Joint bounded design: lr, clipping, block dropout, bias-lr multiplier,
# bias weight decay, augmentation, global weight decay, final dropout.
DESIGN = (
    (0.05, None, 0.0, 0.5, 0.0, "mild", 1e-4, 0.0),
    (0.05, 1.0, 0.1, 1.0, None, "moderate", 5e-4, 0.1),
    (0.05, 2.0, 0.0, 0.5, None, "randaugment_only", 1e-3, 0.4),
    (0.10, None, 0.1, 0.5, None, "moderate", 2e-3, 0.0),
    (0.10, 1.0, 0.0, 1.0, 0.0, "mild", 1e-4, 0.25),
    (0.10, 2.0, 0.1, 1.0, 0.0, "randaugment_only", 5e-4, 0.4),
    (0.15, None, 0.0, 0.5, 0.0, "moderate", 1e-3, 0.1),
    (0.15, 1.0, 0.1, 1.0, None, "randaugment_only", 2e-3, 0.25),
    (0.15, 2.0, 0.0, 0.5, None, "mild", 1e-4, 0.4),
    (0.05, 1.0, 0.1, 0.5, 0.0, "randaugment_only", 5e-4, 0.25),
    (0.10, None, 0.0, 1.0, None, "mild", 1e-3, 0.1),
    (0.15, 2.0, 0.1, 0.5, 0.0, "moderate", 2e-3, 0.4),
    (0.05, 2.0, 0.0, 1.0, None, "mild", 5e-4, 0.0),
    (0.10, 1.0, 0.1, 0.5, None, "randaugment_only", 2e-3, 0.1),
    (0.15, None, 0.0, 1.0, 0.0, "moderate", 1e-3, 0.25),
)

CURRENT_RECIPES = {
    "cifar10": (0.10, 2.0, 0.1, 0.5, 0.0, "randaugment_only", 1e-3, 0.25),
    "cifar100": (0.10, None, 0.0, 1.0, None, "moderate", 1e-3, 0.25),
}

FIXED_RECIPE = {
    "batch_size": 128, "test_batch_size": 512, "eval_every": 5,
    "skip_eval_epochs": 70, "timm_opt": "sgd", "momentum": 0.9,
    "timm_sched": "cosine", "min_lr": 1e-6, "warmup_epoch": 5,
    "warmup_lr": 1e-5, "hflip": 0.5, "timm_train_scale": (1.0, 1.0),
    "timm_train_ratio": (1.0, 1.0), "collapse_monitor_enabled": True,
    "collapse_loss_ema_alpha": 0.3,
}

MISMATCH_FAMILIES = {
    "max_additive": ("additive", "max_abs", "0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"),
    "rms_additive": ("additive", "rms", "0,0.25,0.5,0.75,1.0,1.25"),
    "max_sqrt_additive": ("additive", "max_sqrt", "0,0.02,0.03,0.05,0.07,0.09,0.1"),
    "multiplicative": ("multiplicative", "max_abs", "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4"),
}

FIELDS = (
    "trial_id", "dataset", "study", "phase", "candidate", "seed", "epochs",
    "status", "attempt", "best_accuracy", "best_epoch", "started_at",
    "finished_at", "duration_seconds", "checkpoint", "log", "error",
)


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
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    temp.replace(path)


def candidate_rows(dataset, study):
    count = STUDY_PLANS[(dataset, study)]["candidate_count"]
    values = (CURRENT_RECIPES[dataset],) + DESIGN
    keys = (
        "lr", "max_norm", "dropout_rate", "bias_lr_multiplier",
        "bias_weight_decay", "augmentation", "weight_decay",
        "final_dropout_rate",
    )
    rows = {}
    for index, values_row in enumerate(values[:count], 1):
        rows[f"c{index:02d}"] = dict(zip(keys, values_row))
    rows["c01"]["label"] = "original_recipe"
    return rows


def resolved_recipe(dataset, study, candidate, epochs):
    row = candidate_rows(dataset, study)[candidate]
    return {
        **FIXED_RECIPE, "num_epochs": epochs,
        **{key: value for key, value in row.items() if key not in {"augmentation", "label"}},
        **AUGMENTATIONS[row["augmentation"]],
    }


def encode_value(value):
    if value is None:
        return "none"
    if isinstance(value, tuple):
        return ":".join(str(item) for item in value)
    return str(value).lower() if isinstance(value, bool) else str(value)


def encode_override(dataset, study, candidate, epochs):
    return ",".join(
        f"{key}={encode_value(value)}"
        for key, value in resolved_recipe(dataset, study, candidate, epochs).items()
    )


def make_trial(dataset, study, phase, candidate, seed, epochs):
    trial_id = f"{dataset}__{study}__{phase}__{candidate}__s{seed}"
    return {
        "trial_id": trial_id, "dataset": dataset, "study": study,
        "phase": phase, "candidate": candidate, "seed": seed, "epochs": epochs,
    }


def phase_rows(records, dataset, study, phase, candidate=None):
    return [
        row for row in records.values()
        if row["dataset"] == dataset and row["study"] == study
        and row["phase"] == phase
        and (candidate is None or row["candidate"] == candidate)
    ]


def stable_candidate_statistics(records, dataset, study, phase, expected_seeds):
    stats = []
    for candidate in candidate_rows(dataset, study):
        rows = phase_rows(records, dataset, study, phase, candidate)
        by_seed = {int(row["seed"]): row for row in rows}
        if set(by_seed) != set(expected_seeds):
            continue
        ordered = [by_seed[seed] for seed in expected_seeds]
        if any(row["status"] != "completed" for row in ordered):
            continue
        accuracies = [float(row["best_accuracy"]) for row in ordered]
        stats.append({
            "candidate": candidate, "accuracies": accuracies,
            "mean": statistics.mean(accuracies), "worst": min(accuracies),
            "std": statistics.pstdev(accuracies), "rows": ordered,
        })
    return stats


def rank_stable_candidates(stats):
    return sorted(
        stats,
        key=lambda item: (
            round(item["mean"], 12), item["worst"], -item["std"]
        ),
        reverse=True,
    )


def promoted_candidates(records, dataset, study):
    plan = STUDY_PLANS[(dataset, study)]
    stats = stable_candidate_statistics(
        records, dataset, study, "screen", plan["screen_seeds"]
    )
    return [
        item["candidate"]
        for item in rank_stable_candidates(stats)[:plan["promotion_count"]]
    ]


def winning_candidate(records, dataset, study):
    stats = stable_candidate_statistics(records, dataset, study, "full", FULL_SEEDS)
    ranked = rank_stable_candidates(stats)
    if not ranked:
        raise RuntimeError(f"No stable five-seed full candidate for {dataset}/{study}")
    return ranked[0]["candidate"]


def stage_trials(records, dataset, study, phase):
    if phase == "screen":
        candidates = candidate_rows(dataset, study)
        seeds = STUDY_PLANS[(dataset, study)]["screen_seeds"]
        epochs = 150
    elif phase == "full":
        candidates = promoted_candidates(records, dataset, study)
        if not candidates:
            raise RuntimeError(f"No stable screening candidate for {dataset}/{study}")
        seeds = FULL_SEEDS
        epochs = 300
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return [
        make_trial(dataset, study, phase, candidate, seed, epochs)
        for candidate in candidates for seed in seeds
    ]


def checkpoint_path(search_root, trial):
    model = STUDIES[trial["study"]]
    run = f"custom_noresize_{trial['dataset']}_{model}"
    return (
        search_root / "trials" / trial["trial_id"] / "checkpoints" /
        trial["dataset"] / "custom_noresize" / model / run /
        f"{run}_best_ckpt.pth"
    )


def collapse_path(search_root, trial):
    model = STUDIES[trial["study"]]
    return (
        search_root / "trials" / trial["trial_id"] / "checkpoints" /
        trial["dataset"] / "custom_noresize" / model / "training_collapse.json"
    )


def run_training_trial(args, search_root, trial, previous, active, lock):
    trial_root = search_root / "trials" / trial["trial_id"]
    trial_root.mkdir(parents=True, exist_ok=True)
    log_path = trial_root / "train.log"
    checkpoint = checkpoint_path(search_root, trial)
    collapse_record = collapse_path(search_root, trial)
    attempt = int(previous.get("attempt") or 0) + 1
    started_at = now()
    command = [
        sys.executable, "baseline/train_baseline_cifar.py",
        "--model_name", STUDIES[trial["study"]], "--dataset", trial["dataset"],
        "--data_dir", str(args.data_root), "--output_dir", str(trial_root / "checkpoints"),
        "--case", "custom_noresize", "--pretrained", "false",
        "--seed", str(trial["seed"]),
        "--validation_samples", str(args.validation_samples),
        "--validation_seed", str(args.validation_seed),
        "--override", encode_override(
            trial["dataset"], trial["study"], trial["candidate"], trial["epochs"]
        ),
    ]
    start_time = time.time()
    with log_path.open("a") as log:
        log.write(f"\n===== attempt={attempt} started={started_at} =====\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with lock:
            active[trial["trial_id"]] = process.pid
        return_code = process.wait()
        with lock:
            active.pop(trial["trial_id"], None)

    row = {key: "" for key in FIELDS}
    row.update({key: str(value) for key, value in trial.items()})
    row.update({
        "status": "failed", "attempt": str(attempt), "started_at": started_at,
        "finished_at": now(), "duration_seconds": f"{time.time() - start_time:.1f}",
        "checkpoint": str(checkpoint), "log": str(log_path),
    })
    if collapse_record.exists():
        event = json.loads(collapse_record.read_text())
        row.update({"status": "collapsed", "error": json.dumps(event, sort_keys=True)})
    elif return_code == 0 and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        accuracy = float(payload["acc"])
        row.update({
            "status": (
                "completed" if accuracy > MIN_LEARNED_ACCURACY[trial["dataset"]]
                else "not_learned"
            ),
            "best_accuracy": str(accuracy), "best_epoch": str(int(payload["epoch"])),
        })
        if row["status"] == "not_learned":
            row["error"] = "completed bounded run without rising above near-chance accuracy"
    else:
        row["error"] = f"return_code={return_code}; checkpoint_exists={checkpoint.exists()}"
    return row


def run_parallel_training(args, search_root, records, dataset, study, phase, active, lock):
    trials = stage_trials(records, dataset, study, phase)
    records_path = search_root / "trials.csv"
    while True:
        pending = [
            trial for trial in trials
            if records.get(trial["trial_id"], {}).get("status") not in TERMINAL_STATUSES
            and int(records.get(trial["trial_id"], {}).get("attempt") or 0) < args.max_attempts
        ]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {
                executor.submit(
                    run_training_trial, args, search_root, trial,
                    records.get(trial["trial_id"], {}), active, lock,
                ): trial for trial in pending
            }
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    records[row["trial_id"]] = row
                    write_records(records_path, records)
    unfinished = [
        trial["trial_id"] for trial in trials
        if records.get(trial["trial_id"], {}).get("status") not in TERMINAL_STATUSES
    ]
    if unfinished:
        raise RuntimeError(f"Unfinished {dataset}/{study}/{phase} trials: {unfinished}")


def canonical_winner_trial(search_root, records, dataset, study):
    candidate = winning_candidate(records, dataset, study)
    rows = phase_rows(records, dataset, study, "full", candidate)
    row = max(rows, key=lambda item: float(item["best_accuracy"]))
    trial = make_trial(dataset, study, "full", candidate, int(row["seed"]), 300)
    return candidate, trial, checkpoint_path(search_root, trial)


def run_mismatch(args, search_root, records, dataset, study, family, active, lock):
    _candidate, _trial, checkpoint = canonical_winner_trial(
        search_root, records, dataset, study
    )
    output = search_root / "mismatch" / family
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "evaluation.log"
    mismatch_type, scale, levels = MISMATCH_FAMILIES[family]
    command = [
        sys.executable, "baseline/run_wrn_nobn_mismatch_experiment.py",
        "--output_dir", str(output), "--data_dir", str(args.data_root),
        "--checkpoint_override", str(checkpoint),
        "--model_name_override", STUDIES[study], "--datasets", dataset,
        "--architectures", ARCHITECTURE, "--mismatch_types", mismatch_type,
        "--additive_scale_mode", scale, "--noise_levels", levels,
        "--noisy_trials", "10", "--seed", str(MISMATCH_SEED),
        "--batch_size", "128", "--num_workers", "2", "--case", "custom_noresize",
    ]
    with log_path.open("a") as log:
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with lock:
            active[f"mismatch__{family}"] = process.pid
        return_code = process.wait()
        with lock:
            active.pop(f"mismatch__{family}", None)
    if return_code != 0 or not (output / "full_aggregate.csv").exists():
        raise RuntimeError(f"Mismatch evaluation failed: {family}; see {log_path}")


def run_parallel_mismatch(args, search_root, records, dataset, study, active, lock):
    pending = [
        family for family in MISMATCH_FAMILIES
        if not (search_root / "mismatch" / family / "full_aggregate.csv").exists()
    ]
    with ThreadPoolExecutor(max_workers=min(args.parallelism, len(pending) or 1)) as executor:
        futures = [
            executor.submit(
                run_mismatch, args, search_root, records, dataset, study,
                family, active, lock,
            ) for family in pending
        ]
        for future in as_completed(futures):
            future.result()


def write_summary(search_root, records, dataset, study):
    plan = STUDY_PLANS[(dataset, study)]
    result = {
        "updated_at": now(), "dataset": dataset, "study": study,
        "model_name": STUDIES[study], "plan": plan,
        "reference_accuracy": REFERENCE_ACCURACIES[dataset][f"{study}_current"],
    }
    lines = [
        "# BN-free WRN-28-2 pooling hyperparameter search", "",
        f"- Dataset: `{dataset}`", f"- Study: `{study}`",
        f"- Priority: `{plan['priority']}`", "",
    ]
    screen_stats = rank_stable_candidates(stable_candidate_statistics(
        records, dataset, study, "screen", plan["screen_seeds"]
    ))
    lines.extend(["## Stable screening candidates", "", "| Rank | Candidate | Mean | Worst | Std |", "|---:|---|---:|---:|---:|"])
    for rank, item in enumerate(screen_stats, 1):
        label = candidate_rows(dataset, study)[item["candidate"]].get("label")
        name = "original recipe" if label else item["candidate"]
        lines.append(
            f"| {rank} | {name} | {item['mean'] * 100:.2f}% | "
            f"{item['worst'] * 100:.2f}% | {item['std'] * 100:.2f} |"
        )

    full_stats = rank_stable_candidates(stable_candidate_statistics(
        records, dataset, study, "full", FULL_SEEDS
    ))
    if full_stats:
        winner = full_stats[0]
        recipe = resolved_recipe(dataset, study, winner["candidate"], 300)
        result.update({
            "winner": winner["candidate"], "resolved_recipe": recipe,
            "accuracies": winner["accuracies"], "mean_accuracy": winner["mean"],
            "worst_accuracy": winner["worst"], "std_accuracy": winner["std"],
        })
        label = candidate_rows(dataset, study)[winner["candidate"]].get("label")
        winner_name = "original recipe" if label else winner["candidate"]
        lines.extend([
            "", "## Stable full-training winner", "",
            f"- Winner: **{winner_name}**",
            f"- Five-seed mean: **{winner['mean'] * 100:.2f}%**",
            f"- Worst seed: **{winner['worst'] * 100:.2f}%**",
            f"- Standard deviation: **{winner['std'] * 100:.2f} points**",
        ])
    else:
        lines.extend(["", "## Stable full-training winner", "", "Not available yet."])
    atomic_json(search_root / "result_summary.json", result)
    (search_root / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_ORDER, required=True)
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--search_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data_root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--validation_samples", type=int, default=5000)
    parser.add_argument("--validation_seed", type=int, default=20240826)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--skip_mismatch", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.parallelism < 1:
        raise ValueError("parallelism must be positive")

    dataset, study = args.dataset, args.study
    plan = STUDY_PLANS[(dataset, study)]
    search_root = args.search_root.resolve()
    search_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_or_resumed_at": now(), "dataset": dataset, "study": study,
        "model_name": STUDIES[study], "plan": plan,
        "full_seeds": FULL_SEEDS, "validation_samples": args.validation_samples,
        "validation_seed": args.validation_seed, "parallelism": args.parallelism,
        "screen_epochs": 150, "full_epochs": 300,
        "fixed_recipe": FIXED_RECIPE, "augmentations": AUGMENTATIONS,
        "candidates": candidate_rows(dataset, study),
        "selection_order": ["zero_failures", "mean_accuracy", "worst_seed_accuracy", "lower_std"],
        "finite_budget": True,
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
    active = {}
    lock = threading.Lock()
    stop = threading.Event()
    current = {"dataset": dataset, "study": study, "phase": "startup"}

    def heartbeat():
        while not stop.wait(30):
            with lock:
                running = dict(active)
                status_counts = {}
                for row in records.values():
                    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            atomic_json(search_root / "state.json", {
                "status": "running", "controller_pid": os.getpid(), **current,
                "active_trials": running, "training_status_counts": status_counts,
                "updated_at": now(),
            })

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for phase in ("screen", "full"):
            current["phase"] = phase
            run_parallel_training(
                args, search_root, records, dataset, study, phase, active, lock
            )
            write_summary(search_root, records, dataset, study)
        if not args.skip_mismatch:
            current["phase"] = "mismatch"
            run_parallel_mismatch(
                args, search_root, records, dataset, study, active, lock
            )
        write_summary(search_root, records, dataset, study)
        atomic_json(search_root / "state.json", {
            "status": "completed", "controller_pid": os.getpid(), **current,
            "active_trials": {}, "updated_at": now(),
        })
    except Exception as exc:
        atomic_json(search_root / "state.json", {
            "status": "failed", "controller_pid": os.getpid(), **current,
            "active_trials": dict(active), "error": str(exc), "updated_at": now(),
        })
        raise
    finally:
        stop.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    main()
