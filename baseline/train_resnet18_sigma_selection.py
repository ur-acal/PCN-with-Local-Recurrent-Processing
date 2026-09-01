#!/usr/bin/env python3
"""Train the existing CIFAR-100 ResNet-18 recipe with noisy checkpoint selection."""

import argparse
import csv
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_cifar_configs import build_model, get_baseline_config
from baseline.run_baseline import FixedMismatchHelper
from baseline.train_baseline_cifar import build_trainer_kwargs, parse_kv_overrides
from trainer_timm import TrainerCiFarTimmStyle


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate_fixed_max_sqrt_trials(model, dataloader, evaluate_fn, sigma, trials, base_seed):
    """Evaluate fixed noisy weights without changing model state or training RNG streams."""
    if sigma <= 0:
        raise ValueError("selection sigma must be positive")
    if trials <= 0:
        raise ValueError("selection trials must be positive")

    helper = FixedMismatchHelper(
        model,
        noise_sigma=sigma,
        noise_type="additive",
        noise_to_norm=False,
        include_buffers=False,
        additive_scale_mode="max_sqrt",
    )
    helper.snapshot_clean_state()
    rng_state = _capture_rng_state()
    was_training = model.training
    top1_values = []
    top5_values = []
    try:
        for trial in range(trials):
            helper.restore_clean_state()
            helper.seed = base_seed + trial
            helper.add_noise()
            top1, top5, _, _ = evaluate_fn(dataloader)
            top1_values.append(float(top1))
            if top5 is not None:
                top5_values.append(float(top5))
    finally:
        helper.restore_clean_state()
        model.train(was_training)
        _restore_rng_state(rng_state)

    return {
        "top1_values": top1_values,
        "top5_values": top5_values,
        "top1_mean": float(np.mean(top1_values)),
        "top1_std": float(np.std(top1_values)),
        "top5_mean": float(np.mean(top5_values)) if top5_values else None,
        "top5_std": float(np.std(top5_values)) if top5_values else None,
    }


class ResNet18SigmaSelectionTrainer(TrainerCiFarTimmStyle):
    """Reuse normal training and select checkpoints only by noisy test accuracy."""

    def __init__(self, *args, selection_sigma=0.07, selection_trials=10, selection_seed=123, **kwargs):
        self.selection_sigma = float(selection_sigma)
        self.selection_trials = int(selection_trials)
        self.selection_seed = int(selection_seed)
        super().__init__(*args, **kwargs)

    @property
    def _selection_history_path(self):
        return Path(self.save_path) / self.model_name / "sigma0p07_selection_history.csv"

    def _append_selection_row(self, row):
        path = self._selection_history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        best_acc = float("-inf")
        best_epoch = 0
        best_model_path = None
        last_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch} / {self.num_epochs}")
            train_loss = self.train_one_epoch(epoch)

            if (epoch + 1) % self.eval_every == 0 and epoch >= self.skip_eval_epochs:
                result = evaluate_fixed_max_sqrt_trials(
                    self.model,
                    self.val_dataloader,
                    self.evaluate,
                    sigma=self.selection_sigma,
                    trials=self.selection_trials,
                    base_seed=self.selection_seed,
                )
                last_acc = result["top1_mean"]
                row = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "sigma": self.selection_sigma,
                    "num_noise_trials": self.selection_trials,
                    "noise_seed_start": self.selection_seed,
                    "noisy_top1_mean": result["top1_mean"],
                    "noisy_top1_std": result["top1_std"],
                    "noisy_top5_mean": result["top5_mean"],
                    "noisy_top5_std": result["top5_std"],
                }
                self._append_selection_row(row)
                print(
                    "Selection sigma={:.3f}: top1={:.2f}+/-{:.2f}% top5={:.2f}+/-{:.2f}% over {} trials".format(
                        self.selection_sigma,
                        100.0 * result["top1_mean"],
                        100.0 * result["top1_std"],
                        100.0 * result["top5_mean"],
                        100.0 * result["top5_std"],
                        self.selection_trials,
                    )
                )

                if result["top1_mean"] > best_acc:
                    best_acc = result["top1_mean"]
                    best_epoch = epoch + 1
                    best_model_path = self._save_model_ckpt(
                        best_acc,
                        best_epoch,
                        "_best_sigma0p07_ckpt.pth",
                    )

            self.scheduler.step(epoch + 1)

        self._save_model_ckpt(last_acc, self.num_epochs, "_last_ckpt.pth")
        print(f"----- Train finished, Model Name: {self.model_name} -----")
        print(f"----- Best sigma=0.07 top1: {100.0 * best_acc:.2f}%, Best epoch: {best_epoch} -----")
        print(f"----- Model path: {best_model_path} -----")
        return best_acc, best_epoch, best_model_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="logs/resnet18_ours_sigma0p07_selection/checkpoints")
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--selection_sigma", type=float, default=0.07)
    parser.add_argument("--selection_trials", type=int, default=10)
    parser.add_argument("--selection_seed", type=int, default=123)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--skip_eval_epochs", type=int, default=0)
    parser.add_argument("--override", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    overrides = parse_kv_overrides(args.override)
    overrides.update({"eval_every": args.eval_every, "skip_eval_epochs": args.skip_eval_epochs})
    cfg = get_baseline_config(
        model_name="resnet18",
        pretrained=False,
        case="adapt_noresize_scratch",
        prefer_resize=False,
        extra_overrides=overrides,
    )
    model = build_model("resnet18", cfg, num_classes=100)
    launcher_args = Namespace(
        model_name="resnet18",
        output_dir=args.output_dir,
        dataset="cifar100",
        img_type="rgb",
        distill_method="none",
        distill_alpha=0.0,
        distill_temperature=1.0,
        orig_t_inp=False,
        teacher_input_size=224,
        teacher_center_crop=True,
        validation_samples=0,
        validation_seed=20240826,
    )
    trainer_kwargs = build_trainer_kwargs(launcher_args, cfg, model, teacher_model=None)
    trainer_kwargs.update(
        selection_sigma=args.selection_sigma,
        selection_trials=args.selection_trials,
        selection_seed=args.selection_seed,
    )

    save_root = Path(trainer_kwargs["save_path"])
    save_root.mkdir(parents=True, exist_ok=True)
    with (save_root / "selection_config.json").open("w") as handle:
        json.dump(
            {
                "training_config": cfg,
                "selection": {
                    "metric": "mean unfused frozen-BN max_sqrt additive test accuracy",
                    "sigma": args.selection_sigma,
                    "trials": args.selection_trials,
                    "seed_start": args.selection_seed,
                    "evaluate_clean_during_training": False,
                    "evaluate_train_accuracy_during_training": False,
                    "fixed_noise_realizations_across_epochs": True,
                },
            },
            handle,
            indent=2,
        )

    trainer = ResNet18SigmaSelectionTrainer(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
