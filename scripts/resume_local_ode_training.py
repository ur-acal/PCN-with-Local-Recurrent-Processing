"""Resume PCN pretraining from an explicit epoch-recovery checkpoint.

Restores saved arguments and preserves the model name. Does not start FT.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import train_ode_cifar as training


def apply_training_step_reuse_override(config, cli_override):
    """Apply CLI/env override while keeping a saved value when neither is set."""
    override = cli_override
    if override is None and "REUSE_ACCEPTED_STEP_TRAINING" in os.environ:
        override = training.str2bool(os.environ["REUSE_ACCEPTED_STEP_TRAINING"])
    if override is None:
        config.setdefault("reuse_accepted_step_training", False)
    else:
        config["reuse_accepted_step_training"] = override


def apply_rhs_checkpoint_override(config, cli_override):
    """Apply CLI/env override while keeping a saved value when neither is set."""
    override = cli_override
    if override is None and "CHECKPOINT_ODE_RHS_TRAINING" in os.environ:
        override = training.str2bool(os.environ["CHECKPOINT_ODE_RHS_TRAINING"])
    if override is None:
        config.setdefault("checkpoint_ode_rhs_training", False)
    else:
        config["checkpoint_ode_rhs_training"] = override


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Latest checkpoint or its run directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse_accepted_step_training", type=training.str2bool,
                        default=None)
    parser.add_argument("--checkpoint_ode_rhs_training", type=training.str2bool,
                        default=None)
    args = parser.parse_args()
    path = args.checkpoint
    if path.is_dir():
        paths = list(path.glob("*/*_latest_ckpt.pth"))
        if len(paths) != 1:
            raise ValueError(f"Expected one latest checkpoint in {path}, found {len(paths)}")
        path = paths[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    recovery = checkpoint["training_recovery"]
    config = recovery["config"].copy()
    apply_training_step_reuse_override(config, args.reuse_accepted_step_training)
    apply_rhs_checkpoint_override(config, args.checkpoint_ode_rhs_training)
    model_name = path.parent.name
    config.update(model_name=model_name, ckpt="latest",
                  save_path=str(path.parent.parent),
                  output_save_path=str(path.parent.parent), num_workers=0)
    print(f"Recovery checkpoint: {path}", flush=True)
    print(f"Completed epochs: {checkpoint['epoch']}; target: {config['num_epochs']}", flush=True)
    print(f"Recovery components: {list(recovery['components'])}; DataLoader workers: 0", flush=True)
    if args.dry_run:
        return
    # Use the existing entry point and restore_latest implementation. Avoid its
    # ordinary fine-tuning name generation for this same-stage continuation.
    training.get_args = lambda: argparse.Namespace(**config)
    training.get_model_name = lambda unused: model_name
    del checkpoint, recovery
    training.main()


if __name__ == "__main__":
    main()
