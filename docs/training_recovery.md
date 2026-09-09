# Epoch recovery

Both trainers atomically replace `<model>_latest_ckpt.pth` after each completed
epoch, after validation (when scheduled) and scheduler advancement. This file
contains trainable/QAT weights and buffers, optimizer and scheduler state,
distillation state, completed epoch, metric histories, launch arguments, global
RNG states, and hardware generators/assignments. It is not a flattened inference
checkpoint.

Use the same architecture, training stage, recipe, dataset and device layout.
Explicitly supply the latest file through the existing CNN
`PRETRAIN_RESUME_CKPT` (pretraining) or `MODEL_CKPT` (fine-tuning) environment
variable. These forward to `--resume_checkpoint`. PCN uses its existing
`--model_name`, `--save_path`, and `--ckpt latest` selection. No latest file is
discovered automatically. Ordinary checkpoints retain weights-only loading.

Recovery resumes at the next epoch and restores RNG state after trainer and
teacher setup. Exact reproducibility requires non-persistent data-loader workers,
the same data/worker configuration and deterministic operators; persistent-worker
recovery is rejected rather than pretending to restore inaccessible worker RNGs.
An interrupted epoch is rerun from its beginning.

Existing best/last checkpoint writes and validation scheduling are unchanged.
Latest is deleted only after the existing last-checkpoint writer succeeds.
Completion queues continue to watch last, never latest. Checkpoints made before
this feature cannot recover optimizer/RNG state that was never saved.
