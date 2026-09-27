import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from scripts.resume_local_ode_training import _backfill_validation_defaults
from train_ode_cifar import evaluate_teacher
from cifar_validation import (
    IndexedSubset,
    load_or_create_validation_split,
    require_matching_validation_split,
)


ROOT = Path(__file__).resolve().parents[1]


def balanced_labels(num_classes=10, per_class=20):
    return np.repeat(np.arange(num_classes), per_class)


def test_split_is_persistent_balanced_and_complete(tmp_path):
    labels = balanced_labels()
    manifest = tmp_path / "split.json"
    first = load_or_create_validation_split(
        "cifar10", labels, validation_size=50, seed=4096,
        manifest_path=manifest)
    second = load_or_create_validation_split(
        "cifar10", labels, validation_size=50, seed=999,
        manifest_path=manifest)

    assert first.val_indices == second.val_indices
    assert len(first.train_indices) == 150
    assert len(first.val_indices) == 50
    assert set(first.train_indices).isdisjoint(first.val_indices)
    assert set(first.train_indices) | set(first.val_indices) == set(range(200))
    assert np.bincount(labels[list(first.val_indices)]).tolist() == [5] * 10


def test_manifest_rejects_another_dataset_order(tmp_path):
    labels = balanced_labels()
    manifest = tmp_path / "split.json"
    load_or_create_validation_split(
        "cifar10", labels, validation_size=50, manifest_path=manifest)
    changed = labels.copy()
    changed[[0, -1]] = changed[[-1, 0]]
    with pytest.raises(ValueError, match="labels_checksum"):
        load_or_create_validation_split(
            "cifar10", changed, validation_size=50,
            manifest_path=manifest)


def test_manifest_checksum_detects_edits(tmp_path):
    labels = balanced_labels()
    manifest = tmp_path / "split.json"
    load_or_create_validation_split(
        "cifar10", labels, validation_size=50, manifest_path=manifest)
    payload = json.loads(manifest.read_text())
    payload["validation_indices"][0] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_or_create_validation_split(
            "cifar10", labels, validation_size=50,
            manifest_path=manifest)


def test_indexed_subset_retains_visible_labels():
    inputs = torch.arange(12).view(6, 2)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    dataset = TensorDataset(inputs, labels)
    dataset.targets = labels.tolist()
    subset = IndexedSubset(dataset, [5, 2, 0])
    assert subset.targets == [5, 2, 0]
    assert [int(subset[index][1]) for index in range(3)] == [5, 2, 0]


def test_checkpoint_split_must_match():
    metadata = {
        "dataset": "cifar10",
        "manifest_checksum": "manifest",
        "labels_checksum": "labels",
        "train_size": 45000,
        "validation_size": 5000,
    }
    require_matching_validation_split(
        {"validation_split": dict(metadata)}, metadata)
    mismatched = dict(metadata, manifest_checksum="other")
    with pytest.raises(ValueError, match="manifest_checksum"):
        require_matching_validation_split(
            {"validation_split": mismatched}, metadata)


def test_old_recovery_config_gets_validation_defaults():
    config = _backfill_validation_defaults({"num_epochs": 300})
    assert config["validation_mode"] is False
    assert config["validation_manifest"] is None
    assert config["validation_split_seed"] == 4096


def test_teacher_evaluation_uses_teacher_view_from_paired_batch():
    student_inputs = torch.zeros(2, 4, 16, 16)
    teacher_inputs = torch.ones(2, 3, 224, 224)
    labels = torch.tensor([1, 0])

    class DummyTrainer:
        device = torch.device("cpu")
        orig_t_inp = True
        dataset_name = "cifar10"
        num_classes = 2
        teacher_eval_loader = [(student_inputs, teacher_inputs, labels)]
        val_dataloader = None

        def teacher_forward_for_distillation(self, inputs):
            self.seen_inputs = inputs
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]]), None

    trainer = DummyTrainer()
    accuracy = evaluate_teacher(torch.nn.Identity(), trainer)
    assert accuracy == 1.0
    assert trainer.seen_inputs is teacher_inputs


@pytest.mark.parametrize("validation_value", ["true", "True", "1", "yes"])
def test_pcn_slurm_validation_mode_is_forwarded(validation_value):
    env = dict(
        os.environ,
        VALIDATION_MODE=validation_value,
        TC_DRY_RUN="true",
        TASK="cifar10",
        IMG_TYPE="rgb",
        EXP_PREFIX="validation_smoke",
        PCN_CHAN_0_LIST="16",
        PCN_NUM_LAYERS_LIST="6",
        NUM_COMB_PER_NUM_LAYER="1",
        COMB_SEL_SET="1",
    )
    result = subprocess.run(
        ["bash", "-c", "source ./launch_scripts/slurm_search_config.sh"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "VALIDATION_MODE=true" in result.stdout
    assert "RUN_TAG=validation_smoke_val5k" in result.stdout
    assert "EXP=validation_smoke_val5k_" in result.stdout
    assert "FINAL_EVAL_ONLY=false" in result.stdout
    assert "cifar10_rgb_OldNoTimm_MatchDistill_val5k.pth" in result.stdout


def test_pcn_slurm_validation_mode_normalizes_explicit_checkpoint_roots():
    env = dict(
        os.environ,
        VALIDATION_MODE="true",
        TC_DRY_RUN="true",
        TASK="cifar10",
        IMG_TYPE="rgb",
        EXP_PREFIX="validation_custom_roots",
        PCN_CHAN_0_LIST="16",
        PCN_NUM_LAYERS_LIST="6",
        NUM_COMB_PER_NUM_LAYER="1",
        COMB_SEL_SET="1",
        OUTPUT_SAVE_PATH="/tmp/custom_output",
        PRETRAIN_SAVE_PATH="/tmp/custom_pretrain",
        FT_OUTPUT_SAVE_PATH="/tmp/custom_ft",
    )
    result = subprocess.run(
        ["bash", "-c", "source ./launch_scripts/slurm_search_config.sh"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "OUTPUT_SAVE_PATH=/tmp/custom_output_val5k" in result.stdout
    assert "PRETRAIN_SAVE_PATH=/tmp/custom_pretrain_val5k" in result.stdout
    assert "FT_OUTPUT_SAVE_PATH=/tmp/custom_ft_val5k" in result.stdout


def test_direct_pcn_worker_isolates_validation_outputs():
    source = (ROOT / "launch_scripts/run_kdcrd_then_ft.sbatch").read_text()
    prefix = source[:source.index("# PHASE 1:")]
    prefix = prefix.replace("source activate base", "true")
    prefix = prefix.replace("conda activate scanbase", "true")
    with tempfile.TemporaryDirectory() as directory:
        prefix = "\n".join(
            "LOGDIR=" + shlex.quote(directory)
            if line.startswith("LOGDIR=") else line
            for line in prefix.splitlines())
        command = prefix + r'''
printf '%s\n' "$VALIDATION_MODE" "$EXP" "$RUN_TAG" \
  "$OUTPUT_SAVE_PATH" "$PRETRAIN_SAVE_PATH" "$FT_OUTPUT_SAVE_PATH"
'''
        env = dict(
            os.environ,
            VALIDATION_MODE="True",
            TC_NONIDEALITIES="true",
            TOGGLE_MODE="none",
            SWITCH_INF="false",
            TASK="cifar10",
            IMG_TYPE="rgb",
            EXP_PREFIX="direct_validation",
            SLURM_JOB_ID="fixture",
            COMB_LIST="audit",
        )
        result = subprocess.run(
            ["bash", "-c", command], cwd=ROOT, env=env,
            text=True, capture_output=True, check=True)
    values = result.stdout.strip().splitlines()[-6:]
    assert values == [
        "true",
        "no_bn_PCNetNoBatchNorm_NODE_search_cifar10_rgb_NoCirc_jobfixture_Exp_val5k",
        "direct_validation_val5k",
        "./saved_ckpt_runs/direct_validation_val5k",
        "./saved_ckpt_runs/direct_validation_val5k",
        "./saved_ckpt_runs/direct_validation_val5k",
    ]


def test_direct_pcn_worker_normalizes_explicit_validation_roots():
    source = (ROOT / "launch_scripts/run_kdcrd_then_ft.sbatch").read_text()
    prefix = source[:source.index("# PHASE 1:")]
    prefix = prefix.replace("source activate base", "true")
    prefix = prefix.replace("conda activate scanbase", "true")
    with tempfile.TemporaryDirectory() as directory:
        prefix = "\n".join(
            "LOGDIR=" + shlex.quote(directory)
            if line.startswith("LOGDIR=") else line
            for line in prefix.splitlines())
        command = prefix + r'''
printf '%s\n' "$OUTPUT_SAVE_PATH" "$PRETRAIN_SAVE_PATH" "$FT_OUTPUT_SAVE_PATH"
'''
        env = dict(
            os.environ,
            VALIDATION_MODE="true",
            TC_NONIDEALITIES="true",
            TOGGLE_MODE="none",
            SWITCH_INF="false",
            TASK="cifar10",
            IMG_TYPE="rgb",
            EXP_PREFIX="direct_validation",
            SLURM_JOB_ID="fixture",
            COMB_LIST="audit",
            OUTPUT_SAVE_PATH="/tmp/custom_output",
            PRETRAIN_SAVE_PATH="/tmp/custom_pretrain",
            FT_OUTPUT_SAVE_PATH="/tmp/custom_ft",
        )
        result = subprocess.run(
            ["bash", "-c", command], cwd=ROOT, env=env,
            text=True, capture_output=True, check=True)
    assert result.stdout.strip().splitlines()[-3:] == [
        "/tmp/custom_output_val5k",
        "/tmp/custom_pretrain_val5k",
        "/tmp/custom_ft_val5k",
    ]


@pytest.mark.parametrize("validation_value", ["true", "True", "1", "yes"])
def test_feedforward_slurm_validation_mode_is_forwarded(validation_value):
    command = r'''
sbatch() {
  printf '%s|%s|%s|%s\n' \
    "$VALIDATION_MODE" "$FINAL_EVAL_ONLY" "$EXP_PREFIX" "$TEACHER_CKPT"
}
export -f sbatch
source ./launch_scripts/slurm_search_feedforward_config.sh
'''
    env = dict(
        os.environ,
        REPO_ROOT=str(ROOT),
        VALIDATION_MODE=validation_value,
        TC_FEEDFORWARD="true",
        MODEL_NAME="wrn_16_2_cifar",
        TASK="cifar10",
        IMG_TYPE="rgb",
        EXP_PREFIX="validation_smoke",
    )
    result = subprocess.run(
        ["bash", "-c", command], cwd=ROOT, env=env, text=True,
        capture_output=True, check=True)
    assert result.stdout.strip() == (
        "true|false|validation_smoke_val5k|"
        "./checkpoint/efficientnet_v2_l_cifar10_rgb_"
        "OldNoTimm_MatchDistill_val5k.pth")


def test_rgb_teacher_slurm_validation_paths_and_legacy_epoch_default():
    env = dict(
        os.environ,
        REPO_ROOT=str(ROOT),
        TEACHER_DRY_RUN="true",
        VALIDATION_MODE="true",
    )
    result = subprocess.run(
        ["bash", "launch_scripts/run_rgb_teacher.sbatch"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=True)
    assert "OldNoTimm_MatchDistill_val5k.pth" in result.stdout
    assert "OldNoTimm_MatchDistill_val5k.log" in result.stdout
    assert "--ne 15" in result.stdout


@pytest.mark.parametrize(
    ("module_name", "function_name", "argv"),
    [
        ("train_ode_cifar", "get_args", []),
        (
            "baseline.train_baseline_cifar", "parse_args",
            ["--model_name", "wrn_16_2_cifar", "--dataset", "cifar10"],
        ),
    ],
)
def test_legacy_entry_rejects_implicit_validation_mode(
        module_name, function_name, argv):
    code = f"""
import importlib
import sys
sys.argv = ["entry"] + {argv!r}
module = importlib.import_module({module_name!r})
getattr(module, {function_name!r})()
"""
    env = dict(os.environ, VALIDATION_MODE="true")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        text=True, capture_output=True)
    assert result.returncode == 2
    assert "did not pass --validation_mode explicitly" in result.stderr
