"""Persistent, dataset-order-checked CIFAR train/validation splits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_VERSION = 1
DEFAULT_VALIDATION_SIZE = 5000
DEFAULT_SPLIT_SEED = 4096


def normalize_cifar_name(name: str) -> str:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported CIFAR dataset: {name}")
    return normalized


def extract_labels(dataset: Dataset) -> np.ndarray:
    """Return labels in the dataset's externally visible index order."""
    if hasattr(dataset, "targets"):
        labels = getattr(dataset, "targets")
    elif hasattr(dataset, "train_labels"):
        labels = getattr(dataset, "train_labels")
    elif hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
    elif hasattr(dataset, "_clean_file") and hasattr(dataset, "indices"):
        labels = np.asarray(dataset._clean_file["labels"][:])[np.asarray(dataset.indices)]
    elif isinstance(dataset, torch.utils.data.Subset):
        return extract_labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    elif hasattr(dataset, "dataset"):
        return extract_labels(dataset.dataset)
    else:
        raise AttributeError(
            f"Cannot extract labels from dataset type {type(dataset).__name__}")
    return np.asarray(torch.as_tensor(labels, dtype=torch.long).cpu(), dtype=np.int64)


def labels_checksum(labels: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(labels, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _payload_checksum(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_manifest_path(data_root: str | os.PathLike[str], dataset_name: str) -> Path:
    dataset_name = normalize_cifar_name(dataset_name)
    return Path(data_root).expanduser() / "validation_splits" / (
        f"{dataset_name}_train45k_val5k.json")


@dataclass(frozen=True)
class CIFARValidationSplit:
    dataset_name: str
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    manifest_path: Path
    manifest_checksum: str
    labels_checksum: str

    def checkpoint_metadata(self) -> dict:
        return {
            "mode": "validation",
            "dataset": self.dataset_name,
            "manifest_name": self.manifest_path.name,
            "manifest_checksum": self.manifest_checksum,
            "labels_checksum": self.labels_checksum,
            "train_size": len(self.train_indices),
            "validation_size": len(self.val_indices),
        }


def _stratified_indices(labels: np.ndarray, validation_size: int, seed: int) -> list[int]:
    classes = np.unique(labels)
    if validation_size % len(classes) != 0:
        raise ValueError(
            f"Validation size {validation_size} is not divisible by {len(classes)} classes")
    per_class = validation_size // len(classes)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for class_id in classes.tolist():
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size < per_class:
            raise ValueError(
                f"Class {class_id} has only {candidates.size} samples; need {per_class}")
        chosen.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return sorted(int(index) for index in chosen)


def _validate_payload(payload: dict, dataset_name: str, labels: np.ndarray,
                      validation_size: int) -> None:
    checksum = payload.get("manifest_checksum")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_checksum"}
    if checksum != _payload_checksum(unsigned):
        raise ValueError("CIFAR validation manifest checksum mismatch")
    expected = {
        "version": SPLIT_VERSION,
        "dataset": dataset_name,
        "training_size": int(labels.size),
        "validation_size": validation_size,
        "labels_checksum": labels_checksum(labels),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"CIFAR validation manifest {key} mismatch: "
                f"expected {value!r}, found {payload.get(key)!r}")
    indices = payload.get("validation_indices")
    if not isinstance(indices, list) or len(indices) != validation_size:
        raise ValueError("CIFAR validation manifest has the wrong number of indices")
    if len(set(indices)) != validation_size:
        raise ValueError("CIFAR validation manifest contains duplicate indices")
    if any(not isinstance(index, int) or index < 0 or index >= labels.size
           for index in indices):
        raise ValueError("CIFAR validation manifest contains out-of-range indices")
    class_counts = np.bincount(labels[np.asarray(indices)], minlength=int(labels.max()) + 1)
    expected_per_class = validation_size // class_counts.size
    if not np.all(class_counts == expected_per_class):
        raise ValueError(
            "CIFAR validation manifest is not class balanced: "
            f"expected {expected_per_class} per class")


def load_or_create_validation_split(
        dataset_name: str,
        labels: Sequence[int] | np.ndarray,
        data_root: str | os.PathLike[str] = "../data",
        validation_size: int = DEFAULT_VALIDATION_SIZE,
        seed: int = DEFAULT_SPLIT_SEED,
        manifest_path: str | os.PathLike[str] | None = None,
) -> CIFARValidationSplit:
    """Load a frozen split, or atomically create the standard stratified split."""
    dataset_name = normalize_cifar_name(dataset_name)
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1:
        raise ValueError("CIFAR labels must be one-dimensional")
    path = (Path(manifest_path).expanduser() if manifest_path is not None
            else default_manifest_path(data_root, dataset_name))
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        val_indices = _stratified_indices(labels_array, validation_size, seed)
        unsigned = {
            "version": SPLIT_VERSION,
            "dataset": dataset_name,
            "training_size": int(labels_array.size),
            "validation_size": validation_size,
            "method": "stratified_random",
            "seed": seed,
            "labels_checksum": labels_checksum(labels_array),
            "validation_indices": val_indices,
        }
        payload = dict(unsigned, manifest_checksum=_payload_checksum(unsigned))
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    _validate_payload(payload, dataset_name, labels_array, validation_size)
    val_indices = tuple(int(index) for index in payload["validation_indices"])
    val_set = set(val_indices)
    train_indices = tuple(index for index in range(labels_array.size) if index not in val_set)
    if len(train_indices) + len(val_indices) != labels_array.size:
        raise AssertionError("CIFAR validation split does not cover the training set")
    return CIFARValidationSplit(
        dataset_name=dataset_name,
        train_indices=train_indices,
        val_indices=val_indices,
        manifest_path=path,
        manifest_checksum=payload["manifest_checksum"],
        labels_checksum=payload["labels_checksum"],
    )


def require_matching_validation_split(
        checkpoint: dict,
        expected: dict | None,
        description: str = "checkpoint",
) -> None:
    """Reject a stage handoff made with another or missing validation split."""
    if expected is None:
        return
    actual = checkpoint.get("validation_split") if isinstance(checkpoint, dict) else None
    if not isinstance(actual, dict):
        raise ValueError(
            f"{description} does not contain validation-split metadata")
    for key in ("dataset", "manifest_checksum", "labels_checksum",
                "train_size", "validation_size"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"{description} validation split mismatch for {key}: "
                f"expected {expected.get(key)!r}, found {actual.get(key)!r}")


class IndexedSubset(Dataset):
    """Subset retaining labels for distillation and contrastive samplers."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.labels = torch.as_tensor(
            extract_labels(dataset)[np.asarray(self.indices)], dtype=torch.long)
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]
