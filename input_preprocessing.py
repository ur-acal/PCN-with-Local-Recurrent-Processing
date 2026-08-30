"""Shared student/teacher input preprocessing and run-directory metadata."""

import json
import os
import re

import torch
from torch.utils.data import Dataset


_IQ_TOKEN = re.compile(r"(?:^|_)iq(\d+)(?:_|$)", re.IGNORECASE)
_CTR_TOKEN = re.compile(r"(?:^|_)ctr(?:_|$)", re.IGNORECASE)


def quantize_unit_interval(inputs, bits=None):
    """Uniformly quantize [0, 1] using all 2**bits endpoint-inclusive codes."""
    if bits is None:
        return inputs
    bits = int(bits)
    if bits <= 0:
        raise ValueError("input_quant_bits must be a positive integer or none.")
    max_code = (1 << bits) - 1
    return torch.round(inputs.clamp(0.0, 1.0) * max_code) / max_code


def prepare_student_input(inputs, bits=None, center=False):
    inputs = quantize_unit_interval(inputs, bits)
    return inputs.mul(2.0).sub(1.0) if center else inputs


def preprocessing_suffix(bits=None, center=False):
    suffix = "" if bits is None else f"_iq{int(bits)}"
    return suffix + ("_ctr" if center else "")


def append_preprocessing_suffix(path, bits=None, center=False):
    suffix = preprocessing_suffix(bits, center)
    existing_bits, existing_center = infer_preprocessing_from_path(path)
    if (not suffix or
            (existing_bits == bits and existing_center == bool(center))):
        return str(path)
    return str(path) + suffix


def infer_preprocessing_from_path(path):
    """Infer human-visible `_iq<bits>` and `_ctr` tokens from a run path."""
    text = str(path or "").replace(os.sep, "_")
    matches = list(_IQ_TOKEN.finditer(text))
    bits = int(matches[-1].group(1)) if matches else None
    return bits, bool(_CTR_TOKEN.search(text))


def resolve_preprocessing(path, bits=None, center=None):
    inferred_bits, inferred_center = infer_preprocessing_from_path(path)
    return (
        inferred_bits if bits is None else int(bits),
        inferred_center if center is None else bool(center),
    )


def write_run_config(output_dir, bits=None, center=False):
    """Write human-readable metadata. Runtime code never reads this file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(
            {
                "input_quant_bits": bits,
                "center_student_input": bool(center),
                "student_input_flow": (
                    "Q_b(clamp(x, 0, 1)); then 2*x-1"
                    if center else "Q_b(clamp(x, 0, 1))"
                ),
                "note": "Human-readable metadata only; this file is not read by runtime code.",
            },
            handle,
            indent=2,
        )
        handle.write("\n")


class InputPreprocessedDataset(Dataset):
    """Apply student inference preprocessing to the first item of each sample."""

    def __init__(self, dataset, bits=None, center=False):
        self.dataset = dataset
        self.bits = bits
        self.center = bool(center)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)):
            return prepare_student_input(sample, self.bits, self.center)
        values = list(sample)
        values[0] = prepare_student_input(values[0], self.bits, self.center)
        return tuple(values) if isinstance(sample, tuple) else values
