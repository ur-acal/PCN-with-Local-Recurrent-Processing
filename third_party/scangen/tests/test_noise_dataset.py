"""Unit tests for RawDataset and DataLoader functionality."""

from pathlib import Path
import random

import pytest
import torch
from torch.utils.data import Dataset

from scangen.data.noise_dataset import NoiseCIFARDataset
from scangen.device import get_device
from scangen.pipeline.noise import dnd_noise, sidd_noise


@pytest.mark.data
@pytest.mark.parametrize(
    "tenonehundred,traintest",
    [
        ("cifar10_raw", True), ("cifar10_raw", False),
        ("cifar10_rawdiffusion", True), ("cifar10_rawdiffusion", False),
        ("cifar100_raw", True), ("cifar100_raw", False),
        ("cifar100_rawdiffusion", True), ("cifar100_rawdiffusion", False),
    ]
    )
def test_live_raw_cifar(nn_weights, traintest, tenonehundred):
    """Test that __getitem__ returns correct format."""
    raw_dataset = NoiseCIFARDataset(
        root=Path("."), # The hdf5 files are in the base data directory.
        input_name=tenonehundred + ".h5",
        train=traintest,
        noise_config=dnd_noise(),
        device=get_device(),
    )
    train_cnt = len(raw_dataset.train_indices)
    test_cnt = len(raw_dataset.test_indices)
    assert len(raw_dataset) == train_cnt if traintest else test_cnt
    assert 10 * test_cnt >= train_cnt
    assert 2 * test_cnt <= train_cnt

    for i in random.sample(range(len(raw_dataset)), 5):
        _, clean_raw, noisy_raw, metadata = raw_dataset[i]

        # Check tensor shapes
        assert clean_raw.shape == (4, 16, 16)  # RGGB, half resolution
        assert noisy_raw.shape == (4, 16, 16)  # RGGB, half resolution
        assert isinstance(metadata["label"], int)
