"""Tests for RAW data generation pipeline."""

from pathlib import Path

import pytest
import torch

from scangen.device import get_device
from scangen.pipeline.generator import RAWGenerator
from scangen.pipeline.noise import sidd_noise

class TestRAWGenerator:
    """Test cases for RAW data generator."""

    def test_raw_generator_init_happy(self, mocker, nn_weights, scangen_data) -> None:
        """Test RAW generator initialization."""
        mocker.patch.object(RAWGenerator, "_load_model")
        # Test using the relative path, not the full path `nn_weights`.
        model_path = (scangen_data / "weights/rgb2raw.pth").resolve()
        generator = RAWGenerator(model_path, get_device(), sidd_noise(), (16, 16))
        generator._load_model.assert_called_once() # type: ignore[attr-defined]

    def test_raw_generator_init_missing(self, mocker) -> None:
        """Test RAW generator initialization."""
        mocker.patch.object(RAWGenerator, "_load_model")
        with pytest.raises(FileNotFoundError):
            RAWGenerator(Path("weights/MISSING.pth"), get_device(), sidd_noise(), (128, 128))

    def test_generate_batch_shape(self, mocker, data_dir) -> None:
        """Test batch generation produces correct shapes."""
        fake_weights = data_dir / "weights" / "faked.pth"
        fake_weights.parent.mkdir(parents=True, exist_ok=True)
        with fake_weights.open("w") as fw:
            fw.write("Fake weights")
        mocker.patch.object(RAWGenerator, "_load_model")
        # Mock the model to return expected RAW shape on correct device
        width, height = (16, 16)
        def mock_forward(x):
            return torch.rand(
                x.shape[0], 4, width, height, device=x.device
            )

        generator = RAWGenerator(
            fake_weights,
            device=get_device(),
            noise=sidd_noise(),
            output_dim=(16, 16),
            seed=2934723,
        )
        generator._load_model.assert_called_once() # type: ignore[attr-defined]
        # Fake a generator.model
        generator.model = mock_forward  # type: ignore[assignment]

        # Test input
        rgb_images = torch.rand(2, 3, 256, 256)
        labels = torch.tensor([0, 1])
        label_names = ["class_0", "class_1"]

        clean_raw, noisy_raw, metadata = generator.generate_batch(
            rgb_images, labels, label_names
        )

        assert clean_raw.shape == (2, 4, width, height)
        assert noisy_raw.shape == (2, 4, width, height)
        assert metadata["batch_size"] == 2
        assert metadata["labels"] == [0, 1]
        assert metadata["label_names"] == ["class_0", "class_1"]
