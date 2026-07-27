import json
from pathlib import Path

import pytest
import torch

from scangen.device import get_device
from scangen.generate_dataset import generate_dataset
from scangen.formats.config_format import create_default_config


class TestGenerateDataset:
    """Test cases for complete dataset generation."""

    @pytest.mark.data
    def test_generate_dataset_basic(self, data_dir, mocker) -> None:
        """Test basic dataset generation functionality."""
        # `generate_dataset()` reads from CIFAR, uses a RAWGenerator to transform
        # CIFAR to RAW, and then saves that RAW and metadata. This mocks reading
        # CIFAR and mocks the translation to RAW in order to test just the logic
        # of generate_dataset.
        # The model has one function to generate RAW from RGB.
        mock_model = mocker.MagicMock()
        # Set mock model return value to correct device
        def mock_forward(rgbs, labels, names):
            return (
                torch.rand(rgbs.shape[0], 4, rgbs.shape[2] // 2, rgbs.shape[3] // 2, device=rgbs.device),
                torch.rand(rgbs.shape[0], 4, rgbs.shape[2] // 2, rgbs.shape[3] // 2, device=rgbs.device),
                {"shot_noise": 0.2, "read_noise": 0.01}
            )
        mock_model.generate_batch = mock_forward

        # Mock data loader
        mock_rgb = torch.rand(2, 3, 128, 128)
        mock_labels = torch.tensor([0, 1])
        mock_names = ["airplane", "automobile"]
        rgb_batches = [(mock_rgb, mock_labels, mock_names)]

        config = create_default_config()
        config.dataset.batch_size = 2
        config.noise.type = "pointwise"

        stats = generate_dataset(
            config=config,
            rgb_batches=rgb_batches,
            model=mock_model,
        )

        assert stats["total_images"] == 2
        assert stats["total_batches"] == 1
        assert "generation_time" in stats
        assert "average_time_per_image" in stats

        out_dir = Path(config.output.directory)
        pkl_dir = out_dir / "pkl"
        assert (out_dir / "labels.csv").exists()
        assert (out_dir / "metadata.json").exists()
        assert pkl_dir.exists()

        pkl_files = list(pkl_dir.glob("*.pkl"))
        assert len(pkl_files) == 2
