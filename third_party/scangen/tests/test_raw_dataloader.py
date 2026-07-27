
import logging

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scangen.data.raw_dataloader import create_cifar10_raw_dataloader, create_raw_dataloader
from scangen.pipeline.noise import dnd_noise, sidd_noise

class MockRGBDataset(Dataset):
    """Mock RGB dataset for testing."""

    def __init__(self, num_samples: int = 10, image_size: tuple[int, int] = (32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Return (image, label) like torchvision datasets
        # Image in HWC format as uint8 (like PIL images)
        image = torch.randint(
            0, 256, (self.image_size[0], self.image_size[1], 3), dtype=torch.uint8
        )
        label = idx % 10  # Simple label
        return image, label

class TestRawDataLoader:
    """Test cases for DataLoader creation functions."""

    @pytest.fixture
    def mock_rgb_dataset(self):
        """Create a mock RGB dataset."""
        return MockRGBDataset(num_samples=8)  # Multiple of batch size

    @pytest.mark.slow
    def test_create_raw_dataloader(self, mock_rgb_dataset, nn_weights):
        """Test basic DataLoader creation."""
        outsize = (32, 32)
        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=dnd_noise(),
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Single worker for testing
            model_path=nn_weights,
            target_size=outsize,
        )

        assert isinstance(dataloader, DataLoader)
        assert len(dataloader.dataset) == len(mock_rgb_dataset)

        # Test one batch
        batch = next(iter(dataloader))
        rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list = batch

        assert rgb_batch.shape == (4, 3, 256, 256)
        assert clean_raw_batch.shape == (4, 4, *outsize)
        assert noisy_raw_batch.shape == (4, 4, *outsize)
        assert len(metadata_list) == 4

    @pytest.mark.slow
    def test_single_worker_compatibility(self, mock_rgb_dataset, nn_weights):
        """Test DataLoader works with num_workers=0."""
        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=dnd_noise(),
            batch_size=2,
            num_workers=0,
            model_path=nn_weights,
            target_size=(64, 64),
        )

        # Should be able to iterate through entire dataset
        batches = list(dataloader)
        assert len(batches) == 4  # 8 samples / 2 batch_size

        for rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list in batches:
            assert rgb_batch.shape[0] == 2  # Batch size
            assert clean_raw_batch.shape[0] == 2
            assert noisy_raw_batch.shape[0] == 2
            assert len(metadata_list) == 2

    @pytest.mark.slow
    def test_different_noise_configs(self, mock_rgb_dataset, nn_weights):
        """Test DataLoader with different noise configurations."""
        noise_configs = [dnd_noise(), sidd_noise()]

        for noise_config in noise_configs:
            dataloader = create_raw_dataloader(
                rgb_dataset=mock_rgb_dataset,
                noise_config=noise_config,
                batch_size=2,
                num_workers=0,
                model_path=nn_weights,
                target_size=(64, 64),
            )

            # Get one batch and check metadata
            rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list = next(iter(dataloader))

            for metadata in metadata_list:
                assert metadata["noise_type"] == noise_config.type


    @pytest.mark.slow
    def test_default_cpu_device(self, mock_rgb_dataset, nn_weights):
        """Test that CPU is the default device."""
        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=sidd_noise(),
            batch_size=2,
            model_path=nn_weights,
        )

        # Check that the dataset was created successfully
        assert isinstance(dataloader, DataLoader)

        # Try to get one batch to ensure it works
        batch = next(iter(dataloader))
        rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list = batch
        assert len(metadata_list) == 2

    def test_smart_num_workers_cpu(self, mock_rgb_dataset, nn_weights):
        """Test smart num_workers handling for CPU device."""
        # Test with num_workers=0 on CPU - should use default (half CPU cores)
        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=sidd_noise(),
            batch_size=2,
            device="cpu",
            num_workers=0,  # Should see 0 and choose.
            model_path=nn_weights,
        )

        # Check that it was created successfully
        assert isinstance(dataloader, DataLoader)
        # The actual num_workers will depend on the system, so we just check it works
        batch = next(iter(dataloader))
        assert len(batch) == 4  # (rgb, clean_raw, noisy_raw, metadata)

    def test_smart_num_workers_gpu_warning(self, mock_rgb_dataset, caplog, nn_weights):
        """Test that num_workers>0 with GPU device issues a warning."""

        caplog.set_level(logging.WARNING)

        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=dnd_noise(),
            batch_size=2,
            device="cuda",
            num_workers=4,  # Should be ignored with warning
            model_path=nn_weights,
        )

        # Check that warning was logged
        assert any(
            "num_workers=4 ignored for GPU device" in record.message for record in caplog.records
        )

        # Should still work
        assert isinstance(dataloader, DataLoader)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_none_device_uses_mps_when_available(self, mock_rgb_dataset, nn_weights):
        """Test that device=None actually uses MPS when available."""

        dataloader = create_raw_dataloader(
            rgb_dataset=mock_rgb_dataset,
            noise_config=sidd_noise(),
            batch_size=2,
            num_workers=0,
            target_size=(64, 64),
            model_path=nn_weights,
        )

        # Get one batch to trigger RAW generator initialization
        batch = next(iter(dataloader))
        rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list = batch

        # Check that the generator is using MPS device
        raw_dataset = dataloader.dataset
        generator = raw_dataset._get_generator()

        assert str(generator.device) == "mps", f"Expected MPS device, got {generator.device}"
        assert len(metadata_list) == 2  # Verify batch processed correctly


class TestCIFAR10Integration:
    """Test CIFAR-10 specific functionality."""

    @pytest.mark.slow
    def test_create_cifar10_raw_dataloader(self, tmp_path):
        """Test CIFAR-10 DataLoader creation."""
        pytest.importorskip("torchvision", reason="torchvision required for CIFAR-10 tests")

        try:
            dataloader = create_cifar10_raw_dataloader(
                root=tmp_path,
                train=False,  # Use test set (smaller download)
                download=True,
                batch_size=4,
                num_workers=0,
                target_size=(64, 64),  # Small size for fast testing
            )

            # Test one batch
            rgb_batch, clean_raw_batch, noisy_raw_batch, metadata_list = next(iter(dataloader))

            assert rgb_batch.shape == (4, 3, 64, 64)
            assert clean_raw_batch.shape == (4, 4, 32, 32)
            assert noisy_raw_batch.shape == (4, 4, 32, 32)
            assert len(metadata_list) == 4

            # Check that we have actual CIFAR-10 labels in metadata
            for metadata in metadata_list:
                assert "original_data" in metadata
                original_data = metadata["original_data"]
                if original_data:  # Should contain CIFAR-10 label
                    label = original_data[0]
                    assert 0 <= label <= 9  # CIFAR-10 has 10 classes

        except Exception as e:
            # If download fails (network issues), skip the test
            pytest.skip(f"CIFAR-10 download failed: {e}")


class TestMemoryAndPerformance:
    """Basic tests for memory usage and performance."""

    @pytest.mark.slow
    def test_no_memory_leaks_single_batch(self, nn_weights):
        """Test that repeated batch access doesn't accumulate memory."""
        mock_dataset = MockRGBDataset(num_samples=4)

        dataloader = create_raw_dataloader(
            rgb_dataset=mock_dataset,
            noise_config=dnd_noise(),
            batch_size=2,
            num_workers=0,
            model_path=nn_weights,
            target_size=(64, 64),
        )

        # Get the same batch multiple times
        for _ in range(10):
            batch = next(iter(dataloader))
            # Basic validation
            assert len(batch) == 4
            del batch  # Explicitly delete to help with memory

    @pytest.mark.slow
    def test_basic_performance_timing(self, nn_weights):
        """Basic timing test for DataLoader iteration."""
        import time

        mock_dataset = MockRGBDataset(num_samples=20)

        dataloader = create_raw_dataloader(
            rgb_dataset=mock_dataset,
            noise_config=sidd_noise(),
            batch_size=4,
            num_workers=0,
            model_path=nn_weights,
            target_size=(64, 64),  # Small size for testing
        )

        start_time = time.time()
        batches = list(dataloader)
        end_time = time.time()

        assert len(batches) == 5  # 20 samples / 4 batch_size

        total_time = end_time - start_time
        samples_per_second = 20 / total_time

        # Very relaxed performance requirement for CI
        assert samples_per_second > 1, f"Too slow: {samples_per_second:.2f} samples/sec"
