"""Tests for CIFAR dataset loading and preprocessing."""

import pytest
import torch

from scangen.data.cifar_loader import (
    BatchSampler, CIFARDataset, CIFAR10Dataset, CIFAR100Dataset,
    create_cifar_loader, get_cifar_labels
)


class TestCIFARDataset:
    """Test cases for base CIFAR dataset functionality."""

    def test_cifar_dataset_initialization(self, mocker) -> None:
        """Test CIFAR dataset initialization."""
        mock_cifar10 = mocker.patch("torchvision.datasets.CIFAR10")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=100)
        mock_cifar10.return_value = mock_dataset

        dataset = CIFARDataset(dataset_name="cifar10", target_size=(128, 128))

        assert dataset.dataset_name == "cifar10"
        assert dataset.target_size == (128, 128)
        assert dataset.class_names == get_cifar_labels("cifar10")
        assert len(dataset) == 100

    def test_cifar100_dataset_initialization(self, mocker) -> None:
        """Test CIFAR-100 dataset initialization."""
        mock_cifar100 = mocker.patch("torchvision.datasets.CIFAR100")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=100)
        mock_cifar100.return_value = mock_dataset

        dataset = CIFARDataset(dataset_name="cifar100", target_size=(256, 256))

        assert dataset.dataset_name == "cifar100"
        assert dataset.target_size == (256, 256)
        assert dataset.class_names == get_cifar_labels("cifar100")

    def test_invalid_dataset_name(self) -> None:
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError):
            CIFARDataset(dataset_name="invalid_dataset")

    def test_upscale_image(self) -> None:
        """Test image upscaling functionality."""
        dataset = CIFARDataset.__new__(CIFARDataset)  # Create without __init__
        dataset.target_size = (64, 64)

        # Create test image (3, 32, 32)
        input_image = torch.randn(3, 32, 32)
        upscaled = dataset._upscale_image(input_image)

        assert upscaled.shape == (3, 64, 64)
        assert upscaled.dtype == input_image.dtype

    def test_make_divisible_by_16(self) -> None:
        """Test making images divisible by 16."""
        dataset = CIFARDataset.__new__(CIFARDataset)  # Create without __init__

        # Test image that needs cropping
        input_image = torch.randn(3, 100, 100)  # 100 = 6*16 + 4
        cropped = dataset._make_divisible_by_16(input_image)

        assert cropped.shape[1] % 16 == 0
        assert cropped.shape[2] % 16 == 0
        assert cropped.shape == (3, 96, 96)  # 6*16 = 96

    def test_make_divisible_by_16_already_divisible(self) -> None:
        """Test when image is already divisible by 16."""
        dataset = CIFARDataset.__new__(CIFARDataset)

        input_image = torch.randn(3, 64, 64)  # Already divisible by 16
        cropped = dataset._make_divisible_by_16(input_image)

        assert cropped.shape == (3, 64, 64)
        assert torch.equal(cropped, input_image)

    def test_create_gaussian_kernel(self) -> None:
        """Test Gaussian kernel creation."""
        dataset = CIFARDataset.__new__(CIFARDataset)

        kernel = dataset._create_gaussian_kernel(5, 1.0)

        assert kernel.shape == (3, 1, 5, 5)
        assert kernel.dtype == torch.float32

        # Test kernel normalization (each channel should sum to 1)
        for c in range(3):
            assert abs(kernel[c, 0].sum().item() - 1.0) < 1e-6

    def test_create_gaussian_kernel_even_size(self) -> None:
        """Test Gaussian kernel with even size (should be made odd)."""
        dataset = CIFARDataset.__new__(CIFARDataset)

        kernel = dataset._create_gaussian_kernel(4, 1.0)  # Even size

        assert kernel.shape == (3, 1, 5, 5)  # Should be 5x5 (4+1)


class TestCIFAR10Dataset:
    """Test cases for CIFAR-10 specific functionality."""

    def test_cifar10_dataset_creation(self, mocker) -> None:
        """Test CIFAR-10 dataset creation."""
        mock_cifar10 = mocker.patch("torchvision.datasets.CIFAR10")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=50000)
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(root="./test_data", train=True)

        assert dataset.dataset_name == "cifar10"
        assert dataset.class_names == get_cifar_labels("cifar10")
        assert len(dataset) == 50000

        # Check that CIFAR10 was called with correct parameters
        mock_cifar10.assert_called_once()
        call_args = mock_cifar10.call_args
        assert call_args[1]["root"] == "./test_data"
        assert call_args[1]["train"]

    def test_cifar10_getitem(self, mocker) -> None:
        """Test CIFAR-10 data loading."""
        mock_cifar10 = mocker.patch("torchvision.datasets.CIFAR10")
        mock_image = torch.rand(3, 32, 32)  # CIFAR-10 image
        mock_label = 5  # Class index
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=1)
        mock_dataset.__getitem__ = mocker.MagicMock(return_value=(mock_image, mock_label))
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(target_size=(64, 64), apply_blur=False)
        image, label, label_name = dataset[0]

        assert image.shape == (3, 64, 64)  # Upscaled and cropped
        assert label == 5
        assert label_name == get_cifar_labels("cifar10")[5]  # 'dog'


class TestCIFAR100Dataset:
    """Test cases for CIFAR-100 specific functionality."""

    def test_cifar100_dataset_creation(self, mocker) -> None:
        """Test CIFAR-100 dataset creation."""
        mock_cifar100 = mocker.patch("torchvision.datasets.CIFAR100")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=50000)
        mock_cifar100.return_value = mock_dataset

        dataset = CIFAR100Dataset(root="./test_data", train=False)

        assert dataset.dataset_name == "cifar100"
        assert dataset.class_names == get_cifar_labels("cifar100")
        assert len(dataset) == 50000

        # Check that CIFAR100 was called with correct parameters
        call_args = mock_cifar100.call_args
        assert not call_args[1]["train"]


class TestCreateCifarLoader:
    """Test cases for CIFAR loader creation."""

    def test_create_cifar10_loader(self, mocker) -> None:
        """Test CIFAR-10 loader creation."""
        mock_dataset_class = mocker.patch("scangen.data.cifar_loader.CIFAR10Dataset")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=100)
        mock_dataset_class.return_value = mock_dataset

        loader = create_cifar_loader(
            dataset_name="cifar10", batch_size=16, train=True, target_size=(128, 128)
        )

        assert loader.batch_size == 16
        mock_dataset_class.assert_called_once()

    def test_create_cifar100_loader(self, mocker) -> None:
        """Test CIFAR-100 loader creation."""
        mock_dataset_class = mocker.patch("scangen.data.cifar_loader.CIFAR100Dataset")
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__ = mocker.MagicMock(return_value=100)
        mock_dataset_class.return_value = mock_dataset

        loader = create_cifar_loader(
            dataset_name="cifar100", batch_size=8, train=False, num_workers=1
        )

        assert loader.batch_size == 8
        mock_dataset_class.assert_called_once()

    def test_create_loader_invalid_dataset(self) -> None:
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError):
            create_cifar_loader(dataset_name="invalid")


class TestGetCifarLabels:
    """Test cases for CIFAR label utilities."""

    def test_get_cifar10_labels(self) -> None:
        """Test getting CIFAR-10 labels."""
        labels = get_cifar_labels("cifar10")

        assert len(labels) == 10
        assert labels == get_cifar_labels("cifar10")
        assert "airplane" in labels
        assert "truck" in labels

    def test_get_cifar100_labels(self) -> None:
        """Test getting CIFAR-100 labels."""
        labels = get_cifar_labels("cifar100")

        assert len(labels) == 100
        assert labels == get_cifar_labels("cifar100")
        assert "beaver" in labels
        assert "tractor" in labels

    def test_get_labels_case_insensitive(self) -> None:
        """Test that label retrieval is case insensitive."""
        labels_lower = get_cifar_labels("cifar10")
        labels_upper = get_cifar_labels("CIFAR10")

        assert labels_lower == labels_upper

    def test_get_labels_invalid_dataset(self) -> None:
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError):
            get_cifar_labels("invalid_dataset")


class TestBatchSampler:
    """Test cases for batch sampling functionality."""

    def test_batch_sampler_initialization(self) -> None:
        """Test batch sampler initialization."""
        sampler = BatchSampler(dataset_size=1000, batch_size=32, seed=42)

        assert sampler.dataset_size == 1000
        assert sampler.batch_size == 32

    def test_sample_batch(self) -> None:
        """Test single batch sampling."""
        sampler = BatchSampler(dataset_size=100, batch_size=16, seed=42)

        batch = sampler.sample_batch()

        assert len(batch) == 16
        assert all(0 <= idx < 100 for idx in batch)

    def test_sample_batch_reproducible(self) -> None:
        """Test that sampling is reproducible with seeds."""
        sampler1 = BatchSampler(dataset_size=100, batch_size=10, seed=123)
        sampler2 = BatchSampler(dataset_size=100, batch_size=10, seed=123)

        batch1 = sampler1.sample_batch()
        batch2 = sampler2.sample_batch()

        assert batch1 == batch2

    def test_sample_batch_with_replacement(self) -> None:
        """Test that sampling allows replacement."""
        # Sample more items than dataset size to force replacement
        sampler = BatchSampler(dataset_size=5, batch_size=20, seed=42)

        batch = sampler.sample_batch()

        assert len(batch) == 20
        # Should have duplicates since we're sampling with replacement
        assert len(set(batch)) < 20

    def test_sample_multiple_batches(self) -> None:
        """Test sampling multiple batches."""
        sampler = BatchSampler(dataset_size=100, batch_size=8, seed=42)

        batches = sampler.sample_multiple_batches(num_batches=3)

        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 8
            assert all(0 <= idx < 100 for idx in batch)

    def test_sample_without_seed(self) -> None:
        """Test sampling without explicit seed."""
        sampler = BatchSampler(dataset_size=50, batch_size=5)

        batch1 = sampler.sample_batch()
        batch2 = sampler.sample_batch()

        assert len(batch1) == 5
        assert len(batch2) == 5
        # Different samples (with very high probability)
        assert batch1 != batch2


class TestCIFARConstants:
    """Test cases for CIFAR class constants."""

    def test_cifar10_classes_count(self) -> None:
        """Test CIFAR-10 has 10 classes."""
        assert len(get_cifar_labels("cifar10")) == 10

    def test_cifar100_classes_count(self) -> None:
        """Test CIFAR-100 has 100 classes."""
        assert len( get_cifar_labels("cifar100")) == 100

    def test_no_duplicate_cifar10_classes(self) -> None:
        """Test CIFAR-10 classes are unique."""
        assert len(get_cifar_labels("cifar10")) == len(set(get_cifar_labels("cifar10")))

    def test_no_duplicate_cifar100_classes(self) -> None:
        """Test CIFAR-100 classes are unique."""
        assert len(get_cifar_labels("cifar100")) == len(set(get_cifar_labels("cifar100")))

    def test_cifar_classes_are_strings(self) -> None:
        """Test that all class names are strings."""
        assert all(isinstance(name, str) for name in get_cifar_labels("cifar10"))
        assert all(isinstance(name, str) for name in get_cifar_labels("cifar100"))


class TestImagePreprocessing:
    """Test cases for image preprocessing functionality."""

    def test_gaussian_blur_application(self) -> None:
        """Test Gaussian blur application."""
        dataset = CIFARDataset.__new__(CIFARDataset)
        dataset.blur_kernel = torch.ones(3, 1, 3, 3) / 9.0  # Simple averaging kernel

        input_image = torch.ones(3, 10, 10)
        blurred = dataset._apply_gaussian_blur(input_image)

        assert blurred.shape == input_image.shape
        # Center pixels should be close to 1 (average of surrounding 1s)
        assert torch.allclose(blurred[:, 1:-1, 1:-1], torch.ones(3, 8, 8), atol=0.1)

    def test_preprocessing_pipeline_shapes(self) -> None:
        """Test that preprocessing maintains correct tensor shapes."""
        dataset = CIFARDataset.__new__(CIFARDataset)
        dataset.target_size = (80, 80)  # Will be cropped to 80x80 (divisible by 16)
        dataset.apply_blur = False
        dataset.blur_kernel = None

        # Simulate CIFAR input
        input_image = torch.rand(3, 32, 32)

        # Apply preprocessing steps
        upscaled = dataset._upscale_image(input_image)
        assert upscaled.shape == (3, 80, 80)

        divisible = dataset._make_divisible_by_16(upscaled)
        assert divisible.shape == (3, 80, 80)  # Already divisible by 16
        assert divisible.shape[1] % 16 == 0
        assert divisible.shape[2] % 16 == 0

    def test_edge_case_small_target_size(self) -> None:
        """Test handling of very small target sizes."""
        dataset = CIFARDataset.__new__(CIFARDataset)
        dataset.target_size = (15, 15)  # Smaller than original 32x32

        input_image = torch.rand(3, 32, 32)
        upscaled = dataset._upscale_image(input_image)

        assert upscaled.shape == (3, 15, 15)

        # Should be cropped to 0x0 since 15 < 16
        divisible = dataset._make_divisible_by_16(upscaled)
        assert divisible.shape == (3, 0, 0)

    def test_large_target_size(self) -> None:
        """Test handling of large target sizes."""
        dataset = CIFARDataset.__new__(CIFARDataset)
        dataset.target_size = (512, 512)

        input_image = torch.rand(3, 32, 32)
        upscaled = dataset._upscale_image(input_image)

        assert upscaled.shape == (3, 512, 512)

        divisible = dataset._make_divisible_by_16(upscaled)
        assert divisible.shape == (3, 512, 512)  # 512 is divisible by 16
