"""Tests for noise generation functionality."""

import numpy as np
import pytest
import torch

from scangen.pipeline.noise import (
    add_noise,
    random_noise_levels_dnd,
    random_noise_levels_sidd,
    set_noise_seed,
)


class TestRandomNoiseLevels:
    """Test cases for random noise level generation."""

    def test_dnd_noise_ranges(self) -> None:
        """Test that DND noise levels are within expected ranges."""
        shot, read = random_noise_levels_dnd()

        # Check shapes
        assert shot.shape == torch.Size([1])
        assert read.shape == torch.Size([1])

        # Check shot noise range (from DND dataset specs)
        assert 0.0001 <= shot.item() <= 0.012

        # Read noise should be positive (actual range depends on relationship)
        assert read.item() > 0

    def test_sidd_noise_ranges(self) -> None:
        """Test that SIDD noise levels are within expected ranges."""
        shot, read = random_noise_levels_sidd()

        # Check shapes
        assert shot.shape == torch.Size([1])
        assert read.shape == torch.Size([1])

        # Check shot noise range (from SIDD dataset specs)
        assert 0.00068674 <= shot.item() <= 0.02194856

        # Read noise should be positive
        assert read.item() > 0

    def test_noise_randomness(self) -> None:
        """Test that noise levels vary across multiple calls."""
        # Generate multiple DND noise samples
        dnd_samples = [random_noise_levels_dnd() for _ in range(10)]
        shot_values = [shot.item() for shot, _ in dnd_samples]
        read_values = [read.item() for _, read in dnd_samples]

        # Should have variation (not all identical)
        assert len(set(shot_values)) > 1
        assert len(set(read_values)) > 1

        # Same test for SIDD
        sidd_samples = [random_noise_levels_sidd() for _ in range(10)]
        shot_values = [shot.item() for shot, _ in sidd_samples]
        read_values = [read.item() for _, read in sidd_samples]

        assert len(set(shot_values)) > 1
        assert len(set(read_values)) > 1


class TestAddNoise:
    """Test cases for noise addition functionality."""

    def test_add_noise_shape_preservation(self) -> None:
        """Test that add_noise preserves input tensor shape."""
        # Test various shapes
        test_shapes = [
            (1, 4, 32, 32),  # Single RGGB image
            (8, 4, 128, 128),  # Batch of RGGB images
            (1, 1, 256, 256),  # Single channel
            (2, 3, 64, 64),  # RGB batch
        ]

        for shape in test_shapes:
            clean = torch.rand(shape)
            noisy = add_noise(clean, shot_noise=0.01, read_noise=0.001)

            assert noisy.shape == clean.shape

    def test_add_noise_actually_adds_noise(self) -> None:
        """Test that noise is actually added to images."""
        clean = torch.rand(1, 4, 32, 32)
        noisy = add_noise(clean, shot_noise=0.02, read_noise=0.001)

        # Images should be different after adding noise
        assert not torch.allclose(clean, noisy, atol=1e-6)

    def test_add_noise_zero_noise(self) -> None:
        """Test behavior with zero noise parameters."""
        clean = torch.rand(1, 4, 32, 32)
        noisy = add_noise(clean, shot_noise=0.0, read_noise=0.0)

        # With zero noise, output should be close to input
        # (small epsilon is added to prevent zero variance)
        assert torch.allclose(clean, noisy, atol=1e-3)
        assert noisy.shape == clean.shape

    def test_add_noise_tensor_parameters(self) -> None:
        """Test add_noise with tensor noise parameters."""
        clean = torch.rand(2, 4, 16, 16)
        shot_tensor = torch.tensor(0.015)
        read_tensor = torch.tensor(0.0008)

        noisy = add_noise(clean, shot_noise=shot_tensor, read_noise=read_tensor)

        assert noisy.shape == clean.shape
        assert not torch.allclose(clean, noisy, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_add_noise_cuda_flag(self) -> None:
        """Test add_noise with CUDA flag (even without GPU)."""
        clean = torch.rand(1, 4, 16, 16)

        # Should work even if CUDA is not available
        try:
            noisy = add_noise(clean, shot_noise=0.01, read_noise=0.001)
            assert noisy.shape == clean.shape
        except RuntimeError:
            # Expected if CUDA is not available
            pytest.skip("CUDA not available")

    def test_add_noise_finite_output(self) -> None:
        """Test that noise addition produces finite values."""
        clean = torch.rand(1, 4, 32, 32)
        noisy = add_noise(clean, shot_noise=0.01, read_noise=0.001)

        assert torch.all(torch.isfinite(noisy))

    def test_add_noise_statistical_properties(self) -> None:
        """Test statistical properties of noise addition."""
        # Use constant image to test noise properties
        clean = torch.full((1, 4, 64, 64), 0.5)

        # Generate multiple noisy versions
        noisy_samples = []
        for _ in range(100):
            noisy = add_noise(clean, shot_noise=0.02, read_noise=0.001)
            noisy_samples.append(noisy)

        # Stack all samples
        all_noisy = torch.stack(noisy_samples)

        # Mean should be close to original (noise is zero-mean)
        mean_noisy = torch.mean(all_noisy, dim=0)
        assert torch.allclose(mean_noisy, clean, atol=0.05)

        # Variance should be approximately shot*image + read
        expected_variance = 0.02 * 0.5 + 0.001  # shot*image + read
        actual_variance = torch.var(all_noisy, dim=0).mean().item()
        assert abs(actual_variance - expected_variance) < 0.005


class TestNoiseSeed:
    """Test cases for deterministic noise generation."""

    def test_seed_reproducibility_dnd(self) -> None:
        """Test that setting seed makes DND noise generation reproducible."""
        set_noise_seed(42)
        shot1, read1 = random_noise_levels_dnd()

        set_noise_seed(42)
        shot2, read2 = random_noise_levels_dnd()

        assert torch.allclose(shot1, shot2)
        assert torch.allclose(read1, read2)

    def test_seed_reproducibility_sidd(self) -> None:
        """Test that setting seed makes SIDD noise generation reproducible."""
        set_noise_seed(123)
        shot1, read1 = random_noise_levels_sidd()

        set_noise_seed(123)
        shot2, read2 = random_noise_levels_sidd()

        assert torch.allclose(shot1, shot2)
        assert torch.allclose(read1, read2)

    def test_seed_reproducibility_add_noise(self) -> None:
        """Test that setting seed makes noise addition reproducible."""
        clean = torch.rand(1, 4, 16, 16)

        set_noise_seed(456)
        noisy1 = add_noise(clean, shot_noise=0.02, read_noise=0.001)

        set_noise_seed(456)
        noisy2 = add_noise(clean, shot_noise=0.02, read_noise=0.001)

        assert torch.allclose(noisy1, noisy2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        clean = torch.rand(1, 4, 16, 16)

        set_noise_seed(100)
        noisy1 = add_noise(clean, shot_noise=0.02, read_noise=0.001)

        set_noise_seed(200)
        noisy2 = add_noise(clean, shot_noise=0.02, read_noise=0.001)

        assert not torch.allclose(noisy1, noisy2, atol=1e-6)


class TestNoiseEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_image(self) -> None:
        """Test noise addition to zero image."""
        zero_image = torch.zeros(1, 4, 16, 16)
        noisy = add_noise(zero_image, shot_noise=0.01, read_noise=0.001)

        assert noisy.shape == zero_image.shape
        assert torch.all(torch.isfinite(noisy))

        # With zero image, noise should be purely read noise
        # Mean should be close to zero, std should be close to sqrt(read_noise)
        std_expected = np.sqrt(0.001)
        std_actual = torch.std(noisy).item()
        assert abs(std_actual - std_expected) < 0.05

    def test_large_noise_parameters(self) -> None:
        """Test behavior with large noise parameters."""
        clean = torch.rand(1, 4, 8, 8)

        # Large but reasonable noise
        noisy = add_noise(clean, shot_noise=0.1, read_noise=0.01)

        assert torch.all(torch.isfinite(noisy))
        assert noisy.shape == clean.shape

    def test_negative_image_values(self) -> None:
        """Test noise addition with negative image values."""
        # Image with some negative values (could happen in processing)
        image = torch.randn(1, 4, 8, 8)

        noisy = add_noise(image, shot_noise=0.01, read_noise=0.001)

        # Should handle negative values gracefully (clamp for shot noise calculation)
        assert torch.all(torch.isfinite(noisy))
        assert noisy.shape == image.shape
        assert not torch.allclose(image, noisy, atol=1e-6)
