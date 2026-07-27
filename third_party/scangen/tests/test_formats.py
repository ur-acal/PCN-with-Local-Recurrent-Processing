"""Tests for RAW image format utilities."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from scangen.formats.raw_formats import (
    get_bayer_pattern_info,
    is_image_file,
    is_raw_file,
    load_image,
    load_raw_dict,
    pack_raw,
    save_raw_dict,
    save_raw_png,
    unpack_raw,
)


class TestUnpackRaw:
    """Test cases for unpacking RGGB to Bayer pattern."""

    def test_unpack_raw_shape(self) -> None:
        """Test that unpack_raw produces correct output shape."""
        # Test various batch sizes and dimensions
        test_cases = [
            (1, 4, 32, 32),  # Single image
            (4, 4, 64, 64),  # Small batch
            (8, 4, 16, 16),  # Large batch, small images
        ]

        for batch_size, channels, h, w in test_cases:
            packed = torch.randn(batch_size, channels, h, w)
            bayer = unpack_raw(packed)
            assert isinstance(bayer, torch.Tensor)

            expected_shape = (batch_size, 1, h * 2, w * 2)
            assert bayer.shape == expected_shape

    def test_unpack_raw_channel_placement(self) -> None:
        """Test that RGGB channels are placed in correct Bayer positions."""
        # Create test pattern with distinct values for each channel
        packed = torch.zeros(1, 4, 2, 2)
        packed[0, 0, :, :] = 1.0  # Red channel
        packed[0, 1, :, :] = 2.0  # Green_Red channel
        packed[0, 2, :, :] = 3.0  # Green_Blue channel
        packed[0, 3, :, :] = 4.0  # Blue channel

        bayer = unpack_raw(packed)

        # Check Bayer pattern: R G R G
        #                     G B G B
        #                     R G R G
        #                     G B G B

        # Red positions (even rows, even cols)
        assert torch.all(bayer[0, 0, 0::2, 0::2] == 1.0)

        # Green_Red positions (even rows, odd cols)
        assert torch.all(bayer[0, 0, 0::2, 1::2] == 2.0)

        # Green_Blue positions (odd rows, even cols)
        assert torch.all(bayer[0, 0, 1::2, 0::2] == 3.0)

        # Blue positions (odd rows, odd cols)
        assert torch.all(bayer[0, 0, 1::2, 1::2] == 4.0)

    def test_unpack_raw_dtype_preservation(self) -> None:
        """Test that unpack_raw preserves input dtype."""
        packed_float32 = torch.randn(1, 4, 16, 16, dtype=torch.float32)
        bayer_float32 = unpack_raw(packed_float32)
        assert bayer_float32.dtype == torch.float32

        packed_float16 = torch.randn(1, 4, 16, 16, dtype=torch.float16)
        bayer_float16 = unpack_raw(packed_float16)
        assert bayer_float16.dtype == torch.float16

    def test_unpack_raw_device_preservation(self) -> None:
        """Test that unpack_raw preserves device."""
        packed_cpu = torch.randn(1, 4, 8, 8)
        bayer_cpu = unpack_raw(packed_cpu)
        assert bayer_cpu.device == packed_cpu.device

        # Test with CUDA if available
        if torch.cuda.is_available():
            packed_cuda = torch.randn(1, 4, 8, 8).cuda()
            bayer_cuda = unpack_raw(packed_cuda)
            assert bayer_cuda.device == packed_cuda.device

    def test_unpack_raw_invalid_channels(self) -> None:
        """Test error handling for invalid channel count."""
        with pytest.raises(AssertionError):
            packed_wrong = torch.randn(1, 3, 16, 16)  # Wrong channel count
            unpack_raw(packed_wrong)


class TestPackRaw:
    """Test cases for packing Bayer pattern to RGGB."""

    def test_pack_raw_torch_shape(self) -> None:
        """Test pack_raw with torch tensors."""
        # Test 2D input
        bayer_2d = torch.randn(128, 128)
        packed = pack_raw(bayer_2d)
        assert packed.shape == (4, 64, 64)

        # Test 3D input with channel dimension
        bayer_3d = torch.randn(1, 128, 128)
        packed = pack_raw(bayer_3d)
        assert packed.shape == (4, 64, 64)

        # Test 4D input with batch and channel dimensions
        bayer_4d = torch.randn(1, 1, 128, 128)
        packed = pack_raw(bayer_4d)
        assert packed.shape == (4, 64, 64)

    def test_pack_raw_numpy_shape(self) -> None:
        """Test pack_raw with numpy arrays."""
        # Test 2D input
        bayer_2d = np.random.rand(128, 128)
        packed = pack_raw(bayer_2d)
        assert packed.shape == (64, 64, 4)

        # Test 3D input with channel dimension
        bayer_3d = np.random.rand(128, 128, 1)
        packed = pack_raw(bayer_3d)
        assert packed.shape == (64, 64, 4)

    def test_pack_unpack_round_trip_torch(self) -> None:
        """Test that pack -> unpack is reversible for torch tensors."""
        # Start with packed RGGB
        original_packed = torch.randn(1, 4, 32, 32)

        # Unpack to Bayer, then pack again
        bayer = unpack_raw(original_packed)
        assert isinstance(bayer, torch.Tensor)
        repacked = pack_raw(bayer.squeeze(0).squeeze(0))  # Remove batch and channel dims

        assert isinstance(repacked, torch.Tensor)
        # Should be close to original (within numerical precision)
        assert torch.allclose(original_packed.squeeze(0), repacked, atol=1e-6)

    def test_pack_unpack_round_trip_numpy(self) -> None:
        """Test that pack -> unpack is reversible for numpy arrays."""
        # Create a simple test pattern
        bayer = np.zeros((4, 4))
        bayer[0, 0] = 1.0  # Red position
        bayer[0, 1] = 2.0  # Green position
        bayer[1, 0] = 3.0  # Green position
        bayer[1, 1] = 4.0  # Blue position
        bayer[0, 2] = 1.0  # Red position
        bayer[0, 3] = 2.0  # Green position
        bayer[1, 2] = 3.0  # Green position
        bayer[1, 3] = 4.0  # Blue position
        bayer[2, 0] = 1.0  # Red position
        bayer[2, 1] = 2.0  # Green position
        bayer[3, 0] = 3.0  # Green position
        bayer[3, 1] = 4.0  # Blue position
        bayer[2, 2] = 1.0  # Red position
        bayer[2, 3] = 2.0  # Green position
        bayer[3, 2] = 3.0  # Green position
        bayer[3, 3] = 4.0  # Blue position

        # Pack to RGGB
        packed = pack_raw(bayer)  # Shape: (2, 2, 4)

        # Check that channels extracted correctly
        assert np.allclose(packed[:, :, 0], 1.0)  # Red channel
        assert np.allclose(packed[:, :, 1], 2.0)  # Green_Red channel
        assert np.allclose(packed[:, :, 2], 3.0)  # Green_Blue channel
        assert np.allclose(packed[:, :, 3], 4.0)  # Blue channel

    def test_pack_raw_channel_extraction(self) -> None:
        """Test that correct pixels are extracted for each channel."""
        # Create 4x4 Bayer pattern with known values
        bayer = torch.zeros(4, 4)

        # Set specific values at known positions
        bayer[0, 0] = 10.0  # Red
        bayer[0, 2] = 11.0  # Red
        bayer[2, 0] = 12.0  # Red
        bayer[2, 2] = 13.0  # Red

        bayer[0, 1] = 20.0  # Green_Red
        bayer[0, 3] = 21.0  # Green_Red
        bayer[2, 1] = 22.0  # Green_Red
        bayer[2, 3] = 23.0  # Green_Red

        bayer[1, 0] = 30.0  # Green_Blue
        bayer[1, 2] = 31.0  # Green_Blue
        bayer[3, 0] = 32.0  # Green_Blue
        bayer[3, 2] = 33.0  # Green_Blue

        bayer[1, 1] = 40.0  # Blue
        bayer[1, 3] = 41.0  # Blue
        bayer[3, 1] = 42.0  # Blue
        bayer[3, 3] = 43.0  # Blue

        packed = pack_raw(bayer)  # Shape: (4, 2, 2)

        # Check red channel extraction
        expected_red = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
        assert torch.allclose(packed[0], expected_red)

        # Check green_red channel extraction
        expected_green_red = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
        assert torch.allclose(packed[1], expected_green_red)

        # Check green_blue channel extraction
        expected_green_blue = torch.tensor([[30.0, 31.0], [32.0, 33.0]])
        assert torch.allclose(packed[2], expected_green_blue)

        # Check blue channel extraction
        expected_blue = torch.tensor([[40.0, 41.0], [42.0, 43.0]])
        assert torch.allclose(packed[3], expected_blue)


class TestRawDictIO:
    """Test cases for RAW dictionary save/load operations."""

    def test_save_load_raw_dict(self) -> None:
        """Test saving and loading RAW data dictionaries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_raw.pkl"

            # Create test data
            original_dict = {
                "clean": np.random.rand(4, 64, 64).astype(np.float32),
                "noisy": np.random.rand(4, 64, 64).astype(np.float32),
                "variance": np.random.rand(4, 64, 64).astype(np.float32),
                "metadata": {"camera": "test", "iso": 800},
            }

            # Save and load
            save_raw_dict(original_dict, filepath)
            assert filepath.exists()

            loaded_dict = load_raw_dict(filepath)

            # Check all keys present
            assert set(loaded_dict.keys()) == set(original_dict.keys())

            for key in ["clean", "noisy", "variance"]:
                assert np.array_equal(loaded_dict[key], original_dict[key])  # type: ignore[arg-type]

            # Check nested metadata
            assert loaded_dict["metadata"] == original_dict["metadata"]

    def test_save_raw_dict_creates_directory(self) -> None:
        """Test that save_raw_dict creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "subdir" / "nested" / "test.pkl"

            test_dict = {"data": np.random.rand(4, 16, 16)}
            save_raw_dict(test_dict, nested_path)

            assert nested_path.exists()
            loaded = load_raw_dict(nested_path)
            assert np.array_equal(loaded["data"], test_dict["data"])


class TestImageIO:
    """Test cases for image loading and PNG output."""

    def test_load_image_formats(self) -> None:
        """Test loading different image formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test images in different formats
            test_data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            test_image = Image.fromarray(test_data)

            # Save in different formats
            png_path = temp_path / "test.png"
            jpg_path = temp_path / "test.jpg"

            test_image.save(png_path)
            test_image.save(jpg_path)

            # Load and test
            for img_path in [png_path, jpg_path]:
                loaded = load_image(img_path)

                assert loaded.shape == (64, 64, 3)
                assert loaded.dtype == np.float32
                assert 0 <= loaded.min() <= loaded.max() <= 1

    def test_save_raw_png_packed(self) -> None:
        """Test saving packed RAW data as PNG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_raw.png"

            # Create test packed RAW data
            raw_data = torch.rand(4, 32, 32)  # RGGB format

            save_raw_png(raw_data, output_path, packed=True)

            assert output_path.exists()

            # Load and check it's a valid image
            loaded = Image.open(output_path)
            assert loaded.size == (64, 64)  # Unpacked size
            assert loaded.mode in ["L", "RGB"]  # Grayscale or RGB

    def test_save_raw_png_bayer(self) -> None:
        """Test saving Bayer pattern data as PNG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_bayer.png"

            # Create test Bayer data
            bayer_data = torch.rand(1, 64, 64)

            save_raw_png(bayer_data, output_path, packed=False)

            assert output_path.exists()

            # Load and check
            loaded = Image.open(output_path)
            assert loaded.size == (64, 64)

    def test_save_raw_png_numpy(self) -> None:
        """Test saving numpy RAW data as PNG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_numpy.png"

            # Create numpy test data
            raw_data = np.random.rand(4, 16, 16).astype(np.float32)

            save_raw_png(raw_data, output_path, packed=True)

            assert output_path.exists()

    def test_save_raw_png_creates_directory(self) -> None:
        """Test that save_raw_png creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "subdir" / "test.png"

            raw_data = torch.rand(4, 8, 8)
            save_raw_png(raw_data, nested_path, packed=True)

            assert nested_path.exists()


class TestFileUtils:
    """Test cases for file utility functions."""

    def test_is_raw_file(self) -> None:
        """Test RAW file detection."""
        assert is_raw_file("data.pkl")
        assert is_raw_file("path/to/data.pkl")
        assert is_raw_file(Path("data.pkl"))

        assert not is_raw_file("image.png")
        assert not is_raw_file("document.txt")
        assert is_raw_file("data.PKL")  # Case insensitive

    def test_is_image_file(self) -> None:
        """Test image file detection."""
        assert is_image_file("image.png")
        assert is_image_file("photo.jpg")
        assert is_image_file("picture.jpeg")
        assert is_image_file("Image.PNG")  # Case insensitive

        assert not is_image_file("data.pkl")
        assert not is_image_file("document.txt")
        assert not is_image_file("video.mp4")

    def test_get_bayer_pattern_info(self) -> None:
        """Test Bayer pattern information lookup."""
        # Test RGGB pattern
        rggb_info = get_bayer_pattern_info("rggb")
        assert rggb_info["red"] == (0, 0)
        assert rggb_info["green1"] == (0, 1)
        assert rggb_info["green2"] == (1, 0)
        assert rggb_info["blue"] == (1, 1)

        # Test BGGR pattern
        bggr_info = get_bayer_pattern_info("bggr")
        assert bggr_info["blue"] == (0, 0)
        assert bggr_info["green1"] == (0, 1)
        assert bggr_info["green2"] == (1, 0)
        assert bggr_info["red"] == (1, 1)

        # Test case insensitivity
        rggb_upper = get_bayer_pattern_info("RGGB")
        assert rggb_upper == rggb_info

        # Test invalid pattern
        with pytest.raises(ValueError):
            get_bayer_pattern_info("invalid")


class TestFormatEdgeCases:
    """Test edge cases and error conditions."""

    def test_unpack_raw_single_pixel(self) -> None:
        """Test unpacking with minimal size."""
        packed = torch.randn(1, 4, 1, 1)
        bayer = unpack_raw(packed)

        assert bayer.shape == (1, 1, 2, 2)

    def test_pack_raw_single_pixel(self) -> None:
        """Test packing with minimal size."""
        bayer = torch.randn(2, 2)
        packed = pack_raw(bayer)

        assert packed.shape == (4, 1, 1)

    def test_save_load_empty_dict(self) -> None:
        """Test saving and loading empty dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "empty.pkl"

            empty_dict: dict[str, Any] = {}
            save_raw_dict(empty_dict, filepath)

            loaded = load_raw_dict(filepath)
            assert loaded == {}

    def test_raw_png_extreme_values(self) -> None:
        """Test PNG saving with extreme pixel values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "extreme.png"

            # Create data with extreme values
            raw_data = torch.tensor(
                [
                    [
                        [[-10.0, 100.0], [0.001, 50.0]],
                        [[0.0, 1.0], [0.5, 0.9]],
                        [[1000.0, -5.0], [2.5, 0.1]],
                        [[0.2, 0.8], [100.0, -100.0]],
                    ]
                ]
            )

            # Should handle normalization gracefully for extreme values
            save_raw_png(raw_data, output_path, packed=True, normalize=True)
            assert output_path.exists()

    def test_load_nonexistent_file(self) -> None:
        """Test error handling for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            load_raw_dict("nonexistent.pkl")

        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.png")
