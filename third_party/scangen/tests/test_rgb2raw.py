"""Tests for RGB to RAW conversion pipeline."""

import torch

from scangen.pipeline.rgb2raw import Rgb2Raw, conv, mosaic


class TestMosaicFunction:
    """Test cases for the mosaic function."""

    def test_mosaic_shape(self) -> None:
        """Test that mosaic produces correct output shape."""
        rgb = torch.randn(2, 3, 32, 32)
        rggb = mosaic(rgb)

        expected_shape = (2, 4, 16, 16)
        assert rggb.shape == expected_shape

    def test_mosaic_channels(self) -> None:
        """Test that mosaic extracts correct channel patterns."""
        # Create a simple test pattern where each channel has distinct values
        rgb = torch.zeros(1, 3, 4, 4)
        rgb[0, 0, :, :] = 1.0  # Red channel = 1
        rgb[0, 1, :, :] = 2.0  # Green channel = 2
        rgb[0, 2, :, :] = 3.0  # Blue channel = 3

        rggb = mosaic(rgb)

        # Check that red pixels are extracted correctly (even rows, even cols)
        assert torch.all(rggb[0, 0, :, :] == 1.0)

        # Check that green pixels are extracted correctly
        assert torch.all(rggb[0, 1, :, :] == 2.0)  # Even rows, odd cols
        assert torch.all(rggb[0, 2, :, :] == 2.0)  # Odd rows, even cols

        # Check that blue pixels are extracted correctly (odd rows, odd cols)
        assert torch.all(rggb[0, 3, :, :] == 3.0)

    def test_mosaic_dtype_preservation(self) -> None:
        """Test that mosaic preserves input dtype."""
        rgb_float32 = torch.randn(1, 3, 8, 8, dtype=torch.float32)
        rggb_float32 = mosaic(rgb_float32)
        assert rggb_float32.dtype == torch.float32

        rgb_float16 = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        rggb_float16 = mosaic(rgb_float16)
        assert rggb_float16.dtype == torch.float16


class TestConvFunction:
    """Test cases for the conv helper function."""

    def test_conv_default_padding(self) -> None:
        """Test that conv function creates layer with correct padding."""
        layer = conv(3, 64, 3)

        assert isinstance(layer, torch.nn.Conv2d)
        assert layer.in_channels == 3
        assert layer.out_channels == 64
        assert layer.kernel_size == (3, 3)
        assert layer.padding == (1, 1)

    def test_conv_custom_parameters(self) -> None:
        """Test conv function with custom parameters."""
        layer = conv(16, 32, 5, bias=False, stride=2)

        assert layer.in_channels == 16
        assert layer.out_channels == 32
        assert layer.kernel_size == (5, 5)
        assert layer.padding == (2, 2)
        assert layer.bias is None
        assert layer.stride == (2, 2)


class TestRgb2Raw:
    """Test cases for the Rgb2Raw network."""

    def test_rgb2raw_initialization(self) -> None:
        """Test that Rgb2Raw network initializes without errors."""
        model = Rgb2Raw()
        assert isinstance(model, torch.nn.Module)

    def test_rgb2raw_forward_shape(self) -> None:
        """Test that Rgb2Raw produces correct output shape."""
        model = Rgb2Raw()
        model.eval()

        # Test with different input sizes
        test_cases = [
            (1, 3, 32, 32),
            (2, 3, 64, 64),
            (1, 3, 128, 128),
        ]

        for batch_size, channels, height, width in test_cases:
            rgb_input = torch.randn(batch_size, channels, height, width)

            with torch.no_grad():
                rggb_output = model(rgb_input)

            expected_shape = (batch_size, 4, height // 2, width // 2)
            assert rggb_output.shape == expected_shape, (
                f"Expected {expected_shape}, got {rggb_output.shape}"
            )

    def test_rgb2raw_output_range(self) -> None:
        """Test that Rgb2Raw output is in reasonable range."""
        model = Rgb2Raw()
        model.eval()

        # Input in typical image range [0, 1]
        rgb_input = torch.rand(1, 3, 32, 32)

        with torch.no_grad():
            rggb_output = model(rgb_input)

        # Output should be finite
        assert torch.all(torch.isfinite(rggb_output))

        # Output shouldn't be all zeros (model should do something)
        assert not torch.all(rggb_output == 0)

    def test_rgb2raw_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        model = Rgb2Raw()
        rgb_input = torch.randn(1, 3, 32, 32, requires_grad=True)

        rggb_output = model(rgb_input)
        loss = rggb_output.sum()
        loss.backward()

        # Check that input has gradients
        assert rgb_input.grad is not None
        assert not torch.all(rgb_input.grad == 0)

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_rgb2raw_deterministic(self) -> None:
        """Test that Rgb2Raw gives consistent outputs for same input."""
        model = Rgb2Raw()
        model.eval()

        rgb_input = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            output1 = model(rgb_input)
            output2 = model(rgb_input)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)
