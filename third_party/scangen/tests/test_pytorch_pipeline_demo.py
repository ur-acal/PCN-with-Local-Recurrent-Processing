"""Comprehensive demonstration test for PyTorch training pipeline with scangen DataLoader.

This test validates the complete end-to-end functionality by implementing a simple
denoising training loop using scangen as a DataLoader for real-time RAW generation.
"""
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from scangen.data.raw_dataloader import create_raw_dataloader
from scangen.device import get_device
from scangen.pipeline.noise import dnd_noise

class SimpleRAWDenoiser(nn.Module):
    """Simple CNN for RAW image denoising."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class MockRGBDataset(Dataset):
    """Mock RGB dataset for testing."""

    def __init__(self, num_samples: int = 16):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate consistent but varied test data
        torch.manual_seed(idx)  # Deterministic per index
        image = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
        label = idx % 10
        return image, label


def test_complete_pytorch_pipeline(nn_weights):
    """
    Comprehensive test demonstrating full PyTorch training pipeline
    using scangen as a DataLoader for RAW data generation.

    This test validates:
    1. DataLoader creation and iteration
    2. RAW data generation quality
    3. Integration with PyTorch training loops
    4. Memory stability over multiple iterations
    5. Gradient computation and backpropagation
    """
    # Configuration
    batch_size = 4
    num_epochs = 2
    num_samples = 16
    target_size = (16, 16) # Equivalent to a 32x32 RAW sensor.
    device = get_device()

    # 1. Create mock RGB dataset
    rgb_dataset = MockRGBDataset(num_samples=num_samples)
    # 2. Create RAW DataLoader
    dataloader = create_raw_dataloader(
        rgb_dataset=rgb_dataset,
        noise_config=dnd_noise(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for testing
        model_path=str(nn_weights),
        target_size=target_size,
        device=None,  # Use auto device detection (CUDA if available)
    )
    # 3. Create denoising model
    model = SimpleRAWDenoiser(in_channels=4, out_channels=4)
    model = model.to(device)

    # 4. Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Track training statistics
    epoch_losses = []

    # 5. Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (rgb, clean_raw, noisy_raw, metadata) in enumerate(dataloader):
            # Move data to device
            clean_raw = clean_raw.to(device)
            noisy_raw = noisy_raw.to(device)

            # Validate data shapes and ranges
            assert rgb.shape == (batch_size, 3, 256, 256)
            assert clean_raw.shape == (batch_size, 4, *target_size)
            assert noisy_raw.shape == (batch_size, 4, *target_size)
            assert len(metadata) == batch_size

            # Validate data ranges
            assert torch.all(rgb >= 0) and torch.all(rgb <= 1)
            assert torch.all(clean_raw >= 0) and torch.all(clean_raw <= 1)
            assert torch.all(noisy_raw >= 0) and torch.all(noisy_raw <= 1)

            # Validate noise was actually added
            noise_diff = torch.abs(clean_raw - noisy_raw).mean()
            assert noise_diff > 0, "No noise detected in noisy_raw"

            # Forward pass
            optimizer.zero_grad()
            denoised = model(noisy_raw)
            loss = criterion(denoised, clean_raw)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track statistics
            epoch_loss += loss.item()
            num_batches += 1

            # Validate gradients exist
            has_gradients = any(
                p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
            )
            assert has_gradients, "No gradients computed"

            # Validate metadata
            for meta in metadata:
                assert isinstance(meta, dict)
                assert "noise_type" in meta
                assert meta["noise_type"] == "cycleisp"

            if batch_idx < 2:  # Log first few batches
                # ruff: noqa: T201
                print(
                    f"  Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                    f"loss={loss.item():.4f}, "
                    f"noise_diff={noise_diff.item():.4f}"
                )

        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)

        # Validate we processed all expected batches
        expected_batches = (num_samples + batch_size - 1) // batch_size
        assert num_batches == expected_batches, (
            f"Expected {expected_batches} batches, got {num_batches}"
        )

    # 6. Final validation
    model.eval()
    with torch.no_grad():
        # Test inference on one batch
        test_batch = next(iter(dataloader))
        rgb, clean_raw, noisy_raw, metadata = test_batch

        clean_raw = clean_raw.to(device)
        noisy_raw = noisy_raw.to(device)

        denoised = model(noisy_raw)

        # Validate output shape
        assert denoised.shape == clean_raw.shape

        # Validate output range (should be roughly in [0, 1])
        assert torch.all(denoised >= -0.5) and torch.all(denoised <= 1.5)

        # Calculate denoising performance
        noisy_mse = nn.MSELoss()(noisy_raw, clean_raw).item()
        denoised_mse = nn.MSELoss()(denoised, clean_raw).item()

        # ruff: noqa: T201
        print("\nFinal performance:")
        print(f"  Noisy MSE:    {noisy_mse:.6f}")
        print(f"  Denoised MSE: {denoised_mse:.6f}")
        print(f"  Improvement:  {((noisy_mse - denoised_mse) / noisy_mse * 100):.2f}%")

    # 7. Validate training progressed
    assert len(epoch_losses) == num_epochs

    # Basic check that loss didn't explode
    for loss in epoch_losses:
        assert loss < 100, f"Loss too high: {loss}"
        assert not torch.isnan(torch.tensor(loss)), "NaN loss detected"


def test_memory_stability_extended(nn_weights):
    """Test memory stability over extended training."""
    rgb_dataset = MockRGBDataset(num_samples=12)
    dataloader = create_raw_dataloader(
        rgb_dataset=rgb_dataset,
        noise_config=dnd_noise(),
        batch_size=3,
        num_workers=0,
        model_path=str(nn_weights),
        target_size=(64, 64),
        device=None,  # Use auto device detection (CUDA if available)
    )

    # Run many iterations to check for memory leaks
    for i in range(20):
        for batch_idx, (rgb, clean_raw, noisy_raw, metadata) in enumerate(dataloader):
            # Basic validation
            assert rgb.shape[0] == 3
            assert clean_raw.shape[0] == 3
            assert noisy_raw.shape[0] == 3
            assert len(metadata) == 3

            # Force garbage collection occasionally
            if i % 10 == 0 and batch_idx == 0:
                import gc

                gc.collect()


if __name__ == "__main__":
    # Run the tests when script is executed directly
    nn_weights = Path("data") / "weights" / "rgb2raw.pth"
    test_complete_pytorch_pipeline(nn_weights)
    test_memory_stability_extended(nn_weights)
    # ruff: noqa: T201
    print("tests ran")
