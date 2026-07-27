# PyTorch DataLoader Usage Guide

This guide demonstrates how to use scangen's PyTorch DataLoader interface for real-time RAW data generation during training.

## Quick Start

### Basic Usage

```python
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from scangen.data import RawDataset, create_raw_dataloader

# Method 1: Using convenience function (recommended)
dataloader = create_raw_dataloader(
    rgb_dataset=CIFAR10(root='./data', train=True, download=True),
    noise_config={'type': 'dnd'},
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# Method 2: Manual setup
rgb_dataset = CIFAR10(root='./data', train=True, download=True)
raw_dataset = RawDataset(
    rgb_dataset=rgb_dataset,
    noise_config={'type': 'dnd'},
    target_size=(256, 256),
    device='cpu'
)
dataloader = DataLoader(raw_dataset, batch_size=32, shuffle=True)

# Use in training loop
for rgb, clean_raw, noisy_raw, metadata in dataloader:
    # rgb: (batch_size, 3, 256, 256) - Original RGB images
    # clean_raw: (batch_size, 4, 128, 128) - Clean RAW in RGGB format  
    # noisy_raw: (batch_size, 4, 128, 128) - Noisy RAW with realistic noise
    # metadata: List of dicts with noise parameters per sample
    pass
```

### CIFAR-10 Shortcut

```python
from scangen.data import create_cifar10_raw_dataloader

# Even simpler for CIFAR-10
dataloader = create_cifar10_raw_dataloader(
    root="./data",
    noise_config={'type': 'sidd'},
    batch_size=64,
    num_workers=4
)
```

## Training Example

### Simple Denoising Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from scangen.data import create_cifar10_raw_dataloader

# 1. Create DataLoader
dataloader = create_cifar10_raw_dataloader(
    batch_size=32,
    noise_config={'type': 'dnd'}
)

# 2. Define denoising model
class RAWDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)

model = RAWDenoiser()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# 3. Training loop
for epoch in range(10):
    for rgb, clean_raw, noisy_raw, metadata in dataloader:
        optimizer.zero_grad()
        
        # Denoise the noisy RAW data
        denoised = model(noisy_raw)
        loss = criterion(denoised, clean_raw)
        
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
```

## Configuration Options

### Noise Models

```python
# DND noise (camera sensors)
noise_config = {'type': 'dnd'}

# SIDD noise (smartphone sensors)  
noise_config = {'type': 'sidd'}

# Custom noise levels
noise_config = {
    'type': 'custom',
    'shot_noise': 0.015,
    'read_noise': 0.0012
}
```

### Dataset Configuration

```python
from scangen.data import RawDataset
from torchvision.datasets import CIFAR10

# RGB dataset (any PyTorch Dataset)
rgb_dataset = CIFAR10(root='./data', train=True, download=True)

# RAW dataset wrapper
raw_dataset = RawDataset(
    rgb_dataset=rgb_dataset,
    noise_config={'type': 'dnd'},
    target_size=(256, 256),  # RAW output size
    device='cpu',            # 'cpu', 'cuda', 'mps', or 'auto'
    model_path=None          # Use default weights, or specify path
)
```

### DataLoader Configuration

```python
from scangen.data import create_raw_dataloader

dataloader = create_raw_dataloader(
    rgb_dataset=rgb_dataset,
    noise_config={'type': 'dnd'},
    batch_size=32,
    shuffle=True,
    num_workers=2,           # 0 for debugging, 2-4 for training
    target_size=(256, 256),
    device='cpu',
    pin_memory=True          # Auto-detected if None
)
```

## Data Format

### Input (RGB)
- **Shape**: `(batch_size, 3, height, width)`
- **Type**: `torch.float32`
- **Range**: `[0, 1]`
- **Format**: RGB color channels

### Output (RAW) 
- **Shape**: `(batch_size, 4, height//2, width//2)`
- **Type**: `torch.float32` 
- **Range**: `[0, 1]`
- **Format**: RGGB Bayer pattern, packed into 4 channels
  - Channel 0: R (red)
  - Channel 1: G1 (green, top-left)
  - Channel 2: G2 (green, bottom-right)  
  - Channel 3: B (blue)

### Metadata
Each sample includes metadata with noise parameters:
```python
metadata = {
    'shot_noise': 0.008234,     # Poisson noise level
    'read_noise': 0.000456,     # Gaussian noise level  
    'noise_type': 'dnd',        # Noise model used
    'index': 42,                # Sample index
    'original_data': (label,)   # Original dataset data (labels, etc.)
}
```

## Performance Tips

### Multi-Worker Setup
```python
# Good for training
dataloader = create_raw_dataloader(
    rgb_dataset=dataset,
    noise_config={'type': 'dnd'},
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    batch_size=32
)
```

### Memory Optimization
```python
# For large datasets or limited memory
dataloader = create_raw_dataloader(
    rgb_dataset=dataset,
    noise_config={'type': 'dnd'},
    target_size=(128, 128),  # Smaller images
    device='cpu',            # Generate on CPU
    batch_size=16            # Smaller batches
)
```

### GPU Acceleration
```python
# For maximum performance
dataloader = create_raw_dataloader(
    rgb_dataset=dataset,
    noise_config={'type': 'dnd'}, 
    device='cuda',          # Generate on GPU
    pin_memory=True,        # Fast CPU->GPU transfer
    num_workers=2           # Don't use too many workers with GPU
)
```

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `batch_size`
- Use smaller `target_size`
- Set `device='cpu'`

**"Too slow data loading"**
- Increase `num_workers` (try 2-4)
- Enable `pin_memory=True`
- Use smaller `target_size` for testing

**"Model weights not loading"**
- Check that `data/weights/rgb2raw.pth` exists
- Specify explicit `model_path` if needed

### Performance Targets
- **CPU**: ~30 samples/second
- **GPU**: ~100+ samples/second
- **Memory**: <4GB overhead for DataLoader

## Integration Examples

### With PyTorch Lightning
```python
import pytorch_lightning as pl
from scangen.data import create_cifar10_raw_dataloader

class DenoisingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RAWDenoiser()
        
    def train_dataloader(self):
        return create_cifar10_raw_dataloader(
            batch_size=32,
            num_workers=4,
            noise_config={'type': 'dnd'}
        )
    
    def training_step(self, batch, batch_idx):
        rgb, clean_raw, noisy_raw, metadata = batch
        denoised = self.model(noisy_raw)
        loss = nn.MSELoss()(denoised, clean_raw)
        return loss
```

### Custom Datasets
```python
from torch.utils.data import Dataset
from scangen.data import RawDataset

class MyRGBDataset(Dataset):
    def __len__(self):
        return 1000
        
    def __getitem__(self, idx):
        # Return (image, label) tuple
        # image should be torch.Tensor, shape (H, W, 3) or (3, H, W)
        image = load_my_image(idx)  
        label = load_my_label(idx)
        return image, label

# Wrap with RAW generation
raw_dataset = RawDataset(
    rgb_dataset=MyRGBDataset(),
    noise_config={'type': 'sidd'}
)
```

This guide covers the essential usage patterns for the scangen PyTorch DataLoader interface. The system provides real-time RAW data generation that integrates seamlessly into standard PyTorch training workflows.