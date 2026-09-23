"""Official PyTorch MNIST preprocessing, shared by every stage."""
from pathlib import Path

import torch
from torchvision import datasets, transforms

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MEAN, STD = (0.1307,), (0.3081,)


def mnist_dataset(data_dir=DEFAULT_DATA_DIR, *, train, download=False):
    return datasets.MNIST(
        root=str(data_dir), train=train, download=download,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))


def mnist_loader(data_dir=DEFAULT_DATA_DIR, *, train, batch_size=64,
                 num_workers=4, seed=4096, download=False, limit_samples=0):
    dataset = mnist_dataset(data_dir, train=train, download=download)
    if limit_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(limit_samples, len(dataset))))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed), persistent_workers=False)
