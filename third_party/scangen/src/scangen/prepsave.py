#!/usr/bin/env python
"""
Single script to pre-generate all CIFAR-10 clean RAW data and save to HDF5.
This demonstrates the complete workflow in one place.

Usage:
    python pregenerate_cifar_raw.py
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from scangen.device import get_device
from scangen.pipeline.noise import dnd_noise
from scangen.pipeline.generator import RAWGenerator

LOGGER = logging.getLogger(__name__)

# Paths
CIFAR_ROOT = Path("./data")
MODEL_PATH = Path("data/weights/rgb2raw.pth")  # Adjust to your model path
OUTPUT_HDF5 = Path("./data/cifar10_raw.h5")

# Processing parameters
BATCH_SIZE = 32  # Process images in batches
TARGET_SIZE = (256, 256)  # Final RAW image size
DEVICE = get_device(None)  # Auto-detect best device

# Dataset parameters
DATASET_NAME = "cifar10"

# HDF5 parameters
COMPRESSION = "gzip"  # Use gzip compression for storage efficiency
COMPRESSION_LEVEL = 4  # Balance between compression and speed (1-9)
CHUNK_SIZE = 100  # Number of images per HDF5 chunk

def load_cifar_dataset(name: str, cifar_root: Path, train: bool):
    """Load CIFAR-10 dataset with basic preprocessing."""
    LOGGER.debug(f"Loading CIFAR-10 dataset from {cifar_root} train {train}")
    if name not in ("cifar10", "cifar100"):
        raise ValueError(f"Expected cifar dataset name cifar10 or cifar100, got: {name}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL to tensor [0,1]
    ])

    getter = CIFAR10 if name == "cifar10" else CIFAR100
    
    dataset = getter(
        root=str(cifar_root),
        train=train,
        download=True,
        transform=transform
    )
    
    LOGGER.debug(f"Loaded {len(dataset)} images")
    return dataset


def batches(datasets: list, batch_size: int):
    """Iterates over datasets yielding batches with images, labels, and train flag."""
    for (ds, train) in datasets:
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # Keep original order
            num_workers=0,  # Single-threaded for GPU compatibility
            pin_memory=torch.cuda.is_available()
        )
        for (image, labels) in dataloader:
            yield image, labels, train


def preprocess_rgb_batch(rgb_batch, device: torch.device):
    """Preprocess RGB images to match RAWGenerator input requirements."""
    for (rgb_images, labels, train) in rgb_batch:
        rgb_batch = rgb_batch.to(device)
        
        # Ensure CHW format for entire batch
        if rgb_images.shape[1] != 3:  # Assuming batch is first dimension
            rgb_images = rgb_images.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Resize entire batch to 256x256 (CycleISP input requirement)
        if rgb_images.shape[2:] != (256, 256):
            rgb_images = F.interpolate(
                rgb_images,
                size=(256, 256),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )
        
        yield (torch.clamp(rgb_images, 0.0, 1.0), labels, train)


def clean_raw(rgb_batch, model_path: Path, device: torch.device, target_size: Tuple[int, int]):
    """Initialize the RAWGenerator with dummy noise config."""
    LOGGER.info(f"Initializing RAWGenerator on device: {DEVICE}")
    
    generator = RAWGenerator(
        model_path=model_path,
        device=device,
        noise=dnd_noise(),  # We won't use the noisy output.
        output_dim=target_size,
    )
    # Get class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for (rgb_images, labels, train) in rgb_batch:
        label_names = [class_names[label.item()] for label in labels]
        clean_raw, _, _ = generator.generate_batch(
            rgb_images=rgb_images,
            labels=labels,
            label_names=label_names
        )
        yield clean_raw.cpu().numpy(), labels.numpy(), train


def generate_all_raw_data(dataset, generator):
    """Generate clean RAW data for entire dataset."""
    LOGGER.info("Starting RAW generation...")
    
    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Keep original order
        num_workers=0,  # Single-threaded for GPU compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    all_clean_raw = []
    all_labels = []
    all_label_names = []
    
    # Get class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Preprocess RGB images
        rgb_processed = preprocess_rgb_batch(images)
        rgb_processed = rgb_processed.to(DEVICE)
        
        # Convert labels to label names
        label_names = [class_names[label.item()] for label in labels]
        
        # Generate clean RAW using existing generate_batch method
        with torch.no_grad():
            clean_raw, _, _ = generator.generate_batch(
                rgb_images=rgb_processed,
                labels=labels,
                label_names=label_names
            )
        
        # Move to CPU and convert to numpy for storage
        clean_raw_np = clean_raw.cpu().numpy()
        
        # Store results
        all_clean_raw.append(clean_raw_np)
        all_labels.extend(labels.numpy())
        all_label_names.extend(label_names)
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            processed = min((batch_idx + 1) * BATCH_SIZE, len(dataset))
            LOGGER.info(f"Processed {processed}/{len(dataset)} images")
    
    # Concatenate all batches
    all_clean_raw = np.concatenate(all_clean_raw, axis=0)
    all_labels = np.array(all_labels)
    
    LOGGER.info(f"Generated RAW data shape: {all_clean_raw.shape}")
    LOGGER.info(f"Labels shape: {all_labels.shape}")
    
    return all_clean_raw, all_labels, all_label_names


def save_to_hdf5(clean_raw, labels, label_names, output_path):
    """Save pre-generated RAW data to HDF5 file."""
    LOGGER.info(f"Saving to HDF5: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate chunk shape for efficient access
    n_samples = clean_raw.shape[0]
    chunk_shape = (min(CHUNK_SIZE, n_samples), *clean_raw.shape[1:])
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets with compression
        LOGGER.info("Creating HDF5 datasets...")
        
        # Clean RAW data - main dataset
        clean_dataset = f.create_dataset(
            'clean_raw',
            data=clean_raw,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_LEVEL,
            chunks=chunk_shape,
            dtype='float32'
        )
        clean_dataset.attrs['description'] = 'Clean RAW images in RGGB format'
        clean_dataset.attrs['shape_info'] = 'N x 4 x H x W (RGGB channels)'
        
        # Integer labels
        labels_dataset = f.create_dataset(
            'labels',
            data=labels,
            compression=COMPRESSION,
            dtype='uint8'
        )
        labels_dataset.attrs['description'] = 'Integer class labels'
        
        # String label names - requires special handling
        # Convert to fixed-length ASCII strings for HDF5
        max_len = max(len(name) for name in label_names)
        label_names_ascii = np.array(label_names, dtype=f'S{max_len}')
        
        names_dataset = f.create_dataset(
            'label_names',
            data=label_names_ascii,
            compression=COMPRESSION
        )
        names_dataset.attrs['description'] = 'String class names'
        
        # Add metadata group
        metadata = f.create_group('metadata')
        metadata.attrs['dataset_name'] = DATASET_NAME
        metadata.attrs['is_train'] = TRAIN
        metadata.attrs['target_size'] = TARGET_SIZE
        metadata.attrs['model_path'] = str(MODEL_PATH)
        metadata.attrs['generation_date'] = datetime.now().isoformat()
        metadata.attrs['num_samples'] = n_samples
        metadata.attrs['raw_shape'] = clean_raw.shape
        metadata.attrs['device_used'] = str(DEVICE)
        metadata.attrs['cifar_root'] = str(CIFAR_ROOT)
        
        # Add dataset statistics for validation
        stats = f.create_group('statistics')
        stats.attrs['min_value'] = float(clean_raw.min())
        stats.attrs['max_value'] = float(clean_raw.max())
        stats.attrs['mean_value'] = float(clean_raw.mean())
        stats.attrs['std_value'] = float(clean_raw.std())
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    LOGGER.info(f"HDF5 file saved: {file_size_mb:.2f} MB")
    
    # Print compression ratio
    uncompressed_size_mb = clean_raw.nbytes / (1024 * 1024)
    compression_ratio = uncompressed_size_mb / file_size_mb
    LOGGER.info(f"Compression ratio: {compression_ratio:.2f}x")


def tohdf5(
    rgb_images,
    output_path: Path,
    image_cnt: int,
    image_shape: Tuple[int, int, int] = (4, 256, 256),  # (channels, height, width)
    chunk_size: int = 32,
    compression: str = "gzip",
    compression_level: int = 4,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[int, Path]:
    """
    Initialize an HDF5 file with pre-allocated datasets for storing images.
    
    Args:
        rgb_images: Iterator over input batches to save.
        output_path: Path where the HDF5 file will be created
        image_cnt: Total number of images that will be stored
        image_shape: Shape of each image (C, H, W), default (4, 256, 256) for RGGB RAW
        chunk_size: Size of chunks to write (becomes the chunk size)
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level (1-9 for gzip)
        metadata: Optional dictionary of metadata to store
    
    Returns:
        Tuple of (chunk_size, output_path)
        - chunk_size: Recommended chunk size for writing
        - output_path: Path to the created HDF5 file
    
    Example:
        >>> chunk_size, hdf5_path = initialize_hdf5_file(
        ...     Path("cifar10_raw.h5"),
        ...     image_cnt=60000,
        ...     image_shape=(4, 256, 256),
        ...     batch_size=32
        ... )
        >>> print(f"Write images in chunks of {chunk_size} to {hdf5_path}")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate chunk shape for HDF5
    # Align chunk size with batch size for optimal write performance
    chunk_shape = (chunk_size, *image_shape)
    
    LOGGER.info(f"Initialized HDF5 file: {output_path}")
    LOGGER.debug(f"  Total capacity: {image_cnt} images")
    LOGGER.debug(f"  Image shape: {image_shape}")
    LOGGER.debug(f"  Chunk size: {chunk_size}")
    LOGGER.debug(f"  Compression: {compression} (level {compression_level})")

    with h5py.File(output_path, 'w') as f:
        # Main image dataset - pre-allocated with full size
        images_dataset = f.create_dataset(
            'images',
            shape=(image_cnt, *image_shape),
            dtype='float32',
            chunks=chunk_shape,
            compression=compression,
            compression_opts=compression_level if compression == 'gzip' else None
        )
        images_dataset.attrs['description'] = 'Image data'
        images_dataset.attrs['shape_format'] = '(N, C, H, W)'
        
        # Labels dataset - assuming integer labels
        labels_dataset = f.create_dataset(
            'labels',
            shape=(image_cnt,),
            dtype='uint8',
            chunks=(min(image_cnt, chunk_size * 10),),  # Larger chunks for small data
            compression=compression
        )
        labels_dataset.attrs['description'] = 'Integer class labels'
        
        # Metadata group
        meta_group = f.create_group('metadata')
        meta_group.attrs['total_images'] = image_cnt
        meta_group.attrs['image_shape'] = image_shape
        meta_group.attrs['chunk_size'] = chunk_size
        meta_group.attrs['creation_time'] = datetime.now().isoformat()
        meta_group.attrs['file_version'] = '1.0'
        
        # Add any custom metadata
        if metadata:
            for key, value in metadata.items():
                meta_group.attrs[key] = value

        train_dataset = f.create_dataset(
            'train',
            shape=(image_cnt,),
            dtype='bool',
            chunks=(min(image_cnt, chunk_size * 10),),
            fillvalue=False
        )
        train_dataset.attrs['description'] = 'Train if true else test'
        
        # Track writing progress (initially all false)
        progress_dataset = f.create_dataset(
            'write_progress',
            shape=(image_cnt,),
            dtype='bool',
            chunks=(min(image_cnt, chunk_size * 10),),
            fillvalue=False
        )
        progress_dataset.attrs['description'] = 'Track which images have been written'
    
        chunks_written_cnt = 0
        for (img_batch, label_batch, train) in rgb_images:
            images_chunk = img_batch.astype(np.float32)
            labels_chunk = label_batch.astype(np.uint8)
            img_cnt = images_chunk.shape[0]
            if img_cnt > chunk_size:
                raise ValueError(
                    "Expect image batches to match HDF5 chunk size. "
                    f"Instead see img_cnt {img_cnt} and chunk_size {chunk_size}"
                )
            
            chunk_start_idx = chunks_written_cnt * chunk_size
            chunk_end_idx = chunk_start_idx + img_cnt
            chunks_written_cnt += 1

            images_dataset[chunk_start_idx:chunk_end_idx] = images_chunk
            labels_dataset[chunk_start_idx:chunk_end_idx] = labels_chunk
            train_dataset[chunk_start_idx:chunk_end_idx] = True
            progress_dataset[chunk_start_idx:chunk_end_idx] = True
            f.flush()

def write_chunk_to_hdf5(
    hdf5_path: Path,
    chunk_start_idx: int,
    images_chunk: np.ndarray,
    labels_chunk: np.ndarray,
    label_names_chunk: Optional[np.ndarray] = None,
    verify_write: bool = True
) -> int:
    """
    Write a chunk of images to an existing HDF5 file at specified position.
    
    Args:
        hdf5_path: Path to existing HDF5 file (created by initialize_hdf5_file)
        chunk_start_idx: Starting index where this chunk should be written
        images_chunk: Array of images to write, shape (batch_size, C, H, W)
        labels_chunk: Array of labels to write, shape (batch_size,)
        label_names_chunk: Optional array of label names, shape (batch_size,)
        verify_write: Whether to verify the write was successful
    
    Returns:
        Number of images written
    
    Raises:
        ValueError: If chunk would exceed dataset bounds
        IOError: If write verification fails
    
    Example:
        >>> # Write first batch
        >>> images_batch = np.random.rand(32, 4, 256, 256).astype(np.float32)
        >>> labels_batch = np.random.randint(0, 10, 32, dtype=np.uint8)
        >>> names_batch = np.array([f"class_{i}" for i in labels_batch], dtype='S20')
        >>> 
        >>> n_written = write_chunk_to_hdf5(
        ...     Path("cifar10_raw.h5"),
        ...     chunk_start_idx=0,
        ...     images_chunk=images_batch,
        ...     labels_chunk=labels_batch,
        ...     label_names_chunk=names_batch
        ... )
        >>> print(f"Wrote {n_written} images")
    """
    chunk_size = len(images_chunk)
    
    # Ensure arrays are numpy arrays with correct dtype
    if isinstance(images_chunk, torch.Tensor):
        images_chunk = images_chunk.cpu().numpy()
    if isinstance(labels_chunk, torch.Tensor):
        labels_chunk = labels_chunk.cpu().numpy()
    
    images_chunk = images_chunk.astype(np.float32)
    labels_chunk = labels_chunk.astype(np.uint8)
    
    with h5py.File(hdf5_path, 'r+') as f:  # 'r+' for read/write, file must exist
        # Get datasets
        images_dataset = f['images']
        labels_dataset = f['labels']
        label_names_dataset = f['label_names']
        progress_dataset = f['write_progress']
        
        # Verify bounds
        total_images = images_dataset.shape[0]
        chunk_end_idx = chunk_start_idx + chunk_size
        
        if chunk_end_idx > total_images:
            raise ValueError(
                f"Chunk would exceed dataset bounds: "
                f"trying to write indices {chunk_start_idx}:{chunk_end_idx} "
                f"but dataset size is {total_images}"
            )
        
        # Write the chunk
        images_dataset[chunk_start_idx:chunk_end_idx] = images_chunk
        labels_dataset[chunk_start_idx:chunk_end_idx] = labels_chunk
        
        # Write label names if provided
        if label_names_chunk is not None:
            if isinstance(label_names_chunk[0], str):
                # Convert strings to fixed-length bytes
                label_names_chunk = np.array(label_names_chunk, dtype='S20')
            label_names_dataset[chunk_start_idx:chunk_end_idx] = label_names_chunk
        else:
            # Write empty strings if no names provided
            empty_names = np.array([''] * chunk_size, dtype='S20')
            label_names_dataset[chunk_start_idx:chunk_end_idx] = empty_names
        
        # Mark these indices as written
        progress_dataset[chunk_start_idx:chunk_end_idx] = True
        
        # Verify write if requested
        if verify_write:
            # Read back and compare shapes
            written_data = images_dataset[chunk_start_idx:chunk_end_idx]
            if written_data.shape != images_chunk.shape:
                raise IOError(
                    f"Write verification failed: "
                    f"expected shape {images_chunk.shape}, "
                    f"got {written_data.shape}"
                )
            
            # Optionally verify content (first and last values)
            if not np.allclose(written_data[0, 0, 0, 0], images_chunk[0, 0, 0, 0]):
                raise IOError("Write verification failed: data mismatch")
        
        # Update metadata with last write time
        f['metadata'].attrs[f'last_write_time'] = datetime.now().isoformat()
        f['metadata'].attrs[f'last_write_index'] = chunk_end_idx
        
        # Flush to ensure data is written to disk
        f.flush()
    
    print(f"Wrote {chunk_size} images at indices [{chunk_start_idx}:{chunk_end_idx})")
    
    return chunk_size


# Optional: Helper function to check write progress
def check_write_progress(hdf5_path: Path) -> Dict[str, Any]:
    """
    Check the write progress of an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary with progress information
    """
    with h5py.File(hdf5_path, 'r') as f:
        progress = f['write_progress'][:]
        metadata = f['metadata']
        
        total_images = metadata.attrs['total_images']
        written_count = np.sum(progress)
        
        return {
            'total_images': total_images,
            'written_count': written_count,
            'remaining_count': total_images - written_count,
            'percent_complete': (written_count / total_images) * 100,
            'is_complete': written_count == total_images,
            'last_write_time': metadata.attrs.get('last_write_time', 'N/A'),
            'last_write_index': metadata.attrs.get('last_write_index', 0)
        }


def verify_hdf5(filepath):
    """Verify the saved HDF5 file by reading a sample."""
    LOGGER.info("Verifying saved HDF5 file...")
    
    with h5py.File(filepath, 'r') as f:
        # Print structure
        LOGGER.info("HDF5 Structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                LOGGER.info(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                LOGGER.info(f"  Group: {name}")
                for attr_name, attr_value in obj.attrs.items():
                    LOGGER.info(f"    - {attr_name}: {attr_value}")
        
        f.visititems(print_structure)
        
        # Load and check a sample
        clean_raw = f['clean_raw']
        labels = f['labels']
        label_names = f['label_names']
        
        LOGGER.info(f"\nSample check:")
        LOGGER.info(f"  First image shape: {clean_raw[0].shape}")
        LOGGER.info(f"  First label: {labels[0]}")
        LOGGER.info(f"  First label name: {label_names[0].decode('ascii')}")
        LOGGER.info(f"  Value range: [{clean_raw[0].min():.4f}, {clean_raw[0].max():.4f}]")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        ds = "cifar10"
        batch_cnt = 100
        target_size = (16, 16)
        device = get_device()
        cifar_root = "./data"
        model_path = "./data/weights/rgb2raw.pth"
        output_path = "./data/cifar10_raw.h5"
        train_dataset = load_cifar_dataset(ds, cifar_root, train=True)
        test_dataset = load_cifar_dataset(ds, cifar_root, train=False)
        img_cnt = len(train_dataset) + len(test_dataset)
        load = batches([(train_dataset, True), (test_dataset, False)], batch_cnt)
        # A yield iterator has no len() so set the total.
        progress = tqdm(load, desc="Batches", total=(img_cnt - 1) // batch_cnt + 1)
        resize = preprocess_rgb_batch(progress, device)
        clean = clean_raw(resize, model_path, device, target_size)
        tohdf5(clean, output_path, img_cnt, (4, *target_size), batch_cnt)

        # clean_raw, labels, label_names = generate_all_raw_data(dataset, generator)
        # save_to_hdf5(clean_raw, labels, label_names, OUTPUT_HDF5)
        verify_hdf5(output_path)
        LOGGER.info(f"HDF5 file location: {OUTPUT_HDF5.absolute()}")
        
    except Exception as e:
        LOGGER.error(f"Error during pre-generation: {e}")
        raise

if __name__ == "__main__":
    main()
