"""Command-line interface for ScAN Gen."""

import logging
import os
from pathlib import Path
import tomllib
import traceback

from tqdm import tqdm
import typer
from typer.models import ParameterInfo

from scangen.data import data_directory
from scangen.device import get_device, print_device_info
from scangen.formats.config_format import ConfigFormat
from scangen.formats.config_format import create_default_config
from scangen.pipeline.generator import RAWGenerator
from scangen.preprocess import load_cifar_dataset, batches, preprocess_rgb_batch, clean_raw, tohdf5, verify_hdf5

LOGGER = logging.getLogger("scangen.cli")

app = typer.Typer(name="scangen", help="Generate noisy RAW data from RGB inputs")

def set_debug(bVal):
    logging.basicConfig(level=logging.INFO)
    all_logs = logging.getLogger("scangen")
    if bVal:
        all_logs.setLevel(logging.DEBUG)
    else:
        all_logs.setLevel(logging.INFO)

@app.command()
def download_models() -> None:
    """Download required models and datasets using Pooch."""
    typer.echo("Downloading models... (Not implemented yet)")
    # TODO: Implement Pooch-based model downloading

@app.command()
def pregenerate_raw(
    dataset_name: str = typer.Argument("cifar10", help="Dataset name (cifar10/cifar100)"),
    data_root: Path = typer.Option(Path("."), help="CIFAR directory relative to SCANGENDATA"),
    output_path: Path = typer.Option(Path("cifar10_raw.h5"), help="Output HDF5 file path"),
    debug_limit: int | None = typer.Option(None, help="Limit the number of images for debugging."),
    debug: bool = typer.Option(False, "--debug", "-v", help="Turn on logging"),
) -> None:
    """Generate clean RAW images to make training faster.

    Args:
        dataset_name: "cifar10" or "cifar100".
        data_root: The data directory that contains the `torchvision` dataset relative to SCANGENDATA.
        output_path: The output HDF5 file path, such as "cifar10_raw.h5", relative to SCANGENDATA.
        debug_limit Optional: Limit the number of images for debugging. 0 makes no images.
    """
    data_dir = data_directory()
    set_debug(debug)
    if data_dir is None:
        typer.echo("Looking for environment variable called SCANGENDATA that points to a data directory.")
        raise typer.Exit(2)
    try:
        batch_cnt = 50
        target_size = (16, 16)
        device = get_device()
        model_path = data_dir / Path("weights/rgb2raw.pth")
        train_dataset = load_cifar_dataset(dataset_name, data_dir / data_root, train=True)
        test_dataset = load_cifar_dataset(dataset_name, data_dir / data_root, train=False)
        img_cnt = len(train_dataset) + len(test_dataset)
        load = batches([(train_dataset, True), (test_dataset, False)], batch_cnt, debug_limit)
        # A yield iterator has no len() so set the total.
        progress = tqdm(load, desc="Batches", total=(img_cnt - 1) // batch_cnt + 1)
        resize = preprocess_rgb_batch(progress, device)
        clean = clean_raw(resize, model_path, device, target_size)
        tohdf5(clean, output_path=data_dir / output_path, image_cnt=img_cnt, image_shape=(4, *target_size), chunk_size=batch_cnt)
        verify_hdf5(data_dir / output_path)

    except Exception as e:
        typer.echo(f"Error during pregeneration: {e}", err=True)
        traceback.print_tb(e.__traceback__)
        raise typer.Exit(1) from e


@app.command()
def device_info() -> None:
    """Show available computing devices (CUDA, MPS, CPU)."""
    print_device_info()


@app.command()
def create_config(
    output_path: str = typer.Argument("config.json", help="Output path for configuration file"),
    dataset: str = typer.Option("cifar10", help="Dataset name (cifar10/cifar100)"),
    data_root: str = typer.Option(".", help="Root directory for dataset relative to SCANGENDATA"),
    batch_size: int = typer.Option(32, help="Batch size"),
    noise_type: str = typer.Option("pointwise", help="Noise type (pointwise, cycleisp)"),
    output_dir: str = typer.Option("./output", help="Output directory for config"),
    num_workers: int = typer.Option(4, help="Number of workers for multi-threaded data loading"),
) -> None:
    """Create a default configuration file.

    Args:
        output_path: Path where to save the configuration file
        dataset: Dataset to use (cifar10 or cifar100)
        data_root: Directory relative to SCANGENDATA for storing the dataset
        batch_size: Number of images per batch
        noise_type: Type of noise model to use
        output_dir: Directory for generated data
        num_workers: Number of workers for multi-threaded data loading
    """
    config = create_default_config()

    # Apply command line overrides
    if not isinstance(dataset, ParameterInfo):
        config.dataset.name = str(dataset)
    if not isinstance(data_root, ParameterInfo):
        config.dataset.root = str(data_root)
    if not isinstance(batch_size, ParameterInfo):
        config.dataset.batch_size = int(batch_size)
    if not isinstance(noise_type, ParameterInfo):
        config.noise.type = str(noise_type)
    if not isinstance(output_dir, ParameterInfo):
        config.output.directory = str(output_dir)
    if not isinstance(num_workers, ParameterInfo):
        config.device.num_workers = int(num_workers)

    # Save configuration
    config_path = Path(output_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w") as f:
        f.write(config.model_dump_json(indent=2))
        f.write(os.linesep)

    typer.echo(f"Configuration saved to: {config_path}")
    typer.echo("Edit the file to customize generation parameters")


if __name__ == "__main__":
    app()
