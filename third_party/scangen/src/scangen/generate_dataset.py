import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import cast, Any, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from scangen.pipeline.generator import RAWGenerator

from tqdm import tqdm

from scangen import __version__
from scangen.formats.config_format import ConfigFormat
from scangen.formats.raw_formats import save_raw_dict, save_raw_png
from scangen.formats.json_formats import PathEncoder

LOGGER = logging.getLogger("scangen.generate_dataset")


def generate_dataset(
    config: ConfigFormat,
    rgb_batches,
    model: "RAWGenerator",
) -> dict[str, Any]:
    """Generate complete dataset of RAW images.

    Args:
        config: Configuration data structure
        rgb_batches: A dataset from `create_reproducible_cifar_loader()`
        model: A `RAWGenerator`

    Returns:
        Dictionary with generation statistics and metadata
    """
    output_config = config.output
    output_dir = Path(output_config.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    # Create subdirectories
    (output_dir / "pkl").mkdir(exist_ok=True)
    if output_config.save_png:
        (output_dir / "png" / "clean").mkdir(parents=True, exist_ok=True)
        (output_dir / "png" / "noisy").mkdir(parents=True, exist_ok=True)

    # Track all generated files for CSV
    all_files = []
    stats: Dict[str, Union[int, float, None]] = {
        "total_images": 0,
        "total_batches": 0,
        "generation_time": None,
        "average_time_per_image": None,
    }

    start_time = datetime.now()

    # Process each batch
    for batch_idx, (rgb_images, labels, label_names) in enumerate(
        tqdm(rgb_batches, desc="Generating RAW data")
    ):
        # Generate RAW data for this batch
        clean_raw, noisy_raw, batch_metadata = model.generate_batch(
            rgb_images, labels, label_names
        )

        # Save each image in the batch
        for img_idx in range(rgb_images.shape[0]):
            filename_base = f"batch{batch_idx:04d}_img{img_idx:04d}_{label_names[img_idx]}"

            # Prepare data dictionary
            data_dict = {
                "clean": clean_raw[img_idx].cpu().numpy(),  # (4, H//2, W//2)
                "noisy": noisy_raw[img_idx].cpu().numpy(),
                "label": labels[img_idx].item(),
                "label_name": label_names[img_idx],
                "batch_index": batch_idx,
                "image_index": img_idx,
            }

            # Save as pickle
            pkl_path = output_dir / "pkl" / f"{filename_base}.pkl"
            save_raw_dict(data_dict, pkl_path)

            # Save as PNG if requested
            output_config = config.output
            if output_config.save_png:
                clean_png_path = output_dir / "png" / "clean" / f"{filename_base}.png"
                noisy_png_path = output_dir / "png" / "noisy" / f"{filename_base}.png"

                # Use normalize parameter from config, default to False to match CycleISP
                normalize = output_config.normalize_png
                save_raw_png(
                    clean_raw[img_idx : img_idx + 1],
                    clean_png_path,
                    packed=True,
                    normalize=normalize,
                )
                save_raw_png(
                    noisy_raw[img_idx : img_idx + 1],
                    noisy_png_path,
                    packed=True,
                    normalize=normalize,
                )

            # Track for CSV
            all_files.append(
                {
                    "filename": filename_base,
                    "label": labels[img_idx].item(),
                    "label_name": label_names[img_idx],
                    "batch_index": batch_idx,
                }
            )

            stats["total_images"] = cast(int, stats.get("total_images", 0)) + 1
        stats["total_batches"] += 1

    end_time = datetime.now()
    generation_time = end_time - start_time
    stats["generation_time"] = generation_time.total_seconds()
    if stats["total_images"] and stats["total_images"] > 0 and stats["generation_time"]:
        stats["average_time_per_image"] = stats["generation_time"] / stats["total_images"]
    else:
        stats["average_time_per_image"] = 0.0

    # Save labels CSV
    if output_config.save_labels:
        csv_path = output_dir / "labels.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            for file_info in all_files:
                writer.writerow(
                    {"filename": file_info["filename"], "label": file_info["label"]}
                )

    # Save detailed metadata
    metadata_path = output_dir / "metadata.json"
    full_metadata = {
        "generation_config": {
            "dataset": config.dataset.model_dump(),
            "noise": config.noise.model_dump(),
            "output": output_config.model_dump(),
        },
        "statistics": stats,
        "files": all_files,
        "generation_timestamp": start_time.isoformat(),
        "scangen_version": __version__,  # TODO: Get from package version
    }
    LOGGER.info(f"Writing to {metadata_path}")
    with metadata_path.open("w") as f:
        json.dump(full_metadata, f, cls=PathEncoder, indent=2)

    return stats
