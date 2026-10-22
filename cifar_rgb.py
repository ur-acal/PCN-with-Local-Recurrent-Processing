"""Official RGB ciFAIR loaders and CIFAR/ciFAIR task routing.

ciFAIR replaces only the test set. Training always uses torchvision CIFAR.
Names, release URLs and hashes follow cvjena/cifair/loaders/pytorch/cifair.py.
This module does not implement sensor/RGGB preprocessing.
"""
from torchvision import datasets


RGB_DATASETS = ("cifar10", "cifar100", "cifair10", "cifair100")


def dataset_name(name):
    normalized = (name or "cifar10").lower().replace("-", "").replace("_", "")
    if normalized not in RGB_DATASETS:
        raise ValueError(f"Unsupported CIFAR/ciFAIR dataset: {name}")
    return normalized


def cifar_task(name):
    return dataset_name(name).replace("cifair", "cifar")


def require_native_rgb_config(cfg):
    if cfg.get("in_chans", 3) != 3 or tuple(cfg.get("timm_input_size", (3, 32, 32))) != (3, 32, 32):
        raise ValueError("This evaluator requires native 3-channel 32x32 RGB baselines")


class CiFAIR10(datasets.CIFAR10):
    base_folder = "ciFAIR-10"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip"
    filename = "ciFAIR-10.zip"
    tgz_md5 = "ca08fd390f0839693d3fc45c4e49585f"
    test_list = [["test_batch", "01290e6b622a1977a000eff13650aca2"]]


class CiFAIR100(datasets.CIFAR100):
    base_folder = "ciFAIR-100"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip"
    filename = "ciFAIR-100.zip"
    tgz_md5 = "ddc236ab4b12eeb8b20b952614861a33"
    test_list = [["test", "8130dae8d6fc6a436437f0ebdb801df1"]]


def rgb_dataset_class(name, *, train=False):
    requested = dataset_name(name)
    if not train and requested.startswith("cifair"):
        return CiFAIR100 if cifar_task(requested) == "cifar100" else CiFAIR10
    return datasets.CIFAR100 if cifar_task(requested) == "cifar100" else datasets.CIFAR10


def require_rgb_alias(name, img_type):
    if dataset_name(name).startswith("cifair") and img_type.lower() != "rgb":
        raise ValueError("dataset=cifair10/cifair100 selects sRGB ciFAIR and requires img_type=rgb; "
                         "keep dataset=cifar10/cifar100 for the existing sensor/RGGB pipeline")
