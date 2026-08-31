from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


TINYIMAGENET_ALIASES = {"tinyimagenet", "tiny-imagenet", "tiny_imagenet"}
TINYIMAGENET_NUM_CLASSES = 200
TINYIMAGENET_TRAIN_SAMPLES = 100000
TINYIMAGENET_VAL_SAMPLES = 10000
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


def canonical_tinyimagenet_name(name):
    normalized = str(name).lower()
    if normalized not in TINYIMAGENET_ALIASES:
        raise ValueError(f"Unsupported Tiny ImageNet dataset name: {name}")
    return "tinyimagenet"


class TinyImageNetFlatValDataset(Dataset):
    """Canonical Tiny ImageNet validation split with labels in val_annotations.txt."""

    def __init__(self, val_dir, class_to_idx, transform=None):
        self.val_dir = Path(val_dir)
        self.transform = transform
        images_dir = self.val_dir / "images"
        annotations = self.val_dir / "val_annotations.txt"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Tiny ImageNet validation images not found: {images_dir}")
        if not annotations.is_file():
            raise FileNotFoundError(f"Tiny ImageNet annotations not found: {annotations}")

        labels_by_name = {}
        with annotations.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                labels_by_name[fields[0]] = fields[1]

        unknown = sorted(set(labels_by_name.values()) - set(class_to_idx))
        if unknown:
            raise ValueError(f"Validation annotations contain unknown classes: {unknown[:5]}")

        self.samples = []
        for image_name in sorted(labels_by_name):
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Annotated validation image not found: {image_path}")
            self.samples.append((image_path, class_to_idx[labels_by_name[image_name]]))
        self.targets = [target for _, target in self.samples]
        self.class_to_idx = dict(class_to_idx)
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_tinyimagenet_datasets(
    root,
    train_transform=None,
    val_transform=None,
    validate_counts=True,
):
    root = Path(root).expanduser()
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Tiny ImageNet train directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Tiny ImageNet validation directory not found: {val_dir}")

    train_set = ImageFolder(str(train_dir), transform=train_transform)
    class_dirs = [path for path in val_dir.iterdir() if path.is_dir() and path.name in train_set.class_to_idx]
    if class_dirs:
        val_set = ImageFolder(str(val_dir), transform=val_transform)
        if val_set.class_to_idx != train_set.class_to_idx:
            raise ValueError("Tiny ImageNet train and validation class mappings differ.")
    else:
        val_set = TinyImageNetFlatValDataset(
            val_dir,
            class_to_idx=train_set.class_to_idx,
            transform=val_transform,
        )

    if validate_counts:
        if len(train_set.classes) != TINYIMAGENET_NUM_CLASSES:
            raise ValueError(
                f"Expected {TINYIMAGENET_NUM_CLASSES} Tiny ImageNet classes, "
                f"found {len(train_set.classes)} in {train_dir}."
            )
        if len(train_set) != TINYIMAGENET_TRAIN_SAMPLES:
            raise ValueError(
                f"Expected {TINYIMAGENET_TRAIN_SAMPLES} training images, "
                f"found {len(train_set)} in {train_dir}."
            )
        if len(val_set) != TINYIMAGENET_VAL_SAMPLES:
            raise ValueError(
                f"Expected {TINYIMAGENET_VAL_SAMPLES} validation images, "
                f"found {len(val_set)} in {val_dir}."
            )

    return train_set, val_set
