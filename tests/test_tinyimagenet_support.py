from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import timm
import torch
from PIL import Image
from torchvision import transforms

import baseline.cifar_resnet  # noqa: F401 - register local timm models
from baseline.baseline_cifar_configs import TINYIMAGENET_DEFAULTS
from baseline.run_baseline import _build_eval_transform, infer_num_classes
from tinyimagenet_data import (
    TINYIMAGENET_MEAN,
    TINYIMAGENET_STD,
    TinyImageNetFlatValDataset,
    build_tinyimagenet_datasets,
)
from trainer_tinyimagenet import TrainerTinyImageNetTimmStyle


def _write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def _make_flat_dataset(root: Path):
    classes = ["n00000001", "n00000002"]
    for class_idx, class_name in enumerate(classes):
        _write_image(root / "train" / class_name / "images" / f"train_{class_idx}.JPEG", class_idx * 50)
    annotations = []
    for class_idx, class_name in enumerate(reversed(classes)):
        image_name = f"val_{class_idx}.JPEG"
        _write_image(root / "val" / "images" / image_name, class_idx * 70)
        annotations.append(f"{image_name}\t{class_name}\t0\t0\t64\t64\n")
    (root / "val" / "val_annotations.txt").write_text("".join(annotations))
    return classes


class TinyImageNetSupportTests(unittest.TestCase):
    def test_tiny_trainer_consumes_cifar_validation_subset_arguments(self):
        with patch.object(TrainerTinyImageNetTimmStyle.__mro__[1], "__init__", return_value=None) as parent_init:
            trainer = TrainerTinyImageNetTimmStyle(
                data_root="/tmp/tiny-imagenet-200",
                validation_samples=0,
                validation_seed=123,
                validate_dataset=False,
            )
        forwarded = parent_init.call_args.kwargs
        self.assertNotIn("validation_samples", forwarded)
        self.assertNotIn("validation_seed", forwarded)
        self.assertEqual(trainer.dataset_name, "tinyimagenet")

        with self.assertRaisesRegex(ValueError, "validation_samples must be zero"):
            TrainerTinyImageNetTimmStyle(
                data_root="/tmp/tiny-imagenet-200",
                validation_samples=1,
                validate_dataset=False,
            )

    def test_canonical_flat_validation_uses_training_class_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            classes = _make_flat_dataset(root)
            train_set, val_set = build_tinyimagenet_datasets(root, validate_counts=False)

            self.assertIsInstance(val_set, TinyImageNetFlatValDataset)
            self.assertEqual(train_set.classes, classes)
            self.assertEqual(val_set.class_to_idx, train_set.class_to_idx)
            self.assertEqual(
                val_set.targets,
                [train_set.class_to_idx[classes[1]], train_set.class_to_idx[classes[0]]],
            )

    def test_class_folder_validation_is_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for class_idx, class_name in enumerate(["n00000001", "n00000002"]):
                _write_image(root / "train" / class_name / "images" / "train.JPEG", class_idx * 50)
                _write_image(root / "val" / class_name / "val.JPEG", class_idx * 70)

            train_set, val_set = build_tinyimagenet_datasets(root, validate_counts=False)
            self.assertEqual(train_set.class_to_idx, val_set.class_to_idx)
            self.assertEqual(len(train_set), 2)
            self.assertEqual(len(val_set), 2)

    def test_tiny_eval_transform_is_native_resolution_and_uses_tiny_stats(self):
        model = torch.nn.Identity()
        transform = _build_eval_transform(model, "tinyimagenet", TINYIMAGENET_DEFAULTS)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertEqual(
            [type(item) for item in transform.transforms],
            [transforms.ToTensor, transforms.Normalize],
        )
        normalize = transform.transforms[1]
        self.assertEqual(tuple(normalize.mean), TINYIMAGENET_MEAN)
        self.assertEqual(tuple(normalize.std), TINYIMAGENET_STD)
        self.assertEqual(transform(Image.new("RGB", (64, 64))).shape, (3, 64, 64))

    def test_tiny_recipe_and_class_count_are_pinned(self):
        self.assertEqual(infer_num_classes("tinyimagenet", None), 200)
        self.assertEqual(TINYIMAGENET_DEFAULTS["timm_input_size"], (3, 64, 64))
        self.assertEqual(TINYIMAGENET_DEFAULTS["weight_decay"], 1e-3)
        self.assertEqual(TINYIMAGENET_DEFAULTS["final_dropout_rate"], 0.25)
        self.assertEqual(TINYIMAGENET_DEFAULTS["dropout_rate"], 0.0)
        self.assertEqual(TINYIMAGENET_DEFAULTS["skip_eval_epochs"], 75)

    def test_all_planned_wrn_and_bn_free_factories_build_for_200_classes(self):
        names = [
            "wrn_16_2_cifar", "wrn_16_4_cifar", "wrn_16_8_cifar",
            "wrn_28_2_cifar", "wrn_28_4_cifar", "wrn_40_2_cifar",
            "wrn_16_2_cifar_nobn", "wrn_16_4_cifar_nobn", "wrn_16_8_cifar_nobn",
            "wrn_28_2_cifar_nobn", "wrn_28_4_cifar_nobn", "wrn_40_2_cifar_nobn",
        ]
        for name in names:
            with self.subTest(name=name):
                model = timm.create_model(name, num_classes=200)
                self.assertEqual(model.fc.out_features, 200)

    def test_cifar_style_wrn_keeps_two_reductions_for_64px_input(self):
        model = timm.create_model("wrn_16_2_cifar", num_classes=200)
        model.eval()
        with torch.no_grad():
            features = model.forward_features(torch.zeros(1, 3, 64, 64))
            logits = model.forward_head(features)
        self.assertEqual(tuple(features.shape[-2:]), (16, 16))
        self.assertEqual(tuple(logits.shape), (1, 200))


if __name__ == "__main__":
    unittest.main()
