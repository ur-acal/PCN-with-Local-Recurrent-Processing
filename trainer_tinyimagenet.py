import torch
from timm.data import create_transform
from torchvision import transforms

from tinyimagenet_data import (
    TINYIMAGENET_MEAN,
    TINYIMAGENET_NUM_CLASSES,
    TINYIMAGENET_STD,
    build_tinyimagenet_datasets,
    canonical_tinyimagenet_name,
)
from trainer_imagenet import TrainerImageNetTimmStyle


class TrainerTinyImageNetTimmStyle(TrainerImageNetTimmStyle):
    """Tiny ImageNet data path with the shared timm-style training loop."""

    def __init__(
        self,
        *args,
        tinyimagenet_root=None,
        data_root=None,
        validate_dataset=True,
        **kwargs,
    ):
        root = tinyimagenet_root if tinyimagenet_root is not None else data_root
        if root is None:
            raise ValueError("Please provide tinyimagenet_root or data_root.")
        requested_dataset = kwargs.pop("dataset_name", "tinyimagenet")
        self.tinyimagenet_dataset_name = canonical_tinyimagenet_name(requested_dataset)
        self.validate_tinyimagenet_dataset = bool(validate_dataset)
        validation_samples = int(kwargs.pop("validation_samples", 0))
        kwargs.pop("validation_seed", None)
        if validation_samples:
            raise ValueError(
                "Tiny ImageNet uses its official validation split; "
                "validation_samples must be zero."
            )

        kwargs.setdefault("timm_input_size", (3, 64, 64))
        kwargs.setdefault("timm_mean", TINYIMAGENET_MEAN)
        kwargs.setdefault("timm_std", TINYIMAGENET_STD)
        kwargs.setdefault("interpolation", "bicubic")
        timm_train_scale = kwargs.pop("timm_train_scale", (0.75, 1.0))
        timm_train_ratio = kwargs.pop("timm_train_ratio", (0.75, 4.0 / 3.0))
        color_jitter = kwargs.pop("color_jitter", 0.1)
        super().__init__(
            *args,
            imagenet_root=root,
            num_classes=TINYIMAGENET_NUM_CLASSES,
            use_model_data_config=False,
            timm_train_scale=timm_train_scale,
            timm_train_ratio=timm_train_ratio,
            color_jitter=color_jitter,
            dataset_name="imagenet",
            **kwargs,
        )
        self.dataset_name = self.tinyimagenet_dataset_name

    def _prepare_dataset(self):
        if self.timm_aug:
            transform_train, _ = self._build_rgb_timm_transforms("tinyimagenet")
        else:
            input_size, mean, std, interpolation = self._get_timm_data_config("tinyimagenet")
            transform_train = create_transform(
                input_size=input_size,
                is_training=True,
                use_prefetcher=False,
                no_aug=True,
                interpolation=interpolation,
                mean=mean,
                std=std,
            )
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
        ])

        self.train_set, self.val_set = build_tinyimagenet_datasets(
            self.imagenet_root,
            train_transform=transform_train,
            val_transform=transform_val,
            validate_counts=self.validate_tinyimagenet_dataset,
        )
        self._train_sample_count = len(self.train_set)
        persistent_workers = self.persistent_workers if self.num_workers > 0 else False
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        )
        if self.teacher_eval_loader is None:
            self.teacher_eval_loader = self.val_dataloader
