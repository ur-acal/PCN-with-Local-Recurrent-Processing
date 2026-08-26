"""Train an RGGB EfficientNetV2-L teacher with the timm-style trainer."""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing as TimmRandomErasing
from torchvision.transforms import InterpolationMode

from baseline.baseline_cifar_configs import CASE_DEFAULTS, RGGB_DEFAULTS
from train_teacher import (
    DirectTensorResize,
    load_efficientnet_v2_4ch,
    load_hankyul_efficientnet_v2_4ch,
)
from trainer_timm import TrainerCiFarTimmStyle


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def normalize_img_type(img_type: str) -> str:
    value = img_type.lower()
    if value == "cifair":
        return "CiFAIR"
    if value in {"scangfi", "raw", "_raw"}:
        return "scanGFI"
    raise ValueError("img_type must be scanGFI/raw or CiFAIR")


class RGGBResizeFineTuneTrainer(TrainerCiFarTimmStyle):
    """Timm trainer with native RGGB augmentation followed by 224x224 resize."""

    def __init__(
        self,
        *args,
        source_input_size: int = 16,
        teacher_checkpoint: str,
        teacher_arch_source: str = "hankyul2",
        use_old_augs_for_timm: bool = False,
        use_direct_resize_for_timm_augs: bool = False,
        match_distill_aug_order: bool = False,
        **kwargs,
    ):
        self.source_input_size = source_input_size
        self.teacher_checkpoint = Path(teacher_checkpoint).expanduser()
        self.teacher_arch_source = teacher_arch_source
        self.use_old_augs_for_timm = use_old_augs_for_timm
        self.use_direct_resize_for_timm_augs = use_direct_resize_for_timm_augs
        self.match_distill_aug_order = match_distill_aug_order
        super().__init__(*args, **kwargs)
        self._distill_order_pre_hook = None
        if self.match_distill_aug_order:
            self._distill_order_pre_hook = self.model.register_forward_pre_hook(
                self._resize_normalize_before_teacher_forward
            )

        # Match the legacy teacher-data convention.
        if hasattr(self.train_set, "noisy_inp"):
            self.train_set.noisy_inp = True
        if hasattr(self.val_set, "noisy_inp"):
            self.val_set.noisy_inp = False

    def _resize_normalize_before_teacher_forward(self, _module, args):
        """Resize and normalize at the distillation teacher-input boundary."""
        if not args:
            return None
        inputs = args[0]
        target_size = self.timm_input_size[-2:]
        if not isinstance(inputs, torch.Tensor) or inputs.shape[-2:] == target_size:
            return None
        if inputs.shape[-2:] != (self.source_input_size, self.source_input_size):
            raise ValueError(
                "Distillation-order teacher expected "
                f"{self.source_input_size}x{self.source_input_size} inputs before "
                f"resize, received {tuple(inputs.shape[-2:])}"
            )
        inputs = torch.nn.functional.interpolate(
            inputs, size=target_size, mode="bilinear", align_corners=False
        )
        mean = inputs.new_tensor(self.timm_mean).view(1, -1, 1, 1)
        std = inputs.new_tensor(self.timm_std).view(1, -1, 1, 1)
        inputs = (inputs - mean) / std
        return (inputs, *args[1:])

    def _build_non_rgb_timm_transforms(self, img_type):
        if self.use_old_augs_for_timm:
            target_height, target_width = self.timm_input_size[-2:]
            if target_height != target_width:
                raise ValueError(
                    "The old no-PIL augmentation pipeline requires a square input size"
                )

            # Match train_teacher.py with:
            #   --match_distill_preprocess --train_transform cifar
            #   --train_size 224 --test_size 224 --test_center_crop
            resize = DirectTensorResize(target_height)
            normalize = transforms.Normalize(self.timm_mean, self.timm_std)
            transform_train = transforms.Compose([
                resize,
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(target_height),
                transforms.RandomHorizontalFlip(p=self.hflip),
                normalize,
            ])
            transform_test = transforms.Compose([
                resize,
                transforms.CenterCrop(target_height),
                normalize,
            ])
            return transform_train, transform_test

        train_steps = []

        if self.non_rgb_spatial_aug:
            train_steps.append(
                transforms.RandomCrop(
                    self.source_input_size,
                    padding=self.non_rgb_crop_padding,
                )
            )
            if self.hflip > 0.0:
                train_steps.append(transforms.RandomHorizontalFlip(p=self.hflip))
            if self.vflip > 0.0:
                train_steps.append(transforms.RandomVerticalFlip(p=self.vflip))
            if (
                self.non_rgb_affine_degrees
                or self.non_rgb_affine_translate is not None
                or self.non_rgb_affine_shear is not None
            ):
                train_steps.append(
                    transforms.RandomAffine(
                        degrees=self.non_rgb_affine_degrees,
                        translate=self.non_rgb_affine_translate,
                        shear=self.non_rgb_affine_shear,
                    )
                )

        target_size = self.timm_input_size[-2:]
        if self.use_direct_resize_for_timm_augs or self.match_distill_aug_order:
            if target_size[0] != target_size[1]:
                raise ValueError("DirectTensorResize requires a square input size")
            resize = DirectTensorResize(target_size[0])
        else:
            resize = transforms.Resize(
                target_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        normalize = transforms.Normalize(self.timm_mean, self.timm_std)

        if self.match_distill_aug_order:
            # spatial aug -> RE at 16 -> Mixup/CutMix at 16 (train loop)
            # -> direct resize/normalize (model pre-hook)
            if self.re_prob > 0.0:
                train_steps.append(
                    TimmRandomErasing(
                        probability=self.re_prob,
                        mode=self.re_mode,
                        max_count=self.re_count,
                        device="cpu",
                    )
                )
            return transforms.Compose(train_steps), transforms.Compose([resize, normalize])

        train_steps.extend([resize, normalize])

        if self.re_prob > 0.0:
            train_steps.append(
                TimmRandomErasing(
                    probability=self.re_prob,
                    mode=self.re_mode,
                    max_count=self.re_count,
                    device="cpu",
                )
            )

        transform_train = transforms.Compose(train_steps)
        transform_test = transforms.Compose([resize, normalize])
        return transform_train, transform_test

    def _save_model_ckpt(self, acc, epoch, suffix=""):
        if suffix == "_best_ckpt.pth":
            save_path = self.teacher_checkpoint
        else:
            save_path = self.teacher_checkpoint.with_name(
                self.teacher_checkpoint.stem + "_last.pth"
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.model.state_dict(),
                "net_type": self.model.__class__.__name__,
                "acc": acc,
                "epoch": epoch,
                "dataset_name": self.dataset_name,
                "img_type": self.img_type,
                "arch": "efficientnet_v2_l",
                "arch_source": self.teacher_arch_source,
                "training_recipe": (
                    "rggb_distillation_order_timm_augs"
                    if self.match_distill_aug_order
                    else (
                        "rggb_resize_finetune_old_no_pil_augs"
                        if self.use_old_augs_for_timm
                        else "rggb_resize_finetune"
                    )
                ),
                "use_old_augs_for_timm": self.use_old_augs_for_timm,
                "use_direct_resize_for_timm_augs": self.use_direct_resize_for_timm_augs,
                "match_distill_aug_order": self.match_distill_aug_order,
                "timm_input_size": self.timm_input_size,
                "timm_mean": self.timm_mean,
                "timm_std": self.timm_std,
            },
            save_path,
        )
        return str(save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNetV2-L RGGB teacher with the timm-style trainer"
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--img_type", default="scanGFI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument(
        "--arch_source",
        choices=("hankyul2", "torchvision"),
        default="hankyul2",
        help="EfficientNetV2-L implementation and ImageNet-pretrained weight source.",
    )
    parser.add_argument(
        "--use_old_augs_for_timm",
        type=str2bool,
        default=False,
        help=(
            "Keep the timm model/optimizer/schedule but use the legacy no-PIL "
            "teacher transforms and disable timm-only Mixup, CutMix, label "
            "smoothing, affine augmentation, and Random Erasing."
        ),
    )
    parser.add_argument(
        "--use_direct_resize_for_timm_augs",
        type=str2bool,
        default=False,
        help=(
            "With timm augmentations enabled, use the legacy/distillation direct "
            "bilinear tensor resize for both training and validation."
        ),
    )
    parser.add_argument(
        "--match_distill_aug_order",
        type=str2bool,
        default=False,
        help=(
            "Apply PCN timm augmentation, RE and Mixup/CutMix at native 16x16, "
            "then direct-resize and normalize at the EfficientNet forward. "
            "Random Erasing defaults to 0 in this mode."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--timm_opt", choices=("adamw", "sgd"), default=None)
    parser.add_argument(
        "--timm_sched", choices=("cosine", "multistep"), default=None
    )
    parser.add_argument(
        "--lr_reduce_on",
        default=None,
        help="Comma-separated epoch milestones used by the multistep scheduler.",
    )
    parser.add_argument("--decay_rate", type=float, default=None)
    parser.add_argument(
        "--first_eval_epoch",
        type=int,
        default=None,
        help="First completed epoch at which evaluation is allowed (1-based).",
    )
    parser.add_argument(
        "--re_prob", type=float, default=None,
        help="Override the timm Random Erasing probability (0 disables it).",
    )
    parser.add_argument("--eval_every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.img_type = normalize_img_type(args.img_type)
    if args.data_root:
        os.environ["SCANGEN_DATA_ROOT"] = args.data_root

    cfg = deepcopy(CASE_DEFAULTS["resize_finetune"])
    cfg.update(RGGB_DEFAULTS)
    cfg["timm_input_size"] = (4, 224, 224)
    cfg["pretrained"] = args.pretrained

    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.test_batch_size is not None:
        cfg["test_batch_size"] = args.test_batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        cfg["warmup_epoch"] = args.warmup_epochs
    if args.timm_opt is not None:
        cfg["timm_opt"] = args.timm_opt
    if args.timm_sched is not None:
        cfg["timm_sched"] = args.timm_sched
    if args.lr_reduce_on is not None:
        cfg["lr_reduce_on"] = args.lr_reduce_on
    if args.decay_rate is not None:
        cfg["decay_rate"] = args.decay_rate
    if args.first_eval_epoch is not None:
        if args.first_eval_epoch < 1:
            raise ValueError("--first_eval_epoch must be at least 1")
        cfg["skip_eval_epochs"] = args.first_eval_epoch - 1
    if args.match_distill_aug_order and args.use_old_augs_for_timm:
        raise ValueError(
            "--match_distill_aug_order and --use_old_augs_for_timm are "
            "mutually exclusive"
        )
    if args.use_old_augs_for_timm:
        if args.use_direct_resize_for_timm_augs:
            raise ValueError(
                "--use_direct_resize_for_timm_augs only applies when "
                "--use_old_augs_for_timm=false"
            )
        if args.re_prob not in (None, 0.0):
            raise ValueError(
                "--re_prob must be omitted or 0 when --use_old_augs_for_timm=true"
            )
        cfg.update({
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
            "auto_augment": None,
            "color_jitter": 0.0,
            "re_prob": 0.0,
            "non_rgb_affine_degrees": 0,
            "non_rgb_affine_translate": None,
            "non_rgb_affine_shear": None,
        })
    if args.match_distill_aug_order and args.re_prob is None:
        cfg["re_prob"] = 0.0
    if args.re_prob is not None:
        if not 0.0 <= args.re_prob <= 1.0:
            raise ValueError("--re_prob must be between 0 and 1")
        cfg["re_prob"] = args.re_prob

    num_classes = 100 if args.dataset == "cifar100" else 10
    if args.arch_source == "hankyul2":
        model = load_hankyul_efficientnet_v2_4ch(
            num_classes=num_classes,
            arch="efficientnet_v2_l",
            in_channels=4,
            pretrained=cfg["pretrained"],
        )
    else:
        model = load_efficientnet_v2_4ch(
            num_classes=num_classes,
            arch="efficientnet_v2_l",
            in_channels=4,
            pretrained=cfg["pretrained"],
        )
    model_name = (
        f"efficientnet_v2_l_{args.dataset}_{args.img_type}_"
        f"{args.arch_source}_timm"
    )

    print("Resolved timm teacher configuration:")
    for key in (
        "timm_input_size", "num_epochs", "batch_size", "test_batch_size",
        "lr", "weight_decay", "timm_opt", "timm_sched", "warmup_epoch",
        "label_smoothing", "mixup_alpha", "cutmix_alpha", "re_prob",
    ):
        print(f"  {key}: {cfg[key]}")
    if cfg["timm_sched"] == "multistep":
        print(f"  lr_reduce_on: {cfg.get('lr_reduce_on', '80,122,150,225,262')}")
        print(f"  decay_rate: {cfg.get('decay_rate', 0.1)}")
    print(f"  dataset: {args.dataset}")
    print(f"  img_type: {args.img_type}")
    print(f"  arch_source: {args.arch_source}")
    print(f"  first_eval_epoch: {cfg['skip_eval_epochs'] + 1}")
    print(f"  use_old_augs_for_timm: {args.use_old_augs_for_timm}")
    print(
        "  use_direct_resize_for_timm_augs: "
        f"{args.use_direct_resize_for_timm_augs}"
    )
    print(f"  match_distill_aug_order: {args.match_distill_aug_order}")
    print(
        "  augmentation_pipeline: "
        + ("distill_order_timm_rggb" if args.match_distill_aug_order
           else ("old_no_pil" if args.use_old_augs_for_timm else "timm_rggb"))
    )
    print(f"  checkpoint: {args.checkpoint}")

    trainer = RGGBResizeFineTuneTrainer(
        model=model,
        model_name=model_name,
        save_path=str(Path(args.checkpoint).expanduser().parent),
        batch_size=cfg["batch_size"],
        optim_type="sgd",
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["lr"],
        num_epochs=cfg["num_epochs"],
        warmup_epoch=cfg["warmup_epoch"],
        test_bs=cfg["test_batch_size"],
        max_norm=cfg.get("max_norm"),
        aug=False,
        eval_every=args.eval_every,
        img_type=args.img_type,
        dataset_name=args.dataset,
        distill_method="none",
        distill_alpha=0.0,
        teacher_model=None,
        timm_opt=cfg["timm_opt"],
        momentum=cfg["momentum"],
        timm_sched=cfg["timm_sched"],
        min_lr=cfg["min_lr"],
        warmup_lr=cfg["warmup_lr"],
        decay_rate=cfg.get("decay_rate", 0.1),
        lr_reduce_on=cfg.get("lr_reduce_on", "80,122,150,225,262"),
        timm_aug=True,
        timm_input_size=cfg["timm_input_size"],
        timm_mean=(0.5, 0.5, 0.5, 0.5),
        timm_std=(0.5, 0.5, 0.5, 0.5),
        interpolation="bilinear",
        timm_train_scale=cfg["timm_train_scale"],
        timm_train_ratio=cfg["timm_train_ratio"],
        hflip=cfg["hflip"],
        color_jitter=cfg["color_jitter"],
        auto_augment=cfg["auto_augment"],
        re_prob=cfg["re_prob"],
        label_smoothing=cfg["label_smoothing"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],
        convert_non_rgb_to_rgb=False,
        non_rgb_spatial_aug=cfg["non_rgb_spatial_aug"],
        non_rgb_crop_padding=cfg["non_rgb_crop_padding"],
        non_rgb_affine_degrees=cfg["non_rgb_affine_degrees"],
        non_rgb_affine_translate=cfg["non_rgb_affine_translate"],
        non_rgb_affine_shear=cfg["non_rgb_affine_shear"],
        source_input_size=16,
        teacher_checkpoint=args.checkpoint,
        teacher_arch_source=args.arch_source,
        use_old_augs_for_timm=args.use_old_augs_for_timm,
        use_direct_resize_for_timm_augs=args.use_direct_resize_for_timm_augs,
        match_distill_aug_order=args.match_distill_aug_order,
        is_timm_model=True,
        skip_eval_epochs=cfg["skip_eval_epochs"],
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    trainer.train()


if __name__ == "__main__":
    main()
