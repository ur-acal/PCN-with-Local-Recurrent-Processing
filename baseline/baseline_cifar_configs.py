"""
Configs for CIFAR baseline training with TrainerCiFarTimmStyle.

Cases:
    1a: custom_noresize:
        self-defined CIFAR models registered into timm, e.g.
        resnet20_cifar, resnet56_cifar, wrn_28_10_cifar.
        Train from scratch only.

    1b: adapt_noresize:
        built-in timm models adapted to native CIFAR 32x32 by replacing some
        downsampling layers in the beginning, e.g.
        resnet18, resnet34, resnet50, mobilenetv2_100, vgg19.
        Train from scratch preferred.

    2: resize:
        built-in timm models used as-is with resized CIFAR input, usually 224.
        Train from scratch or fine-tune based on imagenet pretrained models.
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import timm
from copy import deepcopy
from baseline.adapt_model_cifar import adapt_timm_model_to_cifar, is_supported_cifar_adapt_model


CUSTOM_CIFAR_MODELS = {
    "resnet20_cifar",
    "resnet32_cifar",
    "resnet44_cifar",
    "resnet56_cifar",
    "resnet110_cifar",
    "preact_resnet164_cifar",
    "wrn_28_10_cifar",
    "wrn_28_2_cifar_avgpool",
    "wrn_28_2_cifar_avgpool_shortcut",
    "wrn_28_2_cifar_nobn_avgpool",
    "wrn_28_2_cifar_nobn_avgpool_shortcut",
    "pcn_16l96c_cnn_avgpool",
    "wrn_flexible_cifar_avgpool",
    "wrn_flexible_cifar_avgpool_shortcut",
}


ADAPT_NORESIZE_TIMM_MODELS = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext26ts",
    "resnext50_32x4d",
    "seresnet18",
    "seresnet34",
    "seresnet50",
    "mobilenetv2_100",
    "vgg19",
}

RESIZE_FINETUNE_TIMM_MODELS = {
    # EfficientNet baseline
    "efficientnet_b0",

    # Mobile baseline
    "mobilenetv3_small_100",

    # ViT-family baselines
    "vit_tiny_patch16_224",
    "deit_tiny_patch16_224",

    # MLP-style baseline
    "mixer_b16_224",

    # Modern ConvNet baseline
    "convnext_tiny",
}

# -----------------------------------------------------------------------------
# Default configs
# -----------------------------------------------------------------------------
CASE_DEFAULTS = {
    # 1a. Native CIFAR models, no resize, scratch only.
    "custom_noresize": {
        "pretrained": False,
        "timm_input_size": (3, 32, 32),
        "num_epochs": 300,
        "batch_size": 128,
        "test_batch_size": 1024,
        "lr": 0.1,
        "weight_decay": 5e-4,
        "timm_opt": "sgd",
        "momentum": 0.9,
        "timm_sched": "cosine",
        "min_lr": 1e-6,
        "warmup_epoch": 5,
        "warmup_lr": 1e-5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.1,
        "re_prob": 0.25,
        "timm_train_scale": (1.0, 1.0),
        "timm_train_ratio": (1.0, 1.0),
        "hflip": 0.5,
        "skip_eval_epochs": 70,
    },

    # 1b. Built-in timm model adapted to 32x32, scratch.
    "adapt_noresize_scratch": {
        "pretrained": False,
        "timm_input_size": (3, 32, 32),
        "num_epochs": 300,
        "batch_size": 128,
        "test_batch_size": 256,
        "lr": 0.1,
        "weight_decay": 5e-4,
        "timm_opt": "sgd",
        "momentum": 0.9,
        "timm_sched": "cosine",
        "warmup_epoch": 5,
        "warmup_lr": 1e-5,
        "min_lr": 1e-6,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.1,
        "re_prob": 0.25,
        "timm_train_scale": (1.0, 1.0),
        "timm_train_ratio": (1.0, 1.0),
        "hflip": 0.5,
        "skip_eval_epochs": 70,
    },

    # 1b. Built-in timm model adapted to 32x32, ImageNet-pretrained fine-tune.
    # Less canonical than resize fine-tune, because the stem is modified.
    "adapt_noresize_finetune": {
        "pretrained": True,
        "timm_input_size": (3, 32, 32),
        "num_epochs": 100,
        "batch_size": 128,
        "test_batch_size": 256,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "timm_opt": "sgd",
        "momentum": 0.9,
        "timm_sched": "cosine",
        "warmup_epoch": 5,
        "min_lr": 1e-6,
        "warmup_lr": 1e-5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.1,
        "re_prob": 0.25,
        "timm_train_scale": (1.0, 1.0),
        "timm_train_ratio": (1.0, 1.0),
        "hflip": 0.5,
        "skip_eval_epochs": 30,
    },

    # 2. Built-in timm model, resize CIFAR to ImageNet-style input, scratch.
    "resize_scratch": {
        "pretrained": False,
        "timm_input_size": (3, 224, 224),
        "num_epochs": 200,
        "batch_size": 128,
        "test_batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "timm_opt": "adamw",
        "momentum": 0.9,
        "timm_sched": "cosine",
        "warmup_epoch": 5,
        "min_lr": 1e-6,
        "warmup_lr": 1e-6,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.1,
        "re_prob": 0.25,
        "timm_train_scale": (0.75, 1.0),
        "timm_train_ratio": (1.0, 1.0),
        "hflip": 0.5,
        "skip_eval_epochs": 70,
    },

    # 2. Built-in timm model, resize CIFAR to ImageNet-style input, fine-tune.
    "resize_finetune": {
        "pretrained": True,
        "timm_input_size": (3, 224, 224),
        "num_epochs": 50,
        "batch_size": 128,
        "test_batch_size": 256,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "timm_opt": "adamw",
        "momentum": 0.9,
        "timm_sched": "cosine",
        "warmup_epoch": 5,
        "min_lr": 1e-6,
        "warmup_lr": 1e-6,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.1,
        "re_prob": 0.25,
        "timm_train_scale": (0.75, 1.0),
        "timm_train_ratio": (1.0, 1.0),
        "hflip": 0.5,
        "skip_eval_epochs": 20,
    },

    "imagenet1k_scratch": {
        "pretrained": False,
        "timm_input_size": (3, 224, 224),

        "num_epochs": 300,
        "batch_size": 128,
        "test_batch_size": 128,

        "lr": 0.01,
        "weight_decay": 1e-4,
        "timm_opt": "sgd",
        "momentum": 0.9,

        "timm_sched": "cosine",
        "warmup_epoch": 5,
        "min_lr": 1e-6,
        "warmup_lr": 1e-6,

        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,

        "auto_augment": "rand-m9-mstd0.5-inc1",
        "color_jitter": 0.4,
        "re_prob": 0.25,

        "timm_train_scale": (0.08, 1.0),
        "timm_train_ratio": (3.0 / 4.0, 4.0 / 3.0),
        "hflip": 0.5,

        "amp_enabled": True,
        "amp_dtype": "bf16",
        "grad_accum_steps": 2,

        "skip_eval_epochs": 200,
    },
}
# -----------------------------------------------------------------------------
# Some updated configs for rggb data
# -----------------------------------------------------------------------------
RGGB_TO_RGB_EXTRAS = {
    "convert_non_rgb_to_rgb": True,
    "in_chans": 3,
}

RGGB_DEFAULTS = {
    "convert_non_rgb_to_rgb": False,
    "in_chans": 4,
    "timm_input_size": (4, 16, 16),
    # The two are disabled for rggb
    "auto_augment": None,
    "color_jitter": 0.0,
    # decreased from 0.25
    "re_prob": 0.08,
    # Mimicking the other transforms in "auto_augment": "rand-m9-mstd0.5-inc1" that are applicable to rggb
    "non_rgb_spatial_aug": True,
    "non_rgb_crop_padding": 2,

    "non_rgb_affine_degrees": 9,
    "non_rgb_affine_translate": (0.125, 0.125),
    "non_rgb_affine_shear": 5,
}

RGGB_NO_AUG = {
    # disable strong mixing first
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "label_smoothing": 0.0,

    # keep only mild spatial augmentation
    "auto_augment": None,
    "color_jitter": 0.0,
    "re_prob": 0.0,

    "non_rgb_spatial_aug": True,
    "non_rgb_crop_padding": 2,

    # Disable affine first
    "non_rgb_affine_degrees": 0,
    "non_rgb_affine_translate": None,
    "non_rgb_affine_shear": None,
}

RGGB_MILD_AUG = deepcopy(RGGB_NO_AUG)
RGGB_MILD_AUG.update(
    {
        "label_smoothing": 0.05,
        "re_prob": 0.03,
        "non_rgb_affine_degrees": 5,
        "non_rgb_affine_translate": (0.0625, 0.0625),
        "non_rgb_affine_shear": 0,
    }
)

RGGB_MID_AUG = deepcopy(RGGB_MILD_AUG)
RGGB_MID_AUG.update(
    {
        "mixup_alpha": 0.1,
        "cutmix_alpha": 0.5,
        "label_smoothing": 0.05,
    }
)

# -----------------------------------------------------------------------------
# Optional model-specific overrides
# -----------------------------------------------------------------------------


MODEL_OVERRIDES = {
    # Bigger native CIFAR baselines often benefit from longer training.
    "wrn_28_10_cifar": {},
    "preact_resnet164_cifar": {},
    "vgg19": {},
    "mobilenetv2_100": {},
    "wrn_28_2_cifar_avgpool": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "wrn_28_2_cifar_avgpool_shortcut": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "wrn_28_2_cifar_nobn_avgpool": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "wrn_28_2_cifar_nobn_avgpool_shortcut": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "pcn_16l96c_cnn_avgpool": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "wrn_flexible_cifar_avgpool": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
    "wrn_flexible_cifar_avgpool_shortcut": {
        "dropout_rate": 0.0, "final_dropout_rate": 0.25},
}


def infer_case(model_name: str, pretrained: bool = False, prefer_resize: bool = False) -> str:
    """
    Infer the default training case from model_name.

    Rules:
        custom CIFAR model -> custom_noresize
        supported adapted timm model -> adapt_noresize unless prefer_resize=True
        otherwise -> resize
    """
    name = model_name.lower()

    if name in CUSTOM_CIFAR_MODELS:
        return "custom_noresize"

    if name in ADAPT_NORESIZE_TIMM_MODELS and not prefer_resize:
        return "adapt_noresize_finetune" if pretrained else "adapt_noresize_scratch"

    return "resize_finetune" if pretrained else "resize_scratch"


def get_baseline_config(
    model_name: str,
    pretrained: bool = False,
    case: str = "auto",
    prefer_resize: bool = False,
    extra_overrides: dict | None = None,
) -> dict:
    """
    Return merged config:
        case default -> model-specific override -> extra_overrides
    """
    if case == "auto":
        case = infer_case(model_name, pretrained=pretrained, prefer_resize=prefer_resize)

    cfg = deepcopy(CASE_DEFAULTS[case])

    # Explicit pretrained argument wins over default case value except custom_noresize.
    if case == "custom_noresize":
        cfg["pretrained"] = False
    else:
        cfg["pretrained"] = bool(pretrained)

    if model_name in MODEL_OVERRIDES:
        cfg.update(deepcopy(MODEL_OVERRIDES[model_name]))

    if extra_overrides:
        cfg.update(extra_overrides)

    cfg["case"] = case
    cfg["model_name"] = model_name
    return cfg

def build_model(model_name: str, cfg: dict, num_classes: int):
    case = cfg["case"]

    model = timm.create_model(
        model_name,
        pretrained=cfg.get("pretrained", False),
        num_classes=num_classes,
        in_chans=cfg.get("in_chans", 3),
        **({"dropout_rate": cfg["dropout_rate"]}
           if "dropout_rate" in cfg else {}),
        **({"final_dropout_rate": cfg["final_dropout_rate"]}
           if "final_dropout_rate" in cfg else {}),
        **({"intermediate_activation": cfg["intermediate_activation"]}
           if "intermediate_activation" in cfg else {}),
        **({"depth": cfg["wrn_depth"]}
           if cfg.get("wrn_depth") is not None else {}),
        **({"first_stage_channels": cfg["wrn_first_stage_channels"]}
           if cfg.get("wrn_first_stage_channels") is not None else {}),
    )

    if case.startswith("adapt_noresize"):
        if not is_supported_cifar_adapt_model(model_name):
            raise ValueError(f"{model_name} is not supported by adapt_model_cifar.py")
        model = adapt_timm_model_to_cifar(model, model_name)

    return model
