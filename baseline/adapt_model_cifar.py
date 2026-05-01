"""
Helpers to adapt selected built-in timm models to native CIFAR 32x32 input.
This is the change the stem/downsampling of some selected models instead of resizing the input.

Usage:
    import timm
    from adapt_model_cifar import adapt_timm_model_to_cifar

    model = timm.create_model("resnet18", pretrained=False, num_classes=100)
    model = adapt_timm_model_to_cifar(model, "resnet18")

Supported built-in timm model names for this adapter:
    resnet18
    resnet34
    resnet50
    resnext26ts
    resnext50_32x4d
    seresnet18
    seresnet34
    seresnet50
    mobilenetv2_100
    vgg19

Note:
    This is meant for training from scratch on CIFAR without resizing.
    For ImageNet-pretrained fine-tuning, resizing to the pretrained input size is usually safer.

Adapt logic:
    - ResNet/ResNeXt/SEResNet: replace ImageNet 7x7 stride-2 stem + maxpool with
      CIFAR-style 3x3 stride-1 stem and no maxpool.
    - MobileNetV2: reduce early downsampling so 32x32 inputs do not collapse too early.
    - VGG19: usually already works naturally on 32x32; no aggressive stem patch needed.
"""
import torch
import torch.nn as nn


RESNET_LIKE_PREFIXES = (
    "resnet",
    "resnext",
    "seresnet",
    "seresnext",
)

MOBILENETV2_PREFIXES = (
    "mobilenetv2",
)

VGG_PREFIXES = (
    "vgg",
)


SUPPORTED_CIFAR_ADAPTED_TIMM_MODELS = [
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
]


def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _replace_conv_preserve_channels(
    old_conv: nn.Conv2d,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )
    _kaiming_init_conv(new_conv)
    return new_conv


def _set_first_conv_to_cifar_stem(module: nn.Module) -> bool:
    """Find the first Conv2d recursively and replace it by 3x3 stride-1 conv."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _replace_conv_preserve_channels(child, kernel_size=3, stride=1, padding=1))
            return True
        if _set_first_conv_to_cifar_stem(child):
            return True
    return False


def _change_first_n_stride2_convs_to_stride1(module: nn.Module, max_changes: int) -> int:
    """
    Recursively change the first N stride-2 Conv2d layers to stride 1.

    This is mainly used for MobileNetV2. For CIFAR 32x32, keeping all ImageNet
    downsampling stages can collapse the spatial map too early.
    """
    changed = 0
    for child in module.children():
        if changed >= max_changes:
            break
        if isinstance(child, nn.Conv2d):
            if child.stride == (2, 2):
                child.stride = (1, 1)
                changed += 1
        else:
            changed += _change_first_n_stride2_convs_to_stride1(child, max_changes=max_changes - changed)
    return changed


def adapt_resnet_like_to_cifar(model: nn.Module) -> nn.Module:
    """
    Adapt ImageNet-style ResNet/ResNeXt/SEResNet to CIFAR.

    Changes:
      - first conv/stem: usually 7x7 stride 2 -> 3x3 stride 1
      - maxpool -> Identity if present

    Some timm models expose the first conv as model.conv1.
    Others, e.g. resnext26ts in some timm versions, use a different stem layout.
    """
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        model.conv1 = _replace_conv_preserve_channels(
            model.conv1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    else:
        changed = _set_first_conv_to_cifar_stem(model)
        if not changed:
            raise RuntimeError(
                "Could not find first Conv2d to adapt for ResNet-like model."
            )

    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()

    return model


def adapt_mobilenetv2_to_cifar(model: nn.Module, extra_stride1_convs: int = 1) -> nn.Module:
    """
    Adapt MobileNetV2-style models to CIFAR.

    Changes:
      - first conv/stem -> 3x3 stride 1
      - optionally change the next early stride-2 Conv2d to stride 1

    The default extra_stride1_convs=1 removes one additional early downsampling after
    the stem. This is a pragmatic CIFAR adaptation used to avoid overly aggressive
    32 -> 16 -> 8 -> 4 -> 2 -> 1 collapse.
    """
    changed = False

    if hasattr(model, "conv_stem") and isinstance(model.conv_stem, nn.Conv2d):
        model.conv_stem = _replace_conv_preserve_channels(model.conv_stem, kernel_size=3, stride=1, padding=1)
        changed = True
    else:
        changed = _set_first_conv_to_cifar_stem(model)

    if not changed:
        raise RuntimeError("Could not find first conv to adapt for MobileNetV2-style model.")

    if extra_stride1_convs > 0:
        _change_first_n_stride2_convs_to_stride1(model, max_changes=extra_stride1_convs)

    return model


def adapt_vgg_to_cifar(model: nn.Module) -> nn.Module:
    """
    Adapt VGG to CIFAR.

    VGG usually does not have the aggressive 7x7 stride-2 stem problem. In most timm
    versions, creating the model with num_classes already gives a usable classifier.
    Therefore this is intentionally a no-op.
    """
    return model


def adapt_timm_model_to_cifar(model: nn.Module, model_name: str) -> nn.Module:
    """
    Dispatch adapter for selected timm built-in models.

    This modifies the model in-place and returns it.
    """
    name = model_name.lower()

    if name not in SUPPORTED_CIFAR_ADAPTED_TIMM_MODELS:
        raise ValueError(
            f"model_name={model_name} is not in the supported native-CIFAR adapter list. "
            f"Supported: {SUPPORTED_CIFAR_ADAPTED_TIMM_MODELS}"
        )

    if name.startswith(RESNET_LIKE_PREFIXES):
        return adapt_resnet_like_to_cifar(model)

    if name.startswith(MOBILENETV2_PREFIXES):
        return adapt_mobilenetv2_to_cifar(model)

    if name.startswith(VGG_PREFIXES):
        return adapt_vgg_to_cifar(model)

    raise ValueError(f"No CIFAR adapter branch matched model_name={model_name}.")


def is_supported_cifar_adapt_model(model_name: str) -> bool:
    return model_name.lower() in SUPPORTED_CIFAR_ADAPTED_TIMM_MODELS


def list_recommended_cifar_adapted_timm_models():
    return list(SUPPORTED_CIFAR_ADAPTED_TIMM_MODELS)


if __name__ == "__main__":
    import timm

    for model_name in list_recommended_cifar_adapted_timm_models():
        if model_name not in timm.list_models():
            print(f"missing in this timm install: {model_name}")
            continue
        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        model = adapt_timm_model_to_cifar(model, model_name)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"{model_name:24s} -> {tuple(y.shape)}")
