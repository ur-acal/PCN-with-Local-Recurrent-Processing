"""
CIFAR baseline models registered into timm.

Usage:
    import cifar_resnet  # registers models
    import timm

    model = timm.create_model("resnet20_cifar", num_classes=100, pretrained=False)
    model = timm.create_model("wrn_28_10_cifar", num_classes=10, pretrained=False)
    model = timm.create_model("preact_resnet164_cifar", num_classes=100, pretrained=False)

Note:
    When loading those models in run_baseline.py, pass --pretrained  false.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

def _init_cifar_model(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# -----------------------------------------------------------------------------
# CIFAR ResNet v1: ResNet-20/32/44/56
# CIFAR style: depth = 6n + 2, stages [16, 32, 64].
# -----------------------------------------------------------------------------
class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = "A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            if option == "A":
                # Original CIFAR ResNet shortcut: spatial downsample + zero-pad channels.
                pad_ch = planes - in_planes
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::stride, ::stride],
                        (0, 0, 0, 0, pad_ch // 2, pad_ch - pad_ch // 2),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                raise ValueError(f"Unsupported shortcut option: {option}")

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(
        self,
        depth: int = 20,
        num_classes: int = 10,
        in_chans: int = 3,
        base_width: int = 16,
        shortcut_option: str = "A",
        zero_init_last_bn: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (depth - 2) % 6 == 0, "CIFAR ResNet depth should be 6n + 2."
        n = (depth - 2) // 6

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.depth = depth
        self.base_width = base_width
        self.shortcut_option = shortcut_option
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(in_chans, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base_width, n, stride=1)
        self.layer2 = self._make_layer(base_width * 2, n, stride=2)
        self.layer3 = self._make_layer(base_width * 4, n, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 4, num_classes)

        _init_cifar_model(self)
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, CIFARBasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(CIFARBasicBlock(self.in_planes, planes, stride=s, option=self.shortcut_option))
            self.in_planes = planes * CIFARBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x).flatten(1)
        if pre_logits:
            return x
        return self.fc(x)

    def forward(self, x, is_feat: bool = False):
        features = self.forward_features(x)
        outputs = self.forward_head(features)
        if is_feat:
            return [features], outputs
        return outputs

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg"):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.base_width * 4, num_classes) if num_classes > 0 else nn.Identity()


# -----------------------------------------------------------------------------
# Pre-activation ResNet for CIFAR, including PreAct ResNet-164.
# Bottleneck depth = 9n + 2. For 164, n = 18.
# -----------------------------------------------------------------------------
class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out = out + shortcut
        return out


class PreActCIFARResNet(nn.Module):
    def __init__(
        self,
        depth: int = 164,
        num_classes: int = 10,
        in_chans: int = 3,
        base_width: int = 16,
        zero_init_last_bn: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (depth - 2) % 9 == 0, "PreAct bottleneck CIFAR ResNet depth should be 9n + 2."
        n = (depth - 2) // 9

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.depth = depth
        self.base_width = base_width
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(in_chans, base_width, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(base_width, n, stride=1)
        self.layer2 = self._make_layer(base_width * 2, n, stride=2)
        self.layer3 = self._make_layer(base_width * 4, n, stride=2)

        self.bn = nn.BatchNorm2d(base_width * 4 * PreActBottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 4 * PreActBottleneck.expansion, num_classes)

        _init_cifar_model(self)
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, PreActBottleneck):
                    nn.init.zeros_(m.bn3.weight)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBottleneck(self.in_planes, planes, stride=s))
            self.in_planes = planes * PreActBottleneck.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x).flatten(1)
        if pre_logits:
            return x
        return self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg"):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.base_width * 4 * PreActBottleneck.expansion, num_classes) if num_classes > 0 else nn.Identity()


# -----------------------------------------------------------------------------
# WideResNet-28-10 for CIFAR.
# depth = 6n + 4. WRN-28 => n = 4. Widen factor = 10.
# -----------------------------------------------------------------------------
class MaxPoolChannelPad(nn.Module):
    """Parameter-free stride shortcut with symmetric zero channel padding."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        if stride <= 1 or out_channels < in_channels:
            raise ValueError("MaxPoolChannelPad requires stride > 1 and nondecreasing channels")
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        extra = out_channels - in_channels
        self.pad_left = extra // 2
        self.pad_right = extra - self.pad_left

    def forward(self, x):
        x = self.pool(x)
        if self.pad_left or self.pad_right:
            x = F.pad(x, (0, 0, 0, 0, self.pad_left, self.pad_right))
        return x


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
        use_batchnorm: bool = True,
        maxpool_downsample_shortcut: bool = False,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm
        )

        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout_rate = float(dropout_rate)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=not use_batchnorm
        )

        if stride != 1 and maxpool_downsample_shortcut:
            self.shortcut = MaxPoolChannelPad(in_planes, planes, stride=stride)
        elif stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=not use_batchnorm
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class WideResNetCIFAR(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.0,
        num_classes: int = 10,
        in_chans: int = 3,
        base_width: int = 16,
        use_batchnorm: bool = True,
        maxpool_downsample_shortcut: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n + 4."
        n = (depth - 4) // 6
        widths = [base_width, base_width * widen_factor, base_width * 2 * widen_factor, base_width * 4 * widen_factor]

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.depth = depth
        self.widen_factor = widen_factor
        self.dropout_rate = float(dropout_rate)
        self.final_dropout_rate = float(final_dropout_rate)
        self.use_batchnorm = bool(use_batchnorm)
        self.maxpool_downsample_shortcut = bool(maxpool_downsample_shortcut)
        self.out_dim = widths[3]
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(
            in_chans, widths[0], kernel_size=3, stride=1, padding=1, bias=not self.use_batchnorm
        )
        self.layer1 = self._make_layer(widths[1], n, stride=1)
        self.layer2 = self._make_layer(widths[2], n, stride=2)
        self.layer3 = self._make_layer(widths[3], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[3]) if self.use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        _init_cifar_model(self)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                WideBasicBlock(
                    self.in_planes,
                    planes,
                    self.dropout_rate,
                    stride=s,
                    use_batchnorm=self.use_batchnorm,
                    maxpool_downsample_shortcut=self.maxpool_downsample_shortcut,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        if self.final_dropout_rate > 0:
            x = F.dropout(x, p=self.final_dropout_rate, training=self.training)
        x = self.relu(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x).flatten(1)
        if pre_logits:
            return x
        return self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg"):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()


# Register Timm models
@register_model
def resnet20_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for resnet20_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return CIFARResNet(depth=20, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def resnet32_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for resnet32_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return CIFARResNet(depth=32, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def resnet44_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for resnet44_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return CIFARResNet(depth=44, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def resnet56_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for resnet56_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return CIFARResNet(depth=56, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def resnet110_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for resnet110_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return CIFARResNet(depth=110, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def preact_resnet164_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for preact_resnet164_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return PreActCIFARResNet(depth=164, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_28_10_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_10_cifar. Use checkpoint_map/checkpoint_dir instead.")
    return WideResNetCIFAR(depth=28, widen_factor=10, num_classes=num_classes, in_chans=in_chans, **kwargs)

@register_model
def wrn_16_2_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_16_2_cifar.")
    return WideResNetCIFAR(depth=16, widen_factor=2, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_16_4_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_16_4_cifar.")
    return WideResNetCIFAR(depth=16, widen_factor=4, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_16_8_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_16_8_cifar.")
    return WideResNetCIFAR(depth=16, widen_factor=8, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_16_10_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_16_10_cifar.")
    return WideResNetCIFAR(depth=16, widen_factor=10, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_28_2_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_2_cifar.")
    return WideResNetCIFAR(depth=28, widen_factor=2, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_28_4_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_4_cifar.")
    return WideResNetCIFAR(depth=28, widen_factor=4, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_28_5_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_5_cifar.")
    return WideResNetCIFAR(depth=28, widen_factor=5, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_40_2_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_40_2_cifar.")
    return WideResNetCIFAR(depth=40, widen_factor=2, num_classes=num_classes, in_chans=in_chans, **kwargs)


@register_model
def wrn_40_4_cifar(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_40_4_cifar.")
    return WideResNetCIFAR(depth=40, widen_factor=4, num_classes=num_classes, in_chans=in_chans, **kwargs)


def _build_wrn_nobn(depth: int, widen_factor: int, pretrained: bool, num_classes: int, in_chans: int, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for BN-free WRN models.")
    return WideResNetCIFAR(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        in_chans=in_chans,
        use_batchnorm=False,
        **kwargs,
    )


@register_model
def wrn_16_2_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(16, 2, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_16_4_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(16, 4, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_16_8_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(16, 8, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_28_2_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(28, 2, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_28_4_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(28, 4, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_40_2_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(40, 2, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_16_2_cifar_nobn_maxpool_shortcut(
    pretrained: bool = False,
    num_classes: int = 10,
    in_chans: int = 3,
    **kwargs,
):
    return _build_wrn_nobn(
        16, 2, pretrained, num_classes, in_chans, maxpool_downsample_shortcut=True, **kwargs
    )
