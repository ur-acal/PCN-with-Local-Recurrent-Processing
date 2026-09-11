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


class ChannelZeroPad(nn.Module):
    """Parameter-free channel expansion without spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("ChannelZeroPad requires nondecreasing channels")
        extra = out_channels - in_channels
        self.pad_left = extra // 2
        self.pad_right = extra - self.pad_left

    def forward(self, x):
        if self.pad_left or self.pad_right:
            x = F.pad(x, (0, 0, 0, 0, self.pad_left, self.pad_right))
        return x


class AvgPoolChannelPad(nn.Module):
    """Parameter-free average-pool shortcut with symmetric channel padding."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        if stride <= 1 or out_channels < in_channels:
            raise ValueError("AvgPoolChannelPad requires stride > 1 and nondecreasing channels")
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.channel_pad = ChannelZeroPad(in_channels, out_channels)

    def forward(self, x):
        return self.channel_pad(self.pool(x))


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        dropout_rate: float,
        stride: int = 1,
        use_batchnorm: bool = True,
        conv_bias: bool | None = None,
        maxpool_downsample_shortcut: bool = False,
        avgpool_downsample_shortcut: bool = False,
        avgpool_main_downsample: bool = False,
    ):
        super().__init__()
        if conv_bias is None:
            conv_bias = not use_batchnorm
        self.bn1 = nn.BatchNorm2d(in_planes) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=conv_bias
        )

        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout_rate = float(dropout_rate)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1 if avgpool_main_downsample and stride != 1 else stride,
            padding=1,
            bias=conv_bias,
        )
        self.main_downsample = (
            nn.AvgPool2d(kernel_size=stride, stride=stride)
            if avgpool_main_downsample and stride != 1
            else nn.Identity()
        )

        if avgpool_downsample_shortcut and (stride != 1 or in_planes != planes):
            self.shortcut = (
                AvgPoolChannelPad(in_planes, planes, stride=stride)
                if stride != 1
                else ChannelZeroPad(in_planes, planes)
            )
        elif stride != 1 and maxpool_downsample_shortcut:
            self.shortcut = MaxPoolChannelPad(in_planes, planes, stride=stride)
        elif stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=conv_bias
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        out = self.main_downsample(out)
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
        conv_bias: bool | None = None,
        maxpool_downsample_shortcut: bool = False,
        avgpool_downsample_shortcut: bool = False,
        avgpool_main_downsample: bool = False,
        init_mode: str = "wrn",
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
        self.conv_bias = not self.use_batchnorm if conv_bias is None else bool(conv_bias)
        self.maxpool_downsample_shortcut = bool(maxpool_downsample_shortcut)
        self.avgpool_downsample_shortcut = bool(avgpool_downsample_shortcut)
        self.avgpool_main_downsample = bool(avgpool_main_downsample)
        self.out_dim = widths[3]
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(
            in_chans, widths[0], kernel_size=3, stride=1, padding=1, bias=self.conv_bias
        )
        self.layer1 = self._make_layer(widths[1], n, stride=1)
        self.layer2 = self._make_layer(widths[2], n, stride=2)
        self.layer3 = self._make_layer(widths[3], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[3]) if self.use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        if init_mode not in {"wrn", "pytorch"}:
            raise ValueError(f"Unknown WRN initialization: {init_mode}")
        self.init_mode = init_mode
        if init_mode == "wrn":
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
                    conv_bias=self.conv_bias,
                    maxpool_downsample_shortcut=self.maxpool_downsample_shortcut,
                    avgpool_downsample_shortcut=self.avgpool_downsample_shortcut,
                    avgpool_main_downsample=self.avgpool_main_downsample,
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
def wrn_28_2_cifar_avgpool(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_2_cifar_avgpool.")
    return WideResNetCIFAR(
        depth=28,
        widen_factor=2,
        num_classes=num_classes,
        in_chans=in_chans,
        avgpool_downsample_shortcut=True,
        avgpool_main_downsample=True,
        **kwargs,
    )


@register_model
def wrn_28_2_cifar_avgpool_shortcut(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    if pretrained:
        raise ValueError("No registered pretrained weights for wrn_28_2_cifar_avgpool_shortcut.")
    return WideResNetCIFAR(
        depth=28,
        widen_factor=2,
        num_classes=num_classes,
        in_chans=in_chans,
        avgpool_downsample_shortcut=True,
        **kwargs,
    )


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


def _build_wrn_nobn_no_bias(
    depth: int,
    widen_factor: int,
    pretrained: bool,
    num_classes: int,
    in_chans: int,
    **kwargs,
):
    return _build_wrn_nobn(
        depth,
        widen_factor,
        pretrained,
        num_classes,
        in_chans,
        conv_bias=False,
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
def wrn_28_2_cifar_nobn_avgpool(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn(
        28,
        2,
        pretrained,
        num_classes,
        in_chans,
        avgpool_downsample_shortcut=True,
        avgpool_main_downsample=True,
        **kwargs,
    )


@register_model
def wrn_28_2_cifar_nobn_avgpool_shortcut(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn(
        28,
        2,
        pretrained,
        num_classes,
        in_chans,
        avgpool_downsample_shortcut=True,
        **kwargs,
    )


@register_model
def wrn_28_4_cifar_nobn(pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs):
    return _build_wrn_nobn(28, 4, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_16_2_cifar_nobn_no_bias(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn_no_bias(16, 2, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_16_4_cifar_nobn_no_bias(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn_no_bias(16, 4, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_28_2_cifar_nobn_no_bias(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn_no_bias(28, 2, pretrained, num_classes, in_chans, **kwargs)


@register_model
def wrn_28_4_cifar_nobn_no_bias(
    pretrained: bool = False, num_classes: int = 10, in_chans: int = 3, **kwargs
):
    return _build_wrn_nobn_no_bias(28, 4, pretrained, num_classes, in_chans, **kwargs)


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


def _register_wrn_control(row, size):
    from baseline.wrn_control_specs import model_name, model_options
    depth, width = map(int, size.split('_'))
    options = model_options(row)

    def factory(pretrained=False, num_classes=10, in_chans=3, **kwargs):
        if pretrained:
            raise ValueError('WRN controls require a locally trained checkpoint')
        for key, value in options.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f'{factory.__name__} fixes {key}={value}')
        kwargs.update(options)
        return WideResNetCIFAR(depth=depth, widen_factor=width, num_classes=num_classes,
                              in_chans=in_chans, **kwargs)

    factory.__name__ = model_name(row, size)
    globals()[factory.__name__] = register_model(factory)


from baseline.wrn_control_specs import ROWS as _CONTROL_ROWS, SIZES as _CONTROL_SIZES

for _row in _CONTROL_ROWS:
    for _size in _CONTROL_SIZES:
        _register_wrn_control(_row, _size)
