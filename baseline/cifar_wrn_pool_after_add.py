"""Opt-in WRN with addition before selectable average/maximum pooling.

Instantiate WideResNetPoolAfterAddCIFAR with the original model settings and
load its state_dict strictly. Existing WRN registrations are not modified.
Hook block.pre_pool to record the sum before spatial downsampling; clone
hook outputs if retaining them across subsequent in-place activations.
"""

from torch import nn
from torch.nn import functional as F

from .cifar_resnet import ChannelZeroPad, WideBasicBlock, WideResNetCIFAR


class WideBasicBlockPoolAfterAdd(WideBasicBlock):
    """Stride-one main convolutions, zero-pad shortcut, then shared pooling."""

    def __init__(self, in_planes, planes, dropout_rate, stride=1,
                 use_batchnorm=True, conv_bias=None, pool_type="avg"):
        super().__init__(
            in_planes, planes, dropout_rate, stride=stride,
            use_batchnorm=use_batchnorm, conv_bias=conv_bias,
            avgpool_downsample_shortcut=True, avgpool_main_downsample=True,
            pool_type=pool_type,
        )
        self.shortcut = ChannelZeroPad(in_planes, planes)
        self.pre_pool = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        # Preserve the original shortcut input semantics, including in-place
        # ReLU when bn1 is Identity; only rearrange pooling and addition.
        out = self.pre_pool(out + self.shortcut(x))
        return self.main_downsample(out)


class WideResNetPoolAfterAddCIFAR(WideResNetCIFAR):
    """Pooled-main/padded-shortcut WRN; avg is compatible with old checkpoints."""

    def __init__(self, *args, avgpool_downsample_shortcut=True,
                 avgpool_main_downsample=True, maxpool_downsample_shortcut=False,
                 pool_type="avg", **kwargs):
        if (not avgpool_downsample_shortcut or not avgpool_main_downsample
                or maxpool_downsample_shortcut):
            raise ValueError("Pool-after-add requires pooled main and padded shortcuts")
        super().__init__(
            *args, avgpool_downsample_shortcut=True,
            avgpool_main_downsample=True, maxpool_downsample_shortcut=False,
            pool_type=pool_type,
            **kwargs,
        )

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for block_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(WideBasicBlockPoolAfterAdd(
                self.in_planes, planes, self.dropout_rate, stride=block_stride,
                use_batchnorm=self.use_batchnorm, conv_bias=self.conv_bias,
                pool_type=self.pool_type,
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)
