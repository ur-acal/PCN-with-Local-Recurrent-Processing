"""RGB to RAW conversion network adapted from CycleISP.
import math
"""

import logging
from typing import Callable, List, Union

import torch
from torch import nn

LOGGER = logging.getLogger("scangen.pipeline.rgb2raw")


def mosaic(images: torch.Tensor) -> torch.Tensor:
    """Extract RGGB Bayer planes from RGB image.

    Converts RGB images to RGGB Bayer pattern by extracting alternating pixels
    according to the standard Bayer pattern arrangement.

    Args:
        images: RGB image tensor of shape (B, 3, H, W)

    Returns:
        torch.Tensor: RGGB Bayer pattern tensor of shape (B, 4, H//2, W//2)
            where channels are [Red, Green_Red, Green_Blue, Blue]

    Examples:
        >>> rgb = torch.randn(1, 3, 32, 32)
        >>> rggb = mosaic(rgb)
        >>> rggb.shape
        torch.Size([1, 4, 16, 16])
    """
    red = images[:, 0, 0::2, 0::2]  # Red pixels at even rows, even cols
    green_red = images[:, 1, 0::2, 1::2]  # Green pixels at even rows, odd cols
    green_blue = images[:, 1, 1::2, 0::2]  # Green pixels at odd rows, even cols
    blue = images[:, 2, 1::2, 1::2]  # Blue pixels at odd rows, odd cols

    return torch.stack((red, green_red, green_blue, blue), dim=1)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
    padding: int = 1,
    stride: int = 1,
) -> nn.Conv2d:
    """Create a convolutional layer with default padding.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        bias: Whether to include bias term
        padding: Padding size (defaults to kernel_size//2 if not specified)
        stride: Convolution stride

    Returns:
        nn.Conv2d: Configured convolution layer
    """
    if padding == 1 and kernel_size != 3:
        padding = kernel_size // 2
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride
    )


class CALayer(nn.Module):
    """Channel Attention Layer.

    Implements channel attention mechanism using global average pooling
    and squeeze-excitation style channel weighting.
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:
        """Initialize Channel Attention Layer.

        Args:
            channel: Number of input channels
            reduction: Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor."""
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BasicConv(nn.Module):
    """Basic convolution block with optional batch norm and activation."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = False,
        bias: bool = False,
    ) -> None:
        """Initialize BasicConv layer.

        Args:
            in_planes: Input channel count
            out_planes: Output channel count
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            dilation: Dilation rate
            groups: Number of groups for grouped convolution
            relu: Whether to apply ReLU activation
            bn: Whether to apply batch normalization
            bias: Whether to include bias in convolution
        """
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution, optional BN and ReLU."""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """Channel pooling layer that concatenates max and mean across channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool channels by concatenating max and mean."""
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, mean_pool), dim=1)


class SpatialAttentionLayer(nn.Module):
    """Spatial attention mechanism using channel pooling and convolution."""

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize spatial attention layer.

        Args:
            kernel_size: Kernel size for spatial convolution
        """
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to input tensor."""
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class DAB(nn.Module):
    """Dual Attention Block combining spatial and channel attention."""

    def __init__(
        self,
        conv_func: Callable,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool = True,
        bn: bool = False,
        act: nn.Module = nn.ReLU(True),
    ) -> None:
        """Initialize Dual Attention Block.

        Args:
            conv_func: Convolution function to use
            n_feat: Number of features/channels
            kernel_size: Convolution kernel size
            reduction: Channel reduction ratio for attention
            bias: Whether to use bias in convolutions
            bn: Whether to use batch normalization
            act: Activation function
        """
        super().__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(conv_func(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)

        self.SA = SpatialAttentionLayer()
        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dual attention block."""
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res + x


class RRG(nn.Module):
    """Recursive Residual Group containing multiple DAB blocks."""

    def __init__(
        self,
        conv_func: Callable,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        num_dab: int,
    ) -> None:
        """Initialize Recursive Residual Group.

        Args:
            conv_func: Convolution function to use
            n_feat: Number of features/channels
            kernel_size: Convolution kernel size
            reduction: Channel reduction ratio for attention
            act: Activation function
            num_dab: Number of DAB blocks
        """
        super().__init__()

        modules_body = [
            DAB(conv_func, n_feat, kernel_size, reduction, bias=True, bn=False, act=act)
            for _ in range(num_dab)
        ]
        modules_body.append(conv_func(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through recursive residual group."""
        res = self.body(x)
        return res + x


class Rgb2Raw(nn.Module):
    """RGB to RAW conversion network.

    Neural network that converts RGB images to RAW sensor data in RGGB format.
    Uses multiple Recursive Residual Groups (RRG) with Dual Attention Blocks (DAB)
    for high-quality conversion.
    """

    def __init__(self, conv_func: Callable = conv) -> None:
        """Initialize RGB to RAW conversion network.

        Args:
            conv_func: Convolution function to use (defaults to local conv)
        """
        super().__init__()

        # Network configuration
        input_nc = 3  # RGB input channels
        num_rrg = 3  # Number of Recursive Residual Groups
        num_dab = 5  # Number of Dual Attention Blocks per RRG
        n_feats = 96  # Feature channels
        kernel_size = 3  # Convolution kernel size
        reduction = 8  # Channel attention reduction ratio

        act = nn.PReLU(n_feats)

        # Head: Initial feature extraction
        modules_head = [conv_func(input_nc, n_feats, kernel_size=kernel_size, stride=1)]

        # Body: Multiple RRG blocks for feature processing
        modules_body: List[Union[nn.PReLU, RRG]] = [
            RRG(conv_func, n_feats, kernel_size, reduction, act=act, num_dab=num_dab)
            for _ in range(num_rrg)
        ]
        modules_body.append(conv_func(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        # Tail: Final RGB output before mosaicking
        modules_tail = [conv_func(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB input to RAW RGGB output.

        Args:
            x: RGB image tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: RAW RGGB tensor of shape (B, 4, H//2, W//2)
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return mosaic(x)  # Convert to RGGB Bayer pattern
