"""Toggle ResNet ReLU6 must preserve the pretrained cap in voltage coordinates."""
import copy

import pytest
import torch
from torch import nn

from baseline.cifar_resnet import CIFARResNet, WideResNetCIFAR
from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_blocks


@pytest.mark.parametrize("level", [2, 3])
def test_post_activation_relu6_matches_unitless_with_active_caps(level):
    torch.manual_seed(614)
    source = CIFARResNet(depth=8, base_width=2, in_chans=4, num_classes=3,
                        intermediate_activation="relu6").eval()
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.fill_(2.)
                module.bias.fill_(4.5)
    options = dict(R=50e3, C=500e-15, v_dd=0.1, one_over_q=128,
                   toggle_timing_mode="fixed", toggle_y_time=15e-9,
                   z_over_y_time=1, quantize_weights=True, w_bits=8,
                   weight_quant_factor_bits=None, enob=None,
                   enable_spin_variation=False,
                   enable_summing_current_noise=False, enable_coupler_noise=False,
                   enable_dtc_nonideality=False)
    unitless = convert_wide_resnet_to_physical(
        copy.deepcopy(source), physical=False, physical_level=2, **options).eval()
    physical = convert_wide_resnet_to_physical(
        copy.deepcopy(source), physical=True, physical_level=level, **options).eval()
    q = options["v_dd"] / options["one_over_q"]
    assert isinstance(unitless.relu, nn.ReLU6)
    assert physical.relu.max_val == 6*q
    for block in list(iter_physical_blocks(physical))[1:]:
        assert block.act1.max_val == block.act2.max_val == 6*q
        x = torch.tensor([-2., 1., 7., 20.])*q
        torch.testing.assert_close(block.act1(x)/q, x.div(q).clamp(0, 6))
    x = torch.randn(2, 4, 8, 8)*2
    with torch.no_grad():
        torch.testing.assert_close(physical(x), unitless(x), rtol=3e-5, atol=3e-5)


def test_wrn_and_ordinary_relu_conversion_are_unchanged():
    wrn = WideResNetCIFAR(depth=10, base_width=2, widen_factor=1,
                         in_chans=4, num_classes=3,
                         intermediate_activation="relu6")
    model = convert_wide_resnet_to_physical(wrn, physical=True, physical_level=2,
                                           v_dd=0.1, one_over_q=128)
    for block in list(iter_physical_blocks(model))[1:]:
        assert isinstance(block.act1, nn.ReLU6)
        assert isinstance(block.act2, nn.ReLU6)
    resnet = CIFARResNet(depth=8, base_width=2, in_chans=4, num_classes=3)
    stem = resnet.relu
    convert_wide_resnet_to_physical(resnet, physical=True, physical_level=2)
    assert resnet.relu is stem
