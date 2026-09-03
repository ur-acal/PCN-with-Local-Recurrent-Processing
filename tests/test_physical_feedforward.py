import tempfile
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baseline.cifar_resnet import WideBasicBlock, WideResNetCIFAR
from feedforward_validation import FeedForwardCNNValidator
from measured_activation import (
    MEASURED_ACTIVATION_TYPES,
    configure_feedforward_activation_pullback,
    configure_feedforward_measured_activation,
    configure_measured_activation_corner_mode,
    feedforward_measured_activation_factory,
)
from measured_pooling import (
    MeasuredAvgPool2d,
    configure_feedforward_measured_pooling,
)
from physical_feedforward import (
    AveragedPhysicalBasicBlock,
    AveragedFeedForwardPhysicalWrapper,
    PulseFeedForwardPhysicalWrapper,
    PulsePhysicalBasicBlock,
    convert_wide_resnet_to_physical,
)


def test_derived_stage_matches_quantized_convolution():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 8, 8) * 0.01
    for physical_level in (2, 3):
        conv = torch.nn.Conv2d(
            4, 6, 3, stride=2, padding=1, bias=False)
        block_cls = (AveragedPhysicalBasicBlock
                     if physical_level == 2 else PulsePhysicalBasicBlock)
        wrapper_cls = (AveragedFeedForwardPhysicalWrapper
                       if physical_level == 2 else PulseFeedForwardPhysicalWrapper)
        block = block_cls(
            conv,
            R=50e3, C=500e-15, v_dd=10.0, w_bits=5,
            weight_quant_factor_bits=1).eval()
        wrapper = wrapper_cls(block, qat=True).eval()
        actual = wrapper(x)
        original = conv.parametrizations.weight.original
        levels = (original * block.scale1 * 15).round().clamp(-15, 15)
        expected = F.conv2d(
            x, levels / (15 * block.scale1), None, stride=2, padding=1)
        torch.testing.assert_close(actual, expected, atol=5e-8, rtol=1e-6)


def test_unitless_fixed_timing_recipe_matches_plain_convolution():
    torch.manual_seed(7)
    x = torch.randn(2, 4, 8, 8)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1, bias=False)
    reference = F.conv2d(x, conv.weight, padding=1)
    # A one-convolution stem is feedforward y, so z_over_y_time does not
    # multiply its duration.
    gain = 5e-9 / (50e3 * 500e-15)
    conv.weight.data.div_(gain)
    block = AveragedPhysicalBasicBlock(
        conv, physical=False,
        R=50e3, C=500e-15, toggle_timing_mode="fixed",
        toggle_y_time=5e-9, z_over_y_time=3.0).eval()
    torch.testing.assert_close(block(x), reference, atol=2e-6, rtol=2e-5)


def test_fixed_timing_ratio_applies_to_second_feedforward_convolution():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 8, 8)
    conv1 = torch.nn.Conv2d(4, 5, 3, padding=1, bias=False)
    conv2 = torch.nn.Conv2d(5, 6, 3, padding=1, bias=False)
    reference = conv2(conv1(x))
    y_gain = 5e-9 / (50e3 * 500e-15)
    z_gain = 3.0 * y_gain
    conv1.weight.data.div_(y_gain)
    conv2.weight.data.div_(z_gain)
    block = AveragedPhysicalBasicBlock(
        conv1, conv2, physical=False,
        R=50e3, C=500e-15, toggle_timing_mode="fixed",
        toggle_y_time=5e-9, z_over_y_time=3.0).eval()
    torch.testing.assert_close(block(x), reference, atol=3e-5, rtol=3e-5)


class _TinyWideResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.layer1 = torch.nn.Sequential(
            WideBasicBlock(4, 8, 0.0, stride=2))
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(8, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.bn(x))
        return self.fc(self.global_pool(x).flatten(1))


def test_feedforward_validator_unroll_is_numerically_equivalent():
    torch.manual_seed(3)
    inputs = torch.randn(4, 4, 8, 8) * 0.01
    targets = torch.zeros(4, dtype=torch.long)
    model = _TinyWideResNet().eval()
    convert_wide_resnet_to_physical(
        model, physical_level=3, qat=True, R=50e3, C=500e-15,
        v_dd=10.0, w_bits=5, weight_quant_factor_bits=1)
    dense_output = model(inputs)
    loader = DataLoader(
        TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    with tempfile.TemporaryDirectory() as cache_dir:
        FeedForwardCNNValidator(
            model, cache_dir, "cpu", loader, cache_dir)
        expanded_output = model(inputs)
    torch.testing.assert_close(
        expanded_output, dense_output, atol=2e-6, rtol=2e-5)


def test_model_boundary_q_scaling_preserves_bn_free_network_function():
    torch.manual_seed(11)
    inputs = torch.randn(2, 4, 8, 8) * 1e-3
    base = _TinyWideResNet().eval()
    base.layer1[0].bn1 = torch.nn.Identity()
    base.layer1[0].bn2 = torch.nn.Identity()
    base.bn = torch.nn.Identity()
    unit_q = copy.deepcopy(base)
    ten_q = copy.deepcopy(base)
    convert_wide_resnet_to_physical(
        unit_q, physical_level=2, qat=True, R=50e3, C=500e-15,
        v_dd=10.0, one_over_q=10.0, w_bits=5,
        weight_quant_factor_bits=1)
    convert_wide_resnet_to_physical(
        ten_q, physical_level=2, qat=True, R=50e3, C=500e-15,
        v_dd=10.0, one_over_q=1.0, w_bits=5,
        weight_quant_factor_bits=1)
    torch.testing.assert_close(
        ten_q(inputs), unit_q(inputs), atol=2e-6, rtol=2e-5)


def test_physical_block_clamps_its_input_before_activation():
    model = _TinyWideResNet().eval()
    convert_wide_resnet_to_physical(
        model, physical_level=2, qat=False, R=50e3, C=500e-15,
        v_dd=0.1, one_over_q=1.0)
    seen = []
    model.conv1.ode_block.norm1.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()))
    model(torch.full((1, 4, 8, 8), 100.0))
    assert seen and seen[0].abs().max().item() <= 0.100001


def test_feedforward_measured_activation_keeps_final_relu_ideal():
    model = _TinyWideResNet().eval()
    convert_wide_resnet_to_physical(
        model, physical_level=2, physical=False, qat=False,
        R=50e3, C=500e-15, v_dd=0.5, one_over_q=5.0)
    factory = feedforward_measured_activation_factory(
        "hardware_data/relu_current_0p2uA_all.csv", 0.5, corner="TT")
    configure_feedforward_measured_activation(model, factory)
    assert not any(isinstance(module, MEASURED_ACTIVATION_TYPES)
                   for module in model.relu.modules())
    residual_block = model.layer1[0].ode_block
    assert isinstance(residual_block.act1, MEASURED_ACTIVATION_TYPES)
    assert isinstance(residual_block.act2, MEASURED_ACTIVATION_TYPES)


def test_feedforward_measured_pooling_accepts_gaussian_package():
    model = _TinyWideResNet().eval()
    convert_wide_resnet_to_physical(
        model, physical_level=2, physical=False, qat=False,
        R=50e3, C=500e-15, v_dd=0.5, one_over_q=5.0)
    gaussian = {
        "v_grid": torch.tensor([-0.1, 0.1]),
        "mean": torch.tensor([2.0, 2.0]),
        "factor": torch.zeros(2, 2),
        "value_scale": 1.0,
        "quantity": "conductance",
    }
    configure_feedforward_measured_pooling(
        model, enable_nonideality=True, curve_gaussian=gaussian,
        nominal_R=0.5, seed=1)
    assert isinstance(model.global_pool, MeasuredAvgPool2d)
    assert model.global_pool.curve_gaussian
    shortcut = model.layer1[0].ode_block.shortcut
    assert isinstance(shortcut.pool, MeasuredAvgPool2d)
    assert shortcut.pool.curve_gaussian


def test_wrn_feature_forward_supports_shared_feature_kd_contract():
    model = WideResNetCIFAR(
        depth=16, widen_factor=1, num_classes=3, in_chans=4).eval()
    features, logits = model(torch.randn(2, 4, 16, 16), is_feat=True)
    assert len(features) == 1
    assert features[0].ndim == 4
    assert logits.shape == (2, 3)


def test_feedforward_random_per_forward_activation_and_direct_pullback(
        monkeypatch):
    model = _TinyWideResNet().eval()
    convert_wide_resnet_to_physical(
        model, physical_level=2, physical=False, qat=False,
        R=50e3, C=500e-15, v_dd=0.5, one_over_q=5.0)
    factory = feedforward_measured_activation_factory(
        "hardware_data/relu_current_0p2uA_all.csv", 0.5,
        corner="TT")
    configure_feedforward_measured_activation(model, factory)
    count = configure_feedforward_activation_pullback(
        model, mode="direct", q=0.1)
    assert count > 1
    configure_measured_activation_corner_mode(
        model, mode="random_per_forward")

    activations = [
        module for module in model.modules()
        if isinstance(module, MEASURED_ACTIVATION_TYPES)
    ]
    monkeypatch.setattr(
        torch, "randint", lambda *args, **kwargs: torch.tensor([1]))
    model.train()
    model(torch.randn(2, 4, 8, 8))
    selected = activations[0].corner_names[1]
    assert model._last_measured_activation_corner == selected
    assert all(module.active_corner == selected for module in activations)
    assert all(module._coordinate_pullback_scale is not None
               for module in activations)

    model.eval()
    model(torch.randn(2, 4, 8, 8))
    assert model._last_measured_activation_corner is None
    assert all(module.active_corner == module.default_corner
               for module in activations)
