import tempfile
import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baseline.cifar_resnet import (
    AvgPoolChannelPad,
    WideBasicBlock,
    WideResNetCIFAR,
    wrn_28_2_cifar_avgpool,
    wrn_28_2_cifar_avgpool_shortcut,
    wrn_28_2_cifar_nobn_avgpool,
    wrn_28_2_cifar_nobn_avgpool_shortcut,
)
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
    scale_batchnorm_to_physical_domain,
)
from trainer_timm import TrainerCiFarTimmStyle
from trainer import TrainerCiFar


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


def test_biased_unitless_level2_level3_match_fixed_and_derived_timing():
    torch.manual_seed(23)
    R, C = 50e3, 500e-15
    v_dd, one_over_q = 100.0, 10.0
    q = v_dd / one_over_q
    toggle_y_time = 5e-9
    x = torch.randn(1, 2, 3, 3) * 0.02
    base = torch.nn.Conv2d(2, 3, 1, bias=True)
    with torch.no_grad():
        base.weight.mul_(0.1)
        base.bias.mul_(0.1)
    reference = base(x)

    for timing_mode in ("derived", "fixed"):
        gain = (1.0 if timing_mode == "derived" else
                toggle_y_time / (R * C))

        def scaled_conv():
            conv = copy.deepcopy(base)
            if timing_mode == "fixed":
                with torch.no_grad():
                    conv.weight.div_(gain)
                    conv.bias.div_(gain)
            return conv

        unitless = AveragedPhysicalBasicBlock(
            scaled_conv(), physical=False, R=R, C=C,
            v_dd=v_dd, one_over_q=one_over_q,
            toggle_timing_mode=timing_mode,
            toggle_y_time=toggle_y_time, w_bits=9).eval()
        level2 = AveragedFeedForwardPhysicalWrapper(
            AveragedPhysicalBasicBlock(
                scaled_conv(), physical=True, R=R, C=C,
                v_dd=v_dd, one_over_q=one_over_q,
                toggle_timing_mode=timing_mode,
                toggle_y_time=toggle_y_time, w_bits=9),
            qat=True).eval()
        level3 = PulseFeedForwardPhysicalWrapper(
            PulsePhysicalBasicBlock(
                scaled_conv(), physical=True, R=R, C=C,
                v_dd=v_dd, one_over_q=one_over_q,
                toggle_timing_mode=timing_mode,
                toggle_y_time=toggle_y_time, w_bits=9),
            qat=True).eval()

        unitless_out = unitless(x)
        level2_out = level2(q * x) / q
        level3_out = level3(q * x) / q
        torch.testing.assert_close(
            unitless_out, reference, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(
            level2_out, reference, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            level3_out, reference, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            level3_out, level2_out, atol=1e-7, rtol=1e-6)


def test_toy_wrn_with_batchnorm_and_conv_bias_matches_level2_level3():
    """A q-scaled WRN must preserve both BatchNorm and convolution bias."""
    torch.manual_seed(73)
    base = WideResNetCIFAR(
        depth=10, widen_factor=1, num_classes=3, in_chans=4,
        avgpool_downsample_shortcut=True,
        intermediate_activation="relu6").eval()
    with torch.no_grad():
        for module in base.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.mul_(0.2)
                if module.bias is None:
                    module.bias = torch.nn.Parameter(
                        torch.empty(module.out_channels))
                module.bias.uniform_(-0.01, 0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.03, 0.03)
                module.running_var.uniform_(0.7, 1.3)
                module.weight.uniform_(0.15, 0.25)
                module.bias.uniform_(-0.03, 0.03)
        base.fc.weight.mul_(0.2)
        base.fc.bias.mul_(0.2)

    R, C = 50e3, 500e-15
    toggle_y_time = 5e-9
    fixed_scale = toggle_y_time / (R * C)
    v_dd, one_over_q = 100.0, 1000.0
    inputs = torch.randn(1, 4, 8, 8) * 0.02
    reference = base(inputs)

    def converted(level, timing_mode):
        model = copy.deepcopy(base)
        convert_wide_resnet_to_physical(
            model, physical_level=2 if level == 0 else level,
            physical=level != 0, qat=level != 0,
            R=R, C=C, v_dd=v_dd, one_over_q=one_over_q,
            w_bits=9, weight_quant_factor_bits=None,
            toggle_timing_mode=timing_mode,
            toggle_y_time=toggle_y_time, z_over_y_time=1.0)
        if timing_mode == "fixed":
            with torch.no_grad():
                for block in model.modules():
                    if not isinstance(block, AveragedPhysicalBasicBlock):
                        continue
                    for conv in (block.conv1, block.conv2):
                        if conv is None:
                            continue
                        conv.weight.div_(fixed_scale)
                        conv.bias.div_(fixed_scale)
        return model.eval()

    for timing_mode in ("derived", "fixed"):
        unitless = converted(0, timing_mode)(inputs)
        level2 = converted(2, timing_mode)(inputs)
        level3 = converted(3, timing_mode)(inputs)
        torch.testing.assert_close(
            unitless, reference, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(
            level2, reference, atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(
            level3, reference, atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(
            level3, level2, atol=1e-8, rtol=1e-5)


def test_optimizer_composes_stage_bn_and_bias_scaling():
    model = WideResNetCIFAR(
        depth=10, widen_factor=1, num_classes=3, in_chans=4,
        avgpool_downsample_shortcut=True)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.bias is None:
            module.bias = torch.nn.Parameter(torch.zeros(module.out_channels))
    convert_wide_resnet_to_physical(
        model, physical_level=2, physical=True, qat=False,
        R=50e3, C=500e-15, v_dd=1.0, one_over_q=10.0)

    trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
    trainer.model = model
    trainer.scale_train_recipe = True
    trainer.ff_train_scale = 0.6
    trainer.fb_train_scale = 0.2
    trainer.bias_lr_multiplier = 0.5
    trainer.bias_weight_decay = 0.0
    trainer.timm_opt = "sgd"
    trainer.momentum = 0.9
    trainer.opt_eps = None
    trainer.opt_betas = None

    base_lr, base_weight_decay = 0.01, 0.001
    optimizer = trainer._get_optimizer(
        "sgd", base_lr, base_weight_decay)
    assert isinstance(optimizer, torch.optim.SGD)

    settings = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            settings[id(parameter)] = (group["lr"], group["weight_decay"])

    stem = model.conv1.ode_block
    first_block = model.layer1[0].ode_block
    q_sq = model._physical_state_scale ** 2
    def assert_setting(parameter, expected_lr, expected_weight_decay):
        actual_lr, actual_weight_decay = settings[id(parameter)]
        assert math.isclose(actual_lr, expected_lr)
        assert math.isclose(actual_weight_decay, expected_weight_decay)

    assert_setting(
        stem.conv1.weight,
        base_lr / 0.2 ** 2, base_weight_decay * 0.2 ** 2)
    assert_setting(
        stem.conv1.bias, base_lr * 0.5 / 0.2 ** 2, 0.0)
    assert_setting(
        first_block.conv2.weight,
        base_lr / 0.6 ** 2, base_weight_decay * 0.6 ** 2)
    assert_setting(
        first_block.conv2.bias, base_lr * 0.5 / 0.6 ** 2, 0.0)
    assert_setting(first_block.norm1.weight, base_lr * q_sq, 0.0)
    assert_setting(first_block.norm1.bias, base_lr * 0.5 * q_sq, 0.0)


def test_bias_recipe_preserves_selected_timm_optimizer():
    model = torch.nn.Linear(4, 2, bias=True)
    trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
    trainer.model = model
    trainer.scale_train_recipe = False
    trainer.ff_train_scale = 1.0
    trainer.fb_train_scale = 1.0
    trainer.bias_lr_multiplier = 0.5
    trainer.bias_weight_decay = 0.0
    trainer.timm_opt = "adamw"
    trainer.momentum = 0.9
    trainer.opt_eps = None
    trainer.opt_betas = None

    optimizer = trainer._get_optimizer("adamw", 1e-3, 1e-2)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_composed_groups_preserve_pcn_scale_recipe_settings():
    class PCNLikeStages(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.FFconv = torch.nn.Conv2d(2, 2, 1, bias=True)
            self.FBconv = torch.nn.Conv2d(2, 2, 1, bias=True)

    class PCNLikeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = PCNLikeStages()
            self.head = torch.nn.Linear(2, 2)

    trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
    trainer.model = PCNLikeModel()
    trainer.scale_train_recipe = True
    trainer.ff_train_scale = 0.6
    trainer.fb_train_scale = 0.2
    trainer.bias_lr_multiplier = 1.0
    trainer.bias_weight_decay = None

    original_groups = TrainerCiFar._optimizer_parameters(
        trainer, 0.01, 0.001, filter_bias_and_bn=True)
    composed_groups = trainer._composed_optimizer_parameters(0.01, 0.001)

    def settings(groups):
        return {
            id(parameter): (group["lr"], group["weight_decay"])
            for group in groups for parameter in group["params"]
        }

    assert settings(composed_groups) == settings(original_groups)
    assert composed_groups[0]["group_name"].startswith("other")


def test_bn_q_domain_sgd_step_matches_unitless_parameter_update():
    torch.manual_seed(91)
    q = 0.1
    unitless = torch.nn.BatchNorm2d(3).eval()
    physical = copy.deepcopy(unitless).eval()
    scale_batchnorm_to_physical_domain(physical, q)
    physical._physical_state_scale = q

    trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
    trainer.model = physical
    trainer.scale_train_recipe = False
    trainer.ff_train_scale = 1.0
    trainer.fb_train_scale = 1.0
    trainer.bias_lr_multiplier = 1.0
    trainer.bias_weight_decay = None
    trainer.timm_opt = "sgd"
    trainer.momentum = 0.9
    trainer.opt_eps = None
    trainer.opt_betas = None

    lr = 0.01
    physical_optimizer = trainer._get_optimizer("sgd", lr, 0.0)
    unitless_optimizer = torch.optim.SGD(
        unitless.parameters(), lr=lr, momentum=0.9, nesterov=True)
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.randn(2, 3, 4, 4)

    unitless_loss = F.mse_loss(unitless(inputs), targets)
    physical_loss = F.mse_loss(physical(q * inputs) / q, targets)
    unitless_loss.backward()
    physical_loss.backward()
    unitless_optimizer.step()
    physical_optimizer.step()

    torch.testing.assert_close(
        physical.weight / q, unitless.weight, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        physical.bias / q, unitless.bias, atol=1e-7, rtol=1e-6)


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


def test_hardware_wrn_variants_preserve_activation_bn_and_downsampling():
    variants = (
        (wrn_28_2_cifar_avgpool, True, True),
        (wrn_28_2_cifar_avgpool_shortcut, True, False),
        (wrn_28_2_cifar_nobn_avgpool, False, True),
        (wrn_28_2_cifar_nobn_avgpool_shortcut, False, False),
    )
    for factory, uses_bn, pools_main in variants:
        model = factory(num_classes=100, in_chans=4)
        transition = model.layer2[0]
        assert isinstance(transition.relu1, torch.nn.ReLU6)
        assert isinstance(transition.relu2, torch.nn.ReLU6)
        assert type(model.relu) is torch.nn.ReLU
        assert isinstance(transition.shortcut, AvgPoolChannelPad)
        assert isinstance(transition.main_downsample, (
            torch.nn.AvgPool2d if pools_main else torch.nn.Identity))
        assert isinstance(transition.bn1, (
            torch.nn.BatchNorm2d if uses_bn else torch.nn.Identity))
        assert (transition.conv1.bias is None) is uses_bn


def test_avgpool_main_path_survives_physical_conversion_and_is_measured():
    torch.manual_seed(29)
    model = WideResNetCIFAR(
        depth=16, widen_factor=1, num_classes=3, in_chans=4,
        avgpool_downsample_shortcut=True,
        avgpool_main_downsample=True,
        intermediate_activation="relu6").eval()
    dense = copy.deepcopy(model)
    inputs = torch.randn(2, 4, 16, 16) * 1e-3
    convert_wide_resnet_to_physical(
        model, physical_level=2, physical=False, qat=False,
        R=50e3, C=500e-15, v_dd=10.0, one_over_q=1.0)
    torch.testing.assert_close(
        model(inputs), dense(inputs), atol=2e-6, rtol=2e-5)

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
    transition = model.layer2[0].ode_block
    assert isinstance(transition.main_downsample, MeasuredAvgPool2d)
    assert isinstance(transition.shortcut.pool, MeasuredAvgPool2d)


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
