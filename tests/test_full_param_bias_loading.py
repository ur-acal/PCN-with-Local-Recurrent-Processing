"""Full-param loading preserves biases without importing hardware buffers."""
import pytest
import torch
from torch import nn

from data_utils import load_and_register_buffer
from inference_utils import load_and_prepare_model
from pc_conv import PCConvReLU6
from pc_model import PCNetNoBatchNorm
from test_toggle_physical_head import base_model


@pytest.mark.parametrize("original_weight", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_weight_only_preserves_bias_and_ignores_hardware_buffers(original_weight, nested):
    linear = nn.Linear(4, 3)
    model = nn.Sequential(linear) if nested else linear
    linear.register_buffer("hardware_scale", torch.tensor(7.))
    prefix = "0." if nested else ""
    weight_key = "parametrizations.weight.original" if original_weight else "weight"
    state = {
        prefix + weight_key: torch.full_like(linear.weight, .25),
        prefix + "bias": torch.full_like(linear.bias, 1.2345),
        prefix + "hardware_scale": torch.tensor(99.),
    }
    load_and_register_buffer(model, state, "cpu", load_weight_only=True)
    torch.testing.assert_close(linear.weight, state[prefix + weight_key], rtol=0, atol=0)
    torch.testing.assert_close(linear.bias, state[prefix + "bias"], rtol=0, atol=0)
    assert linear.hardware_scale.item() == 7.


@pytest.mark.parametrize("kind", ["normal", "full_param", "recovery"])
@pytest.mark.parametrize("bias", [False, True])
def test_checkpoint_loader_preserves_classifier_parameters(tmp_path, kind, bias):
    source = base_model(bias=bias)
    with torch.no_grad():
        source.linear.weight.fill_(.25)
        if bias:
            source.linear.bias.fill_(1.2345)
    state = source.state_dict()
    if kind != "normal":
        state = {k.removesuffix("weight") + "parametrizations.weight.original"
                 if k.endswith(".weight") else k: v for k, v in state.items()}
    payload = dict(net=state, init_args=source.init_args, net_type="PCNetNoBatchNorm")
    if kind == "recovery":
        payload.update(training_recovery={}, checkpoint_weight_format="full_param")
    path = tmp_path / ("model_full_param_best_ckpt.pth" if kind == "full_param"
                       else "model_latest_ckpt.pth")
    torch.save(payload, path)
    loaded = load_and_prepare_model(str(path), "cpu", model_struct=PCNetNoBatchNorm,
                                   pc_conv_layer=PCConvReLU6, fuse_bn=False, noise_level=0)
    for name, param in source.named_parameters():
        torch.testing.assert_close(dict(loaded.named_parameters())[name], param, rtol=0, atol=0)
    assert (loaded.linear.bias is not None) == bias
