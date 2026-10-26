"""CNN bias is an ideal current term, not part of nonlinear-R convolution."""
import pytest
import torch
from torch import nn

from physical_feedforward_tc import TCPhysicalBasicBlock
from test_tc_curve_groups import package
from test_tc_unified_backend import original_off


@pytest.mark.parametrize('empirical', [False, True])
@pytest.mark.parametrize('method,n', [('shared', 0), ('shared', 3), ('shared', 15),
                                    ('loop', 0), ('grouped', 0)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bias_current_and_gradients_match_independent_expression(empirical, method, n, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dtype = torch.float64 if device == 'cpu' else torch.float32
    conv = nn.Conv2d(2, 2, 1, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([.2, -.4, 0., .8], device=device).reshape_as(conv.weight))
        conv.bias.copy_(torch.tensor([.02, -.03], device=device))
    block = TCPhysicalBasicBlock(conv, one_shot_conv=True,
        tc_conv_method=method, tc_curve_top_n=n).to(device=device, dtype=dtype)
    curves = package(device, empirical)
    if empirical:
        from tc_empirical import EmpiricalTCResistanceCurves
        curves = EmpiricalTCResistanceCurves(**vars(curves))
    block._tc_resistance_package = curves
    data = torch.linspace(.01, .04, 16, device=device, dtype=dtype).reshape(2, 2, 2, 2)
    actual_input = data.clone().requires_grad_()
    actual = block._current(conv, actual_input)
    draw = block._tc_samples['conv1']
    if torch.is_tensor(draw):
        from tc_empirical import EmpiricalLevels
        draw = EmpiricalLevels.from_gaussian(block._tc_curve_package.v_grid, draw)
    reference_input = data.clone().requires_grad_()
    reference = original_off(block, conv, reference_input, draw)
    # Initial non-QAT block has unit stage scale; bias enters once as q*b.
    reference = reference + (block.q * conv.bias).view(1, -1, 1, 1)
    actual_grads = torch.autograd.grad(actual.square().sum(), (actual_input, conv.weight, conv.bias))
    expected_grads = torch.autograd.grad(reference.square().sum(), (reference_input, conv.weight, conv.bias))
    tolerance = dict(atol=1e-5, rtol=1e-4) if device == 'cuda' else dict(atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(actual, reference, **tolerance)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


def test_actual_no_bn_resnet44_nonlinear_forward_and_backward():
    from baseline.cifar_resnet import resnet44_cifar_nobn
    from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_blocks
    torch.set_num_threads(1)
    torch.manual_seed(73)
    model = resnet44_cifar_nobn(num_classes=10).double()
    stem_bias = model.conv1.bias
    convert_wide_resnet_to_physical(model, tc_options={'one_shot_conv': True},
        R=1e4, C=49e-15, v_dd=.1, one_over_q=32)
    assert model.conv1.block.conv1.bias is stem_bias
    for block in iter_physical_blocks(model):
        block._tc_resistance_package = package()
    inputs = (torch.rand(2, 3, 8, 8, dtype=torch.float64)*.03).requires_grad_()
    logits = model(inputs)
    logits.square().mean().backward()
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    assert stem_bias.grad is not None and torch.isfinite(stem_bias.grad).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_pcn_bias_rejection_is_unchanged():
    from ode_pc import ODEBlockPC
    block = TCPhysicalBasicBlock(nn.Conv2d(2, 2, 1, bias=True))
    with pytest.raises(ValueError, match='bias-free'):
        ODEBlockPC._tc_dense_conv(block, block.conv1, torch.ones(1, 2, 2, 2), object())
