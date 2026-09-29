from argparse import Namespace

from train_ode_cifar import _constr_model_name, parse_n0n1n2


def test_two_stage_depths_are_parsed_independently():
    inp = [4] + [32] * 13 + [32] + [64] * 7
    out = [32] + [32] * 13 + [64] + [64] * 7
    assert parse_n0n1n2(inp, out) == "13l7l0"


def test_one_stage_pool_position_is_encoded_in_model_name():
    args = Namespace(
        pcn="PCNetNoBatchNorm", pc_conv="PCConvReLU6", offset_eps=0.0,
        ode_block="ToggleODEXInitFFFB", method="dopri5", t_end=1.75,
        tol=1e-4, weight_decay=1e-3, batch_size=128, cosine_t0=None,
        learning_rate=0.01, dataset="cifar100", kernel_size=3, stride=1,
        inp_channels=[4] + [56] * 18, out_channels=[56] * 19,
        dropout=0.25, max_pool=[0] + [0] * 8 + [1] + [0] * 9,
        tie_method=None, distill_method="srrl", distill_alpha=0.3,
        distill_temperature=2.0, img_type="CiFAIR", rggb_to_rgb=False,
        model_name=None, ode_wrapper=None, noise_level=None, timm_trainer=True)
    name = _constr_model_name(args)
    assert "19Layers18l0l0_1Pool10_" in name
