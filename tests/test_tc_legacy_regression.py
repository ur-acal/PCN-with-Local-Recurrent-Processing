"""CPU snapshots captured before TC nonideality implementation.

Baseline: colab_all 3ea543408fc958f44eab1e0bd92bc97ab017034a, clean code
worktree. New TC nonidealities are absent/off. These snapshots intentionally
exercise the existing parent wrappers also inherited by toggle classes.
"""
import unittest
import torch

from ode_pc import (
    ODEXInitFFFB, S2NoisyIYAsXZAs0, ODEWrapper1State, ODEWrapper2State,
    QATWrapper1State, QATWrapper2State, ToggleODEXInitFFFB,
    TogglePulseODEXInitFFFB, ToggleWrapper1State, ToggleQATWrapper1State,
    TogglePulseWrapper1State,
)
from pc_conv import PCConvReLU6


CASES = {
    "tc1": (ODEXInitFFFB, ODEWrapper1State),
    "tc2": (S2NoisyIYAsXZAs0, ODEWrapper2State),
    "tc1_qat": (ODEXInitFFFB, QATWrapper1State),
    "tc2_qat": (S2NoisyIYAsXZAs0, QATWrapper2State),
    "toggle": (ToggleODEXInitFFFB, ToggleWrapper1State),
    "toggle_qat": (ToggleODEXInitFFFB, ToggleQATWrapper1State),
    "pulse": (TogglePulseODEXInitFFFB, TogglePulseWrapper1State),
}


def capture(case, noisy=False, tc_flag=None):
    torch.manual_seed(1049)
    pc = PCConvReLU6(inp_chan=2, out_chan=2, kernel_size=1, padding=0,
                     cls=2, bypass=False, tie_weights=False, tie_bp=False,
                     layer_idx=0)
    with torch.no_grad():
        pc.FFconv.weight.copy_(torch.tensor([.23, -.07, .11, .31]).reshape_as(pc.FFconv.weight))
        pc.FBconv.weight.copy_(torch.tensor([.19, .05, -.03, .27]).reshape_as(pc.FBconv.weight))
    block_cls, wrapper_cls = CASES[case]
    kwargs = dict(toggle_n_cycles=5, odexinit_scaling_mode="direct")
    if noisy:
        kwargs.update(enable_spin_variation=True, sigma_spin=.1,
                      spin_variation_seed=11, enable_summing_current_noise=True,
                      summing_current_p=.6e-12, summing_noise_seed=13,
                      enable_coupler_noise=True, coupler_noise_p=.6e-12,
                      coupler_noise_seed=17)
    block = block_cls(pc_conv=pc, noise_level=0., method="dopri5",
                      t_end=.3, tol=1e-6, sde_noise_type="addi", **kwargs)
    wrapper_options = {} if tc_flag is None else dict(tc_nonidealities=tc_flag)
    wrapper = wrapper_cls(ode_block=block, R=1e4, R_max=150e3,
                          C=49e-15, v_dd=.1, state_bound=1., w_bits=5,
                          weight_quant_factor_bits=None, enob=None,
                          thermal_noise=False, is_first=True, is_last=True, **wrapper_options)
    x = torch.tensor([.1, .2, .3, .4, .15, .25, .35, .45]).reshape(1, 2, 2, 2)
    x.requires_grad_()
    block.train("qat" in case)
    y = block(x)
    y.square().sum().backward()
    result = {"output": y.detach().flatten().tolist(),
              "input_grad": x.grad.flatten().tolist()}
    if "qat" in case:
        result["ff_grad"] = block.FFconv.parametrizations.weight.original.grad.flatten().tolist()
        result["fb_grad"] = block.FBconv.parametrizations.weight.original.grad.flatten().tolist()
    return result


# Filled from the pre-implementation capture, never regenerated during tests.
SNAPSHOTS = {
    "tc1": {
        "output": [.0999373272, .2003229707, .3007086217, .4010942578, .1553293169, .2593282759, .3633272350, .4673261940],
        "input_grad": [.2065950632, .4127243459, .6188536882, .8249829412, .3171322048, .5288640857, .7405959964, .9523279071]},
    "tc2": {
        "output": [.1000856608, .2005053908, .3009251058, .4013448060, .1541696042, .2572816908, .3603938520, .4635059834],
        "input_grad": [.2054235786, .4105042517, .6155849695, .8206656575, .3135230243, .5227668285, .7320109010, .9412547946]},
    "toggle": {
        "output": [.1001147851, .2006089091, .3011030555, .4015971720, .1547611058, .2583126426, .3618641794, .4654156864],
        "input_grad": [.2062430382, .4120944142, .6179460287, .8237973452, .3154909313, .5260792971, .7366677523, .9472559094]},
    "pulse": {
        "output": [.0937226564, .1887012124, .2836790383, .3786568940, .1562773585, .2612988055, .3663223684, .4713431001],
        "input_grad": [.1905870736, .3810484409, .5715101361, .7619701028, .3156964779, .5262418389, .7367967963, .9473425150]},
    "toggle_noisy": {
        "output": [.1000337079, .2006422281, .3020487726, .4030734003, .1547808647, .2580935061, .3626850247, .4629786015],
        "input_grad": [.2065753937, .4122399390, .6185745597, .8241955638, .3157072961, .5241434574, .7378873229, .9410902858]},
    "pulse_noisy": {
        "output": [.0928765163, .1911724359, .2841277719, .3772424459, .1580617875, .2589210272, .3668749630, .4694152176],
        "input_grad": [.1895483583, .3855663836, .5715342760, .7561314106, .3199197948, .5210642815, .7370286584, .9404754639]},
}
SNAPSHOTS["tc1_qat"] = dict(SNAPSHOTS["tc1"],
    ff_grad=[.0311336797, .0873595253, .0360211954, .1017924622],
    fb_grad=[.0723701343, .0619957745, .0980381370, .0844133273])
SNAPSHOTS["tc2_qat"] = dict(SNAPSHOTS["tc2"],
    ff_grad=[.0248464495, .0604768842, .0295155477, .0721799731],
    fb_grad=[.0561694466, .0491853468, .0673106313, .0591489300])
SNAPSHOTS["toggle_qat"] = dict(SNAPSHOTS["toggle"],
    ff_grad=[.0284274369, .0681436956, .0340146385, .0819132030],
    fb_grad=[.0641883239, .0565519594, .0758871436, .0670902953])
SNAPSHOTS["toggle_qat_noisy"] = dict(SNAPSHOTS["toggle_noisy"],
    ff_grad=[.0271902811, .0703446716, .0300805271, .0774368495],
    fb_grad=[.0586458556, .0524387397, .0695393533, .0621925071])


class LegacyRegressionTests(unittest.TestCase):
    def test_existing_tc_and_toggle_snapshots(self):
        for case, noisy in ([(c, False) for c in CASES] +
                            [("toggle", True), ("toggle_qat", True), ("pulse", True)]):
            key = case + ("_noisy" if noisy else "")
            with self.subTest(case=key):
                actual = capture(case, noisy)
                expected = SNAPSHOTS[key]
                self.assertEqual(actual.keys(), expected.keys())
                for field in actual:
                    torch.testing.assert_close(torch.tensor(actual[field]), torch.tensor(expected[field]),
                                               rtol=3e-5, atol=2e-7)
