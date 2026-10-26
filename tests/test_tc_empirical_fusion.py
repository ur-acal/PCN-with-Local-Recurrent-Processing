"""CUDA native-grid TC empirical fusion, including complete PCN forwards."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from tc_empirical import prepare_empirical_curves
from toggle_edge_inference import triton, empirical_edge_forward
from validation import MVMConv, Validator, conv2d_to_matrix_fixed_padding
from test_tc_dense_training import make_block
from ode_pc import ODEWrapper1State, ODEWrapper2State

BANK = Path(__file__).resolve().parents[1]/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'


@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA/Triton required')
class TCEmpiricalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.package = prepare_empirical_curves(str(BANK), levels=torch.arange(16,device='cuda')/15,
            R=13300., R_max=200000., dtype=torch.float32, device='cuda')

    def test_native_per_level_edges_zero_signed_and_fallbacks(self):
        kernel = torch.tensor([0., -1/15, 2/15, -3/15, 4/15, 1., -1., 0., 7/15],
                              device='cuda').reshape(1,1,3,3)
        mat, _, _ = conv2d_to_matrix_fixed_padding((1,4,4),kernel,padding=1)
        m = MVMConv(mat,dict(inp_chan=1,out_chan=1,ker_h=3,ker_w=3,stride=1,padding=1)).eval()
        m.enable_csv(None,None,None,None,nn.Hardtanh(-.1,.1),13300.,
                     tc_curve_package=self.package, nonlinear_R_curve_seed=7)
        assignment = m.nonlinear_R_curve_assignment.clone()
        for batch in (1,4,128):
            x = torch.linspace(-.2,.2,16*batch,device='cuda').reshape(batch,1,4,4)
            with torch.no_grad():
                m._tc_fused_edges = False
                expected = m(x)
                del m._tc_fused_edges  # Actual default must use the fused path.
                rng = torch.cuda.get_rng_state()
                fused = empirical_edge_forward(m,x.reshape(batch,-1).T,m._tc_signed_mat,13300.)
                self.assertIsNotNone(fused)
                actual = m(x)
                torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-6)
                self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
                self.assertTrue(torch.equal(assignment,m.nonlinear_R_curve_assignment))
                m._tc_measure_coupler_energy = True
                self.assertIsNone(empirical_edge_forward(m,x.reshape(batch,-1).T,m._tc_signed_mat,13300.))
                m._tc_measure_coupler_energy = False

    def test_complete_expanded_pcn_logits_both_states(self):
        for two in (False, True):
            torch.manual_seed(82)
            blocks, wrappers = [], []
            for index in range(2):
                b = make_block(two).cuda()
                b.clean_params = {name: value.cuda() for name,value in b.clean_params.items()}
                b.layer_idx = index
                b.integration_time = b.integration_time.cuda()
                if isinstance(b.FBconv,nn.ConvTranspose2d):
                    old = b.FBconv
                    b.FBconv = nn.Conv2d(2,2,1,bias=False).cuda()
                    b.FBconv.weight.data.copy_(old.weight.data.transpose(0,1))
                cls = ODEWrapper2State if two else ODEWrapper1State
                w = cls(ode_block=b,tc_nonidealities=True,R=13300.,R_max=200000.,
                    C=49e-15,k=1e3,v_dd=.1,state_bound=1.,w_bits=5,thermal_noise=False,
                    is_first=index==0,is_last=index==1,nonlinear_R=True,
                    nonlinear_R_curve_sampling='empirical_with_replacement',
                    tc_empirical_curve_bank=str(BANK), nonlinear_R_curve_seed=13)
                blocks.append(b); wrappers.append(w)
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.PcConvs=nn.ModuleList(blocks)
                    self.head=nn.Linear(2,3).cuda()
                def forward(self,x):
                    for block in self.PcConvs:x=block(x)
                    return self.head(x.mean((2,3)))
            model = Net().eval()
            x = torch.rand(4,2,3,3,device='cuda')*.08
            with tempfile.TemporaryDirectory() as folder, torch.no_grad():
                Validator(model,folder,'cuda',[(x,torch.zeros(4,device='cuda'))],folder,wrapper=wrappers)
                model.eval()  # Newly installed MVM children otherwise start in training mode.
                modules=[m for m in model.modules() if isinstance(m,MVMConv)]
                for m in modules:m._tc_fused_edges=False
                expected=model(x)
                for m in modules:del m._tc_fused_edges
                actual=model(x)
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))

    def test_complete_expanded_cnn_logits(self):
        from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
        from feedforward_validation import FeedForwardCNNValidator
        b=TCPhysicalBasicBlock(nn.Conv2d(2,2,1,bias=False).cuda(),
            R=13300.,C=49e-15,v_dd=.1,tc_method='euler',tc_num_steps=10,
            tc_distribution='empirical_with_replacement',tc_empirical_curve_bank=str(BANK))
        wrapper=TCFeedForwardPhysicalWrapper(b,quantize_weights=True)
        wrapper.configure_nonlinear_R_inference('/ignored',curve_seed=19)
        model=nn.Sequential(wrapper,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,3).cuda()).eval()
        x=torch.rand(4,2,4,4,device='cuda')*.08
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            FeedForwardCNNValidator(model,folder,'cuda',[(x,torch.zeros(4,device='cuda'))],folder,wrapper=[wrapper])
            model.eval()
            modules=[m for m in model.modules() if isinstance(m,MVMConv)]
            for m in modules:m._tc_fused_edges=False
            expected=model(x)
            for m in modules:del m._tc_fused_edges
            actual=model(x)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))

    def test_expanded_analog_head(self):
        from final_linear import AnalogLinear
        from physical_feedforward_tc import TCPhysicalBasicBlock, TCFeedForwardPhysicalWrapper
        block=TCPhysicalBasicBlock(nn.Conv2d(2,3,1,bias=False).cuda(),
            R=13300.,C=49e-15,v_dd=.1,one_shot_conv=True,
            tc_distribution='empirical_with_replacement',tc_empirical_curve_bank=str(BANK))
        wrapper=TCFeedForwardPhysicalWrapper(block)
        wrapper.configure_nonlinear_R_inference('/ignored',curve_seed=19)
        head=AnalogLinear(2,3,bias=False).cuda().eval()
        head.configure(physical=True,q=.1,v_dd=.1,template=block,family='tc',R=13300.,C=49e-15)
        head.begin_evaluation_trial()
        with torch.no_grad():
            x=torch.rand(4,2,device='cuda')*.03
            head(x)
            head._expanded_module._tc_fused_edges=False
            expected=head(x)
            del head._expanded_module._tc_fused_edges
            actual=head(x)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
