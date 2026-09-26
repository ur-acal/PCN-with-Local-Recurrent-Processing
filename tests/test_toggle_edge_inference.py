"""Bitwise lookup checks: unequal grids, boundary clipping and fallbacks."""
import unittest
from types import SimpleNamespace

import torch
from validation import MVMConv
from toggle_edge_inference import empirical_resistance, empirical_edge_forward, triton


@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA/Triton required')
class EmpiricalLookupTests(unittest.TestCase):
    def module(self):
        grid = torch.tensor([[-.5, -.2, 0., .1, .5],
                             [-.4, -.1, .3, float('inf'), float('inf')]], device='cuda')
        torch.manual_seed(17)
        return SimpleNamespace(
            training=False, _toggle_pulse_edges=True, _toggle_fused_lookup=True,
            proj_fn=None, nonlinear_R_curve_bank_v_grid=grid,
            nonlinear_R_curve_bank_R_left=torch.rand(2,4,device='cuda')*50000+10000,
            nonlinear_R_curve_bank_R_slope=torch.randn(2,4,device='cuda')*10000,
            nonlinear_R_curve_bank_lengths=torch.tensor([5,3],device='cuda'))

    def test_exact_lookup_and_no_rng_consumption(self):
        m = self.module()
        for projection in (None, torch.nn.Hardtanh(-.25,.25)):
            m.proj_fn = projection
            for batch in (1,4,128):
                v = torch.linspace(-.8,.8,257*batch,device='cuda').reshape(batch,257).T
                index = torch.arange(257,device='cuda') % 2
                with torch.no_grad():
                    m._toggle_fused_lookup = False
                    expected = MVMConv._get_curve_bank_R_eff(m,v,index)
                    m._toggle_fused_lookup = True
                    rng = torch.cuda.get_rng_state()
                    actual = empirical_resistance(m,v,index)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                    self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            # Exact knots, rails and adjacent representable floats.
            knots = torch.tensor([-.5,-.2,0.,.1,.5],device='cuda')
            v = torch.stack([knots,torch.nextafter(knots,torch.full_like(knots,float('inf')))])
            with torch.no_grad():
                m._toggle_fused_lookup = False
                expected = MVMConv._get_curve_bank_R_eff(m,v,torch.zeros(2,device='cuda',dtype=torch.long))
                m._toggle_fused_lookup = True
                torch.testing.assert_close(empirical_resistance(m,v,torch.zeros(2,device='cuda',dtype=torch.long)), expected,atol=0,rtol=0)

    def test_fallbacks(self):
        m = self.module()
        v = torch.ones(1,2,device='cuda')
        index = torch.zeros(1,device='cuda',dtype=torch.long)
        self.assertIsNone(empirical_resistance(m,v,index))  # gradients enabled
        with torch.no_grad():
            for name,value in (('training',True),('_toggle_pulse_edges',False),
                               ('_toggle_fused_lookup',False),('_tc_measure_coupler_energy',True),
                               ('proj_fn',torch.nn.ReLU())):
                original = getattr(m,name,None)
                setattr(m,name,value)
                self.assertIsNone(empirical_resistance(m,v,index))
                setattr(m,name,original)
            self.assertIsNone(empirical_resistance(m,v.double(),index))
            self.assertIsNone(empirical_resistance(m,v.cpu(),index.cpu()))

    def test_default_enabled_only_for_expanded_toggle(self):
        m = self.module()
        del m._toggle_fused_lookup
        v = torch.zeros(1,4,device='cuda')
        index = torch.zeros(1,device='cuda',dtype=torch.long)
        with torch.no_grad():
            self.assertIsNotNone(empirical_resistance(m,v,index))
            del m._toggle_pulse_edges
            self.assertIsNone(empirical_resistance(m,v,index))

    def test_fused_signed_fractional_zero_edges_and_fallbacks(self):
        from validation import conv2d_to_matrix_fixed_padding
        kernel = torch.tensor([1.,-.5,0.,.3,-1.,.8,0.,-.2,1.]).reshape(1,1,3,3)
        mat,_,_ = conv2d_to_matrix_fixed_padding((1,3,3),kernel,padding=1)
        m = MVMConv(mat.cuda(),dict(padding=1,stride=1,ker_h=3,ker_w=3,inp_chan=1,out_chan=1)).eval()
        bank = self.module()
        for key,value in vars(bank).items():
            if key.startswith('nonlinear_R_curve_bank_'):setattr(m,key,value)
        m._toggle_pulse_edges = True
        m.nonlinear_R_curve_sharing = 'per_coupler'
        m.nonlinear_R_curve_sampling = 'empirical_with_replacement'
        m.nonlinear_R_curve_assignment = torch.arange(mat.values().numel(),device='cuda')%2
        m.nonlinear_R_curve_row_ids = torch.arange(9,device='cuda').repeat_interleave(m.mat.crow_indices().diff())
        # Non-integer pulse values exercise duty overlap and multiplicative mismatch.
        values = m.mat.values()*.713
        values[::3] = 0
        weights = torch.sparse_csr_tensor(m.mat.crow_indices(),m.mat.col_indices(),values,size=m.mat.shape)
        for projection in (None,torch.nn.Hardtanh(-.25,.25)):
            m.proj_fn = projection
            for batch in (1,4,128):
                x = torch.linspace(-.8,.8,9*batch,device='cuda').reshape(batch,9).T
                with torch.no_grad():
                    m._toggle_fused_edges = m._toggle_fused_lookup = False
                    expected = m._forward_pulse_per_edge(x,weights,50000.)
                    m._toggle_fused_edges = True
                    rng = torch.cuda.get_rng_state()
                    actual = empirical_edge_forward(m,x,weights,50000.)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=2e-6)
                    self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            with torch.no_grad():
                zero = torch.sparse_csr_tensor(weights.crow_indices(),weights.col_indices(),torch.zeros_like(values),size=weights.shape)
                self.assertEqual(empirical_edge_forward(m,x,zero,50000.).count_nonzero(),0)
                m._tc_measure_coupler_energy = True
                self.assertIsNone(empirical_edge_forward(m,x,weights,50000.))
                m._tc_measure_coupler_energy = False
                m.train()
                self.assertIsNone(empirical_edge_forward(m,x,weights,50000.))
                m.eval()
        self.assertIsNone(empirical_edge_forward(m,x,weights,50000.)) # gradients enabled


if __name__ == '__main__':
    unittest.main()
