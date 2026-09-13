"""Sanity-check the Jacobian alignment statistic independently of the model."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tc_shared_sensitivity import alignment_metrics


class AlignmentTests(unittest.TestCase):
    def test_exact_operator_curve_derivative_matches_finite_difference(self):
        from test_tc_dense_training import make_block,synthetic_package
        b=make_block();b.q_hi=3;b.weight_scale=1.;b.v_dd=.1
        package=synthetic_package(b);b._tc_curve_package=package;b._tc_conv_method='grouped'
        conv=torch.nn.Conv2d(3,2,1,bias=False,dtype=torch.float64)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([1.,-2/3,0.,1/3,-1.,2/3]).reshape_as(conv.weight))
        x=torch.tensor([.02,-.04,.06],dtype=torch.float64).reshape(1,3,1,1)
        d=torch.ones(3,3,dtype=torch.float64,requires_grad=True)
        def loss(distortion):
            curves=package.programmed_resistances[:,None]/distortion
            return b._tc_dense_conv(conv,x,curves).square().sum()
        grad=torch.autograd.grad(loss(d),d)[0]
        direction=torch.arange(1,10,dtype=d.dtype).reshape_as(d)/10
        h=1e-5
        numerical=(loss(d.detach()+h*direction)-loss(d.detach()-h*direction))/(2*h)
        torch.testing.assert_close((grad*direction).sum(),numerical,rtol=1e-7,atol=1e-12)

    def test_aligned_and_cancelling_sensitivities(self):
        torch.manual_seed(12)
        p=torch.tensor([.2,.3,.5],dtype=torch.float64)
        total=torch.randn(8,1,5,dtype=torch.float64)
        aligned=total*p[None,:,None]
        stats=alignment_metrics(aligned,p)
        self.assertLess(stats['relative_residual'],1e-14)
        torch.testing.assert_close(torch.tensor(stats['fitted_coefficients']),p.float())
        cancel=total*torch.tensor([1.,-1.,1.])[None,:,None]
        stats=alignment_metrics(cancel,p)
        self.assertGreater(stats['relative_residual'],.5)
        self.assertLess(stats['code_cosines'][1],-.99)
        self.assertLess(stats['best_coefficient_residual'],1e-14)

if __name__=='__main__':unittest.main()
