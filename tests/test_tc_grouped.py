"""Loop/grouped equivalence, including input and QAT gradient paths."""
import unittest
import torch
from torch import nn
from test_tc_dense_training import make_block, wrap, synthetic_package


class GroupedTCTests(unittest.TestCase):
    def test_conv_geometry_outputs_and_gradients(self):
        for transpose in (False,True):
            for groups in (1,2):
                for stride in (1,2):
                    b=make_block(); b.q_hi=3; b.weight_scale=1.; b.v_dd=.1
                    p=synthetic_package(b); b._tc_curve_package=p
                    curves=p.means*(1+torch.tensor([1.,2.,3.])[:,None]*p.v_grid)
                    cls=nn.ConvTranspose2d if transpose else nn.Conv2d
                    conv=cls(2,4,3,stride=stride,padding=2,dilation=2,groups=groups,
                        bias=False,dtype=torch.float64,**({'output_padding':stride-1} if transpose else {}))
                    with torch.no_grad():
                        conv.weight.copy_((torch.arange(conv.weight.numel()).reshape_as(conv.weight)%7-3)/3)
                    x=torch.linspace(-.15,.15,64,dtype=torch.float64).reshape(2,2,4,4).requires_grad_()
                    results=[]
                    for mode in ('loop','grouped'):
                        b._tc_conv_method=mode
                        y=b._tc_dense_conv(conv,x,curves)
                        g=torch.autograd.grad(y.square().sum(),(x,conv.weight))
                        results.append([y,*g])
                    for a,c in zip(*results):torch.testing.assert_close(a,c,rtol=1e-10,atol=1e-12)

    def test_qat_weight_gradients_both_states(self):
        for two in (False,True):
            b=make_block(two); w=wrap(b,tc_conv_method='grouped')
            p=synthetic_package(b); b._tc_curve_package=p
            curves=p.means*(1+p.v_grid)
            for conv in (b.FFconv,b.FBconv):
                x=torch.full((2,2,3,3),.03,requires_grad=True)
                param=conv.parametrizations.weight.original
                results=[]
                for mode in ('loop','grouped'):
                    b._tc_conv_method=mode
                    y=b._tc_dense_conv(conv,x,curves)
                    results.append([y,*torch.autograd.grad(y.square().sum(),(x,param))])
                for a,c in zip(*results):torch.testing.assert_close(a,c,rtol=2e-5,atol=1e-8)

if __name__=='__main__':unittest.main()
