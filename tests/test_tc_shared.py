"""Method-2a sampling, solve lifetime, geometry and differentiable QAT checks."""
from dataclasses import replace
import argparse
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from tc_cli import add_tc_arguments, wrapper_options
from tc_nonidealities import TCSharedCurve
from test_tc_dense_training import make_block, wrap, synthetic_package


class TCSharedTests(unittest.TestCase):
    def test_histogram_uniform_and_conditional_curve_draw(self):
        b=make_block(); b.q_hi=3; b.weight_scale=1.
        p=synthetic_package(b)
        codes=torch.tensor([0]*40+[1]*8+[3]*2)
        for mode, expected in [('histogram',torch.tensor([.8,0.,.2])),
                               ('uniform',torch.tensor([1/3]*3))]:
            gen=torch.Generator().manual_seed(18)
            draws=[p.sample_shared(codes,sampling=mode,generator=gen) for _ in range(3000)]
            picked=torch.stack([d.code for d in draws])
            frequencies=torch.bincount(picked,minlength=4)[1:]/len(draws)
            torch.testing.assert_close(frequencies,expected,atol=.035,rtol=0)
            residual=torch.stack([d.resistance-p.means[d.code-1] for d in draws])
            torch.testing.assert_close(torch.cov(residual.T),p.covariance,atol=6,rtol=.08)
            for d in draws[:10]:
                torch.testing.assert_close(d.nominal_R,p.programmed_resistances[d.code-1])
                self.assertFalse(d.resistance.requires_grad)

    def test_zero_and_replay_and_positive_guard(self):
        b=make_block(); b.q_hi=3; b.weight_scale=1.
        p=synthetic_package(b); g=torch.Generator().manual_seed(3)
        state=g.get_state().clone()
        self.assertIsNone(p.sample_shared(torch.zeros(10,dtype=torch.long),generator=g))
        self.assertTrue(torch.equal(state,g.get_state()))
        a=p.sample_shared(torch.tensor([1,2,3]),generator=g)
        g.set_state(state); c=p.sample_shared(torch.tensor([1,2,3]),generator=g)
        torch.testing.assert_close(a.resistance,c.resistance)
        guarded=replace(p,means=-torch.ones_like(p.means),factor=torch.zeros_like(p.factor))
        d=guarded.sample_shared(torch.tensor([1]),generator=g)
        self.assertTrue(torch.all(d.resistance==p.floor_ohms))
        with self.assertRaises(ValueError):p.sample_shared(torch.tensor([1]),sampling='bad')

    def test_single_conv_matches_reference_geometry_gradients(self):
        for transpose in (False,True):
            for groups in (1,2):
                b=make_block(); b.q_hi=3; b.weight_scale=1.; b.v_dd=.1
                p=synthetic_package(b); b._tc_curve_package=p
                nominal=p.programmed_resistances[1]
                curve=TCSharedCurve(torch.tensor(2),nominal*(1+2*p.v_grid),nominal)
                cls=nn.ConvTranspose2d if transpose else nn.Conv2d
                extra={'output_padding':1} if transpose else {}
                conv=cls(2,4,3,stride=2,padding=2,dilation=2,groups=groups,bias=False,dtype=torch.float64,**extra)
                x=torch.linspace(-.15,.15,100,dtype=torch.float64).reshape(2,2,5,5).requires_grad_()
                got=b._tc_dense_conv(conv,x,curve)
                corrected=x/(1+2*x.clamp(-.1,.1))
                fn=F.conv_transpose2d if transpose else F.conv2d
                expected=fn(corrected,conv.weight,stride=2,padding=2,dilation=2,groups=groups,**extra)
                torch.testing.assert_close(got,expected)
                for left,right in zip(torch.autograd.grad(got.square().sum(),(x,conv.weight)),
                                      torch.autograd.grad(expected.square().sum(),(x,conv.weight))):
                    torch.testing.assert_close(left,right)

    def test_both_state_qat_and_solve_lifetimes(self):
        for two in (False,True):
            b=make_block(two); w=wrap(b,tc_conv_method='shared',tc_curve_sampling='histogram')
            b._tc_curve_package=synthetic_package(b)
            b._tc_curve_generator=torch.Generator().manual_seed(31)
            first=b._tc_curves_for_solve()
            second=b._tc_curves_for_solve()
            self.assertFalse(torch.equal(first['FFconv'].resistance,first['FBconv'].resistance))
            self.assertFalse(torch.equal(first['FFconv'].resistance,second['FFconv'].resistance))
            b.eval(); self.assertIs(b._tc_curves_for_solve(),second)
            b.reset_tc_curves(); self.assertIsNot(b._tc_curves_for_solve(),second)
            for name in ('FFconv','FBconv'):
                conv=getattr(b,name); x=torch.full((2,2,3,3),.03,requires_grad=True)
                y=b._tc_dense_conv(conv,x,first[name])
                again=b._tc_dense_conv(conv,x,first[name])
                torch.testing.assert_close(y,again,rtol=0,atol=0)
                grads=torch.autograd.grad(y.square().sum(),(x,conv.parametrizations.weight.original))
                for grad in grads:
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(),0)
            # Expanded modules own independent per-edge draws, regardless of method.
            for conv in (b.FFconv,b.FBconv):conv._tc_curve_package=b._tc_curve_package
            state=b._tc_curve_generator.get_state().clone()
            self.assertIsNone(b._tc_curves_for_solve())
            self.assertTrue(torch.equal(state,b._tc_curve_generator.get_state()))

    def test_cli_forwarding_and_validation(self):
        from tc_cli import NOISE_KEYS
        parser=argparse.ArgumentParser(); add_tc_arguments(parser)
        args=parser.parse_args(['--tc_conv_method','shared','--tc_curve_sampling','uniform'])
        for key in NOISE_KEYS:
            if not hasattr(args,key):setattr(args,key,None)
        options=wrapper_options(args)
        self.assertEqual(options['tc_conv_method'],'shared')
        self.assertEqual(options['tc_curve_sampling'],'uniform')
        with self.assertRaises(ValueError):wrap(make_block(),tc_curve_sampling='bad')

    def test_shared_full_solve_noise_and_backward_replay(self):
        for two in (False,True):
            for sampling in ('histogram','uniform'):
                results=[]
                for regenerate in (False,True):
                    b=make_block(two)
                    w=wrap(b,tc_conv_method='shared',tc_curve_sampling=sampling,
                        enable_spin_variation=True,enable_summing_current_noise=True,
                        enable_coupler_noise=True,spin_variation_seed=11,
                        summing_noise_seed=12,coupler_noise_seed=13,enable_measured_activation=True)
                    b._tc_curve_package=synthetic_package(b)
                    b._tc_curve_generator=torch.Generator().manual_seed(14)
                    b.option_aca['regenerate_graph']=regenerate
                    x=torch.full((2,2,2,2),.2,requires_grad=True)
                    y=b(x)
                    state=b._tc_curve_generator.get_state().clone()
                    params=[conv.parametrizations.weight.original for conv in (b.FFconv,b.FBconv)]
                    grads=torch.autograd.grad(y.square().sum(),[x]+params)
                    self.assertTrue(torch.equal(state,b._tc_curve_generator.get_state()))
                    for g in grads:
                        self.assertTrue(torch.isfinite(g).all())
                        self.assertGreater(float(g.abs().sum()),0)
                    results.append([y,*grads])
                for left,right in zip(*results):torch.testing.assert_close(left,right,rtol=2e-5,atol=1e-8)

if __name__=='__main__':unittest.main()
