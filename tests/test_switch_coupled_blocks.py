"""Coupled local flow geometry, fresh reads, wrapper and scheduling contracts."""
import unittest
from copy import deepcopy
from unittest.mock import patch
import torch
from torch import nn
from switch import ODEXInitFFFBPixelSwitchExplicit as Lie
from switch import ODEXInitFFFBPixelSwitchEfficient as Jacobi
from switch import ODEXInitFFFBPixelSwitchStrang as Strang
from switch import ODEXInitFFFBPixelSwitchYoshida4 as Yoshida
from switch import ODEXInitFFFBPixelSwitchParallel as Parallel
from switch import aca_ode_solve


def make(cls=Lie, m=2):
    b=cls.__new__(cls); nn.Module.__init__(b)
    b.block_size=m; b.scale_RHS=False; b.n_iters=1
    b.FBconv=nn.Conv2d(1,1,3,padding=1,bias=False).double()
    b.FFconv=nn.Conv2d(1,1,3,padding=1,bias=False).double()
    with torch.no_grad():
        b.FBconv.weight.fill_(0.1); b.FFconv.weight.fill_(0.1)
    b.fb_kh=b.fb_kw=b.ff_kh=b.ff_kw=3
    b.fb_pad_h=b.fb_pad_w=b.ff_pad_h=b.ff_pad_w=1
    b.FBconv_copies=[deepcopy(b.FBconv) for _ in range(9)]
    b.block_FBconv_copies=[deepcopy(b.FBconv) for _ in range((m+2)**2)]
    b.act_fn=nn.ReLU(); b.init_y=lambda x:x.clone()
    b.integration_time=torch.tensor([0.,0.01],dtype=torch.float64)
    b.option_aca={'t0':0.,'t1':0.01,'method':'dopri5','rtol':1e-9,'atol':1e-9,'h':None}
    return b


def euler(fn,y,opts):
    return torch.stack([y,y+float(opts['t1']-opts['t0'])*fn(0.,y)])


class CoupledBlockTests(unittest.TestCase):
    def test_geometry_rhs_and_edges(self):
        y=torch.rand(2,1,7,9,dtype=torch.float64)
        for m in (1,2,3,4):
            b=make(m=m); coverage=torch.zeros(7,9)
            full=b.FFconv(b.act_fn(b.FBconv(y)))
            for block in b.iter_spatial_blocks(7,9):
                i,j,k,l=block; coverage[i:k,j:l]+=1
                rhs=b._make_coupled_block_ode_fn(block)(0,y)
                expected=torch.zeros_like(y);expected[:,:,i:k,j:l]=full[:,:,i:k,j:l]
                torch.testing.assert_close(rhs,expected,atol=1e-14,rtol=1e-14)
            self.assertTrue((coverage==1).all())
            geo=b.block_geometry((3,3,3+m,3+m),(12,12))
            self.assertEqual((geo['ff_count'],geo['fb_count']),(m*m,(m+2)**2))

    def test_active_mutual_influence(self):
        for m in (2,3,4):
            b=make(m=m);y=torch.ones(1,1,6,6,dtype=torch.float64)
            fn=b._make_coupled_block_ode_fn((1,1,1+m,1+m))
            z=y.clone();z[:,:,1,1]+=1
            self.assertGreater(float((fn(0,z)-fn(0,y))[0,0,1,2]),0)
            z=y.clone();z[:,:,1,2]+=1
            self.assertGreater(float((fn(0,z)-fn(0,y))[0,0,1,1]),0)

    def test_flow_signed_wrapper_and_outside(self):
        for cls in (Lie,Jacobi,Parallel,Strang,Yoshida):
            for m in (1,2,3,4):
                b=make(cls,m);y=torch.ones(1,1,5,7,dtype=torch.float64)
                original=b._make_ode_fn
                def factory(x):
                    raw=original(x)
                    return lambda t,state:3*raw(t,2*state)
                b._make_ode_fn=factory
                block=(0,0,m,m); raw=b._make_coupled_block_ode_fn(block)(0,y)
                for duration in (.001,-.001):
                    with patch('switch.aca_ode_solve',side_effect=euler):
                        out=b.local_block_flow(y,y,block,duration)
                    torch.testing.assert_close(out,y+duration*6*raw)
                    self.assertTrue(torch.equal(out[:,:,m:,:],y[:,:,m:,:]))
                    self.assertTrue(torch.equal(out[:,:,:,m:],y[:,:,:,m:]))
                    self.assertFalse(hasattr(b,'_active_spatial_block'))

    @torch.no_grad()
    def test_whole_block_is_full_coupled_solve(self):
        torch.set_num_threads(1)
        b=make(m=4);y=torch.rand(1,1,3,4,dtype=torch.float64)
        expected=aca_ode_solve(lambda t,z:b.FFconv(b.act_fn(b.FBconv(z))),
                               y,b._build_interval_option_aca(0,.01,y))[-1]
        out=b.local_block_flow(y,y,(0,0,3,4),.01)
        torch.testing.assert_close(out,expected,atol=1e-12,rtol=1e-12)
        # Outside state must not drift even through the solver's projector.
        b.option_aca['proj_fn']=lambda z:z.clamp(max=.5)
        y=torch.ones(1,1,5,7,dtype=torch.float64)
        out=b.local_block_flow(y,y,(1,1,3,3),.01)
        mask=torch.ones_like(y,dtype=torch.bool);mask[:,:,1:3,1:3]=False
        self.assertTrue(torch.equal(out[mask],y[mask]))

    def test_transposed_fb_and_alternative_ff_geometry(self):
        b=make(m=3)
        b.FBconv=nn.ConvTranspose2d(1,1,3,padding=1,bias=False).double()
        b.block_FBconv_copies=[deepcopy(b.FBconv) for _ in range(25)]
        y=torch.rand(1,1,7,9,dtype=torch.float64)
        out=b._make_coupled_block_ode_fn((2,2,5,5))(0,y)
        full=b.FFconv(b.act_fn(b.FBconv(y)))
        torch.testing.assert_close(out[:,:,2:5,2:5],full[:,:,2:5,2:5])
        b.FFconv=nn.Conv2d(1,1,1,bias=False).double();b.ff_kh=b.ff_kw=1
        self.assertEqual(b.block_geometry((2,2,5,5),(9,9))['fb_count'],9)

    def test_frozen_vs_sequential(self):
        y=torch.ones(1,1,3,5,dtype=torch.float64)
        b=make(m=2)
        with patch('switch.aca_ode_solve',side_effect=euler):
            frozen=b._run_coupled_sweep(y,frozen=True)
            sequential=b._run_coupled_sweep(y,frozen=False)
        full=b.FFconv(b.act_fn(b.FBconv(y)))
        torch.testing.assert_close(frozen,y+.01*full)
        self.assertGreater(float((sequential-frozen).abs().max()),0)

    def test_symmetric_and_yoshida_durations(self):
        for cls,factor in ((Strang,2),(Yoshida,6)):
            b=make(cls,2);y=torch.ones(1,1,3,5,dtype=torch.float64);visits=[]
            b.local_block_flow=lambda x,y,block,duration:(visits.append((block,duration)) or y)
            b._run_explicit_pixel_switch(y)
            blocks=b.iter_spatial_blocks(3,5)
            self.assertEqual(len(visits),len(blocks)*factor)
            for block in blocks:
                self.assertAlmostEqual(sum(d for p,d in visits if p==block),.01)
            self.assertEqual([p for p,d in visits[:2*len(blocks)]],blocks+blocks[::-1])

if __name__=='__main__': unittest.main()
