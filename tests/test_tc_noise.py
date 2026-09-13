"""TC diffusion, static gains and accepted-step lifecycle (CPU software)."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from tc_nonidealities import TCNoiseLifecycle
from test_tc_dense_training import make_block, wrap
from TorchDiffEqPack.odesolver import odesolve


def tape(ref, fb=True):
    coeff=[(torch.full_like(ref,.02),torch.full_like(ref,.03))]
    generators={(key,source):torch.Generator().manual_seed(10+i*17+j)
                for i,key in enumerate([0,'fb']) for j,source in enumerate(['sum','coupler'])}
    return TCNoiseLifecycle(coeff,generators,coeff[0] if fb else None,ref)


class TCNoiseTests(unittest.TestCase):
    def test_nominal_maps_padding_sign_and_transpose(self):
        b=make_block()
        for transpose in (False,True):
            cls=nn.ConvTranspose2d if transpose else nn.Conv2d
            conv=cls(1,1,3,padding=1,bias=False)
            conv.weight.data.fill_(-.5);conv.weight.data[:,:,1,1]=0
            x=torch.ones(2,1,3,3)
            expected=torch.tensor([[1.5,2.5,1.5],[2.5,4.,2.5],[1.5,2.5,1.5]])[None,None]
            torch.testing.assert_close(b._tc_nominal_sum(conv,x),expected)

    def test_diffusion_units_and_no_legacy_noise(self):
        for two in (False,True):
            b=make_block(two)
            w=wrap(b,enable_summing_current_noise=True,enable_coupler_noise=True,
                   summing_current_p=.7e-12,coupler_noise_p=.4e-12)
            x=torch.ones(2,2,2,2)*.02
            if two:
                w.C_ff=2*w.C_fb
            context,_,_=b._tc_prepare_noise(x,two)
            y=b.init_y(x);y=y[0] if two else y
            fb=b._tc_nominal_sum(b.FBconv,y);ff=b._tc_nominal_sum(b.FFconv,fb)
            caps=(w.C_ff,w.C_fb) if two else (w.C,)
            for index,(s,c) in enumerate(context.coefficients):
                counts=(ff,fb)[index]
                expected=(.7e-12**2+.4e-12**2*counts)*5/caps[index]**2
                torch.testing.assert_close(s.square()+c.square(),expected)
            self.assertEqual(b.offset_eps,0.)
            if not two:
                s,c=context.fb_coefficients
                expected=b._tc_fb_integral.variance_A2*5*((.7/.6)**2+(.4/.6)**2*fb)
                torch.testing.assert_close(s.square()+c.square(),expected,atol=0,rtol=1e-5)

    def test_spin_training_eval_lifetime_and_independence(self):
        b=make_block();w=wrap(b,enable_spin_variation=True,spin_variation_seed=5)
        x=torch.ones(3,2,2,2)*.02
        _,sy,sz=b._tc_prepare_noise(x)
        self.assertEqual(sy.shape,(1,2,2,2))
        self.assertFalse(torch.equal(sy,sz))
        _,next_y,_=b._tc_prepare_noise(x)
        self.assertFalse(torch.equal(sy,next_y))
        b.eval()
        _,held,_=b._tc_prepare_noise(x)
        torch.testing.assert_close(next_y,held)
        b.reset_tc_spin()
        _,fresh,_=b._tc_prepare_noise(x)
        self.assertFalse(torch.equal(held,fresh))

    def test_fb_held_samples_replay_and_empirical_variance(self):
        ref=torch.zeros(40000)
        ctx=tape(ref)
        fb=ctx.fb_current().clone()
        torch.testing.assert_close(ctx.fb_current(),fb)
        self.assertAlmostEqual(fb.var().item(),.02**2+.03**2,delta=.00004)
        noise=ctx.normal(ref,0)
        self.assertAlmostEqual(noise.var().item(),1.,delta=.04)
        ctx.accepted()
        self.assertFalse(torch.equal(ctx.fb_current(),fb))
        ctx.restart()
        torch.testing.assert_close(ctx.fb_current(),fb)
        torch.testing.assert_close(ctx.normal(ref,0),noise)

    def test_fixed_partial_step_and_replay(self):
        for h in (.4,2.):
            ref=torch.zeros(2)
            ctx=tape(ref,fb=False)
            def rhs(t,y):return torch.zeros_like(y)
            rhs.tc_context=ctx
            options=dict(method='euler',t0=0.,t1=1.,h=h,t_eval=[1.],
                         noise_type='addi',eps=(.02**2+.03**2)**.5)
            solver=odesolve(rhs,ref,options,return_solver=True)
            got=solver.integrate(ref,0.,t_eval=[1.])
            expected=torch.zeros_like(ref)
            steps=[.4,.4,.2] if h<1 else [1.]
            for i,dt in enumerate(steps):
                expected+=ctx.tape[(i,0)]*dt**.5
            torch.testing.assert_close(got.reshape(-1,2)[-1],expected,atol=1e-7,rtol=1e-5)
            ctx.restart()
            again=solver.integrate(ref,0.,t_eval=[1.])
            torch.testing.assert_close(got,again)

    def test_adaptive_rejection_holds_fb_and_endpoint_noise(self):
        for endpoint in (False,True):
            ref=torch.zeros(2)
            ctx=tape(ref)
            seen=[]
            def rhs(t,y):
                value=ctx.fb_current()
                seen.append((ctx.index,value.clone()))
                return value
            rhs.tc_context=ctx
            opts=dict(method='dopri5',t0=0.,t1=1.,h=.4,t_eval=[1.],
                      eps=(.02**2+.03**2)**.5,noise_type='addi',end_point_mode=endpoint)
            solver=odesolve(rhs,ref,opts,return_solver=True)
            original=solver.adapt_stepsize
            calls=[0]
            def reject_once(y,yn,error,h,**kwargs):
                calls[0]+=1
                if calls[0]==1:return h/2,False,True
                return original(y,yn,error,h,**kwargs)
            with patch.object(solver,'adapt_stepsize',side_effect=reject_once):
                result,steps=solver.integrate(ref,0.,t_eval=[1.],return_steps=True)
            self.assertGreater(calls[0],1)
            for index,value in seen:
                torch.testing.assert_close(value,ctx.tape[(index,'fb')])
            expected=torch.zeros_like(ref);last=0.
            for i,point in enumerate(steps):
                dt=float(point)-last
                expected+=ctx.tape[(i,'fb')]*dt+ctx.tape[(i,0)]*dt**.5
                last=float(point)
            torch.testing.assert_close(result.reshape(-1,2)[-1],expected,rtol=1e-5,atol=1e-7)

    def test_real_tc_noisy_backward_both_states(self):
        for two in (False,True):
            b=make_block(two)
            w=wrap(b,enable_spin_variation=True,enable_summing_current_noise=True,
                   enable_coupler_noise=True,spin_variation_seed=11,
                   summing_noise_seed=12,coupler_noise_seed=13)
            x=torch.full((2,2,2,2),.2,requires_grad=True)
            y=b(x);y.sum().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            for conv in (b.FFconv,b.FBconv):
                self.assertTrue(torch.isfinite(conv.parametrizations.weight.original.grad).all())

    def test_graph_regeneration_replays_noise_and_gradient(self):
        outputs=[]; gradients=[]
        for regenerate in (False,True):
            ref=torch.ones(2,requires_grad=True)*.01
            ctx=tape(ref)
            scale=torch.tensor(.3,requires_grad=True)
            def rhs(t,y):return scale*y+ctx.fb_current()
            rhs.tc_context=ctx
            out=odesolve(rhs,ref,dict(method='dopri5',t0=0.,t1=1.,h=.2,
                t_eval=[1.],eps=(.02**2+.03**2)**.5,noise_type='addi',regenerate_graph=regenerate))
            outputs.append(out.detach())
            gradients.append(torch.autograd.grad(out.sum(),scale)[0])
        torch.testing.assert_close(*outputs)
        torch.testing.assert_close(*gradients)

    def test_unrolled_maps_use_clean_programming(self):
        from validation import MVMConv, conv2d_to_matrix_fixed_padding
        b=make_block()
        conv=nn.Conv2d(1,1,3,padding=1,bias=False)
        conv.weight.data.fill_(.5)
        mat,_,_=conv2d_to_matrix_fixed_padding((1,3,3),conv.weight.detach(),padding=1)
        m=MVMConv(mat,dict(out_chan=1,ker_h=3,ker_w=3,stride=1,padding=1))
        m.clean_mat_values=mat.values().clone()
        expected=b._tc_nominal_sum(conv,torch.ones(2,1,3,3))
        mat.values().mul_(7)  # must not change diffusion derived from programming
        torch.testing.assert_close(b._tc_nominal_sum(m,torch.ones(2,1,3,3)),expected)


if __name__=='__main__':unittest.main()
