"""Histogram grouping, per-convolution draws, gradients and checkpoint replay."""
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ode_pc import ODEBlockPC
from tc_curve_groups import sample_groups, group_convolution, validate_top_n
from tc_nonidealities import TCResistanceCurves


def package(device='cpu', empirical=False):
    grid = torch.linspace(-.1,.1,5,device=device)
    nominal = 150000/torch.arange(1,16,device=device)
    means = nominal[:,None]*(1+.2*grid[None,:])
    p = TCResistanceCurves(grid,torch.arange(16,device=device)/15,nominal,nominal,
        means,torch.eye(5,device=device)*3.,'synthetic','synthetic')
    if empirical:
        from dataclasses import replace
        rows = means.repeat_interleave(4,0)
        bank = dict(v_grid=grid.expand(60,-1).contiguous(),R_left=rows[:,:-1].contiguous(),
            R_slope=((rows[:,1:]-rows[:,:-1])/(grid[1:]-grid[:-1])).contiguous(),
            lengths=torch.full((60,),5,device=device),draws_per_code=4,
            corner_indices={'test':torch.arange(60,device=device).reshape(15,4)})
        p = replace(p,empirical_bank=bank)
    return p


class CurveGroupsTests(unittest.TestCase):
    def test_validation(self):
        for n in (-1,16,True,.5):
            with self.assertRaises(ValueError):validate_top_n(n,15)
        self.assertEqual(validate_top_n(15,15),15)
        self.assertEqual(validate_top_n(31,31),31)

    def test_full_level_endpoint(self):
        from tc_empirical import EmpiricalLevels
        p=package(empirical=True)
        conv=nn.Conv2d(1,15,1,bias=False)
        with torch.no_grad():conv.weight.copy_(torch.arange(1,16).reshape_as(conv.weight)/15)
        owner=SimpleNamespace(v_dd=.1,q_hi=15,weight_scale=1.,_tc_curve_package=p,
                              _tc_conv_method='grouped')
        for name in ('_get_quant_magnitude_levels','_values_to_level_idx'):
            setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
        codes=owner._values_to_level_idx(conv.weight.detach())
        draw=sample_groups(p,codes,14,generator=torch.Generator().manual_seed(23))
        reordered=torch.empty_like(draw.corrections.indices)
        reordered[draw.representative_codes-1]=draw.corrections.indices
        full=EmpiricalLevels(draw.corrections.bank,reordered)
        x=torch.randn(2,1,4,4)*.03
        torch.testing.assert_close(group_convolution(owner,conv,x,draw),
            ODEBlockPC._tc_dense_conv(owner,conv,x,full),atol=1e-6,rtol=1e-5)

    def test_gaussian_interpolation_floor_is_preserved(self):
        from tc_shared_correction import batched_empirical_correction, empirical_correction
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            p=package(device)
            draw=sample_groups(p,torch.arange(16,device=device).reshape(1,1,4,4),3)
            self.assertEqual(draw.floor,p.floor_ohms)
            # Exercise the interpolation guard itself, including a zero raw lookup.
            draw.corrections.bank['R_left'].zero_()
            draw.corrections.bank['R_slope'].zero_()
            x=torch.randn(2,1,4,4,device=device,requires_grad=True)*.03
            constants=draw.corrections.prepare(x,draw.nominal)
            ref=torch.cat([empirical_correction(x,*values,.1,fused=False,floor=100.)
                           for values in constants],dim=1)
            fused=batched_empirical_correction(x,draw.corrections,draw.nominal,.1,floor=100.)
            if fused is not None:
                torch.testing.assert_close(fused,ref,atol=1e-5,rtol=1e-4)
                a=torch.autograd.grad(ref.sum(),x,retain_graph=True)[0]
                b=torch.autograd.grad(fused.sum(),x)[0]
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
            self.assertTrue(torch.isfinite(ref).all())

    def test_histogram_masks_and_corner(self):
        codes=torch.tensor([0,1,1,2,2,3,4])
        for empirical in (False,True):
            p=package(empirical=empirical)
            for n in (0,1,2,3,14,15):
                draw=sample_groups(p,codes,n,generator=torch.Generator().manual_seed(5),
                                   factor='test' if empirical else None)
                self.assertEqual(draw.top_codes.tolist(),[1,2,3,4][:min(n,4)])
                if n:
                    self.assertTrue(torch.equal(draw.mask.sum(0),torch.ones_like(codes)))
                    self.assertTrue(draw.mask[-1,0])
                else:
                    self.assertIsNone(draw.mask)
                if draw.representative_codes.numel()>draw.top_codes.numel():
                    self.assertNotIn(int(draw.representative_codes[-1]),draw.top_codes.tolist())
                if empirical:
                    self.assertTrue(torch.equal(draw.corrections.indices//4+1,draw.representative_codes))
            self.assertIsNone(sample_groups(p,torch.zeros(7,dtype=torch.long),3))

    def test_zero_avoids_group_ranking_and_preserves_sampling_rng(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            codes=torch.tensor([0,1,1,2,2,3,4],device=device)
            for empirical in (False,True):
                for sampling in ('histogram','uniform'):
                    p=package(device,empirical=empirical)
                    generator=torch.Generator(device=device).manual_seed(91)
                    reference=torch.Generator(device=device).manual_seed(91)
                    hist=torch.bincount(codes,minlength=16)[1:]
                    probabilities=hist.float() if sampling=='histogram' else torch.ones(15,device=device)
                    selected=torch.multinomial(probabilities,1,generator=reference)
                    if empirical:
                        index=selected*4+torch.randint(4,(1,),device=device,generator=reference)
                    else:
                        resistance=p.sample(selected+1,generator=reference)
                    with patch('torch.argsort',side_effect=AssertionError('N=0 must not rank')):
                        draw=sample_groups(p,codes,0,generator=generator,sampling=sampling)
                    self.assertIsNone(draw.mask)
                    self.assertEqual(draw.representative_codes.tolist(),(selected+1).tolist())
                    if empirical:torch.testing.assert_close(draw.corrections.indices,index)
                    else:torch.testing.assert_close(draw.corrections.bank['R_left'],resistance[:,:-1])
                    self.assertTrue(torch.equal(generator.get_state(),reference.get_state()))
                    before=generator.get_state()
                    self.assertIsNone(sample_groups(p,torch.zeros_like(codes),0,generator=generator))
                    self.assertTrue(torch.equal(before,generator.get_state()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cpu_package_with_cuda_weights_outputs_and_gradients(self):
        for empirical in (False,True):
            p=package(empirical=empirical)
            for n in (0,1,3,14):
                codes=torch.tensor([0,1,1,2],device='cuda').reshape(2,2,1,1)
                cpu=sample_groups(p,codes.cpu(),n,generator=torch.Generator().manual_seed(71))
                gpu=sample_groups(p,codes,n,generator=torch.Generator().manual_seed(71))
                torch.testing.assert_close(cpu.corrections.indices,gpu.corrections.indices)
                if n:self.assertEqual(gpu.mask.device.type,'cuda')
                else:self.assertIsNone(gpu.mask)
                module=nn.Conv2d(2,2,1,bias=False).cuda()
                with torch.no_grad():module.weight.copy_(codes/15)
                if cpu.mask is not None:cpu.mask=cpu.mask.cuda()
                owner=SimpleNamespace(v_dd=.1)
                x=(torch.rand(2,2,3,3,device='cuda')*.05).requires_grad_()
                a=group_convolution(owner,module,x,cpu)
                b=group_convolution(owner,module,x,gpu)
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                for ga,gb in zip(torch.autograd.grad(a.sum(),(x,module.weight)),
                                 torch.autograd.grad(b.sum(),(x,module.weight))):
                    torch.testing.assert_close(ga,gb,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cpu_built_pcn_moved_to_cuda_forward_and_backward(self):
        from test_tc_multilevel_empirical import EmpiricalTCTests
        fixture=EmpiricalTCTests()
        for two in (False,True):
            for n in (0,1,3):
                wrapper=fixture.wrapper(two,True).cuda()
                # Existing fixture time metadata is a plain attribute, too.
                wrapper.orig_integration_time=wrapper.orig_integration_time.cuda()
                block=wrapper.ode_block
                block._tc_conv_method='shared'
                block._tc_curve_top_n=n
                x=(torch.rand(2,2,2,2,device='cuda')*.05).requires_grad_()
                y=wrapper(x)
                gradients=torch.autograd.grad(y.square().sum(),(x,*wrapper.parameters()),allow_unused=True)
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))

    def test_operator_and_checkpoint(self):
        torch.set_num_threads(1)
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for empirical in (False,True):
                p=package(device,empirical)
                for transpose in (False,True):
                    conv=(nn.ConvTranspose2d if transpose else nn.Conv2d)(2,2,3,padding=1,bias=False).to(device)
                    with torch.no_grad():conv.weight.copy_(torch.arange(conv.weight.numel(),device=device).reshape_as(conv.weight).remainder(16)/15)
                    owner=SimpleNamespace(v_dd=.1,q_hi=15,weight_scale=1.,_tc_curve_package=p)
                    for name in ('_get_quant_magnitude_levels','_values_to_level_idx'):
                        setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
                    for n in (0,1,2,3,14):
                        codes=owner._values_to_level_idx(conv.weight.detach())
                        draw=sample_groups(p,codes,n,generator=torch.Generator(device=device).manual_seed(8))
                        x=(torch.randn(2,2,4,4,device=device)*.03).requires_grad_()
                        reference=None
                        for fused,recompute in ((False,False),(True,False),(True,True)):
                            owner._tc_fused_correction=fused
                            conv.zero_grad();x.grad=None
                            fun=lambda v:group_convolution(owner,conv,v,draw)
                            out=checkpoint(fun,x,use_reentrant=False) if recompute else fun(x)
                            out.square().sum().backward()
                            current=(out.detach().clone(),x.grad.clone(),conv.weight.grad.clone())
                            if reference is None:reference=current
                            for a,b in zip(current,reference):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
                            # Zero-code weights retain the shared path's nonzero STE gradient.
                            self.assertGreater(conv.weight.grad[codes==0].abs().sum().item(),0)

    def test_complete_ode_checkpoint_and_sampling_lifetime(self):
        from test_tc_multilevel_empirical import EmpiricalTCTests
        fixture=EmpiricalTCTests()
        for two in (False,True):
            for n in (0,1,2,3):
                a,b=fixture.wrapper(two),fixture.wrapper(two,True)
                for wrapped in (a,b):
                    wrapped.ode_block._tc_conv_method='shared'
                    wrapped.ode_block._tc_curve_top_n=n
                xa=(torch.rand(2,2,2,2)*.1).requires_grad_()
                xb=xa.detach().clone().requires_grad_()
                ya,yb=a(xa),b(xb)
                ya.square().sum().backward();yb.square().sum().backward()
                torch.testing.assert_close(ya,yb,rtol=0,atol=0)
                torch.testing.assert_close(xa.grad,xb.grad,rtol=1e-4,atol=1e-5)
                for pa,pb in zip(a.parameters(),b.parameters()):
                    if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-5)
                block=a.ode_block
                training_draws=block._tc_curve_samples
                block.eval()
                first=block._tc_curves_for_solve()
                if n:
                    self.assertIsNot(first,training_draws)
                    self.assertTrue(all(not d.training_draw for d in first.values() if d is not None))
                self.assertIs(first,block._tc_curves_for_solve())
                self.assertIsNot(first['FFconv'],first['FBconv'])
                block.reset_tc_curves()
                self.assertIsNot(first,block._tc_curves_for_solve())

    def test_cnn_and_head_forwarding(self):
        from physical_feedforward_tc import TCPhysicalBasicBlock
        for n in (0,1,2,3):
            block=TCPhysicalBasicBlock(nn.Conv2d(2,2,1,bias=False),one_shot_conv=True,
                tc_curve_top_n=n)
            self.assertEqual(block.tc_curve_top_n,n)
            # Exercise the same dense implementation called by CNNs/analog heads.
            block._tc_resistance_package=package(empirical=True)
            with torch.no_grad():block.conv1.weight.copy_(torch.tensor([0.,1.,2.,3.]).reshape_as(block.conv1.weight)/15)
            block.conv1.weight.requires_grad_()
            x=torch.rand(2,2,4,4,requires_grad=True)*.1
            out=block._current(block.conv1,x)
            out.sum().backward()
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(block.conv1.weight.grad).all())

    def test_analog_head_inherits_top_n(self):
        from test_tc_multilevel_empirical import EmpiricalTCTests
        from final_linear import AnalogLinear
        fixture=EmpiricalTCTests()
        for two in (False,True):
            template=fixture.wrapper(two).ode_block
            template._tc_curve_top_n=2
            template._tc_conv_method='shared'
            head=AnalogLinear(2,3,bias=False)
            head.configure(physical=True,q=.1,v_dd=.1,template=template,
                family='tc',R=13300.,C=49e-15,timing='derived',base_time=1e-9)
            head(torch.rand(4,2)*.01).sum().backward()
            self.assertEqual(head._circuit.tc_curve_top_n,2)
            self.assertTrue(torch.isfinite(head.weight.grad).all())


if __name__=='__main__':unittest.main()
