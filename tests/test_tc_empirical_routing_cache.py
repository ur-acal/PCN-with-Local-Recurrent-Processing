"""Solve-local empirical routing cache: lifetime, gradients and replay."""
import gc
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import torch
from torch import nn

from ode_pc import ODEBlockPC
from tc_empirical import EmpiricalLevels
import test_tc_multilevel_empirical as empirical_tests


class EmpiricalRoutingCacheTests(unittest.TestCase):
    def test_reuse_invalidation_and_no_autograd_graph_retention(self):
        draw=EmpiricalLevels({},torch.arange(15))
        classify=Mock(side_effect=lambda w:(w.abs()*15).round().long())
        weight=nn.Parameter(torch.tensor([.2,.4]))
        codes,mask=draw.weight_routing(weight,classify,(15,1.))
        self.assertIs(draw.weight_routing(weight,classify,(15,1.))[0],codes)
        self.assertEqual(classify.call_count,1)
        self.assertFalse(codes.requires_grad or mask.requires_grad)
        self.assertEqual(mask.dtype,torch.bool)
        self.assertEqual(codes.numel()*codes.element_size()+mask.numel()*mask.element_size(),23*weight.numel())
        with torch.no_grad():weight.add_(1/15)
        updated,_=draw.weight_routing(weight,classify,(15,1.))
        self.assertEqual(classify.call_count,2)
        self.assertFalse(torch.equal(updated,codes))
        draw.weight_routing(weight,classify,(15,.5))
        self.assertEqual(classify.call_count,3)
        other=weight.detach().clone().requires_grad_()*2
        reference=weakref.ref(other)
        draw.weight_routing(other,classify,(15,1.))
        del other
        gc.collect()
        self.assertIsNone(reference())
        fresh=EmpiricalLevels({},torch.arange(15))
        fresh.weight_routing(weight,classify,(15,1.))
        self.assertEqual(classify.call_count,5)

    def test_inference_and_dynamic_settings_fall_back(self):
        draw=EmpiricalLevels({},torch.arange(15))
        classify=Mock(side_effect=lambda w:(w.abs()*15).round().long())
        normal_weight=nn.Parameter(torch.tensor([.2,.4]))
        with torch.inference_mode():
            weight=torch.tensor([.2,.4])
            for _ in range(2):draw.weight_routing(weight,classify,(15,1.))
            for _ in range(2):draw.weight_routing(normal_weight,classify,(15,1.))
        weight=torch.tensor([.2,.4])
        for _ in range(2):draw.weight_routing(weight,classify,(15,torch.tensor(1.)))
        self.assertEqual(classify.call_count,6)
        self.assertFalse(draw._routing)
        _,mask=draw.weight_routing(normal_weight,classify,(15,1.))
        (normal_weight.unsqueeze(0)*mask).sum().backward()
        self.assertTrue(torch.isfinite(normal_weight.grad).all())

    def test_conv_and_transpose_outputs_and_gradients(self):
        helper=empirical_tests.EmpiricalTCTests()
        helper.setUpClass()
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():continue
            package=helper.package
            draw=EmpiricalLevels(package.sample_levels().bank,package.sample_levels().indices)
            owner=SimpleNamespace(q_hi=15,weight_scale=1.,v_dd=.1,
                _tc_curve_package=package,_tc_fused_correction=True)
            for name in ('_get_quant_magnitude_levels','_values_to_level_idx'):
                setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
            for transpose in (False,True):
                module=(nn.ConvTranspose2d if transpose else nn.Conv2d)(4,6,3,padding=1,
                    groups=2,bias=False).to(device)
                with torch.no_grad():module.weight.copy_((module.weight*60).round().clamp(-15,15)/15)
                data=torch.rand(2,4,4,4,device=device)*.2-.1
                for method in ('loop','grouped'):
                    owner._tc_conv_method=method
                    answers=[]
                    for cached in (False,True):
                        owner._tc_cache_level_routing=cached
                        x=data.clone().requires_grad_()
                        y=ODEBlockPC._tc_dense_conv(owner,module,x,draw)
                        answers.append((y.detach(),*torch.autograd.grad(y.square().sum(),(x,module.weight))))
                    for actual,expected in zip(answers[1],answers[0]):
                        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)

    def test_complete_pcn_both_states_checkpoint_replay(self):
        helper=empirical_tests.EmpiricalTCTests()
        helper.setUpClass()
        for two in (False,True):
            for checkpoint in (False,True):
                answers=[];counts=[]
                for cached in (False,True):
                    model=helper.wrapper(two,checkpoint)
                    block=model.ode_block
                    block._tc_conv_method='grouped'
                    block._tc_cache_level_routing=cached
                    classify=Mock(wraps=block._values_to_level_idx)
                    block._values_to_level_idx=classify
                    torch.manual_seed(83)
                    x=(torch.rand(4,2,3,3)*.08).requires_grad_()
                    y=model(x)
                    gradients=torch.autograd.grad(y.square().mean(),(x,*model.parameters()),allow_unused=True)
                    answers.append((y.detach(),*[g for g in gradients if g is not None]))
                    counts.append(classify.call_count)
                self.assertEqual(counts[1],2) # one independent FF/FB routing each
                self.assertGreater(counts[0],counts[1])
                for actual,expected in zip(answers[1],answers[0]):
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)


if __name__=='__main__':unittest.main()
