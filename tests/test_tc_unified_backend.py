"""Independent original-OFF arithmetic versus the consolidated dense backend."""
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from ode_pc import ODEBlockPC
from tc_curve_groups import CurveGroups, sample_groups
from tc_nonidealities import TCSharedCurve
from test_tc_curve_groups import package


def original_off(owner, module, x, draw):
    """Frozen reference expression/serial sum; no optimized correction helpers."""
    p=owner._tc_curve_package
    floor=p.floor_ohms
    if isinstance(draw,TCSharedCurve):
        g=p.v_grid.to(x);r=draw.resistance.to(x)
        rows=[(g,r[:-1],r.diff()/g.diff(),draw.nominal_R.to(x))]
        mask=None
    else:
        if isinstance(draw,CurveGroups):
            bank,indices=draw.corrections.bank,draw.corrections.indices
            nominal,mask,floor=draw.nominal,draw.mask,draw.floor
        else:
            bank,indices,nominal=draw.bank,draw.indices,p.programmed_resistances
            codes=owner._values_to_level_idx(module.weight.detach())
            mask=codes.unsqueeze(0)==torch.arange(1,nominal.numel()+1,device=x.device).reshape(-1,1,1,1,1)
            if p.empirical_bank is not None:floor=0.
        rows=[]
        for code,index in enumerate(indices):
            length=int(bank['lengths'][index])
            rows.append((bank['v_grid'][index,:length].to(x),bank['R_left'][index,:length-1].to(x),
                         bank['R_slope'][index,:length-1].to(x),nominal[code].to(x)))
    out=None
    for index,(g,left,slope,rnom) in enumerate(rows):
        q=x.clamp(-owner.v_dd,owner.v_dd).clamp(g[0],g[-1])
        interval=(torch.bucketize(q.contiguous(),g.contiguous())-1).clamp(0,g.numel()-2)
        resistance=left[interval]+slope[interval]*(q-g[interval])
        if floor:resistance=resistance.clamp_min(floor)
        corrected=x*rnom/resistance
        weight=module.weight if mask is None else module.weight*mask[index]
        if isinstance(module,nn.ConvTranspose2d):
            y=F.conv_transpose2d(corrected,weight,None,module.stride,module.padding,
                                module.output_padding,module.groups,module.dilation)
        else:y=F.conv2d(corrected,weight,None,module.stride,module.padding,module.dilation,module.groups)
        out=y if out is None else out+y
    return out


class UnifiedBackendTests(unittest.TestCase):
    def test_single_full_level_still_masks_zero_code_gradients(self):
        from dataclasses import replace
        from tc_empirical import EmpiricalLevels
        from tc_curve_groups import group_convolution
        p=package(empirical=True)
        p=replace(p,levels=p.levels[:2],programmed_resistances=p.programmed_resistances[:1],
                  means=p.means[:1],mean_column_resistances=p.mean_column_resistances[:1])
        draw=EmpiricalLevels(p.empirical_bank,torch.tensor([0]))
        module=nn.Conv2d(1,2,1,bias=False)
        with torch.no_grad():module.weight.copy_(torch.tensor([0.,1.]).reshape_as(module.weight))
        owner=SimpleNamespace(v_dd=.1,q_hi=1,weight_scale=1.,_tc_curve_package=p)
        for name in ('_get_quant_magnitude_levels','_values_to_level_idx'):
            setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
        x=torch.ones(1,1,2,2)*.01
        y=group_convolution(owner,module,x,draw)
        gradient=torch.autograd.grad(y.sum(),module.weight)[0]
        self.assertEqual(float(gradient[0]),0.)
        self.assertGreater(float(gradient[1]),0.)

    def test_identical_prepared_curve_has_identical_single_group_backend(self):
        from tc_empirical import EmpiricalLevels
        from tc_curve_groups import group_convolution
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            p=package(device)
            module=nn.Conv2d(2,2,1,bias=False).to(device)
            gaussian=p.sample_shared(torch.ones_like(module.weight,dtype=torch.long))
            native=EmpiricalLevels.from_gaussian(p.v_grid,gaussian.resistance[None])
            empirical=CurveGroups(native,gaussian.nominal_R.reshape(1),None,
                torch.ones(15,device=device),torch.empty(0,device=device),gaussian.code.reshape(1),p.floor_ohms)
            owner=SimpleNamespace(v_dd=.1,_tc_curve_package=p)
            data=torch.rand(2,2,3,3,device=device)*.2-.1
            answers=[]
            for draw in (gaussian,empirical):
                x=data.clone().requires_grad_()
                y=group_convolution(owner,module,x,draw)
                answers.append((y,*torch.autograd.grad(y.square().sum(),(x,module.weight))))
            for actual,expected in zip(answers[1],answers[0]):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_single_group_never_uses_packed_kernel_and_supports_inference_tensors(self):
        from tc_curve_groups import group_convolution
        with torch.inference_mode():
            for empirical in (False,True):
                p=package('cpu',empirical)
                module=nn.Conv2d(2,2,1,bias=False)
                codes=torch.ones_like(module.weight,dtype=torch.long)
                draw=(sample_groups(p,codes,0) if empirical else p.sample_shared(codes))
                owner=SimpleNamespace(v_dd=.1,_tc_curve_package=p)
                x=torch.rand(2,2,3,3)*.2-.1
                with patch('tc_shared_correction.batched_empirical_correction',side_effect=AssertionError('one group must use scalar correction')):
                    actual=group_convolution(owner,module,x,draw)
                torch.testing.assert_close(actual,original_off(owner,module,x,draw),atol=0,rtol=0)

    def test_all_n_against_original_off_outputs_and_gradients(self):
        torch.set_num_threads(1)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for empirical in (False,True):
                p=package(device,empirical)
                for transpose in (False,True):
                    module=(nn.ConvTranspose2d if transpose else nn.Conv2d)(2,2,3,padding=1,bias=False).to(device)
                    with torch.no_grad():module.weight.copy_(torch.arange(module.weight.numel(),device=device).reshape_as(module.weight).remainder(16)/15)
                    owner=SimpleNamespace(q_hi=15,weight_scale=1.,v_dd=.1,_tc_curve_package=p)
                    for method in ('_get_quant_magnitude_levels','_values_to_level_idx'):
                        setattr(owner,method,MethodType(getattr(ODEBlockPC,method),owner))
                    codes=owner._values_to_level_idx(module.weight.detach())
                    data=torch.rand(2,2,3,3,device=device)*.2-.1
                    for n in range(16):
                        generator=torch.Generator(device=device).manual_seed(37)
                        draw=(p.sample_shared(codes,generator=generator) if n==0 and not empirical else
                              sample_groups(p,codes,n,generator=generator))
                        answers=[]
                        for kind in ('original_off','off','on'):
                            owner._tc_fused_correction=kind=='on'
                            owner._tc_grouped_convolution=kind=='on'
                            x=data.clone().requires_grad_()
                            y=(original_off(owner,module,x,draw) if kind=='original_off' else
                               ODEBlockPC._tc_dense_conv(owner,module,x,draw))
                            answers.append((y.detach(),*torch.autograd.grad(y.square().sum(),(x,module.weight))))
                        for actual,expected in zip(answers[1],answers[0]):
                            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                        for actual,expected in zip(answers[2],answers[1]):
                            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)


if __name__=='__main__':unittest.main()
