"""Empirical correction fusion + grouped convolution: forward/backward parity."""
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ode_pc import ODEBlockPC
from tc_empirical import prepare_empirical_curves
from tc_shared_correction import empirical_correction
from test_tc_multilevel_empirical import EmpiricalTCTests

BANK=Path(__file__).resolve().parents[1]/'hardware_data/mc_45_corners/pseudo_res_PVT_scanned_vprog'
ATOL,RTOL=1e-5,1e-4


def original_loop(owner,module,x,draw):
    """Literal pre-optimization empirical arithmetic, without optimized helpers."""
    from utils import interpolate_R_eff
    package=owner._tc_curve_package
    codes=owner._values_to_level_idx(module.weight.detach())
    output=None
    for code,row in enumerate(draw.indices.unbind()):
        grid=draw.bank['v_grid'][row].to(x)
        left=draw.bank['R_left'][row].to(x)
        slope=draw.bank['R_slope'][row].to(x)
        query=x.clamp(-owner.v_dd,owner.v_dd).clamp(grid[0],grid[-1])
        r=interpolate_R_eff(query,grid,package.programmed_resistances[code:code+1],
                            left[:,None],slope[:,None],0)
        corrected=x*package.programmed_resistances[code].to(x)/r
        weight=module.weight*(codes==code+1)
        if isinstance(module,nn.ConvTranspose2d):
            y=F.conv_transpose2d(corrected,weight,None,module.stride,module.padding,
                                 module.output_padding,module.groups,module.dilation)
        else:
            y=F.conv2d(corrected,weight,None,module.stride,module.padding,module.dilation,module.groups)
        output=y if output is None else output+y
    return output


def original_gaussian_grouped(owner,module,source,curves):
    """Frozen pre-change Gaussian grouped arithmetic; independent of git HEAD."""
    from tc_nonidealities import positive_resistance
    package=owner._tc_curve_package
    weight=module.weight
    codes=owner._values_to_level_idx(weight.detach())
    grid=package.v_grid.to(source)
    table=curves.to(source).T.contiguous()
    slope=(table[1:]-table[:-1])/(grid[1:]-grid[:-1])[:,None]
    query=source.clamp(-owner.v_dd,owner.v_dd).clamp(grid[0],grid[-1])
    count=package.means.shape[0]
    intervals=(torch.bucketize(query,grid)-1).clamp(0,grid.numel()-2)
    resistance=table[intervals]+slope[intervals]*(query-grid[intervals]).unsqueeze(-1)
    resistance=positive_resistance(resistance,package.floor_ohms)
    corrected=source.unsqueeze(-1)*package.programmed_resistances.to(source)/resistance
    packed_input=corrected.permute(0,4,1,2,3).reshape(
        source.shape[0],count*source.shape[1],*source.shape[2:])
    mask=codes.unsqueeze(0)==torch.arange(1,count+1,device=weight.device).reshape(-1,1,1,1,1)
    packed_weight=(weight.unsqueeze(0)*mask).reshape(count*weight.shape[0],*weight.shape[1:])
    result=F.conv2d(packed_input,packed_weight,None,module.stride,module.padding,
                   module.dilation,count*module.groups)
    return result.reshape(source.shape[0],count,module.out_channels,*result.shape[-2:]).sum(1)


class EmpiricalTrainingOptimizationTests(unittest.TestCase):
    def test_operator_forward_input_and_weight_gradients(self):
        torch.set_num_threads(1)
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():continue
            package=prepare_empirical_curves(str(BANK),levels=torch.arange(16)/15,
                R=13300.,R_max=200000.,dtype=torch.float32,device=device)
            owner=SimpleNamespace(q_hi=15,weight_scale=1.,v_dd=.1,_tc_curve_package=package)
            for name in ('_values_to_level_idx','_get_quant_magnitude_levels'):
                setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
            draw=package.sample_levels(generator=torch.Generator(device=device).manual_seed(10))
            for transpose in (False,True):
                for groups in (1,2):
                    torch.manual_seed(31)
                    cls=nn.ConvTranspose2d if transpose else nn.Conv2d
                    options=dict(stride=2,padding=1,groups=groups,bias=False)
                    if transpose:options['output_padding']=1
                    module=cls(4,6,3,**options).to(device)
                    with torch.no_grad():module.weight.copy_((module.weight*50).round().clamp(-15,15)/15)
                    data=torch.rand(2,4,5,5,device=device)*.3-.15
                    def snapshot(method,fused,old=False):
                        x=data.clone().requires_grad_()
                        owner._tc_conv_method=method;owner._tc_fused_correction=fused
                        y=(original_loop(owner,module,x,draw) if old else
                           ODEBlockPC._tc_dense_conv(owner,module,x,draw))
                        gradients=torch.autograd.grad(y.square().mean(),(x,module.weight))
                        return (y.detach(),*gradients)
                    expected=snapshot('loop',False,True)
                    for method,fused in (('loop',False),('loop',True),('grouped',False),('grouped',True)):
                        for actual,reference in zip(snapshot(method,fused),expected):
                            torch.testing.assert_close(actual,reference,atol=ATOL,rtol=RTOL)

    def test_native_boundaries_backward_and_cpu_fallback(self):
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():continue
            grid=torch.tensor([-.1,-.03,.02,.1],device=device)
            left=torch.tensor([20000.,21000.,19000.],device=device)
            slope=torch.tensor([1e4,-2e4,3e4],device=device)
            nominal=torch.tensor(20000.,device=device)
            values=torch.tensor([-.2,-.1,-.03,0.,.02,.1,.2],device=device)
            answers=[]
            for fused in (False,True):
                x=values.clone().requires_grad_()
                y=empirical_correction(x,grid,left,slope,nominal,.1,fused=fused)
                answers.append((y.detach(),torch.autograd.grad(y.sum(),x)[0]))
            for actual,reference in zip(answers[1],answers[0]):
                torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)

    def test_complete_pcn_forward_gradients_and_checkpoint_replay(self):
        helper=EmpiricalTCTests()
        helper.setUpClass()
        for two in (False,True):
            for checkpoint in (False,True):
                snapshots=[]
                for method,fused in (('loop',False),('loop',True),('grouped',False),('grouped',True)):
                    model=helper.wrapper(two,checkpoint)
                    model.ode_block._tc_conv_method=method
                    model.ode_block._tc_fused_correction=fused
                    torch.manual_seed(62)
                    x=(torch.rand(4,2,3,3)*.08).requires_grad_()
                    y=model(x)
                    gradients=torch.autograd.grad(y.square().mean(),(x,*model.parameters()),allow_unused=True)
                    snapshots.append((y.detach(),*[g for g in gradients if g is not None]))
                for snapshot in snapshots[1:]:
                    for actual,reference in zip(snapshot,snapshots[0]):
                        torch.testing.assert_close(actual,reference,atol=ATOL,rtol=RTOL)

    def test_gaussian_grouped_matches_literal_prechange_code(self):
        from tc_nonidealities import TCResistanceCurves
        levels=torch.arange(16,dtype=torch.float64)/15
        nominal=13300./levels[1:]
        grid=torch.tensor([-.1,0.,.1],dtype=torch.float64)
        package=TCResistanceCurves(grid,levels,nominal,nominal,
            nominal[:,None].expand(-1,3),torch.eye(3,dtype=torch.float64)*100.,'','')
        owner=SimpleNamespace(q_hi=15,weight_scale=1.,v_dd=.1,
                              _tc_curve_package=package,_tc_conv_method='grouped')
        for name in ('_values_to_level_idx','_get_quant_magnitude_levels'):
            setattr(owner,name,MethodType(getattr(ODEBlockPC,name),owner))
        module=nn.Conv2d(4,6,3,padding=1,groups=2,bias=False).double()
        curves=package.sample_levels(generator=torch.Generator().manual_seed(17))
        data=torch.rand(2,4,5,5,dtype=torch.float64)*.15-.075
        snapshots=[]
        for function in (original_gaussian_grouped,ODEBlockPC._tc_dense_conv):
            x=data.clone().requires_grad_()
            y=function(owner,module,x,curves)
            snapshots.append((y.detach(),*torch.autograd.grad(y.square().sum(),(x,module.weight))))
        torch.testing.assert_close(snapshots[1][0],snapshots[0][0],atol=0,rtol=0)
        # Consolidating the vectorized and per-curve reference changes only
        # FP64 backward accumulation order (observed max absolute 2.78e-17).
        for actual,expected in zip(snapshots[1][1:],snapshots[0][1:]):
            torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-12)

    def test_complete_cnn_forward_and_gradients(self):
        from baseline.cifar_resnet import WideResNetCIFAR
        from physical_feedforward import convert_wide_resnet_to_physical,iter_physical_wrappers
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():continue
            snapshots=[]
            for method,fused in (('loop',False),('loop',True),('grouped',False),('grouped',True)):
                torch.manual_seed(53)
                model=convert_wide_resnet_to_physical(
                    WideResNetCIFAR(depth=10,base_width=2,widen_factor=1,num_classes=3,
                        use_batchnorm=False,conv_bias=False,avgpool_main_downsample=True,
                        avgpool_downsample_shortcut=True),
                    qat=True,physical_level=2,R=13300.,C=49e-15,v_dd=.1,
                    tc_options=dict(tc_method='euler',tc_num_steps=3,tc_conv_method=method,
                        tc_distribution='empirical_with_replacement',tc_empirical_curve_bank=str(BANK))).to(device)
                wrappers=list(iter_physical_wrappers(model))
                package=wrappers[0].configure_nonlinear_R_training('/ignored',curve_seed=14)
                for w in wrappers:
                    w.install_nonlinear_R_training_package(package)
                    w.block._tc_fused_correction=fused
                x=(torch.rand(4,3,8,8,device=device)*.08).requires_grad_()
                y=model(x)
                gradients=torch.autograd.grad(y.square().mean(),(x,*model.parameters()),allow_unused=True)
                snapshots.append((y.detach(),*[g for g in gradients if g is not None]))
            for snapshot in snapshots[1:]:
                self.assertEqual(len(snapshot),len(snapshots[0]))
                for actual,expected in zip(snapshot,snapshots[0]):
                    torch.testing.assert_close(actual,expected,atol=ATOL,rtol=RTOL)


if __name__=='__main__':unittest.main()
