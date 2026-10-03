"""New-head coordinate audit: unitless checkpoint versus physical FT initialization.

The backbone is deliberately small and unquantized to isolate the new head.
No noise, measured pooling, nonlinear R, spin variation or DTC imperfections.
Reference head is independently written gain*F.linear, never Analog/DigitalLinear.
All four physical-head settings compare to the SAME unchanged unitless reference.
"""
import argparse
import copy
import csv
import itertools
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from final_linear import select_model_head,configure_feedforward_head
from pc_model import PCNetNoBatchNorm,logits_for_loss
from pc_conv import PCConvReLU6
from ode_pc import (make_ode_block,ToggleODEXInitFFFB,ODEXInitFFFB,S2NoisyIYAsXZAs0,
                    ToggleWrapper1State,ODEWrapper1State,ODEWrapper2State)
from ode_pc import ToggleQATWrapper1State,QATWrapper1State,QATWrapper2State
from inference_utils import load_and_prepare_model
from physical_feedforward import convert_wide_resnet_to_physical,iter_physical_blocks
from measured_activation import PiecewiseLinearActivation
from distillation.srrl import SRRLLoss


def activation(q=None):
    a=PiecewiseLinearActivation(ROOT/'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv',
        v_dd=.5,corner='MC18',normalize_positive_endpoint=False,fuse_measured_activation=False).cuda()
    if q is not None: a.set_coordinate_pullback_scale(q)
    return a


def pair(family,kind,bias,measured,timing,factor,quant,clamp,path,stress=False,backbone_qat=False):
    torch.manual_seed(117)
    q=.1
    tc=family.startswith('tc')
    R,C=(1e4,49e-15) if tc else (50e3,500e-15)
    gain=10e-9/(R*C) if kind=='analog' and timing=='fixed' else 1.
    cfg=dict(final_head_quantize=quant,final_head_clamp=clamp)
    if stress and kind=='digital':
        cfg['final_accumulator_bits']=4
    if family.endswith('ff'):
        from baseline.cifar_resnet import WideResNetCIFAR
        base=WideResNetCIFAR(depth=10,widen_factor=1,num_classes=3,in_chans=3,
            use_batchnorm=False,avgpool_downsample_shortcut=True)
        if not bias: base.fc.bias=None
        name='fc'
    else:
        base=PCNetNoBatchNorm(inp_channels=[3,4],out_channels=[4,4],max_pool=[False,False],
            num_classes=3,pc_conv_layer=PCConvReLU6,kernel_size=1,padding=0,
            first_bn=False,avg_pooling=True,dropout=0.,tie_weights=False,tie_bp=False,
            bypass=False,linear_bias=bias,measured_activation_scope='pc_only')
        name='linear'
    # A controlled small checkpoint, not a claim about trained accuracy.
    with torch.no_grad():
        for p in base.parameters(): p.mul_(.5)
        getattr(base,name).weight.div_(gain)
        if bias: getattr(base,name).bias.div_(gain)
        if stress and kind=='analog':
            getattr(base,name).weight.mul_(200.)
            if bias: getattr(base,name).bias.mul_(200.)
    select_model_head(base,kind,cfg)
    payload=dict(net=base.state_dict(),final_head=dict(type=kind,config=cfg))
    if hasattr(base,'init_args'):
        payload.update(init_args=base.init_args,net_type='PCNetNoBatchNorm')
    torch.save(payload,path)
    reference=copy.deepcopy(base).cuda()
    if family.endswith('ff'):
        physical=copy.deepcopy(base)
        physical.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['net'])
        for model,isphysical in ((reference,False),(physical,True)):
            convert_wide_resnet_to_physical(model,physical=isphysical,qat=backbone_qat and isphysical,physical_level=2,
                R=R,C=C,v_dd=.5,one_over_q=5,w_bits=5,weight_quant_factor_bits=None,
                toggle_timing_mode=timing,toggle_y_time=10e-9,z_over_y_time=1,
                tc_options=dict(one_shot_conv=True) if tc else None)
        physical.cuda()
        args=SimpleNamespace(v_dd=.5,one_over_q=5,R=R,C=C,w_bits=5,
            weight_quant_factor_bits=factor,tc_feedforward=tc,toggle_timing_mode=timing,
            toggle_y_time=10e-9,scale_train_recipe=1 if timing=='fixed' else 0)
        configure_feedforward_head(physical,args)
        if measured:
            reference.relu=activation(q)
            physical.relu=activation()
        pool_name='global_pool'
        blocks=list(iter_physical_blocks(physical))
    else:
        block=ToggleODEXInitFFFB if not tc else ODEXInitFFFB if family=='tc1' else S2NoisyIYAsXZAs0
        ode=dict(ode_block=block,method='rk4' if tc else 'dopri5',t_end=.35,
            n_steps=32 if tc else 5,tol=1e-10,
            toggle_timing_mode=timing,toggle_y_time=10e-9,z_over_y_time=1,
            toggle_timing_R=R,toggle_timing_C=C,odexinit_scaling_mode='direct')
        reference.device=torch.device('cuda')
        make_ode_block(reference,**ode)
        for unitless_block in reference.PcConvs:
            unitless_block.eps_scale=None
            unitless_block.offset_eps=0.
        wrappers={}
        wrapper_class = (ToggleQATWrapper1State if not tc else QATWrapper1State if family=='tc1' else QATWrapper2State) if backbone_qat else (
            ToggleWrapper1State if not tc else ODEWrapper1State if family=='tc1' else ODEWrapper2State)
        with torch.no_grad():
            physical=load_and_prepare_model(str(path),'cuda',model_struct=PCNetNoBatchNorm,
            pc_conv_layer=PCConvReLU6,fuse_bn=False,noise_level=0,wrappers=wrappers,
            ode_params=ode,ode_wrapper_params=dict(
                ode_wrapper=wrapper_class,
                quantize=False,R=R,R_max=R*15,C=C,v_dd=.5,one_over_q=5,w_bits=5,
                weight_quant_factor_bits=factor,thermal_noise=False,nonlinear_R=False,
                tc_nonidealities=tc,tc_conv_method='shared'))
        if measured:
            reference.set_non_pc_activation('final_activation',activation(q))
            physical.set_non_pc_activation('final_activation',activation())
        pool_name='global_avg_pool2d'
        blocks=list(physical.PcConvs)
        if not clamp:
            for wrapper in wrappers['wrappers']:
                wrapper.proj_fn=nn.Identity()
                for key in ('orig_option_aca','orig_option_init','orig_option_patch'):
                    if hasattr(wrapper,key):
                        getattr(wrapper,key)['proj_fn']=nn.Identity()
    if not clamp:
        for block in blocks:
            block.project_state=lambda x:x
            for key in ('option_aca','option_init','option_patch'):
                if hasattr(block,key): getattr(block,key)['proj_fn']=nn.Identity()
    # Independent unitless classifier; retain identical named trainable parameters.
    def independent_linear(self,x):
        return gain*F.linear(x,self.weight,self.bias)
    head=getattr(reference,name)
    head.forward=MethodType(independent_linear,head)
    head.physical=False
    return reference,physical,name,pool_name,q,gain


def run(config,path,dtype=torch.float32,stress=False,backbone_qat=False):
    reference,physical,name,pool_name,q,gain=pair(*config,path,stress=stress,backbone_qat=backbone_qat)
    reference.to(dtype=dtype);physical.to(dtype=dtype)
    family,kind,bias,measured,timing,factor,quant,clamp=config
    channels=getattr(reference,name).in_features
    torch.manual_seed(51)
    aux0=SRRLLoss(channels,4).cuda().to(dtype=dtype)
    teacher=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,3)).cuda().to(dtype=dtype).requires_grad_(False)
    initial=[copy.deepcopy(m.state_dict()) for m in (reference,physical)]
    auxiliary=copy.deepcopy(aux0.state_dict())
    rows=[]
    for srrl in (False,True):
        torch.manual_seed(98)
        x=torch.rand(2,3,8,8,device='cuda').to(dtype=dtype)
        labels=torch.tensor([0,2],device='cuda')
        target=torch.rand(2,4,8 if not family.endswith('ff') else 2,8 if not family.endswith('ff') else 2,device='cuda').to(dtype=dtype)
        teacher_logits=teacher(target)
        records=[]
        for index,model in enumerate((reference,physical)):
            model.load_state_dict(initial[index]); model.train(); model.zero_grad(set_to_none=True)
            aux=copy.deepcopy(aux0); aux.load_state_dict(auxiliary)
            pooled=[]
            hook=getattr(model,pool_name).register_forward_hook(lambda m,i,o:pooled.append(o))
            torch.manual_seed(99)
            features,raw=model(x,is_feat=True)
            logits=raw if index==0 else logits_for_loss(raw,model)
            ce=F.cross_entropy(logits,labels)
            kd=aux(features[-1],target,teacher_logits,teacher) if srrl else logits.new_tensor(0.)
            loss=ce+.3*kd
            loss.backward(); hook.remove()
            # Compare the same latent trainable weights before/after production
            # QAT parametrization, not the transient quantized tensors.
            params={n.replace('.parametrizations.weight.original','.weight'):p
                    for n,p in model.named_parameters()}
            params.update({'srrl.'+n:p for n,p in aux.named_parameters()})
            gradients={n:p.grad.detach().clone() for n,p in params.items() if p.grad is not None}
            # Actual production optimizer grouping (fixed-time head W/b scaling).
            from trainer_timm import TrainerCiFarTimmStyle
            trainer=object.__new__(TrainerCiFarTimmStyle)
            trainer.model=model; trainer.scale_train_recipe=timing=='fixed'
            trainer.ff_train_scale=trainer.fb_train_scale=1.
            model.linear_train_scale=gain
            trainer.bias_lr_multiplier=1.; trainer.bias_weight_decay=None
            groups=trainer._composed_optimizer_parameters(.001,.001)
            groups.append(dict(params=aux.parameters(),lr=.001,weight_decay=0.))
            optimizer=torch.optim.SGD(groups,momentum=.9)
            optimizer.step()
            records.append(dict(feature=features[-1].detach(),gap=pooled[0].detach()/(q if index else 1),
                logits=logits.detach(),ce=ce.detach(),srrl=kd.detach(),loss=loss.detach(),
                gradients=gradients,updated={n:p.detach().clone() for n,p in params.items()}))
        a,b=records
        row=dict(zip(('family','head','bias','measured','timing','factor_bits','quant','clamp'),config),use_srrl=srrl)
        passed=True
        row['dtype']=str(dtype)
        row['stress']=stress
        row['backbone_qat']=backbone_qat
        for key in ('feature','gap','logits','ce','srrl','loss'):
            row[key+'_max_abs']=float((a[key]-b[key]).abs().max())
            passed &= bool(torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all())
            if not quant and not clamp:
                passed &= torch.allclose(a[key],b[key],rtol=2e-4,atol=2e-6)
        for key in ('gradients','updated'):
            assert a[key].keys()==b[key].keys(),(a[key].keys()-b[key].keys(),b[key].keys()-a[key].keys())
            row[key+'_max_abs']=max(float((a[key][n]-b[key][n]).abs().max()) for n in a[key])
            passed &= all(bool(torch.isfinite(a[key][n]).all() and torch.isfinite(b[key][n]).all()) for n in a[key])
            if not quant and not clamp:
                # Same float32 gradient tolerance as the existing head-pipeline
                # regression. SRRL BN amplifies sub-ULP feature differences;
                # --numerical-recheck independently checks them in float64.
                passed &= all(torch.allclose(a[key][n],b[key][n],rtol=5e-4,atol=5e-6) for n in a[key])
        row['required_match']=not quant and not clamp
        row['passed']=bool(passed)
        row['unitless_logits']=a['logits'].tolist();row['physical_restored_logits']=b['logits'].tolist()
        rows.append(row)
    return rows


def write_summary(output,rows):
    """Keep the comparison definition beside the numerical evidence."""
    valid=[r for r in rows if 'family' in r]
    fields=['family','head','bias','measured','timing','factor_bits','quant','clamp',
            'use_srrl','feature_max_abs','gap_max_abs','logits_max_abs','ce_max_abs',
            'srrl_max_abs','loss_max_abs','gradients_max_abs','updated_max_abs',
            'required_match','passed']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(valid)
    required=[r for r in valid if r['required_match']]
    text=[
        'NEW-HEAD PRETRAIN -> PHYSICAL FT COORDINATE CHECK',
        'A: unitless model with independent ideal gain*linear(W,b).',
        'B: same checkpoint weights mapped through the production physical model and new head.',
        'B features, GAP and logits are converted to model coordinates before comparing to A.',
        'A remains unchanged in all four quantization/clamping cases.',
        'CE-only and CE+actual SRRL: compare losses, latent parameter gradients and one SGD update.',
        'Synthetic small checkpoints and batch=2; not a convergence/accuracy experiment.',
        'TC uses matched RK4 grids, not production adaptive-grid equivalence.',
        'Quantization/clamping ON rows are deviation measurements, NOT equality claims.',
        'Clamping OFF leaves intrinsic ReLU transfer and finite ADC/bias code limits unchanged.',
        f'Rows: {len(rows)}; required-equivalence checks: {len(required)}; failures: {sum(not r["passed"] for r in rows)}.',
        'For full-backbone QAT, see the separate companion run; primary run isolates head quantization.',
        '',
        'Maximum absolute differences (over families/configurations; units are model coordinates):']
    for head,quant,clamp in itertools.product(('analog','digital'),(False,True),(False,True)):
        group=[r for r in valid if (r['head'],r['quant'],r['clamp'])==(head,quant,clamp)]
        if group:
            metrics=', '.join(f'{key}={max(r[key+"_max_abs"] for r in group):.8g}'
                              for key in ('feature','gap','logits','loss','gradients','updated'))
            text.append(f'{head}, quant={quant}, clamp={clamp}: {metrics}')
    (output/'summary.txt').write_text('\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--numerical-recheck',action='store_true')
    p.add_argument('--stress',action='store_true')
    p.add_argument('--backbone-qat',action='store_true',
                   help='Additional quantized rows with production backbone QAT as well as head QAT')
    p.add_argument('--dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if args.summarize_only:
        write_summary(args.output,json.loads((args.output/'results.json').read_text()))
        return
    assert torch.cuda.is_available(),'GPU required'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);logging.disable(logging.WARNING)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metadata.json').write_text(json.dumps(dict(
        python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        dtype=args.dtype,batch_size=2,tf32=False,
        reference='Unitless production backbone, independent ideal gain*F.linear head; identical checkpoint parameters.',
        physical='Production physical conversion with analog/digital head; feature/GAP/logit outputs restored to model coordinates.',
        noise=False,measured_pooling=False,nonlinear_R=False,spin_variation=False,
        backbone_quantization=args.backbone_qat,head_quantization='four-case sweep',
        tc_integrator='RK4, 32 coordinate-matched steps to isolate mapping from adaptive grid selection',
        tc_pcn_backend='TC current mapping, nonlinear R/spin/current noise disabled',
        toggle_cycles=5,stress=args.stress,
        output_tolerance=dict(rtol=2e-4,atol=2e-6),
        gradient_tolerance=dict(rtol=5e-4,atol=5e-6)),indent=2)+'\n')
    rows=[]
    with tempfile.TemporaryDirectory() as d:
        for family in ('toggle','tc1','tc2','toggle_ff','tc_ff'):
            settings=[('derived',None)] if family.startswith('tc') else [('derived',None),('derived',1),('fixed',None),('fixed',1)]
            for kind,bias,measured,(timing,factor),quant,clamp in itertools.product(
                ('analog','digital'),(False,True),(False,True),settings,(False,True),(False,True)):
                if args.smoke and (bias is False or measured or quant or clamp or factor is not None or timing!='derived'): continue
                if args.numerical_recheck and (family!='toggle_ff' or not measured or quant or clamp or factor is not None or timing!='fixed'): continue
                if args.stress and (not bias or factor is not None or timing!='derived'): continue
                if args.backbone_qat and not quant: continue
                config=(family,kind,bias,measured,timing,factor,quant,clamp)
                try:
                    rows.extend(run(config,Path(d)/'pretrain.pth',getattr(torch,args.dtype),args.stress,args.backbone_qat))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    rows.append(dict(config=config,error=str(e),passed=False))
                (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(f'{family}: {len(rows)} rows; failures {sum(not r["passed"] for r in rows)}',flush=True)
    print('Evidence:',args.output/'results.json')
    write_summary(args.output,rows)
    if any(not r['passed'] for r in rows):
        raise SystemExit(1)


if __name__=='__main__': main()
