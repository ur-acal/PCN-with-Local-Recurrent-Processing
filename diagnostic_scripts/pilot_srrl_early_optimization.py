"""Small common-initialization optimization pilot, not a full PT reproduction.

Default: 512 fixed augmented examples, 80 updates, no warmup, two constant
learning rates. Optional warmup uses subset epochs, not full-data epochs.
Production fixed-time initialization and optimizer parameter transforms are
reused. All artifacts isolated; no production checkpoint is saved.
"""
import argparse
import copy
import json
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from study_relu_srrl_gap import (ROOT,seed,checkpoint,PCConvReLU6,ToggleODEXInitFFFB,
    make_ode_block,configure_model_head,configure_unitless_measured_activation,
    SRRLLoss,quantize_unit_interval,CurveStats)
from pc_model import PCNetNoBatchNorm
from train_ode_cifar import scale_toggle_initial_weights
from trainer_timm import TrainerCiFarTimmStyle
from probe_srrl_task_interactions import intervene


def fresh(classes,family,initial_seed):
    seed(initial_seed)
    meta=torch.load(checkpoint(classes,'1022'),map_location='cpu',weights_only=False)
    args=copy.deepcopy(meta['init_args']['model_args'])
    args.update(meta['init_args']['kwargs']);args['dropout']=.25
    model=PCNetNoBatchNorm(**args).cuda()
    configure_model_head(model,physical=False,q=.1,v_dd=.5,timing='fixed',base_time=10e-9,R=50e3,C=500e-15)
    scale_toggle_initial_weights(model,8/7,.4)
    if model.final_head_type=='analog':
        with torch.no_grad():model.linear.weight.div_(.4)
    ideal=family in ['ideal','ideal6_all']
    base='1022' if family.startswith('1022_') else family
    model=make_ode_block(model,ode_block=ToggleODEXInitFFFB,noise_level=0.,method='dopri5',
        t_end=1.75,tol=1e-4,n_steps=5,toggle_n_cycles=5,toggle_fast_path=True,
        toggle_timing_mode='fixed',toggle_y_time=10e-9,z_over_y_time=1.,toggle_timing_R=50e3,
        toggle_timing_C=500e-15,odexinit_scaling_mode='direct',
        unitless_measured_pullback_mode='none' if ideal else 'direct',unitless_pullback_q=.1)
    if not ideal:
        configure_unitless_measured_activation(model,
            str(ROOT/'hardware_data/mc_45_corners'/f'{base}_RELU_Voltage'/'tt_25_1.csv'),
            'MC18',10,False,physical_v_dd=.5,pullback_scale=.1,
            interpolation='piecewise_linear',fuse_measured_activation=True)
    kd=SRRLLoss(96,1280).cuda()
    if family=='ideal6_all':model.final_activation=torch.nn.ReLU6()
    if family=='1022_no_upper':intervene(model,'upper_linear_all')
    if family=='1022_no_upper_final':intervene(model,'upper_linear_final')
    if family=='1022_no_upper_pc':intervene(model,'upper_linear_pc')
    if family=='1022_zero_offset':
        from measured_activation import PiecewiseLinearActivation
        # Subtract the measured origin value, retaining the upper saturation.
        for mod in model.modules():
            if isinstance(mod,PiecewiseLinearActivation):
                with torch.no_grad():
                    offset=(mod.voltage_scale*mod._interpolate(
                        mod.vin.new_zeros(1),mod.curves[mod.active_corner])/.1).item()
                mod.register_forward_hook(lambda m,args,out,offset=offset: (out-offset).clamp_min(0))
    return model,kd


@torch.no_grad()
def evaluate(model,kd,x,labels,t,w,b,record_curves=False):
    # Same training-mode objective, fresh masks fixed across variants.
    model.train();kd.train()
    values=[];stats=CurveStats(model) if record_curves else None
    for i in range(0,len(x),128):
        seed(7000+i)
        f,logits=model(quantize_unit_interval(x[i:i+128].cuda(),12),is_feat=True)
        z=kd.connector(f[-1]).mean((2,3))
        values.append(dict(pred=F.mse_loss(F.linear(z,w,b),F.linear(t['pool'][i:i+128].cuda(),w,b)).item(),
            stat=F.mse_loss(z,t['stat'][i:i+128].cuda()).item(),
            ce=F.cross_entropy(logits,labels[i:i+128].cuda()).item()))
    result={k:sum(r[k] for r in values)/len(values) for k in values[0]}
    if stats is not None:result['original_curve_occupancy']=stats.finish()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--classes',type=int,required=True)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--seed',type=int,default=4096)
    p.add_argument('--variants',nargs='+',default=['ideal','0906','1022'])
    p.add_argument('--lrs',nargs='+',type=float,default=[.002,.01])
    p.add_argument('--pred-weight',type=float,default=1.)
    p.add_argument('--output-name',default='early_optimization_pilot.json')
    p.add_argument('--record-curves',action='store_true')
    p.add_argument('--warmup-epochs',type=int,default=0,
                   help='Diagnostic epochs over the 512-image subset (4 updates/epoch), not full-data epochs.')
    a=p.parse_args();folder=ROOT/'results/relu_srrl_gap'/f'cifar{a.classes}'
    cache=torch.load(folder/'matched_augmented.pt',weights_only=False)
    x,labels=cache['inputs'],cache['labels'];t=cache['targets']['12'];w,b=cache['w'].cuda(),cache['b'].cuda()
    results={};initial=None;initial_kd=None
    for lr in a.lrs:
        for family in a.variants:
            model,kd=fresh(a.classes,family,a.seed)
            params={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            if initial is None:initial=params
            assert params.keys()==initial.keys()
            assert all(torch.equal(p,initial[n]) for n,p in params.items())
            kd_params={n:p.detach().cpu().clone() for n,p in kd.named_parameters()}
            if initial_kd is None:initial_kd=kd_params
            assert all(torch.equal(p,initial_kd[n]) for n,p in kd_params.items())
            helper=SimpleNamespace(model=model,scale_train_recipe=True,ff_train_scale=8/7,
                fb_train_scale=.4,bias_lr_multiplier=1.,bias_weight_decay=None)
            groups=TrainerCiFarTimmStyle._composed_optimizer_parameters(helper,lr,.001)
            opt=create_optimizer_v2(groups,opt='sgd',lr=lr,weight_decay=.001,momentum=.9)
            first=opt.param_groups[0]
            opt.add_param_group(dict(params=list(kd.parameters()),lr=first['lr'],
                                     weight_decay=first['weight_decay'],momentum=.9))
            target_lrs=[g['lr'] for g in opt.param_groups]
            history=[dict(step=0,**evaluate(model,kd,x,labels,t,w,b,a.record_curves))]
            for step in range(a.steps):
                i=(step%4)*128
                if a.warmup_epochs:
                    fraction=min((step//4)/a.warmup_epochs,1.)
                    for group,target_lr in zip(opt.param_groups,target_lrs):
                        group['lr']=1e-5+(target_lr-1e-5)*fraction
                seed(12000+step)
                model.train();kd.train();opt.zero_grad()
                f,logits=model(quantize_unit_interval(x[i:i+128].cuda(),12),is_feat=True)
                z=kd.connector(f[-1]).mean((2,3))
                pred=F.mse_loss(F.linear(z,w,b),F.linear(t['pool'][i:i+128].cuda(),w,b))
                stat=F.mse_loss(z,t['stat'][i:i+128].cuda())
                ce=F.cross_entropy(logits,labels[i:i+128].cuda())
                loss=a.pred_weight*pred+stat+ce
                if not torch.isfinite(loss):
                    history.append(dict(step=step+1,nonfinite=True));break
                loss.backward();opt.step()
                if step+1 in [10,20,40,a.steps]:
                    r=dict(step=step+1,**evaluate(model,kd,x,labels,t,w,b,a.record_curves));history.append(r)
                    print(a.classes,lr,family,{k:v for k,v in r.items() if k!='original_curve_occupancy'},flush=True)
            results[f'{family}_lr{lr}']=history
            (folder/a.output_name).write_text(json.dumps(dict(config=vars(a),results=results),indent=2)+'\n')
            del model,kd,opt;torch.cuda.empty_cache()


if __name__=='__main__':main()
