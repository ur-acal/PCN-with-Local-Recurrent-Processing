"""Matched-data SRRL/augmentation interventions on isolated recovery clones.

No source checkpoint or production configuration is changed. Frozen probes and
short fixed-data optimization tests are not claimed to reproduce epoch12's
exact minibatches. All reported prediction/stat losses are unweighted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.data.mixup import mixup_target
from timm.optim import create_optimizer_v2
from study_relu_srrl_gap import (seed, teacher, teacher_cache, PCConvReLU6,
    load_and_prepare_model, configure_model_head, make_ode_block,
    ToggleODEXInitFFFB, configure_unitless_measured_activation, SRRLLoss, CurveStats)
from probe_srrl_task_interactions import grouped_gradients
from trainer_timm import TrainerCiFarTimmStyle
from train_ode_cifar import CASE_DEFAULTS, RGGB_DEFAULTS
from input_preprocessing import quantize_unit_interval
from measured_activation import PiecewiseLinearActivation

EVAL_SOURCE = None


def load_student(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    c = d['training_recovery']['config']
    model = load_and_prepare_model(str(path), 'cuda', pc_conv_layer=PCConvReLU6,
        fuse_bn=False, conv_only=False, noise_level=0., noise_to_bn=False, noise_to_linear=False)
    configure_model_head(model, physical=False, q=.1, v_dd=.6,
                         timing='fixed', base_time=10e-9, R=67e3, C=500e-15)
    model = make_ode_block(model, ode_block=ToggleODEXInitFFFB, noise_level=0.,
        method='dopri5', t_end=1.75, tol=1e-4, n_steps=5, toggle_n_cycles=5,
        toggle_fast_path=True, toggle_timing_mode='fixed', toggle_y_time=10e-9,
        z_over_y_time=1., toggle_timing_R=67e3, toggle_timing_C=500e-15,
        odexinit_scaling_mode='direct', unitless_measured_pullback_mode='direct', unitless_pullback_q=.1)
    configure_unitless_measured_activation(model,
        str(ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv'),
        'MC18',10,False,physical_v_dd=.6,pullback_scale=.1,
        interpolation='piecewise_linear',fuse_measured_activation=True)
    kd=SRRLLoss(96,1280).cuda()
    kd.load_state_dict(d['training_recovery']['aux']['_feature_kd_loss'],strict=True)
    model.dropout=.25
    return model,kd,d


def make_optimizer(model,kd,d,fix_aux=False):
    c=d['training_recovery']['config']
    rc=c['R']*c['C'];ff=c['toggle_y_time']/(rc*(c['t_end']/c['toggle_n_cycles']));fb=c['toggle_y_time']*c['z_over_y_time']/rc
    cfg=SimpleNamespace(model=model,scale_train_recipe=True,ff_train_scale=ff,fb_train_scale=fb,
        bias_lr_multiplier=1.,bias_weight_decay=None)
    groups=TrainerCiFarTimmStyle._composed_optimizer_parameters(cfg,c['learning_rate'],c['weight_decay'])
    opt=create_optimizer_v2(groups,opt='sgd',lr=c['learning_rate'],weight_decay=c['weight_decay'],momentum=.9)
    opt.add_param_group(dict(params=list(kd.parameters())))
    saved=d['training_recovery']['components']['optimizer']
    assert [len(g['params']) for g in opt.param_groups]==[len(g['params']) for g in saved['param_groups']]
    opt.load_state_dict(saved)
    if fix_aux:opt.param_groups[-1]['lr']=opt.param_groups[0]['lr']
    return opt


def losses(model,kd,x,labels,target,w,b):
    features,logits=model(quantize_unit_interval(x.cuda(),12),is_feat=True)
    f=features[-1];z=kd.connector(f).mean((2,3));pred=F.linear(z,w,b)
    terms=dict(ce=-(labels.cuda()*logits.log_softmax(1)).sum(1).mean(),
        stat=F.mse_loss(z,target['stat'].cuda()),
        pred=F.mse_loss(pred,F.linear(target['pool'].cuda(),w,b)))
    return terms,f


def inputs():
    data=torch.load(ROOT/'results/relu_srrl_gap/cifar10/inputs.pt',weights_only=False)
    x=data['train_inputs'][:512];y=data['train_labels'][:512]
    cfg=CASE_DEFAULTS['custom_noresize'].copy();cfg.update(RGGB_DEFAULTS);cfg.update(re_prob=0.,vflip=0.)
    transform,_=TrainerCiFarTimmStyle._build_non_rgb_timm_transforms(SimpleNamespace(**cfg),'CiFAIR')
    seed(4096);spatial=torch.stack([transform(v) for v in x])
    datasets={'raw':(x,mixup_target(y,10,lam=1.,smoothing=.1)),
              'spatial':(spatial,mixup_target(y,10,lam=1.,smoothing=.1))}
    for name,ma,ca in [('mixup',.2,0.),('cutmix',0.,1.),('production_mix',.2,1.)]:
        seed(4096)
        mix=Mixup(mixup_alpha=ma,cutmix_alpha=ca,prob=1.,switch_prob=.5,mode='batch',label_smoothing=.1,num_classes=10)
        xx=[];yy=[]
        for xb,yb in zip(spatial.split(128),y.split(128)):
            a,b=mix(xb.clone(),yb);xx.append(a);yy.append(b)
        datasets[name]=(torch.cat(xx),torch.cat(yy))
    return datasets


def target_batch(t,i):
    return {k:v[i:i+128] for k,v in t.items()}


@torch.no_grad()
def evaluate(model,kd,x,y,t,w,b):
    # BN updates confined to a clone, with identical dropout seeds per condition.
    # Toggle forward caches can contain non-leaf graph tensors: do not deepcopy
    # an already-executed model. Reconstruct a separate model, then copy states.
    m,k,_=load_student(EVAL_SOURCE)
    m.load_state_dict(model.state_dict(),strict=True);k.load_state_dict(kd.state_dict(),strict=True)
    if getattr(model,'_diagnostic_upper_linear',False):upper_linear(m)
    m.train();k.train()
    values=[];stats=CurveStats(m)
    for i in range(0,len(x),128):
        seed(7000+i);terms,f=losses(m,k,x[i:i+128],y[i:i+128],target_batch(t,i),w,b)
        values.append({key:float(v) for key,v in terms.items()})
    result={key:sum(v[key] for v in values)/len(values) for key in values[0]}
    result['final_activation']=stats.finish().get('final_activation')
    del m,k
    return result


def upper_linear(model):
    model._diagnostic_upper_linear=True
    handles=[]
    for m in model.modules():
        if not isinstance(m,PiecewiseLinearActivation):continue
        old=m._coordinate_pullback_scale;m.set_coordinate_pullback_scale(.1)
        with torch.no_grad():anchor=float(m(torch.tensor(5.1,device='cuda')))
        m.set_coordinate_pullback_scale(old)
        handles.append(m.register_forward_hook(lambda mod,args,out,a=anchor:
            torch.where(args[0]>5.1,a+args[0]-5.1,out)))
    return handles


def main():
    global EVAL_SOURCE
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--post-checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--resume-pilots',action='store_true');a=p.parse_args()
    EVAL_SOURCE=a.checkpoint
    a.output.mkdir(parents=True,exist_ok=a.resume_pilots)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if a.resume_pilots:
        cache=torch.load(a.output/'matched_inputs_targets.pt',weights_only=False)
        datasets,targets=cache['inputs'],cache['targets']
        report=json.loads((a.output/'report.json').read_text())
        assert report['checkpoint_sha256']==digest
        state=torch.load(ROOT/'checkpoint/efficientnet_v2_l_cifar10_CiFAIR_OldNoTimm_MatchDistill.pth',map_location='cpu',weights_only=False,mmap=True)['net']
        w=state['classifier.1.weight'].cuda();b=state['classifier.1.bias'].cuda();del state
    else:
        datasets=inputs();net=teacher(10)
        targets={name:teacher_cache(net,x,12) for name,(x,y) in datasets.items()}
        w=net.classifier[-1].weight.detach().clone();b=net.classifier[-1].bias.detach().clone()
        del net;torch.cuda.empty_cache()
        torch.save(dict(inputs=datasets,targets=targets),a.output/'matched_inputs_targets.pt')
        report=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=digest,
            note='512 matched images, fixed augmented realizations, original losses; short pilots are not a full training rerun.',frozen={},pilots={})
    def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for label,path in [('before',a.checkpoint),('after',a.post_checkpoint)]:
        if a.resume_pilots:break
        if path is None:continue
        model,kd,d=load_student(path);model.train();kd.train();report['frozen'][label]={}
        for condition,(x,y) in datasets.items():
            # Reset BN state and dropout realization for each condition/batch.
            rows=[];grad=[]
            for i in range(0,len(x),128):
                kd.load_state_dict(d['training_recovery']['aux']['_feature_kd_loss'])
                seed(7000+i);terms,f=losses(model,kd,x[i:i+128],y[i:i+128],target_batch(targets[condition],i),w,b)
                grad.append(grouped_gradients(terms,model,f))
                rows.append({key:float(v.detach()) for key,v in terms.items()})
            row=dict(losses={key:sum(v[key] for v in rows)/len(rows) for key in rows[0]},gradients=grad)
            report['frozen'][label][condition]=row;save();print('FROZEN',label,condition,row['losses'],flush=True)
        del model,kd,d;torch.cuda.empty_cache()
    for case in ['baseline','aux_lr_corrected','pred_weight0p2','upper_linear','spatial_only']:
        model,kd,d=load_student(a.checkpoint);model.train();kd.train()
        opt=make_optimizer(model,kd,d,case=='aux_lr_corrected')
        condition='spatial' if case=='spatial_only' else 'production_mix'
        x,y=datasets[condition];t=targets[condition];handles=upper_linear(model) if case=='upper_linear' else []
        row=dict(lrs=[g['lr'] for g in opt.param_groups],history=[])
        for step in range(a.steps):
            i=(step%4)*128;seed(12000+step);opt.zero_grad()
            terms,f=losses(model,kd,x[i:i+128],y[i:i+128],target_batch(t,i),w,b)
            loss=terms['ce']+terms['stat']+(.2 if case=='pred_weight0p2' else 1.)*terms['pred']
            loss.backward();opt.step()
            if step+1 in [20,40,a.steps]:
                v=evaluate(model,kd,x,y,t,w,b);row['history'].append(dict(step=step+1,**v));print('PILOT',case,step+1,{k:v[k] for k in ['ce','stat','pred']},flush=True)
        for h in handles:h.remove()
        model._diagnostic_upper_linear=False
        # Also evaluate modified-activation weights with ORIGINAL 1022 curves.
        row['original_1022_evaluation']=evaluate(model,kd,x,y,t,w,b)
        report['pilots'][case]=row;save()
        del model,kd,opt,d;torch.cuda.empty_cache()
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==digest
    (a.output/'COMPLETE').write_text('Matched probes and pilots complete; source checkpoint unchanged.\n')


if __name__=='__main__':main()
