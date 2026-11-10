"""Matched task/objective and curve-intervention diagnostics, not production FT.

All model/BN changes are in-memory clones. No checkpoint or production config
is written. Counterfactual curve runs retain the original timing and weights.
"""
import argparse
import copy
import hashlib
import json
import torch
import torch.nn.functional as F
from study_relu_srrl_gap import (ROOT, seed, student, teacher, teacher_cache,
    checkpoint, quantize_unit_interval, PiecewiseLinearActivation)
from probe_srrl_curve_gradients import augment
from decompose_srrl_task_gap import decompose


def grouped_gradients(losses, model, feature):
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad and not n.startswith('linear.')]
    params = [p for _,p in named]
    gradients = {k:torch.autograd.grad(v,params+[feature],retain_graph=True,allow_unused=True)
                 for k,v in losses.items()}
    def pair(g,h,idx):
        dot=aa=bb=feature.new_tensor(0.)
        for i in idx:
            if g[i] is not None: aa=aa+g[i].detach().square().sum()
            if h[i] is not None: bb=bb+h[i].detach().square().sum()
            if g[i] is not None and h[i] is not None: dot=dot+(g[i].detach()*h[i].detach()).sum()
        return dict(norm_a=aa.sqrt().item(),norm_b=bb.sqrt().item(),
            cosine=(dot/(aa*bb).sqrt().clamp_min(1e-20)).item())
    groups={'backbone':list(range(len(params))), 'feature':[len(params)],
        'first_block':[i for i,(n,_) in enumerate(named) if n.startswith('PcConvs.0.')],
        'last_block':[i for i,(n,_) in enumerate(named) if n.startswith('PcConvs.15.')]}
    return {group:{a+'__'+b:pair(gradients[a],gradients[b],indices)
            for a,b in [('ce','pred'),('ce','stat'),('stat','pred')]}
            for group,indices in groups.items()}


def intervene(model, variant):
    if variant == 'baseline':return []
    old = PiecewiseLinearActivation(str(ROOT/'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv'),
        v_dd=.5,corner='MC18',normalize_positive_endpoint=False,fuse_measured_activation=False).cuda()
    old.set_coordinate_pullback_scale(.1)
    new = PiecewiseLinearActivation(str(ROOT/'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv'),
        v_dd=.5,corner='MC18',normalize_positive_endpoint=False,fuse_measured_activation=False).cuda()
    new.set_coordinate_pullback_scale(.1)
    y0=new(torch.tensor(4.25,device='cuda')).item()
    def hook(mod,args,out):
        x=args[0]
        if variant.startswith('upper_linear'):return torch.where(x>4.25,y0+(x-4.25),out)
        if variant == 'ideal6_all':return x.clamp(0,6)
        if variant == '0906_all':return old(x)
        if variant == 'q_half_all':return 2*new(x*.5)
        if variant == 'nearzero0906_all':
            blend=((1-x)/.5).clamp(0,1)
            return out+blend*(old(x)-out)
        raise ValueError(variant)
    handles=[]
    for name,mod in model.named_modules():
        if (isinstance(mod,PiecewiseLinearActivation)
                and (variant!='upper_linear_final' or name=='final_activation')
                and (variant!='upper_linear_pc' or name!='final_activation')):
            handles.append(mod.register_forward_hook(hook))
    return handles


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--classes',type=int,choices=[10,100],required=True)
    p.add_argument('--samples',type=int,default=512)
    a=p.parse_args()
    folder=ROOT/'results/relu_srrl_gap'/f'cifar{a.classes}'
    data=torch.load(folder/'inputs.pt',weights_only=False)
    x,labels=augment(data['train_inputs'][:a.samples],data['train_labels'][:a.samples],a.classes)
    cache=folder/'matched_augmented.pt'
    if cache.exists():
        stored=torch.load(cache,weights_only=False)
        assert torch.equal(stored['inputs'],x) and torch.equal(stored['labels'],labels)
        targets,w,b=stored['targets'],stored['w'].cuda(),stored['b'].cuda()
    else:
        net=teacher(a.classes)
        targets={str(bits):teacher_cache(net,x,bits) for bits in [None,12]}
        w=net.classifier[-1].weight.detach().clone();b=net.classifier[-1].bias.detach().clone()
        del net;torch.cuda.empty_cache()
        torch.save(dict(inputs=x,labels=labels,targets=targets,w=w.cpu(),b=b.cpu()),cache)
    results={}
    cases=[(f,'baseline',True) for f in ['ideal','0906','1022']]+[
        ('1022',v,False) for v in ['upper_linear_final','upper_linear_all','q_half_all',
                                'nearzero0906_all','ideal6_all','0906_all']]
    for family,variant,compute_grads in cases:
        source=checkpoint(a.classes,family);digest=hashlib.sha256(source.read_bytes()).hexdigest()
        model,kd,_=student(a.classes,family)
        model.dropout=.25;model.train();kd.train()
        handles=intervene(model,variant)
        bits=None if family=='ideal' else 12
        target=targets[str(bits)]
        losses_out=[];gradient_out=[];pred_out=[];teacher_out=[];logits_out=[]
        for i in range(0,len(x),128):
            seed(7000+i)
            with torch.set_grad_enabled(compute_grads):
                feats,logits=model(quantize_unit_interval(x[i:i+128].cuda(),bits),is_feat=True)
                feat=feats[-1]
                z=kd.connector(feat).mean((2,3))
                pred=F.linear(z,w,b)
                tl=F.linear(target['pool'][i:i+128].cuda(),w,b)
                losses=dict(ce=F.cross_entropy(logits,labels[i:i+128].cuda()),
                    stat=F.mse_loss(z,target['stat'][i:i+128].cuda()),pred=F.mse_loss(pred,tl))
                if compute_grads:gradient_out.append(grouped_gradients(losses,model,feat))
            losses_out.append({k:v.item() for k,v in losses.items()})
            pred_out.append(pred.detach().cpu());teacher_out.append(tl.detach().cpu());logits_out.append(logits.detach().cpu())
        row=dict(losses={k:sum(r[k] for r in losses_out)/len(losses_out) for k in losses_out[0]},
            gradients=gradient_out,head_type=type(model.linear).__name__,
            head_gain=getattr(model.linear,'recipe_gain',1.),
            error_decomposition=decompose(torch.cat(pred_out),torch.cat(teacher_out),labels.argmax(1)))
        results[family+'_'+variant]=row
        torch.save(dict(pred=torch.cat(pred_out),target=torch.cat(teacher_out),logits=torch.cat(logits_out)),
                   folder/f'interaction_{family}_{variant}.pt')
        for h in handles:h.remove()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        print(a.classes,family,variant,row['losses'],flush=True)
        (folder/'task_interactions.json').write_text(json.dumps(results,indent=2)+'\n')
        del model,kd;torch.cuda.empty_cache()


if __name__=='__main__':main()
