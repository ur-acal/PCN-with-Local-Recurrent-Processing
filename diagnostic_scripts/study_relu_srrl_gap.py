"""Compare six archived PCN PT models at the SRRL interface.

Read-only source checkpoints. Diagnostic feature caches, reports, and optional
connector refits stay under the explicitly selected output directory.
"""
import argparse
import copy
import functools
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_l
import inference_utils
from inference_utils import get_eval_data, load_and_prepare_model
from input_preprocessing import quantize_unit_interval
from pc_conv import PCConvReLU6
from ode_pc import ToggleODEXInitFFFB, make_ode_block
from final_linear import configure_model_head
from train_ode_cifar import configure_unitless_measured_activation
from measured_activation import PiecewiseLinearActivation
from distillation.srrl import SRRLLoss


RUNS = {
    (10, 'ideal'): 'coupler_v2_CiFAIR10_qf1_noENOB_matchdistill_srrl_noRE_ftlr0p002_e200_toggle_odexinit',
    (100, 'ideal'): 'coupler_v2_CiFAIR100_qf1_noENOB_matchdistill_srrl_noRE_toggle_odexinit',
    (10, '0906'): 'C10_C96_454_16L_allReLU_analog_biasfalse_warmup5PreFT_iq12_toggle_odexinit',
    (100, '0906'): 'C100_C96_454_16L_allReLU_digital_biasfalse_iq12_toggle_odexinit',
    (10, '1022'): 'C10_C96L16_cc_randSpin_fullRange_1022ReLU_vdd0p5_analog_finalOnly_warmPT5_iq12_toggle_odexinit',
    (100, '1022'): 'C100_C96L16_cc_randSpin_fullRange_1022ReLU_vdd0p5_digital_finalOnly_iq12_toggle_odexinit',
}


def seed(value=4096):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def checkpoint(classes, family):
    paths = list((ROOT/'saved_ckpt_runs'/RUNS[classes, family]).glob('TIMMPCNet*1REP/*_last_ckpt.pth'))
    assert len(paths) == 1, paths
    return paths[0]


def student(classes, family):
    path = checkpoint(classes, family)
    meta = torch.load(path, map_location='cpu', weights_only=False)
    model = load_and_prepare_model(str(path), 'cuda', pc_conv_layer=PCConvReLU6,
        fuse_bn=False, conv_only=False, noise_level=0., noise_to_bn=False, noise_to_linear=False)
    measured = family != 'ideal'
    if measured:
        configure_model_head(model, physical=False, q=.1, v_dd=.5,
                             timing='fixed', base_time=10e-9, R=50e3, C=500e-15)
    model = make_ode_block(model, ode_block=ToggleODEXInitFFFB, noise_level=0.,
        method='dopri5', t_end=1.75, tol=1e-4, n_steps=5, toggle_n_cycles=5,
        toggle_fast_path=True, toggle_timing_mode='fixed' if measured else 'derived',
        toggle_y_time=10e-9, z_over_y_time=1. if measured else 3.,
        toggle_timing_R=50e3, toggle_timing_C=500e-15, odexinit_scaling_mode='direct',
        unitless_measured_pullback_mode='direct' if measured else 'none', unitless_pullback_q=.1)
    if measured:
        configure_unitless_measured_activation(model,
            str(ROOT/'hardware_data/mc_45_corners'/f'{family}_RELU_Voltage'/'tt_25_1.csv'),
            'MC18', 10, False, physical_v_dd=.5, pullback_scale=.1,
            interpolation='piecewise_linear', fuse_measured_activation=True)
    kd = SRRLLoss(96, 1280).cuda()
    kd.load_state_dict(meta['feature_kd']['state_dict'], strict=True)
    assert meta['feature_kd']['beta'] == 1.
    model.eval()
    kd.eval()
    return model, kd, meta


def teacher(classes):
    p = ROOT/'checkpoint'/f'efficientnet_v2_l_cifar{classes}_CiFAIR_OldNoTimm_MatchDistill.pth'
    d = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
    with torch.device('meta'):
        net = efficientnet_v2_l(weights=None, num_classes=classes)
        c = net.features[0][0]
        net.features[0][0] = torch.nn.Conv2d(4, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    net.load_state_dict(d['net'], strict=True, assign=True)
    net = net.cuda().eval()
    net.requires_grad_(False)
    return net


class CurveStats:
    def __init__(self, model):
        self.acc = {}
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, PiecewiseLinearActivation):
                self.handles.append(mod.register_forward_pre_hook(functools.partial(self.hook, name)))

    def hook(self, name, mod, inputs):
        x = inputs[0].detach()
        q = mod._coordinate_pullback_scale
        assert q is not None
        # Input x is unitless. The production transform is q*x/voltage_scale.
        z = q*x/mod.voltage_scale
        vin = mod.vin
        v = mod.curves[mod.active_corner]
        idx = torch.searchsorted(vin, z.contiguous(), right=True).clamp(1, len(vin)-1)
        slope = (v[idx]-v[idx-1])/(vin[idx]-vin[idx-1])
        slope = torch.where((z < vin[0]) | (z > vin[-1]), 0., slope)
        vals = torch.stack([x.new_tensor(x.numel()), (x <= 0).sum(),
            ((x > 1) & (slope < .1)).sum(), (x >= 4.5).sum(),
            slope.abs().sum(), x.sum(), x.square().sum()]).double()
        self.acc[name] = self.acc.get(name, torch.zeros_like(vals)) + vals

    def finish(self):
        for h in self.handles:
            h.remove()
        out = {}
        for name, v in self.acc.items():
            n, nonpos, upper, high, slope, sm, sq = v.cpu().tolist()
            out[name] = dict(elements=n, nonpositive_fraction=nonpos/n,
                upper_low_slope_fraction=upper/n, input_ge4p5_fraction=high/n,
                mean_abs_slope=slope/n, input_mean=sm/n, input_rms=(sq/n)**.5)
        return out


def get_data(classes, count, train):
    seed()
    loader = get_eval_data(test_bs=128, img_type='CiFAIR', task=f'cifar{classes}', train=train,
                           input_quant_bits=None, center_student_input=False)
    xx, yy = [], []
    for x, y in loader:
        xx.append(x.cpu()); yy.append(y.cpu())
        if sum(len(v) for v in xx) >= count:
            break
    return torch.cat(xx)[:count], torch.cat(yy)[:count]


@torch.no_grad()
def teacher_cache(net, inputs, bits):
    pool, stat = [], []
    for chunk in inputs.split(32):
        x = quantize_unit_interval(chunk.cuda(), bits)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        f = net.features((x-.5)/.5)
        assert f.shape[1:] == (1280, 7, 7)
        pool.append(f.mean((2, 3)).cpu())
        stat.append(F.adaptive_avg_pool2d(f, (4, 4)).mean((2, 3)).cpu())
    return dict(pool=torch.cat(pool), stat=torch.cat(stat))


def predict(connector, features, w, b):
    z = connector(features).mean((2, 3))
    return z, F.linear(z, w, b)


@torch.no_grad()
def metrics(kd, f, target, w, b, batch_bn=False):
    kd = copy.deepcopy(kd)
    kd.train(batch_bn)
    sums = {}
    n = len(f)
    for i in range(0, n, 128):
        x = f[i:i+128].cuda()
        t = target['pool'][i:i+128].cuda()
        ts = target['stat'][i:i+128].cuda()
        z, pred = predict(kd.connector, x, w, b)
        tl = F.linear(t, w, b)
        e = z-t
        err = pred-tl
        vals = dict(pred_mse=err.square().mean(1),
            centered_pred_mse=(err-err.mean(1, keepdim=True)).square().mean(1),
            common_logit_offset_mse=err.mean(1).square(),
            feature_mse=e.square().mean(1), stat_mse=(z-ts).square().mean(1),
            teacher_feature_energy=t.square().mean(1), teacher_logit_energy=tl.square().mean(1),
            feature_cosine=F.cosine_similarity(z,t,dim=1),
            connector_feature_energy=z.square().mean(1))
        for k,v in vals.items():sums[k]=sums.get(k,0.)+v.sum().item()
    out={k:v/n for k,v in sums.items()}
    out['normalized_feature_mse']=out['feature_mse']/out['teacher_feature_energy']
    out['classifier_amplification']=out['pred_mse']/out['feature_mse']
    out['normalized_pred_mse']=out['pred_mse']/out['teacher_logit_energy']
    out['isotropic_classifier_amplification']=w.square().sum().item()/len(w)
    return out


def refit(kd, train_f, train_t, test_f, test_t, w, b, epochs):
    seed()
    fitted = copy.deepcopy(kd)
    optimizer = torch.optim.Adam(fitted.parameters(), lr=.001)
    train_f=train_f.cuda();tp=train_t['pool'].cuda();ts=train_t['stat'].cuda()
    targets=F.linear(tp,w,b)
    history=[]
    for epoch in range(1, epochs+1):
        fitted.train()
        order=torch.randperm(len(train_f),device='cuda')
        loss_total=0.
        for ix in order.split(128):
            z,pred=predict(fitted.connector,train_f[ix],w,b)
            loss=F.mse_loss(pred,targets[ix])+F.mse_loss(z,ts[ix])
            optimizer.zero_grad();loss.backward();optimizer.step()
            loss_total+=loss.item()*len(ix)
        if epoch in {1,5,10,20,50,epochs}:
            row=dict(epoch=epoch,train_loss=loss_total/len(train_f),
                     heldout=metrics(fitted,test_f,test_t,w,b))
            history.append(row)
            print('REFIT',epoch, row['train_loss'], row['heldout']['pred_mse'], flush=True)
    return history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--classes',type=int,choices=[10,100],required=True)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--replay-samples',type=int,default=10000)
    p.add_argument('--refit-epochs',type=int,default=50)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    inference_utils.MyNoiseCIFARDataset=functools.partial(inference_utils.MyNoiseCIFARDataset,seed=0)
    seed()
    inputs,labels=get_data(a.classes,max(a.samples,a.replay_samples),False)
    train_inputs,train_labels=get_data(a.classes,a.samples,True)
    # Save exact inputs once: subsequent diagnostic stages use identical samples.
    torch.save(dict(test_inputs=inputs[:a.samples],test_labels=labels[:a.samples],
                    train_inputs=train_inputs,train_labels=train_labels),a.output/'inputs.pt')
    net=teacher(a.classes)
    w=net.classifier[-1].weight.detach().clone();b=net.classifier[-1].bias.detach().clone()
    targets={}
    for bits in [None,12]:
        targets[str(bits)]={
            'test':teacher_cache(net,inputs[:a.samples],bits),
            'train':teacher_cache(net,train_inputs,bits)}
        print('TEACHER_CACHED',a.classes,bits,flush=True)
    torch.save(dict(targets=targets,w=w.cpu(),b=b.cpu()),a.output/'teacher.pt')
    del net;torch.cuda.empty_cache()
    reports={}
    for family in ['ideal','0906','1022']:
        seed()
        path=checkpoint(a.classes,family)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        model,kd,meta=student(a.classes,family)
        bits=None if family=='ideal' else 12
        target=targets[str(bits)]
        fs=[];correct=0;ce=0.
        stats=CurveStats(model)
        with torch.no_grad():
            for i in range(0,len(inputs),128):
                x=quantize_unit_interval(inputs[i:i+128].cuda(),bits)
                feats,logits=model(x,is_feat=True)
                assert feats[-1].shape[1:] == (96,4,4)
                correct+=logits.argmax(1).eq(labels[i:i+128].cuda()).sum().item()
                ce+=F.cross_entropy(logits,labels[i:i+128].cuda(),reduction='sum').item()
                if i<a.samples:fs.append(feats[-1][:a.samples-i].cpu())
                if i+128 >= a.samples and stats.handles:
                    activation_stats=stats.finish();stats.handles=[]
                if (i//128+1)%10==0:print('REPLAY',a.classes,family,i+len(x),correct/(i+len(x)),flush=True)
            train_f=[]
            for x in train_inputs.split(128):
                feat,_=model(quantize_unit_interval(x.cuda(),bits),is_feat=True)
                train_f.append(feat[-1].cpu())
        f=torch.cat(fs);train_f=torch.cat(train_f)
        report=dict(checkpoint=str(path),sha256=digest,recorded_accuracy=meta['acc'],epoch=meta['epoch'],
            replay_samples=len(inputs),replay_accuracy=correct/len(inputs),ce=ce/len(inputs),
            teacher_input_bits=bits,diagnostic_samples=len(f),
            eval_connector=metrics(kd,f,target['test'],w,b),
            batch_stats_connector=metrics(kd,f,target['test'],w,b,batch_bn=True),
            activation_stats=activation_stats if family!='ideal' else {})
        torch.save(dict(train_features=train_f,test_features=f,kd=kd.cpu().state_dict()),a.output/f'{family}_features.pt')
        kd=kd.cuda()
        print('BASE',a.classes,family,json.dumps({k:v for k,v in report.items() if k!='activation_stats'}),flush=True)
        if a.refit_epochs:
            report['refit']=refit(kd,train_f,target['train'],f,target['test'],w,b,a.refit_epochs)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        reports[family]=report
        (a.output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
        del model,kd;torch.cuda.empty_cache()
    print('DONE',a.output/'report.json',flush=True)


if __name__=='__main__':main()
