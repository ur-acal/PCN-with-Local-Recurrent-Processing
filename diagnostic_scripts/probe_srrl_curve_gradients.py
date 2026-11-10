"""Read-only, matched-data probe of SRRL gradients through two measured curves.

Counterfactual slope metrics change neither the forward pass nor its backward:
they multiply the observed incoming gradient by each curve's local derivative.
All BN updates are confined to freshly loaded in-memory diagnostic models.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from timm.data import Mixup

from study_relu_srrl_gap import (ROOT, seed, student, teacher, teacher_cache,
    checkpoint, quantize_unit_interval, PiecewiseLinearActivation)
from trainer_timm import TrainerCiFarTimmStyle


def augment(x, y, classes):
    cfg = SimpleNamespace(timm_input_size=(4, 16, 16), non_rgb_spatial_aug=True,
        non_rgb_crop_padding=2, hflip=.5, vflip=0., non_rgb_affine_degrees=9,
        non_rgb_affine_translate=(.125, .125), non_rgb_affine_shear=5, re_prob=0.)
    transform, _ = TrainerCiFarTimmStyle._build_non_rgb_timm_transforms(cfg, 'CiFAIR')
    mix = Mixup(mixup_alpha=.2, cutmix_alpha=1., prob=1., switch_prob=.5,
                mode='batch', label_smoothing=.1, num_classes=classes)
    seed()
    xx, yy = [], []
    for xb, yb in zip(x.split(128), y.split(128)):
        xb = torch.stack([transform(v) for v in xb])
        xb, yb = mix(xb, yb)
        xx.append(xb); yy.append(yb)
    return torch.cat(xx), torch.cat(yy)


@torch.no_grad()
def slope(x, q, mod):
    vin = mod.vin
    vals = mod.curves[mod.active_corner]
    z = q * x / mod.voltage_scale
    ix = torch.searchsorted(vin, z.contiguous(), right=True).clamp(1, len(vin)-1)
    d = (vals[ix]-vals[ix-1])/(vin[ix]-vin[ix-1])
    yy = vals[ix-1] + (z.clamp(vin[0], vin[-1])-vin[ix-1])*d
    valid = (z >= vin[0]) & (z <= vin[-1]) & (yy.abs()*mod.voltage_scale <= mod.v_dd)
    return torch.where(valid, d, 0.)


class GradientProbe:
    def __init__(self, model, references):
        self.stats = {}
        self.references = references
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, PiecewiseLinearActivation) and (
                    name.startswith('PcConvs.0.') or name.startswith('PcConvs.15.') or name == 'final_activation'):
                self.handles.append(mod.register_forward_hook(functools.partial(self.forward, name)))

    def forward(self, name, mod, args, output):
        if not output.requires_grad:
            return
        x = args[0].detach()
        q = mod._coordinate_pullback_scale
        derivatives = {fam: slope(x, q, ref) for fam, ref in self.references.items()}
        def backward(g):
            with torch.no_grad():
                energy = g.square()
                row = torch.stack([energy.sum(),
                    (energy*derivatives['0906'].square()).sum(),
                    (energy*derivatives['1022'].square()).sum(),
                    (energy*(x <= 0)).sum(),
                    (energy*(x >= 4.5)).sum(),
                    energy.new_tensor(energy.numel()),
                    (derivatives['0906'].abs()<1e-6).sum(),
                    (derivatives['1022'].abs()<1e-6).sum(),
                    (energy*(derivatives['0906'].abs()<1e-6)).sum(),
                    (energy*(derivatives['1022'].abs()<1e-6)).sum()]).double()
                self.stats[name] = self.stats.get(name, torch.zeros_like(row)) + row
        output.register_hook(backward)

    def finish(self):
        for h in self.handles:
            h.remove()
        out = {}
        for name, row in self.stats.items():
            total, old, new, negative, upper, n, dead_old, dead_new, de_old, de_new = row.cpu().tolist()
            # No incoming gradient means normalized energy metrics are undefined.
            # JSON null distinguishes this from a measured zero transmission.
            def energy_fraction(value):
                return value / total if total > 0 else None
            out[name] = dict(incoming_gradient_energy=total,
                transmitted_energy_0906=old, transmitted_energy_1022=new,
                rms_gain_0906=(old/total)**.5 if total > 0 else None,
                rms_gain_1022=(new/total)**.5 if total > 0 else None,
                negative_input_gradient_energy_fraction=energy_fraction(negative),
                upper_input_gradient_energy_fraction=energy_fraction(upper),
                near_zero_slope_fraction_0906=dead_old/n,
                near_zero_slope_fraction_1022=dead_new/n,
                gradient_energy_on_near_zero_slope_0906=energy_fraction(de_old),
                gradient_energy_on_near_zero_slope_1022=energy_fraction(de_new))
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--classes', type=int, required=True, choices=[10,100])
    p.add_argument('--samples', type=int, default=512)
    a = p.parse_args()
    folder = ROOT/'results/relu_srrl_gap'/f'cifar{a.classes}'
    data = torch.load(folder/'inputs.pt', weights_only=False)
    x = data['train_inputs'][:a.samples]; y = data['train_labels'][:a.samples]
    aug, soft_labels = augment(x, y, a.classes)
    inputs = dict(unaugmented=x, augmented=aug)
    net = teacher(a.classes)
    targets = {k:teacher_cache(net, v, 12) for k,v in inputs.items()}
    w = net.classifier[-1].weight.detach().clone()
    b = net.classifier[-1].bias.detach().clone()
    del net; torch.cuda.empty_cache()
    reference_models = {fam:student(a.classes, fam)[0] for fam in ['0906','1022']}
    refs = {fam:next(m for m in model.modules() if isinstance(m, PiecewiseLinearActivation))
            for fam, model in reference_models.items()}
    reports = {}
    for family in ['0906', '1022']:
        path = checkpoint(a.classes, family)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for condition, xx in inputs.items():
            model, kd, _ = student(a.classes, family)
            model.dropout = .25
            model.train(); kd.train()
            probe = GradientProbe(model, refs)
            totals = dict(pred=0., stat=0., ce=0.)
            for i in range(0, len(xx), 128):
                seed(7000+i)
                model.zero_grad(); kd.zero_grad()
                features, logits = model(quantize_unit_interval(xx[i:i+128].cuda(),12), is_feat=True)
                z = kd.connector(features[-1]).mean((2,3))
                t = targets[condition]['pool'][i:i+128].cuda()
                ts = targets[condition]['stat'][i:i+128].cuda()
                pred = F.mse_loss(F.linear(z,w,b),F.linear(t,w,b))
                stat = F.mse_loss(z,ts)
                labels = soft_labels[i:i+128].cuda() if condition == 'augmented' else y[i:i+128].cuda()
                ce = F.cross_entropy(logits,labels)
                pred.backward()
                for key, val in [('pred',pred),('stat',stat),('ce',ce)]:
                    totals[key] += val.item()*len(z)/len(xx)
            row = dict(samples=len(xx), dropout=.25, loss=totals, gradient=probe.finish())
            reports[f'{family}_{condition}'] = row
            print(a.classes,family,condition,json.dumps(row),flush=True)
            del model,kd;torch.cuda.empty_cache()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    (folder/'gradient_probe.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
