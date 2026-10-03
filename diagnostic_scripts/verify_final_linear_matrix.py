"""Persist numerical evidence for all four classifier arithmetic cases.

No datasets/training jobs. Reference formulas do not call either head's forward.
The stress fixture deliberately reaches analog rails / ADC endpoints. A separate
8-bit accumulator stress test exists in tests/test_final_linear.py; this audit
keeps the production 32-bit bias/accumulator defaults.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from final_linear import AnalogLinear, DigitalLinear
from measured_activation import PiecewiseLinearActivation
from physical_feedforward import AveragedPhysicalBasicBlock
from physical_feedforward_tc import TCPhysicalBasicBlock


def run_case(device, kind, mapping, measured, stress, quantize, clamp):
    torch.manual_seed(71)
    q, rail = .1, .5
    R, C = (1e4, 49e-15) if mapping == 'tc_derived' else (50e3, 500e-15)
    timing = 'fixed' if mapping == 'toggle_fixed' else 'derived'
    cls = TCPhysicalBasicBlock if mapping == 'tc_derived' else AveragedPhysicalBasicBlock
    block = cls(nn.Conv2d(4, 3, 1, bias=False), R=R, C=C, v_dd=rail, one_over_q=5).to(device)
    head = (AnalogLinear if kind == 'analog' else DigitalLinear)(4, 3,
        config=dict(final_head_quantize=quantize, final_head_clamp=clamp)).to(device)
    head.configure(physical=True, q=q, v_dd=rail, template=block,
                   family='tc' if mapping == 'tc_derived' else 'toggle',
                   timing=timing, base_time=10e-9, R=R, C=C)
    with torch.no_grad():
        head.weight.copy_(head.weight.abs() * (30 if stress else .2))
        head.bias.mul_(.2)
    W = head.weight.detach().clone().requires_grad_()
    b = head.bias.detach().clone().requires_grad_()
    x = torch.tensor([[.2, .9, -.2, .5], [.7, .3, 1., -.1]], device=device)
    x = x * (10 if stress else 1)
    if measured:
        act = PiecewiseLinearActivation(
            ROOT/'hardware_data/mc_45_corners/0906_RELU_Voltage/tt_25_1.csv',
            v_dd=rail, corner='MC18', normalize_positive_endpoint=False,
            fuse_measured_activation=False).to(device)
        v = act(q*x).detach()
    else:
        v = F.relu(q*x)
    v = v.requires_grad_()
    u = (v.detach()/q).requires_grad_()
    gain = 10e-9/(R*C) if kind == 'analog' and timing == 'fixed' else 1.
    ideal = gain*F.linear(u, W, b)
    raw = head(v)
    actual = head.logits_for_loss(raw)
    with torch.no_grad():
        source = v.clamp(-rail, rail) if clamp else v
        if kind == 'analog':
            aug = torch.cat((W,b[:,None]),1)
            scale = aug.abs().max().reciprocal() if quantize else aug.new_tensor(1.)
            encoded = (aug*scale*15).round().clamp(-15,15)/15 if quantize else aug
            voltage = gain*F.linear(torch.cat((source,source.new_full((2,1),q)),1),encoded)/scale
            expected = (voltage.clamp(-rail,rail) if clamp else voltage)/q
        elif quantize:
            sv, sw = rail/128, 127/W.double().abs().max()
            codes = (v.double()/sv).round().clamp(-128,127)
            K = (W.double()*sw).round().clamp(-127,127)
            bias_code = (b.double()*q*sw/sv).round().clamp(-2**31,2**31-1)
            acc = F.linear(codes,K,bias_code)
            if clamp:
                acc = acc.clamp(-2**31,2**31-1)
            expected = (acc*sv/(q*sw)).float()
        else:
            expected = F.linear(source,W,q*b)/q
    error = float((actual-expected).detach().abs().max())
    delta = float((actual-ideal).detach().abs().max())
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    labels = torch.tensor([0,2], device=device)
    F.cross_entropy(actual,labels).backward()
    gradient_error = None
    if not quantize and not clamp:
        F.cross_entropy(ideal,labels).backward()
        pairs = [(head.weight.grad,W.grad),(head.bias.grad,b.grad),(v.grad*q,u.grad)]
        gradient_error = max(float((a-b).abs().max()) for a,b in pairs)
        for a,b in pairs:
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(actual,ideal,rtol=1e-5,atol=2e-6)
    return dict(device=device,head=kind,mapping=mapping,measured=measured,
        stress=stress,quantization=quantize,clamping=clamp,
        max_reference_error=error,max_ideal_deviation=delta,
        max_coordinate_gradient_error=gradient_error,
        actual_logits=actual.detach().cpu().tolist(),reference_logits=expected.cpu().tolist(),
        ideal_logits=ideal.detach().cpu().tolist(),
        analog_rail_hits=int((raw.abs()>=rail).sum()) if kind=='analog' else None,
        adc_endpoint_inputs=int((v.abs()>=rail).sum()) if kind=='digital' else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing to silently skip')
    torch.set_num_threads(1)
    rows=[run_case(args.device,*case) for case in itertools.product(
        ('analog','digital'),('toggle_derived','toggle_fixed','tc_derived'),
        (False,True),(False,True),(False,True),(False,True))]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'{args.device}.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=[key for key in rows[0] if not key.endswith('_logits')]
    with (args.output/f'{args.device}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} cases passed; raw values: {args.output}/{args.device}.json')
    for kind,quant,clamp in itertools.product(('analog','digital'),(False,True),(False,True)):
        selected=[r for r in rows if (r['head'],r['quantization'],r['clamping'])==(kind,quant,clamp)]
        print(kind,quant,clamp,'reference error',max(r['max_reference_error'] for r in selected),
              'ideal deviation',max(r['max_ideal_deviation'] for r in selected),
              'gradient error',max((r['max_coordinate_gradient_error'] or 0) for r in selected))


if __name__=='__main__':
    main()
