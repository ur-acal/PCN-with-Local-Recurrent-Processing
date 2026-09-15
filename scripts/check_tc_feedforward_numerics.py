#!/usr/bin/env python3
"""Reproducible CPU comparisons; prints JSON, never launches training."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from physical_feedforward_tc import TCPhysicalBasicBlock

torch.set_num_threads(1)
torch.manual_seed(141)
rows=[]
for timing in ('derived','fixed'):
    conv=nn.Conv2d(2,2,1,bias=False).double()
    conv.weight.data.copy_(torch.tensor([3.,-6.,0.,12.]).reshape_as(conv.weight)/15)
    a=TCPhysicalBasicBlock(conv,R=1e4,C=49e-15,v_dd=.1,one_shot_conv=True,
                          toggle_timing_mode=timing,toggle_y_time=5e-9).double()
    b=copy.deepcopy(a);b.one_shot_conv=False
    x=torch.rand(4,2,3,3,dtype=torch.float64)*.001
    xa=x.clone().requires_grad_();xb=x.clone().requires_grad_()
    ya,yb=a(xa),b(xb)
    ya.sum().backward();yb.sum().backward()
    row=dict(timing=timing,output_max_abs=float((ya-yb).detach().abs().max()),
             input_gradient_max_abs=float((xa.grad-xb.grad).abs().max()),
             weight_gradient_max_abs=float((a.conv1.weight.grad-b.conv1.weight.grad).abs().max()))
    for m in (a,b):
        m.enable_coupler_noise=True;m.enable_summing_current_noise=True
        m.summing_current_p=.6e-12;m.coupler_noise_p=.6e-12
    samples=torch.zeros(20000,2,1,1,dtype=torch.float64)
    _,eps=a._noise_context(a.conv1,samples,samples,'z')
    duration=a._stage_duration(samples,'z')
    row['duration_seconds']=float(duration)
    row['noise_expected_variance']=(duration*eps.flatten().square()).tolist()
    with torch.no_grad():
        for label,m in [('one_shot',a),('dopri5',b)]:
            y=m(samples).reshape(-1,2)
            row[label+'_noise_mean']=y.mean(0).tolist()
            row[label+'_noise_variance']=y.var(0).tolist()
    rows.append(row)
print(json.dumps(dict(dtype='float64 (legacy solver time bookkeeping may use float32)',
                      samples=20000,results=rows),indent=2))
