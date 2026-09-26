"""Isolate TC/toggle interpolation on identical edge counts and input shapes.

This is an operator timing diagnostic, not a comparison of model accuracy.
Only a process-local alternative lookup is installed, then restored.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from validation import MVMConv, conv2d_to_matrix_fixed_padding
from tc_nonidealities import prepare_tc_resistance_curves
from utils import load_mc_res_curve_bank

p=argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
torch.manual_seed(4096)
d=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['net']
prefix='PcConvs.1.FFconv.parametrizations.weight.'
w=d[prefix+'original']; scale=d[prefix+'0.s_w']
w=(w*scale*15).round().clamp(-15,15)/15
mat,_,_=conv2d_to_matrix_fixed_padding((16,32,32),w.cuda(),padding=1,stride=1)
meta=dict(padding=1,stride=1,ker_h=3,ker_w=3,inp_chan=16,out_chan=16)
tc=MVMConv(mat,meta)
pkg=prepare_tc_resistance_curves('hardware_data/res_vs_vin_10k_150k.csv',
    'hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv',
    levels=torch.linspace(0,1,16),R=1e4,R_max=150e3,device='cuda',dtype=torch.float32)
tc.enable_csv(None,None,None,None,lambda x:x.clamp(-.1,.1),1e4,tc_curve_package=pkg,
              nonlinear_R_curve_seed=4096)
bank=load_mc_res_curve_bank('hardware_data/mc_45_corners/coupler_monte_v2/tt_-20_0.csv',
                            quantity='conductance',device='cuda')
toggle=MVMConv(mat,meta)
toggle.enable_csv(None,None,None,None,lambda x:x.clamp(-.1,.1),5e4,curve_bank=bank,
                  nonlinear_R_curve_sharing='per_coupler',nonlinear_R_curve_seed=4096)
report=dict(scope='one real 16->16 3x3 kernel expanded at 32x32; identical edge workload',
    edges=mat.values().numel(),tc_grid=pkg.v_grid.numel(),toggle_grid=bank['v_grid'].shape[-1],rows=[])
original=tc._get_gaussian_curve_R_eff
def direct(self,v,group,nominal_R):
    if self.proj_fn is not None:v=self.proj_fn(v)
    grid=self.nonlinear_R_curve_gaussian_v_grid
    q=v.clamp(min=grid[0],max=grid[-1])
    idx=(torch.searchsorted(grid.contiguous(),q.contiguous(),right=False)-1).clamp_(0,grid.numel()-2)
    table=self.nonlinear_R_curve_gaussian_R_normalized
    if getattr(self, '_tc_curve_row_index', None) is not None:
        group=self._tc_curve_row_index[group]
    left=table[group[:,None],idx]; right=table[group[:,None],idx+1]
    fraction=(q-grid[idx])/(grid[idx+1]-grid[idx])
    return nominal_R*(left+fraction*(right-left))
with torch.no_grad():
 for bs in (4,32,128):
    x=torch.rand(bs,16,32,32,device='cuda')*.2-.1
    outputs={}
    for mode in ('tc_original','tc_direct_lookup','toggle_empirical','ideal_sparse'):
        tc._get_gaussian_curve_R_eff=(types.MethodType(direct,tc) if mode=='tc_direct_lookup' else original)
        if mode.startswith('tc'): fn=lambda:tc(x)
        elif mode=='toggle_empirical':fn=lambda:toggle.forward_pulse(x,tc._tc_signed_mat)
        else:fn=lambda:torch.sparse.mm(mat,x.reshape(bs,-1).T)
        fn();torch.cuda.synchronize(); times=[]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            begin=time.perf_counter();y=fn();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        if mode.startswith('tc'):outputs[mode]=y
        row=dict(batch_size=bs,mode=mode,seconds=times,peak=torch.cuda.max_memory_allocated())
        if bs in (4,128) and mode in ('tc_original','toggle_empirical'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn();torch.cuda.synchronize()
            row['operator_profile']=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=12)
        report['rows'].append(row);print(json.dumps(row),flush=True)
    report.setdefault('errors',[]).append(float((outputs['tc_original']-outputs['tc_direct_lookup']).abs().max()))
    # index_add_ uses CUDA atomic sums, so repeated full convolutions are not
    # bitwise deterministic even with the original lookup.
    torch.testing.assert_close(outputs['tc_original'],outputs['tc_direct_lookup'],rtol=1e-5,atol=2e-6)
Path(a.output).write_text(json.dumps(report,indent=2))
