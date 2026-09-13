"""Diagnostic only: isolate the shared-curve activation correction at real B128 shapes.

Captures inputs from one ordinary nonlinear-R student benchmark. No model updates
or production implementation changes. Coarse curves and guard bypass are timing
experiments only, not proposed hardware settings.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ode_pc import ODEBlockPC as ODEBlock
from tc_nonidealities import TCSharedCurve, positive_resistance

p = argparse.ArgumentParser()
p.add_argument('--output-dir', default='results/tc_correction_profile')
p.add_argument('--full-variant', choices=['original','guard_bypass'],
               help='Separate full-model timing check; bypass is diagnostic only.')
a = p.parse_args()
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
if a.full_variant:
    import tc_nonidealities
    if a.full_variant == 'guard_bypass':
        tc_nonidealities.positive_resistance = lambda values, floor_ohms=1e-6: values.clamp_min(floor_ohms)
    target = out/('full_'+a.full_variant+'.json')
    sys.argv = ['benchmark','--mode','all','--effects','nonlinear','--conv-method','shared',
                '--batch','128','--no-checkpoint','--count-stages','--tol','1e-6',
                '--repeats','5','--output',str(target)]
    runpy.run_path(str(ROOT/'scripts/benchmark_tc_checkpointed_training.py'))
    data=json.loads(target.read_text())
    data['diagnostic_variant']=a.full_variant
    target.write_text(json.dumps(data,indent=2))
    sys.exit(0)
captures = {}
original = ODEBlock._tc_dense_conv

def capture(self, module, source, curves):
    key = (self.layer_idx, 'FF' if module is self.FFconv else 'FB')
    if isinstance(curves, TCSharedCurve) and key not in captures:
        captures[key] = dict(source=source.detach().clone(), grid=self._tc_curve_package.v_grid,
                             table=curves.resistance.detach().clone(), nominal=curves.nominal_R,
                             rail=self.v_dd, floor=self._tc_curve_package.floor_ohms)
    return original(self, module, source, curves)

ODEBlock._tc_dense_conv = capture
sys.argv = ['benchmark', '--mode', 'all', '--effects', 'nonlinear', '--conv-method', 'shared',
            '--batch', '128', '--no-checkpoint', '--tol', '1e-6', '--repeats', '0',
            '--output', str(out/'capture.json')]
try:
    state = runpy.run_path(str(ROOT/'scripts/benchmark_tc_checkpointed_training.py'))
finally:
    ODEBlock._tc_dense_conv = original
state['model'].zero_grad(set_to_none=True)

def correction(x, item, knots=None, guard=True, annotate=False):
    from contextlib import nullcontext
    scope = torch.profiler.record_function if annotate else lambda _: nullcontext()
    grid, table = item['grid'], item['table']
    if knots:
        # Downsample outside measured operation in the benchmark below.
        grid, table = item['coarse'][knots]
    with scope('correction/query_clamp'):
        query = x.clamp(-item['rail'], item['rail']).clamp(grid[0], grid[-1])
    with scope('correction/bucketize'):
        index = (torch.bucketize(query, grid)-1).clamp(0, grid.numel()-2)
    with scope('correction/interpolate'):
        slope = (table[1:]-table[:-1])/(grid[1:]-grid[:-1])
        resistance = table[index]+slope[index]*(query-grid[index])
    with scope('correction/positive_guard'):
        resistance = positive_resistance(resistance, item['floor']) if guard else resistance.clamp_min(item['floor'])
    with scope('correction/multiply_divide'):
        return x*item['nominal']/resistance

result = dict(note='Isolated correction, no convolution/QAT/solver. All 32 captured FF/FB shapes once per sweep.',
              cases={}, captured_shapes={str(k):list(v['source'].shape) for k,v in captures.items()})
for item in captures.values():
    item['coarse'] = {}
    for n in (13,4):
        idx = torch.linspace(0,item['grid'].numel()-1,n,device='cuda').round().long()
        item['coarse'][n] = (item['grid'][idx],item['table'][idx])

variants = [('original',None,True),('guard_bypass',None,False),('13_knots',13,True),('4_knots',4,True)]
timings = {name: {'forward':[], 'forward_backward':[]} for name,_,_ in variants}
for iteration in range(12):
    for name,knots,guard in (variants if iteration%2==0 else variants[::-1]):
        for backward in (False,True):
            torch.cuda.synchronize()
            start=time.perf_counter()
            for item in captures.values():
                x=item['source'].detach().requires_grad_(backward)
                with torch.set_grad_enabled(backward):
                    y=correction(x,item,knots,guard)
                    if backward:y.sum().backward()
            torch.cuda.synchronize()
            if iteration>=2:
                timings[name]['forward_backward' if backward else 'forward'].append(time.perf_counter()-start)
import statistics
for name,t in timings.items():
    result['cases'][name]={k:dict(median_seconds=statistics.median(v),seconds=v) for k,v in t.items()}

# Explicitly record that coarsening changes the operation, not merely its speed.
for name,knots,guard in variants:
    errors=[]
    with torch.no_grad():
        for item in captures.values():
            ref=correction(item['source'],item)
            new=correction(item['source'],item,knots,guard)
            errors.append(float((new-ref).norm()/ref.norm().clamp_min(1e-30)))
    result['cases'][name]['max_relative_correction_error']=max(errors)

item=max(captures.values(),key=lambda item:item['source'].numel())
result['profile_shape']=list(item['source'].shape)
result['original_knots']=item['grid'].numel()
saved=[]
def pack(t):
    saved.append(dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size()))
    return t
x=item['source'].detach().requires_grad_(True)
with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
    correction(x,item).sum().backward()
result['saved_tensors']=saved
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        x=item['source'].detach().requires_grad_(True)
        correction(x,item,annotate=True).sum().backward()
    torch.cuda.synchronize()
events=prof.key_averages()
(out/'operators_cpu.txt').write_text(events.table(sort_by='self_cpu_time_total',row_limit=35))
(out/'operators_cuda.txt').write_text(events.table(sort_by='self_cuda_time_total',row_limit=35))
result['regions']=[dict(name=e.key,calls=e.count,cpu_us=e.cpu_time_total,
                         self_cpu_us=e.self_cpu_time_total,device_us=e.device_time_total)
                    for e in events if e.key.startswith('correction/')]
(out/'microbenchmark.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ('cases','regions','profile_shape','original_knots')},indent=2))
