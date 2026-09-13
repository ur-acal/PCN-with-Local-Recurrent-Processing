"""Diagnostic-only endpoint/norm controls; never patches production source files."""
import inspect
import argparse
import json
import logging
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'docs/switched_dynamics_diagnostic_snapshot'), str(ROOT)]
import torch
import ode_pc
import switch
from debug_strang_real_block import build
from inference_utils import get_test_data
from TorchDiffEqPack.odesolver import odesolve
from TorchDiffEqPack.odesolver import adaptive_grid_solver as ag
from TorchDiffEqPack.odesolver.base import ODESolver

logging.disable(logging.WARNING)
OUT = ROOT / 'results/yoshida_cause_diagnostic.jsonl'
ORIGINAL = ag.AdaptiveGridSolver.integrate_search_grids
source = textwrap.dedent(inspect.getsource(ORIGINAL))
source = source.replace('h_current = h_new  # .clone().detach()',
    'h_new = min(float(h_new), float(abs(self.t1 - t_current)))\n'
    '                h_current = h_new  # capped diagnostic endpoint')
# Equality must record the final endpoint rather than starting another zero step.
source = source.replace('> torch.abs(self.t_end - self.t0)',
                        '>= torch.abs(self.t_end - self.t0)')
scope = dict(vars(ag))
exec(source, scope)
CAPPED = scope['integrate_search_grids']


def emit(row):
    print(json.dumps(row), flush=True)
    with OUT.open('a') as fp:
        fp.write(json.dumps(row) + '\n')


def set_tol(block, tol):
    block.tol = tol
    block.option_aca.update(rtol=tol, atol=tol)


def relative(y, ref):
    return float((y-ref).norm()/ref.norm())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['lie','strang','yoshida'])
    parser.add_argument('--modes', nargs='+', default=['legacy','endpoint','active_norm'])
    parser.add_argument('--iters', nargs='+', type=int, default=[5,10])
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--remove-projection', action='store_true')
    args=parser.parse_args()
    torch.manual_seed(4096)
    x, _ = next(iter(get_test_data(test_bs=1, img_type='scanGFI', task='cifar100')))
    x = x.cuda()
    refnet, _ = build(ode_pc.ODEXInitFFFB, 1)
    refblock = refnet.PcConvs[0]
    if args.remove_projection:
        refblock.option_aca.pop('proj_fn',None)
    ref = refblock(x)
    refs = {'legacy': ref}
    ag.AdaptiveGridSolver.integrate_search_grids = CAPPED
    for tol in (1e-6, 1e-8):
        set_tol(refblock, tol)
        refs[str(tol)] = refblock(x)
        emit({'kind':'reference', 'endpoint':'capped', 'tol':tol,
              'remove_projection':args.remove_projection,
              'error_legacy':relative(refs[str(tol)], ref)})
    ag.AdaptiveGridSolver.integrate_search_grids = ORIGINAL
    methods = [('lie', switch.ODEXInitFFFBPixelSwitchExplicit),
               ('strang', switch.ODEXInitFFFBPixelSwitchStrang),
               ('yoshida', switch.ODEXInitFFFBPixelSwitchYoshida4)]
    for name, cls in methods:
        if name not in args.methods:
            continue
        net, wrappers = build(cls, 1)
        block = net.PcConvs[0]
        if args.remove_projection:
            block.option_aca.pop('proj_fn',None)
        for n in args.iters:
            for mode in args.modes:
                block.n_iters = n
                set_tol(block, args.tol)
                ag.AdaptiveGridSolver.integrate_search_grids = (
                    ORIGINAL if mode == 'legacy' else CAPPED)
                original_adapt = ag.AdaptiveGridSolver.adapt_stepsize
                if mode == 'active_norm':
                    def active_adapt(self, y, yn, error, *args, **kwargs):
                        p = getattr(block, '_strang_active_pixel',
                                    getattr(block, '_explicit_active_pixel', None))
                        if p is not None:
                            a,b = p
                            y = tuple(v[:,:,a:a+1,b:b+1] for v in y)
                            yn = tuple(v[:,:,a:a+1,b:b+1] for v in yn)
                            error = tuple(v[:,:,a:a+1,b:b+1] for v in error)
                        return original_adapt(self, y, yn, error, *args, **kwargs)
                    ag.AdaptiveGridSolver.adapt_stepsize = active_adapt
                stats = {'solves':0, 'nfe':0, 'projection_changed':0,
                         'projection_calls':0, 'max_projection_change':0.,
                         'overshoot':0, 'interpolations':0}
                oldsolve = switch.aca_ode_solve
                oldinterp = ODESolver.interpolate
                oldproj = ODESolver.mult_noisy_update_and_proj
                def projection(self, h, y_current):
                    out = oldproj(self, h, y_current)
                    delta = max(float((a-b).abs().max()) for a,b in zip(out,y_current))
                    stats['projection_calls'] += 1
                    stats['projection_changed'] += int(delta > 0)
                    stats['max_projection_change'] = max(stats['max_projection_change'],delta)
                    return out
                def interpolation(self, t0,t1,te,*args,**kwargs):
                    stats['interpolations'] += 1
                    stats['overshoot'] += int(float(t1)>float(te) and float(te)>0)
                    return oldinterp(self,t0,t1,te,*args,**kwargs)
                def solve(fn,y0,options,*args,**kwargs):
                    stats['solves'] += 1
                    def counted(t,y):
                        stats['nfe'] += 1
                        return fn(t,y)
                    return oldsolve(counted,y0,options,*args,**kwargs)
                switch.aca_ode_solve = solve
                ODESolver.interpolate = interpolation
                ODESolver.mult_noisy_update_and_proj = projection
                start=time.perf_counter()
                try:
                    y=block(x)
                    torch.cuda.synchronize()
                    emit(dict(kind='case', method=name, n=n, mode=mode,
                        tol=args.tol, remove_projection=args.remove_projection,
                        error_legacy=relative(y,ref),
                        error_capped_ref=relative(y,refs[str(1e-8)]),
                        runtime=time.perf_counter()-start, **stats))
                finally:
                    switch.aca_ode_solve=oldsolve
                    ODESolver.interpolate=oldinterp
                    ODESolver.mult_noisy_update_and_proj=oldproj
                    ag.AdaptiveGridSolver.adapt_stepsize=original_adapt
        del net
    ag.AdaptiveGridSolver.integrate_search_grids = ORIGINAL
    emit({'kind':'complete'})


if __name__=='__main__':
    main()
