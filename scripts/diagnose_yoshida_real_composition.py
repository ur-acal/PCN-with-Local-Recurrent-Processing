"""Real wrapped block: independently assembled composition and negative solve."""
import sys
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'docs/switched_dynamics_diagnostic_snapshot'),str(ROOT)]
from diagnose_yoshida_cause import CAPPED, ag
from debug_strang_real_block import build
from inference_utils import get_test_data
from TorchDiffEqPack.odesolver import odesolve
from switch import ODEXInitFFFBPixelSwitchYoshida4

@torch.no_grad()
def main():
    torch.manual_seed(4096)
    x,_=next(iter(get_test_data(test_bs=1,img_type='scanGFI',task='cifar100')))
    x=x.cuda()
    net,wrappers=build(ODEXInitFFFBPixelSwitchYoshida4,1)
    block=net.PcConvs[0]
    w=wrappers[0]
    xin=w.proj_fn(w.inp_scale*x)
    ag.AdaptiveGridSolver.integrate_search_grids=CAPPED
    y0=block.init_y(xin)
    total=block._get_total_horizon()[2]
    opts=block._build_interval_option_aca(0.,total/10,y0)
    signed=block._make_signed_fixed_pixel_ode_fn(xin,0,0,-1.)
    got=odesolve(signed,y0,opts)[-1]
    positive=block._make_fixed_pixel_ode_fn(xin,0,0)
    expected=odesolve(lambda t,y:-positive(t,y),y0,opts)[-1]
    negative_diff=float((got-expected).abs().max())
    del block._strang_active_pixel
    production=block._run_explicit_pixel_switch(xin)
    y=y0.clone()
    H,W=y.shape[-2:]
    P=H*W
    a=1/(2-2**(1/3))
    b=1-2*a
    for coefficient in (a,b,a):
        for order in (range(P),range(P-1,-1,-1)):
            for p in order:
                h,w=divmod(p,W)
                block._strang_active_pixel=(h,w)
                base=block._make_ode_fn(xin)
                sign=1 if coefficient>0 else -1
                opts=block._build_interval_option_aca(0.,abs(coefficient)*total/2,y)
                y=odesolve(lambda t,state:sign*base(t,state),y,opts)[-1]
                del block._strang_active_pixel
    result={'negative_solve_max_diff':negative_diff,
            'independent_composition_max_diff':float((y-production).abs().max()),
            'independent_composition_relative_diff':float((y-production).norm()/production.norm())}
    (ROOT/'results/yoshida_real_composition_control.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
