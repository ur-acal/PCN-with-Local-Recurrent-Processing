"""Compare switching currents and isolated layer errors on common reference inputs."""
import json
import sys
import argparse
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'docs/switched_dynamics_diagnostic_snapshot'),str(ROOT)]
from debug_strang_real_block import build
from ode_pc import ODEXInitFFFB
from switch import ODEXInitFFFBPixelSwitchStrang as Strang, aca_ode_solve

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--m3-screen',action='store_true')
    parser.add_argument('--deep-comparison',action='store_true');args=parser.parse_args()
    x=torch.load(ROOT/'results/coupled_switch_legacy.pt',weights_only=True)['input'].cuda()
    full,fw=build(ODEXInitFFFB,5)
    inputs={}
    hooks=[]
    for i,b in enumerate(full.PcConvs):
        b.option_aca.update(rtol=1e-6,atol=1e-6)
        def capture(module,args,index=i): inputs[index]=args[0].clone()
        hooks.append(b.register_forward_pre_hook(capture))
    full(x)
    for h in hooks:h.remove()
    path=ROOT/'results/coupled_layer_diagnosis.json'
    rows=json.loads(path.read_text()) if (args.m3_screen or args.deep_comparison) and path.exists() else []
    for m in ((3,) if args.m3_screen else (1,3)):
        class Configured(Strang):
            def __init__(self,**kw):super().__init__(block_size=m,**kw)
        net,wraps=build(Configured,5)
        for i,b in enumerate(net.PcConvs):
            if args.deep_comparison and i not in (5,11,15):continue
            b.option_aca.update(rtol=1e-8,atol=1e-8)
            inp=inputs[i];xp=wraps[i].proj_fn(wraps[i].inp_scale*inp)
            f=full.PcConvs[i];xf=fw[i].proj_fn(fw[i].inp_scale*inp)
            y=b.init_y(xp);yf=f.init_y(xf)
            assert torch.equal(y,yf)
            rhs_full=f._make_ode_fn(xf)(0.,y)
            summed=torch.zeros_like(y)
            try:
                for block in b.iter_spatial_blocks(*y.shape[-2:]):
                    b._active_spatial_block=block
                    if m==1:b._strang_active_pixel=block[:2]
                    summed+=b._make_ode_fn(xp)(0.,y)
            finally:
                for attr in ('_active_spatial_block','_strang_active_pixel'):
                    if hasattr(b,attr):delattr(b,attr)
            error=None
            if not args.m3_screen or i<3:
                reference=aca_ode_solve(f._make_ode_fn(xf),yf,f.option_aca)[-1]
                out=b._run_explicit_pixel_switch(xp)
                error=float((out-reference).norm()/reference.norm().clamp_min(1e-30))
            row={'m':m,'layer':i,'shape':list(y.shape),
                 'rhs_relative_error':float((summed-rhs_full).norm()/rhs_full.norm().clamp_min(1e-30)),
                 'layer_relative_error':error}
            rows.append(row);print(json.dumps(row),flush=True)
            (ROOT/'results/coupled_layer_diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
