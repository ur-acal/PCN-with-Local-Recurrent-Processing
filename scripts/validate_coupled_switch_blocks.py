"""Real trained-block regression and size sweep for coupled switching."""
import sys,json,argparse,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'docs/switched_dynamics_diagnostic_snapshot'),str(ROOT)]
from debug_strang_real_block import build, CKPT
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB
from switch import ODEXInitFFFBPixelSwitchExplicit as Lie, ODEXInitFFFBPixelSwitchStrang as Strang, ODEXInitFFFBPixelSwitchEfficient as Jacobi, aca_ode_solve

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',action='store_true')
    p.add_argument('--local-only',action='store_true');args=p.parse_args()
    path=ROOT/'results/coupled_switch_legacy.pt'
    if args.capture:
        torch.manual_seed(4096)
        x,_=next(iter(get_test_data(test_bs=1,img_type='scanGFI',task='cifar100')))
        x=x.cuda(); saved={'input':x.cpu(),'outputs':{}}
        for name,cls in [('full',ODEXInitFFFB),('jacobi',Jacobi),('lie',Lie),('strang',Strang)]:
            net,_=build(cls,2);b=net.PcConvs[0];b.option_aca.update(rtol=1e-8,atol=1e-8)
            saved['outputs'][name]=b(x).cpu()
            print('captured',name,flush=True)
        torch.save(saved,path);return
    saved=torch.load(path,weights_only=True);x=saved['input'].cuda();ref=saved['outputs']['full'].cuda()
    if args.local_only:
        class Four(Lie):
            def __init__(self,**kw):super().__init__(block_size=4,**kw)
        net,wraps=build(Four,2);b=net.PcConvs[0]
        b.option_aca.update(rtol=1e-8,atol=1e-8)
        xp=wraps[0].proj_fn(wraps[0].inp_scale*x);initial=b.init_y(xp)
        _,_,total=b._get_total_horizon();checks=[]
        for m in (1,2,3,4):
            b.block_size=m
            for block in (b.iter_spatial_blocks(*initial.shape[-2:])[0],
                          b.iter_spatial_blocks(*initial.shape[-2:])[-1]):
                i,j,k,l=block;mask=torch.zeros_like(initial,dtype=torch.bool)
                mask[:,:,i:k,j:l]=True
                out=b.local_block_flow(xp,initial,block,total/100)
                assert torch.equal(out[~mask],initial[~mask])
                row={'m':m,'block':block,'outside_bitwise_unchanged':True,
                     'active_change_max':float((out-initial)[mask].abs().max())}
                if m>1 and l-j>1:
                    b._active_spatial_block=block
                    fn=b._make_ode_fn(xp)
                    base=fn(0,initial)
                    for src,dst in ((j,j+1),(j+1,j)):
                        perturbed=initial.clone();perturbed[:,:,i,src]+=1e-4
                        influence=float((fn(0,perturbed)-base)[:,:,i,dst].abs().max())
                        assert influence>0
                        row[f'influence_{src}_to_{dst}']=influence
                    del b._active_spatial_block
                checks.append(row)
        (ROOT/'results/coupled_switch_local_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(checks),flush=True);return
    full,full_wraps=build(ODEXInitFFFB,2);fb=full.PcConvs[0]
    fb.option_aca.update(rtol=1e-8,atol=1e-8)
    fx=full_wraps[0].proj_fn(full_wraps[0].inp_scale*x)
    physical_ref=aca_ode_solve(fb._make_ode_fn(fx),fb.init_y(fx),fb.option_aca)[-1]
    rows=[]
    for name,cls in [('jacobi',Jacobi),('lie',Lie),('strang',Strang)]:
        for m in (1,2,3,4):
            class Configured(cls):
                def __init__(self,**kw):super().__init__(block_size=m,**kw)
            net,wraps=build(Configured,2);b=net.PcConvs[0];b.option_aca.update(rtol=1e-8,atol=1e-8)
            t=time.perf_counter();y=b(x);torch.cuda.synchronize()
            row={'method':name,'m':m,'n_iters':2,'tol':1e-8,'error':float((y-ref).norm()/ref.norm()),'seconds':time.perf_counter()-t}
            xp=wraps[0].proj_fn(wraps[0].inp_scale*x)
            physical=b._run_explicit_pixel_switch(xp)
            row['physical_state_relative_error']=float((physical-physical_ref).norm()/physical_ref.norm())
            if name=='lie':
                initial=b.init_y(xp)
                _,_,total=b._get_total_horizon()
                checks=[]
                for block in (b.iter_spatial_blocks(*initial.shape[-2:])[0],
                              b.iter_spatial_blocks(*initial.shape[-2:])[-1]):
                    i,j,k,l=block; mask=torch.zeros_like(initial,dtype=torch.bool)
                    mask[:,:,i:k,j:l]=True
                    flowed=b.local_block_flow(xp,initial,block,total/100)
                    assert torch.equal(flowed[~mask],initial[~mask])
                    checks.append({'block':block,'outside_bitwise_unchanged':True})
                row['real_local_checks']=checks
            if m==1:row['legacy_difference']=float((y-saved['outputs'][name].cuda()).norm()/saved['outputs'][name].cuda().norm())
            if m==1 and name!='jacobi':
                current=b.local_block_flow
                def legacy_flow(x,y,block,duration):
                    i,j,_,_=block
                    fn=b._make_fixed_pixel_ode_fn(x,i,j)
                    out=aca_ode_solve(fn,y,b._build_interval_option_aca(0,duration,y))[-1]
                    for attr in ('_explicit_active_pixel','_strang_active_pixel'):
                        if hasattr(b,attr):delattr(b,attr)
                    return out
                b.local_block_flow=legacy_flow
                legacy=b._run_explicit_pixel_switch(xp)
                b.local_block_flow=current
                row['prequant_legacy_relative_difference']=float((physical-legacy).norm()/legacy.norm())
                row['prequant_legacy_max_abs_difference']=float((physical-legacy).abs().max())
                b.double()
                for conv in b.FBconv_copies:conv.double()
                xp=xp.double()
                b.local_block_flow=current
                physical64=b._run_explicit_pixel_switch(xp)
                b.local_block_flow=legacy_flow
                legacy64=b._run_explicit_pixel_switch(xp)
                b.local_block_flow=current
                row['float64_prequant_legacy_relative_difference']=float((physical64-legacy64).norm()/legacy64.norm())
            rows.append(row);print(json.dumps(row),flush=True)
            (ROOT/'results/coupled_switch_validation.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
