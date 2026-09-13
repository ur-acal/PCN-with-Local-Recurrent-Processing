"""Exact local flows of the real block's frozen-activation linearization."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'docs/switched_dynamics_diagnostic_snapshot'),str(ROOT)]
from debug_strang_real_block import build
from inference_utils import get_test_data
from ode_pc import ODEXInitFFFB

def main():
    torch.manual_seed(4096)
    x,_=next(iter(get_test_data(test_bs=1,img_type='scanGFI',task='cifar100')))
    with torch.no_grad():
        net,wraps=build(ODEXInitFFFB,1)
    block=net.PcConvs[0]
    w=wraps[0]
    xin=w.proj_fn(w.inp_scale*x.cuda())
    y0=block.init_y(xin).detach().double()
    block.double()
    fn=block._make_ode_fn(xin.double())
    T=float(block.option_aca['t1']-block.option_aca['t0'])
    shape=y0.shape
    def vector(v):return (T*fn(torch.zeros((),device=v.device,dtype=v.dtype),v.reshape(shape))).flatten()
    v=y0.flatten()
    J=torch.autograd.functional.jacobian(vector,v,vectorize=True).detach()
    b=vector(v).detach()-J@v
    d=v.numel()
    aug=torch.zeros(d+1,d+1,device=v.device,dtype=v.dtype)
    aug[:d,:d]=J
    aug[:d,d]=b
    ref=(torch.matrix_exp(aug)@torch.cat([v,v.new_ones(1)]))[:d]
    C,H,W=shape[1:]
    P=H*W
    indices=torch.arange(d,device=v.device).reshape(C,P).T
    rows=J[indices]
    diag=torch.stack([rows[p,:,indices[p]] for p in range(P)])
    a=1/(2-2**(1/3));bb=1-2*a
    results=[]
    with torch.no_grad():
        for n in (5,10,20,40):
            for method in ('lie','strang','yoshida'):
                coeffs=[1.] if method!='yoshida' else [a,bb,a]
                caches={}
                for coeff in coeffs:
                    dt=coeff/n/(1 if method=='lie' else 2)
                    local=torch.zeros(P,2*C,2*C,device=v.device,dtype=v.dtype)
                    local[:,:C,:C]=diag*dt
                    local[:,:C,C:]=torch.eye(C,device=v.device,dtype=v.dtype)*dt
                    ee=torch.matrix_exp(local)
                    caches[coeff]=(ee[:,:C,:C],ee[:,:C,C:])
                y=v.clone()
                max_reverse_gain=0.
                trace=[]
                for iteration in range(n):
                    for coeff in coeffs:
                        E,M=caches[coeff]
                        if coeff<0:max_reverse_gain=float(torch.linalg.matrix_norm(E,ord=2).max())
                        orders=[range(P)] if method=='lie' else [range(P),range(P-1,-1,-1)]
                        for order in orders:
                            for p in order:
                                idx=indices[p]
                                u=y[idx].clone()
                                c=rows[p]@y+b[idx]-diag[p]@u
                                y[idx]=E[p]@u+M[p]@c
                        if method=='yoshida' and n in (5,10):
                            trace.append({'iteration':iteration,'coefficient':coeff,
                                          'state_norm':float(y.norm())})
                row={'method':method,'n':n,'relative_error':float((y-ref).norm()/ref.norm()),
                     'max_negative_local_operator_norm':max_reverse_gain,
                     'initial_norm':float(v.norm()),'reference_norm':float(ref.norm()),
                     'stage_trace':trace}
                results.append(row)
                print(json.dumps(row),flush=True)
                (ROOT/'results/yoshida_frozen_exact.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
