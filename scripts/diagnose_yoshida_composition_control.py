"""Independent exact-flow check of production Lie/Strang/Yoshida composition."""
import json
import sys
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import switch

torch.set_default_dtype(torch.float64)
A=torch.tensor([[-.7,.4,.2],[-.3,-.5,.6],[.5,-.2,-.4]])
x=torch.tensor([.2,-.1,.3]).reshape(1,1,1,3)
reference=(torch.matrix_exp(A)@x.flatten()).reshape_as(x)
rows=[]
oldsolve=switch.aca_ode_solve
def exact_solve(fn,y,options):
    basis=torch.eye(3).reshape(3,1,1,3)
    mat=fn(torch.tensor(0.),basis).reshape(3,3).T
    result=(torch.matrix_exp(mat*float(options['t1']-options['t0']))@y.flatten()).reshape_as(y)
    return torch.stack([y,result])
switch.aca_ode_solve=exact_solve
try:
    for name,cls in [('lie',switch.ODEXInitFFFBPixelSwitchExplicit),
                     ('strang',switch.ODEXInitFFFBPixelSwitchStrang),
                     ('yoshida',switch.ODEXInitFFFBPixelSwitchYoshida4)]:
        block=cls.__new__(cls)
        nn.Module.__init__(block)
        block.scale_RHS=False
        block.integration_time=torch.tensor([0.,1.])
        block.option_aca={'t0':torch.tensor(0.),'t1':torch.tensor(1.),'h':None}
        block.init_y=lambda y:y.clone()
        def factory(inp):
            def fn(t,y):
                _,p=block._time_to_active_pixel(t,1,3)
                full=y.reshape(-1,3)@A.T
                out=torch.zeros_like(full)
                out[:,p]=full[:,p]
                return out.reshape_as(y)
            return fn
        block._make_ode_fn=factory
        for n in (1,2,5,10,20,40):
            block.n_iters=n
            out=block._run_explicit_pixel_switch(x)
            rows.append({'method':name,'n':n,'relative_error':float((out-reference).norm()/reference.norm())})
finally:
    switch.aca_ode_solve=oldsolve
path=ROOT/'results/yoshida_exact_composition_control.json'
path.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
