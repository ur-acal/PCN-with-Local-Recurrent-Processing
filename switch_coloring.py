"""Geometry-derived independent block groups and optional colored schedules.

Dependency analysis assumes spatially pointwise activation/physical transforms
between FB and FF; it rejects unsupported convolution geometry. It deliberately
uses structural support (including zero-weight taps), not data-dependent zeros.
No training or existing switching implementation is modified.
"""
from itertools import combinations
import torch
from torch import nn
from switch import ODEXInitFFFBPixelSwitchExplicit, ODEXInitFFFBPixelSwitchStrang


from spatial_coloring import sites, output_shape, read_sites, dependency_regions, interaction_graph, validate_groups, color_blocks

def plan_for(operator, shape):
    blocks=tuple(operator.iter_spatial_blocks(*shape))
    deps=dependency_regions(blocks,shape,operator.FBconv,operator.FFconv)
    groups=color_blocks(blocks,deps)
    return blocks,deps,groups


def installed_rhs(operator,x,y,block):
    names=('_active_spatial_block','_explicit_active_pixel','_strang_active_pixel')
    saved={n:getattr(operator,n) for n in names if hasattr(operator,n)}
    try:
        operator._active_spatial_block=block
        i,j,k,l=block
        fn=operator._make_fixed_pixel_ode_fn(x,i,j) if k-i==l-j==1 else operator._make_ode_fn(x)
        return fn(0.,y)
    finally:
        for n in names:
            if n in saved:setattr(operator,n,saved[n])
            elif hasattr(operator,n):delattr(operator,n)


@torch.no_grad()
def validate_numerical_independence(operator,x,y,atol=1e-10,rtol=1e-8,max_pairs_per_color=None):
    """Every same-color pair, in both directions, through installed wrapped RHS."""
    blocks,deps,groups=plan_for(operator,y.shape[-2:]);checks=0
    for group in groups:
        for pair_index,(i,j) in enumerate(combinations(group,2)):
            if max_pairs_per_color is not None and pair_index>=max_pairs_per_color:break
            for source,target in ((i,j),(j,i)):
                z=y.clone();a,b,c,d=blocks[source]
                z[:,:,a:c,b:d]+=0.137+0.071*torch.sin(z[:,:,a:c,b:d])
                a,b,c,d=blocks[target]
                before=installed_rhs(operator,x,y,blocks[target])[:,:,a:c,b:d]
                after=installed_rhs(operator,x,z,blocks[target])[:,:,a:c,b:d]
                torch.testing.assert_close(after,before,atol=atol,rtol=rtol)
                checks+=1
    return checks


def color_flow(operator,x,y,blocks,group,duration):
    """All local flows read the same snapshot; commit disjoint outputs together."""
    pending=[]
    for index in group:
        block=blocks[index];i,j,k,l=block
        out=operator.local_block_flow(x,y,block,duration)
        pending.append((block,out[:,:,i:k,j:l].clone()))
    result=y.clone()
    for (i,j,k,l),value in pending:result[:,:,i:k,j:l]=value
    return result


class ColorSchedule:
    symmetric=False
    def color_plan(self,shape):
        # Geometry is fixed after model construction; cache per feature-map shape.
        cache=getattr(self,'_color_plans',None)
        if cache is None:self._color_plans={};cache=self._color_plans
        key=tuple(shape)
        if key not in cache:cache[key]=plan_for(self,key)
        return cache[key]

    def _run_explicit_pixel_switch(self,x,full_traj=False):
        y=self.init_y(x);self.integration_time=self.integration_time.type_as(x)
        t0,t1,total=self._get_total_horizon();blocks,deps,groups=self.color_plan(y.shape[-2:])
        validate_groups(blocks,deps,groups)
        # Keep the central halves separate: no change to legacy subsolve durations.
        order=list(groups)+list(reversed(groups)) if self.symmetric else list(groups)
        duration=total/self.n_iters/(2 if self.symmetric else 1)
        states=[y];times=[y.new_tensor(t0)]
        for iteration in range(self.n_iters):
            for group in order:y=color_flow(self,x,y,blocks,group,duration)
            if full_traj:states.append(y);times.append(y.new_tensor(t0+(iteration+1)*total/self.n_iters))
        return (torch.stack(states),torch.stack(times)) if full_traj else y


class ColorLie(ColorSchedule,ODEXInitFFFBPixelSwitchExplicit):
    pass


class ColorStrang(ColorSchedule,ODEXInitFFFBPixelSwitchStrang):
    symmetric=True
