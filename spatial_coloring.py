"""Solver-independent structural dependency graph and coloring for FB/FF operators.

Assumes pointwise intervening transforms; uses conservative nonzero structural
support, independent of learned weight values and integration order.
"""
from itertools import combinations
from torch import nn

def sites(block):
    i,j,k,l=block
    return {(a,b) for a in range(i,k) for b in range(j,l)}


def output_shape(conv, shape):
    if not isinstance(conv,(nn.Conv2d,nn.ConvTranspose2d)) or conv.padding_mode!='zeros':
        raise ValueError('Only zero-padded Conv2d/ConvTranspose2d are supported')
    out=[]
    for axis,size in enumerate(shape):
        k,s,p,d=conv.kernel_size[axis],conv.stride[axis],conv.padding[axis],conv.dilation[axis]
        out.append((size-1)*s-2*p+d*(k-1)+conv.output_padding[axis]+1 if isinstance(conv,nn.ConvTranspose2d)
                   else (size+2*p-d*(k-1)-1)//s+1)
    return tuple(out)


def read_sites(conv, outputs, input_shape):
    """Exact structural spatial preimage of convolution output coordinates."""
    result=set()
    for i,j in outputs:
        for kh in range(conv.kernel_size[0]):
            for kw in range(conv.kernel_size[1]):
                if isinstance(conv,nn.ConvTranspose2d):
                    a=i+conv.padding[0]-kh*conv.dilation[0]
                    b=j+conv.padding[1]-kw*conv.dilation[1]
                    if a%conv.stride[0] or b%conv.stride[1]:continue
                    a//=conv.stride[0];b//=conv.stride[1]
                else:
                    a=i*conv.stride[0]-conv.padding[0]+kh*conv.dilation[0]
                    b=j*conv.stride[1]-conv.padding[1]+kw*conv.dilation[1]
                if 0<=a<input_shape[0] and 0<=b<input_shape[1]:result.add((a,b))
    return result


def dependency_regions(blocks, shape, fb, ff):
    middle=output_shape(fb,shape)
    if output_shape(ff,middle)!=tuple(shape):raise ValueError('FF(FB(state)) must preserve spatial shape')
    coverage=set()
    for b in blocks:
        coords=sites(b)
        if not coords or not coords<=sites((0,0,*shape)) or coverage & coords:raise ValueError('Invalid or overlapping blocks')
        coverage |= coords
    if coverage!=sites((0,0,*shape)):raise ValueError('Blocks must partition the state')
    # Include self for pointwise wrapper terms, even if kernels have no self tap.
    return [read_sites(fb,read_sites(ff,sites(b),middle),shape)|sites(b) for b in blocks]


def interaction_graph(blocks, dependencies):
    if len(blocks)!=len(dependencies):raise ValueError('Dependency count mismatch')
    graph=[set() for _ in blocks];writes=[sites(b) for b in blocks]
    for i,j in combinations(range(len(blocks)),2):
        if writes[i]&dependencies[j] or writes[j]&dependencies[i]:graph[i].add(j);graph[j].add(i)
    return graph


def validate_groups(blocks, dependencies, groups):
    ids=[i for group in groups for i in group]
    if sorted(ids)!=list(range(len(blocks))) or any(not g for g in groups):raise ValueError('Colors must partition block IDs exactly once')
    for group in groups:
        for i,j in combinations(group,2):
            if sites(blocks[i])&dependencies[j] or sites(blocks[j])&dependencies[i]:
                raise ValueError(f'Color contains interacting blocks {i}, {j}')


def color_blocks(blocks, dependencies):
    """Deterministic raster-ID greedy graph coloring; independent of integrator."""
    graph=interaction_graph(blocks,dependencies);colors=[];groups=[]
    for i in range(len(blocks)):
        forbidden={colors[j] for j in graph[i] if j<i};c=0
        while c in forbidden:c+=1
        colors.append(c)
        if c==len(groups):groups.append([])
        groups[c].append(i)
    validate_groups(blocks,dependencies,groups)
    return tuple(tuple(g) for g in groups)


