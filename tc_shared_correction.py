"""Fused TC shared-curve input correction; no changes to toggle or per-edge paths.

CUDA float32 uses one kernel, saving one local-derivative tensor for first-order
backward. Other devices/dtypes use the same differentiable PyTorch expression.
Curve constants are sampled without gradients; no new STE is introduced.
"""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _correct(X, G, T, S, R, Y, D, N: tl.constexpr, K: tl.constexpr,
                 RAIL: tl.constexpr, FLOOR: tl.constexpr, GRAD: tl.constexpr, STEPS: tl.constexpr,
                 BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        x = tl.load(X+i, i<N, other=0)
        g0, g1 = tl.load(G), tl.load(G+K-1)
        q0 = tl.minimum(tl.maximum(x, -RAIL), RAIL)
        q = tl.minimum(tl.maximum(q0, g0), g1)
        # torch.bucketize(right=False): first grid value >= query.
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), K, tl.int32)
        for _ in range(STEPS):
            mid = (lo+hi)//2
            v = tl.load(G+mid, mid<K, other=float('inf'))
            right = v < q
            lo = tl.where(right, mid+1, lo)
            hi = tl.where(right, hi, mid)
        idx = tl.minimum(tl.maximum(lo-1, 0), K-2)
        slope = tl.load(S+idx)
        raw = tl.load(T+idx)+slope*(q-tl.load(G+idx))
        r = tl.maximum(raw, FLOOR)
        nominal = tl.load(R)
        y = tl.div_rn(x*nominal, r)
        y = tl.where(x != x, float('nan'), y)
        tl.store(Y+i, y, i<N)
        if GRAD:
            active = (x>=-RAIL)&(x<=RAIL)&(q0>=g0)&(q0<=g1)&(raw>=FLOOR)
            derivative = tl.div_rn(nominal,r)-tl.where(active, tl.div_rn(y,r)*slope, 0.)
            derivative = tl.where(x != x, float('nan'), derivative)
            tl.store(D+i, derivative, i<N)


class _Correction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid, table, slope, nominal, rail, floor):
        x = x.contiguous()
        output = torch.empty_like(x)
        need_grad = ctx.needs_input_grad[0]
        derivative = torch.empty_like(x) if need_grad else output
        _correct[(triton.cdiv(x.numel(),256),)](
            x, grid, table, slope, nominal, output, derivative,
            x.numel(), grid.numel(), rail, floor, need_grad, grid.numel().bit_length(), 256,
            enable_fp_fusion=False)
        if need_grad:
            ctx.save_for_backward(derivative)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return grad*ctx.saved_tensors[0], None, None, None, None, None, None


def shared_correction(x, curve, grid, rail, floor):
    grid, table, slope, nominal = curve.prepare(grid, x, floor)
    if (triton is not None and x.is_cuda and x.dtype == torch.float32
            and not any(t.requires_grad for t in (grid,table,slope,nominal))):
        return _Correction.apply(x,grid,table,slope,nominal,float(rail),float(floor))
    query = x.clamp(-rail,rail).clamp(grid[0],grid[-1])
    idx = (torch.bucketize(query,grid)-1).clamp(0,grid.numel()-2)
    resistance = table[idx]+slope[idx]*(query-grid[idx])
    return x*nominal/resistance.clamp_min(floor)
