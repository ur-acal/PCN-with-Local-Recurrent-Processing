"""Shared training input correction for TC and opt-in averaged toggle FT.

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
                 BLOCK: tl.constexpr, STRICT_RAIL: tl.constexpr = False,
                 PACKED: tl.constexpr = False, COUNT: tl.constexpr = 1,
                 PER_SAMPLE: tl.constexpr = 0, GRID_STRIDE: tl.constexpr = 0,
                 TABLE_STRIDE: tl.constexpr = 0, SLOPE_STRIDE: tl.constexpr = 0,
                 LENGTHS=None):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        if PACKED:
            level = tl.program_id(1)
            G = G+level*GRID_STRIDE
            T = T+level*TABLE_STRIDE
            S = S+level*SLOPE_STRIDE
            R = R+level
            length = K if LENGTHS is None else tl.load(LENGTHS+level).to(tl.int32)
            output_i = ((i//PER_SAMPLE)*COUNT+level)*PER_SAMPLE+i%PER_SAMPLE
        else:
            length = K
            output_i = i
        x = tl.load(X+i, i<N, other=0)
        g0, g1 = tl.load(G), tl.load(G+length-1)
        q0 = tl.minimum(tl.maximum(x, -RAIL), RAIL)
        q = tl.minimum(tl.maximum(q0, g0), g1)
        # torch.bucketize(right=False): first grid value >= query.
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), 0, tl.int32)+length
        for _ in range(STEPS):
            mid = (lo+hi)//2
            v = tl.load(G+mid, mid<length, other=float('inf'))
            right = v < q
            lo = tl.where(right, mid+1, lo)
            hi = tl.where(right, hi, mid)
        idx = tl.minimum(tl.maximum(lo-1, 0), length-2)
        slope = tl.load(S+idx)
        raw = tl.load(T+idx)+slope*(q-tl.load(G+idx))
        r = tl.maximum(raw, FLOOR)
        nominal = tl.load(R)
        y = tl.div_rn(x*nominal, r)
        y = tl.where(x != x, float('nan'), y)
        tl.store(Y+output_i, y, i<N)
        if GRAD:
            if STRICT_RAIL:
                rail_active = (x>-RAIL)&(x<RAIL)
            else:
                rail_active = (x>=-RAIL)&(x<=RAIL)
            active = rail_active&(q0>=g0)&(q0<=g1)&(raw>=FLOOR)
            derivative = tl.div_rn(nominal,r)-tl.where(active, tl.div_rn(y,r)*slope, 0.)
            derivative = tl.where(x != x, float('nan'), derivative)
            tl.store(D+output_i, derivative, i<N)

    @triton.jit
    def _correct_packed_backward(GRAD, D, OUT, N: tl.constexpr, COUNT: tl.constexpr,
                                 PER_SAMPLE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        result = tl.full((BLOCK,), 0., tl.float32)
        # Explicit fixed-order accumulation, matching the reverse traversal of
        # the separate correction branches rather than a parallel tree reduction.
        for level in tl.static_range(COUNT-1, -1, -1):
            j = ((i//PER_SAMPLE)*COUNT+level)*PER_SAMPLE+i%PER_SAMPLE
            result = result+tl.load(GRAD+j, i<N, other=0)*tl.load(D+j, i<N, other=0)
        tl.store(OUT+i, result, i<N)


class _Correction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid, table, slope, nominal, rail, floor, strict_rail=False):
        x = x.contiguous()
        output = torch.empty_like(x)
        need_grad = ctx.needs_input_grad[0]
        derivative = torch.empty_like(x) if need_grad else output
        _correct[(triton.cdiv(x.numel(),256),)](
            x, grid, table, slope, nominal, output, derivative,
            x.numel(), grid.numel(), rail, floor, need_grad, grid.numel().bit_length(), 256,
            STRICT_RAIL=strict_rail,
            enable_fp_fusion=False)
        if need_grad:
            ctx.save_for_backward(derivative)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return (grad*ctx.saved_tensors[0],) + (None,) * (len(ctx.needs_input_grad)-1)


class _BatchedCorrection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid, table, slope, nominal, lengths, rail, floor=0.):
        x = x.contiguous()
        count = nominal.numel()
        output = x.new_empty(x.shape[0], count*x.shape[1], *x.shape[2:])
        need_grad = ctx.needs_input_grad[0]
        derivative = torch.empty_like(output) if need_grad else output
        per_sample = x.numel()//x.shape[0]
        _correct[(triton.cdiv(x.numel(),256),count)](
            x,grid,table,slope,nominal,output,derivative,
            x.numel(),grid.shape[1],rail,float(floor),need_grad,grid.shape[1].bit_length(),256,
            PACKED=True,COUNT=count,PER_SAMPLE=per_sample,GRID_STRIDE=grid.stride(0),
            TABLE_STRIDE=table.stride(0),SLOPE_STRIDE=slope.stride(0),LENGTHS=lengths,
            enable_fp_fusion=False)
        if need_grad:
            ctx.save_for_backward(derivative)
            ctx.input_shape = x.shape
            ctx.count = count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        derivative, = ctx.saved_tensors
        output = grad.new_empty(ctx.input_shape)
        _correct_packed_backward[(triton.cdiv(output.numel(),256),)](
            grad.contiguous(),derivative,output,output.numel(),ctx.count,
            output.numel()//output.shape[0],256,enable_fp_fusion=False)
        return (output,) + (None,)*(len(ctx.needs_input_grad)-1)


def batched_empirical_correction(x, curves, programmed_resistances, rail, *, floor=0.):
    """Return packed corrections, or None to retain the existing scalar path."""
    if triton is None or not x.is_cuda or x.dtype != torch.float32 or x.ndim != 4 or not x.numel():
        return None
    if hasattr(curves, 'prepare_batched'):
        if any(t.requires_grad for t in (programmed_resistances,
                *(curves.bank[name] for name in ('v_grid','R_left','R_slope')))):
            return None
        grid,left,slope,nominal,lengths = curves.prepare_batched(x,programmed_resistances)
    else:
        grid,left,slope,nominal,lengths = curves
    if any(t.requires_grad for t in (grid,left,slope,nominal)):
        return None
    return _BatchedCorrection.apply(x,grid,left,slope,nominal,lengths,float(rail),float(floor))


def shared_correction(x, curve, grid, rail, floor):
    grid, table, slope, nominal = curve.prepare(grid, x, floor)
    return empirical_correction(x,grid,table[:-1],slope,nominal,rail,floor=floor)


def empirical_correction(x, grid, left, slope, nominal, rail, *, fused=True,
                         strict_rail=False, floor=0.):
    """Reuse the first-order correction kernel on one prepared native-grid curve.

    Measured empirical curves are already positive. Use their stored slopes
    directly rather than reconstructing/regridding resistance knots.
    """
    if (fused and triton is not None and x.is_cuda and x.dtype == torch.float32
            and all(t.device == x.device and t.dtype == x.dtype and t.is_contiguous()
                    and not t.requires_grad for t in (grid,left,slope,nominal))):
        return _Correction.apply(x,grid,left,slope,nominal,float(rail),float(floor),strict_rail)
    # Hardtanh and clamp differ at the rail endpoints in backward.
    query = (torch.nn.functional.hardtanh(x, -rail, rail) if strict_rail
             else x.clamp(-rail,rail)).clamp(grid[0],grid[-1])
    idx = (torch.bucketize(query.contiguous(),grid)-1).clamp(0,grid.numel()-2)
    resistance = left[idx]+slope[idx]*(query-grid[idx])
    if floor:
        resistance = resistance.clamp_min(floor)
    return x*nominal/resistance


def toggle_empirical_correction(x, package, curve_index, cache):
    """Return a fused result, or None so toggle executes its literal reference.

    Only identity or symmetric native Hardtanh projection is supported. Selected
    constants are prepared once per realization; this function never samples.
    Curve data is trusted as in the reference path; unsupported kernels fall back.
    """
    import math
    if triton is None or not x.is_cuda or x.dtype != torch.float32:
        return None
    projection = package.get('proj_fn')
    strict_rail = False
    rail = float('inf')
    if projection is not None:
        if type(projection) is not torch.nn.Hardtanh or projection.inplace:
            return None
        rail = float(projection.max_val)
        if not math.isfinite(rail) or rail <= 0 or float(projection.min_val) != -rail:
            return None
        strict_rail = True
    key = (id(package), curve_index, x.device, x.dtype)
    if key not in cache:
        length = int(package['lengths'][curve_index].item())
        if length < 2:
            return None
        grid = package['v_grid'][curve_index, :length].to(x).contiguous()
        left = package['R_left'][curve_index, :length-1].to(x).contiguous()
        slope = package['R_slope'][curve_index, :length-1].to(x).contiguous()
        nominal = x.new_tensor(package['R'])
        if any(t.requires_grad for t in (grid, left, slope)):
            return None
        cache[key] = (grid, left, slope, nominal)
    return empirical_correction(x, *cache[key], rail, strict_rail=strict_rail)
