"""Inference-only Gaussian per-edge interpolation/current/reduction fusion.

Unsupported devices, dtypes, projections and energy observers use the original
PyTorch implementation. No changes to empirical banks, toggle, or QAT gradients.
"""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _edges(X, COL, ROW, W, GRID, CURVES, CURVE_ROW, OUT,
               E: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
               XS0: tl.constexpr, XS1: tl.constexpr, R: tl.constexpr,
               LOW: tl.constexpr, HIGH: tl.constexpr, STEPS: tl.constexpr,
               BLOCK: tl.constexpr, COMPACT: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        edge, batch = i//B, i%B
        weight = tl.load(W+edge, edge<E, other=0.)
        valid = (edge<E) & (weight != 0.)
        col = tl.load(COL+edge, valid, other=0)
        row = tl.load(ROW+edge, valid, other=0)
        x = tl.load(X+col*XS0+batch*XS1, valid, other=0.)
        query = tl.minimum(tl.maximum(x, LOW), HIGH)
        query = tl.minimum(tl.maximum(query, tl.load(GRID)), tl.load(GRID+K-1))
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), K, tl.int32)
        for _ in range(STEPS):
            mid = (lo+hi)//2
            value = tl.load(GRID+mid, mid<K, other=float('inf'))
            right = value < query
            lo = tl.where(right, mid+1, lo)
            hi = tl.where(right, hi, mid)
        idx = tl.minimum(tl.maximum(lo-1, 0), K-2)
        curve_row = tl.load(CURVE_ROW+edge, valid, other=0) if COMPACT else edge
        left = tl.load(CURVES+curve_row*K+idx, valid, other=1.)
        right = tl.load(CURVES+curve_row*K+idx+1, valid, other=1.)
        vl, vr = tl.load(GRID+idx), tl.load(GRID+idx+1)
        fraction = tl.div_rn(query-vl, vr-vl)
        resistance = R * (left + fraction*(right-left))
        contribution = tl.div_rn(weight*x*R, resistance)
        tl.atomic_add(OUT+row*B+batch, contribution, valid, sem='relaxed')


def gaussian_edge_forward(module, x, weight, nominal_R):
    """Return None to request the reference path; otherwise the summed current."""
    curves = module.nonlinear_R_curve_gaussian_R_normalized
    projection = module.proj_fn
    if (triton is None or torch.is_grad_enabled() or module.training or not x.is_cuda
            or x.dtype != torch.float32 or curves is None
            or module.nonlinear_R_curve_sharing != 'per_coupler'
            or getattr(module, '_tc_curve_package', None) is None
            or getattr(module, '_tc_measure_coupler_energy', False)
            or not getattr(module, '_tc_fused_edges', True)
            or (projection is not None and not isinstance(projection, torch.nn.Hardtanh))):
        return None
    grid = module.nonlinear_R_curve_gaussian_v_grid
    if curves.dtype != x.dtype or not curves.is_contiguous() or not grid.is_contiguous():
        return None
    values = weight.values()
    if values.numel() != module.mat.values().numel():
        raise ValueError('TC fused edge weights must match physical connectivity.')
    out = x.new_zeros((module.mat.shape[0], x.shape[1]))
    curve_rows = getattr(module, '_tc_curve_row_index', None)
    if values.numel():
        _edges[(triton.cdiv(values.numel()*x.shape[1], 256),)](
            x, module.mat.col_indices(), module.nonlinear_R_curve_row_ids,
            values, grid, curves, curve_rows if curve_rows is not None else module.mat.col_indices(),
            out, values.numel(), x.shape[1], grid.numel(),
            x.stride(0), x.stride(1), float(nominal_R),
            float(projection.min_val) if projection is not None else -float('inf'),
            float(projection.max_val) if projection is not None else float('inf'),
            grid.numel().bit_length(), 256, curve_rows is not None, enable_fp_fusion=False)
    return out
