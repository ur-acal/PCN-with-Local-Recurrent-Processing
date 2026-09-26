"""Inference-only empirical lookup and per-edge current fusion.

No sampling, pulse generation, noise or state updates occur here.
Unsupported cases return None for the original implementation.
Enabled for supported toggle inference; GPU reductions are not bitwise stable.
"""
import torch
from tc_edge_inference import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _resistance(value, curve, GRID, LENGTH, LEFT, SLOPE,
                    K: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr,
                    STEPS: tl.constexpr):
        length = tl.load(LENGTH + curve)
        query = tl.minimum(tl.maximum(value, LOW), HIGH)
        query = tl.minimum(tl.maximum(query, tl.load(GRID + curve*K)),
                           tl.load(GRID + curve*K + length-1))
        lo = tl.full(value.shape, 0, tl.int32)
        hi = tl.full(value.shape, K, tl.int32)
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            point = tl.load(GRID + curve*K + mid, mid < K, other=float('inf'))
            right = point < query
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        idx = tl.minimum(tl.maximum(lo - 1, 0), length - 2)
        left_v = tl.load(GRID + curve*K + idx)
        left_r = tl.load(LEFT + curve*(K-1) + idx)
        slope = tl.load(SLOPE + curve*(K-1) + idx)
        return left_r + slope * (query - left_v)

    @triton.jit(do_not_specialize=['N'])
    def _lookup(V, INDEX, GRID, LENGTH, LEFT, SLOPE, OUT,
                N, B: tl.constexpr, K: tl.constexpr,
                VS0: tl.constexpr, VS1: tl.constexpr,
                LOW: tl.constexpr, HIGH: tl.constexpr,
                STEPS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, batch = i // B, i % B
        valid = i < N * B
        curve = tl.load(INDEX + row, valid, other=0)
        value = tl.load(V + row * VS0 + batch * VS1, valid, other=0.)
        resistance = _resistance(value, curve, GRID, LENGTH, LEFT, SLOPE,
                                 K, LOW, HIGH, STEPS)
        tl.store(OUT + i, resistance, valid)

    @triton.jit(do_not_specialize=['E'])
    def _edges(X, ACTIVE, COL, ROW, W, ASSIGN, GRID, LENGTH, LEFT, SLOPE, OUT,
               E, B: tl.constexpr, K: tl.constexpr,
               XS0: tl.constexpr, XS1: tl.constexpr, R: tl.constexpr,
               LOW: tl.constexpr, HIGH: tl.constexpr,
               STEPS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        active, batch = i//B, i%B
        valid = active<E
        edge = tl.load(ACTIVE+active, valid, other=0)
        weight = tl.load(W+edge, valid, other=0.)
        col = tl.load(COL+edge, valid, other=0)
        row = tl.load(ROW+edge, valid, other=0)
        curve = tl.load(ASSIGN+edge, valid, other=0)
        source = tl.load(X+col*XS0+batch*XS1, valid, other=0.)
        resistance = _resistance(source, curve, GRID, LENGTH, LEFT, SLOPE,
                                 K, LOW, HIGH, STEPS)
        contribution = tl.div_rn(weight*source*R, resistance)
        tl.atomic_add(OUT+row*B+batch, contribution, valid, sem='relaxed')


def _supported(module, voltage):
    if (triton is None or torch.is_grad_enabled() or module.training
            or not getattr(module, '_toggle_pulse_edges', False)
            or getattr(module, '_tc_measure_coupler_energy', False)
            or not voltage.is_cuda or voltage.dtype != torch.float32
            or voltage.ndim != 2):
        return False
    projection = module.proj_fn
    if projection is not None and not isinstance(projection, torch.nn.Hardtanh):
        return False
    grid = module.nonlinear_R_curve_bank_v_grid
    left = module.nonlinear_R_curve_bank_R_left
    slope = module.nonlinear_R_curve_bank_R_slope
    lengths = module.nonlinear_R_curve_bank_lengths
    if (any(t is None or t.device != voltage.device or not t.is_contiguous()
            for t in (grid, left, slope, lengths))
            or any(t.dtype != torch.float32 for t in (grid, left, slope))
            or left.shape != (grid.shape[0], grid.shape[1]-1)
            or slope.shape != left.shape):
        return False
    return True


def empirical_resistance(module, voltage, curve_index):
    if not getattr(module, '_toggle_fused_lookup', True) or not _supported(module, voltage):
        return None
    projection = module.proj_fn
    grid = module.nonlinear_R_curve_bank_v_grid
    left = module.nonlinear_R_curve_bank_R_left
    slope = module.nonlinear_R_curve_bank_R_slope
    lengths = module.nonlinear_R_curve_bank_lengths
    index = torch.as_tensor(curve_index, device=voltage.device,
                            dtype=torch.int64).reshape(-1).contiguous()
    if index.numel() != voltage.shape[0]:
        return None
    result = torch.empty_like(voltage, memory_format=torch.contiguous_format)
    if result.numel():
        _lookup[(triton.cdiv(result.numel(), 256),)](
            voltage, index, grid, lengths, left, slope, result,
            voltage.shape[0], voltage.shape[1], grid.shape[1],
            voltage.stride(0), voltage.stride(1),
            float(projection.min_val) if projection is not None else -float('inf'),
            float(projection.max_val) if projection is not None else float('inf'),
            grid.shape[1].bit_length(), 256, enable_fp_fusion=False)
    return result


def empirical_edge_forward(module, x, weight, nominal_R):
    """Fused per-physical-edge current; pulse values already include DTC/mismatch."""
    if (not getattr(module, '_toggle_fused_edges', True)
            or not _supported(module, x)
            or module.nonlinear_R_curve_sharing != 'per_coupler'
            or module.nonlinear_R_curve_sampling != 'empirical_with_replacement'
            or weight.layout != torch.sparse_csr):
        return None
    values = weight.values()
    if values.numel() != module.mat.values().numel():
        raise ValueError('Toggle fused pulse weights must match physical connectivity.')
    assignment = module.nonlinear_R_curve_assignment
    if (values.dtype != x.dtype or not values.is_contiguous()
            or assignment is None or not assignment.is_contiguous()):
        return None
    grid = module.nonlinear_R_curve_bank_v_grid
    projection = module.proj_fn
    out = x.new_zeros((module.mat.shape[0], x.shape[1]))
    active = torch.nonzero(values != 0, as_tuple=False).reshape(-1)
    if active.numel() and x.shape[1]:
        _edges[(triton.cdiv(active.numel()*x.shape[1], 256),)](
            x, active, module.mat.col_indices(), module.nonlinear_R_curve_row_ids,
            values, assignment, grid, module.nonlinear_R_curve_bank_lengths,
            module.nonlinear_R_curve_bank_R_left, module.nonlinear_R_curve_bank_R_slope,
            out, active.numel(), x.shape[1], grid.shape[1],
            x.stride(0), x.stride(1), float(nominal_R),
            float(projection.min_val) if projection is not None else -float('inf'),
            float(projection.max_val) if projection is not None else float('inf'),
            grid.shape[1].bit_length(), 256, enable_fp_fusion=False)
    return out
