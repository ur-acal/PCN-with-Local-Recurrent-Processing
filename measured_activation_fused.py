"""Fused CUDA evaluation for fixed measured piecewise-linear activations.

The characterized grids and curve values are fixed buffers.  CUDA float32
therefore needs one elementwise interpolation kernel in each of forward and
backward.  Unsupported devices and dtypes continue to use the ordinary
PyTorch implementation in :mod:`measured_activation`.
"""

import warnings

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    from triton.language import math as tl_math
except ImportError:
    triton = None


_warned_cuda_fallback = False


if triton is not None:
    @triton.jit
    def _piecewise_linear_forward(
            X, GRID, VALUES, CURVE_INDICES, V_DD, SCALE, GRID_SPACING,
            PULLBACK, Y, N: tl.constexpr, K: tl.constexpr,
            INDEX_COUNT: tl.constexpr, UNIFORM: tl.constexpr,
            BANKED: tl.constexpr, HAS_PULLBACK: tl.constexpr,
            SEARCH_STEPS: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0)

        v_dd = tl.load(V_DD)
        scale = tl.load(SCALE)
        pullback = tl.load(PULLBACK) if HAS_PULLBACK else 1.0
        x_scaled = x * pullback if HAS_PULLBACK else x
        raw_characterized = tl_math.div_rn(x_scaled, scale)
        grid_min = tl.load(GRID)
        grid_max = tl.load(GRID + K - 1)
        characterized = tl.minimum(
            tl.maximum(raw_characterized, grid_min), grid_max)

        if UNIFORM:
            spacing = tl.load(GRID_SPACING)
            position = tl_math.div_rn(
                characterized - grid_min, spacing)
            # characterized is clamped to the grid, so truncation is floor.
            left = position.to(tl.int32)
        else:
            # torch.searchsorted(grid, value, right=True) - 1
            lo = tl.full((BLOCK,), 0, tl.int32)
            hi = tl.full((BLOCK,), K, tl.int32)
            for _ in range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                grid_value = tl.load(
                    GRID + mid, mask=mid < K, other=float("inf"))
                move_right = grid_value <= characterized
                lo = tl.where(move_right, mid + 1, lo)
                hi = tl.where(move_right, hi, mid)
            left = lo - 1
        left = tl.minimum(tl.maximum(left, 0), K - 2)

        if BANKED:
            curve_index = tl.load(
                CURVE_INDICES + offsets % INDEX_COUNT,
                mask=mask, other=0).to(tl.int64)
            value_offset = curve_index * K + left
        else:
            value_offset = left
        x_left = tl.load(GRID + left)
        x_right = tl.load(GRID + left + 1)
        y_left = tl.load(VALUES + value_offset)
        y_right = tl.load(VALUES + value_offset + 1)
        fraction = tl_math.div_rn(
            characterized - x_left, x_right - x_left)
        y_characterized = y_left + fraction * (y_right - y_left)
        output_before_clamp = scale * y_characterized
        output = tl.minimum(tl.maximum(output_before_clamp, -v_dd), v_dd)
        if HAS_PULLBACK:
            output = tl_math.div_rn(output, pullback)
        output = tl.where(x != x, float("nan"), output)
        tl.store(Y + offsets, output, mask=mask)

    @triton.jit
    def _piecewise_linear_backward(
            GRAD, X, GRID, VALUES, CURVE_INDICES, V_DD, SCALE,
            GRID_SPACING, PULLBACK, GRAD_X,
            N: tl.constexpr, K: tl.constexpr,
            INDEX_COUNT: tl.constexpr, UNIFORM: tl.constexpr,
            BANKED: tl.constexpr, HAS_PULLBACK: tl.constexpr,
            SEARCH_STEPS: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        grad = tl.load(GRAD + offsets, mask=mask, other=0.0)
        x = tl.load(X + offsets, mask=mask, other=0.0)

        v_dd = tl.load(V_DD)
        scale = tl.load(SCALE)
        pullback = tl.load(PULLBACK) if HAS_PULLBACK else 1.0
        x_scaled = x * pullback if HAS_PULLBACK else x
        raw_characterized = tl_math.div_rn(x_scaled, scale)
        grid_min = tl.load(GRID)
        grid_max = tl.load(GRID + K - 1)
        characterized = tl.minimum(
            tl.maximum(raw_characterized, grid_min), grid_max)

        if UNIFORM:
            spacing = tl.load(GRID_SPACING)
            position = tl_math.div_rn(
                characterized - grid_min, spacing)
            left = position.to(tl.int32)
        else:
            lo = tl.full((BLOCK,), 0, tl.int32)
            hi = tl.full((BLOCK,), K, tl.int32)
            for _ in range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                grid_value = tl.load(
                    GRID + mid, mask=mid < K, other=float("inf"))
                move_right = grid_value <= characterized
                lo = tl.where(move_right, mid + 1, lo)
                hi = tl.where(move_right, hi, mid)
            left = lo - 1
        left = tl.minimum(tl.maximum(left, 0), K - 2)

        if BANKED:
            curve_index = tl.load(
                CURVE_INDICES + offsets % INDEX_COUNT,
                mask=mask, other=0).to(tl.int64)
            value_offset = curve_index * K + left
        else:
            value_offset = left
        x_left = tl.load(GRID + left)
        x_right = tl.load(GRID + left + 1)
        y_left = tl.load(VALUES + value_offset)
        y_right = tl.load(VALUES + value_offset + 1)
        fraction = tl_math.div_rn(
            characterized - x_left, x_right - x_left)
        y_characterized = y_left + fraction * (y_right - y_left)
        output_before_clamp = scale * y_characterized

        # Preserve the operation order of the unfused PyTorch autograd graph:
        # pullback division, physical-output clamp, output scale,
        # interpolation, input clamp, input scale, and finally the coordinate
        # pullback.
        output_active = ((output_before_clamp >= -v_dd) &
                         (output_before_clamp <= v_dd))
        if HAS_PULLBACK:
            grad = tl_math.div_rn(grad, pullback)
        grad = tl.where(output_active, grad, 0.0)
        grad = grad * scale
        grad = grad * (y_right - y_left)
        grad = tl_math.div_rn(grad, x_right - x_left)
        input_active = ((raw_characterized >= grid_min) &
                        (raw_characterized <= grid_max))
        grad = tl.where(input_active, grad, 0.0)
        grad = tl_math.div_rn(grad, scale)
        if HAS_PULLBACK:
            grad = grad * pullback
        # torch.clamp propagates a NaN output but gives that input zero
        # gradient; preserve the same backward behavior in the fused path.
        grad = tl.where(x != x, 0.0, grad)
        tl.store(GRAD_X + offsets, grad, mask=mask)


    class _PiecewiseLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, grid, values, curve_indices, v_dd, scale,
                    grid_spacing, pullback, uniform, banked, has_pullback):
            x_contiguous = x.contiguous()
            output = torch.empty_like(x_contiguous)
            block = 256
            _piecewise_linear_forward[(triton.cdiv(x_contiguous.numel(), block),)](
                x_contiguous, grid, values, curve_indices, v_dd, scale,
                grid_spacing, pullback, output,
                x_contiguous.numel(), grid.numel(), curve_indices.numel(),
                uniform, banked, has_pullback,
                grid.numel().bit_length(), block,
                enable_fp_fusion=False)
            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(
                    x_contiguous, grid, values, curve_indices, v_dd, scale,
                    grid_spacing, pullback)
                ctx.uniform = uniform
                ctx.banked = banked
                ctx.has_pullback = has_pullback
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            (x, grid, values, curve_indices, v_dd, scale,
             grid_spacing, pullback) = ctx.saved_tensors
            grad_contiguous = grad_output.contiguous()
            grad_x = torch.empty_like(grad_contiguous)
            block = 256
            _piecewise_linear_backward[(triton.cdiv(grad_x.numel(), block),)](
                grad_contiguous, x, grid, values, curve_indices, v_dd, scale,
                grid_spacing, pullback, grad_x,
                grad_x.numel(), grid.numel(), curve_indices.numel(),
                ctx.uniform, ctx.banked, ctx.has_pullback,
                grid.numel().bit_length(), block,
                enable_fp_fusion=False)
            return grad_x, None, None, None, None, None, None, None, None, None, None


def fused_piecewise_linear(
        x, grid, values, curve_indices, v_dd, scale, grid_spacing,
        pullback, uniform):
    """Return a fused result, or ``None`` when the Triton path is inapplicable."""
    global _warned_cuda_fallback
    banked = curve_indices is not None
    has_pullback = pullback is not None
    fixed_tensors = (grid, values, v_dd, scale, grid_spacing)
    if (triton is None and x.is_cuda and x.dtype == torch.float32 and
            not _warned_cuda_fallback):
        warnings.warn(
            "Triton is unavailable; measured piecewise-linear activation is "
            "using the unfused PyTorch CUDA path.", RuntimeWarning)
        _warned_cuda_fallback = True
    if (triton is None or not x.is_cuda or x.dtype != torch.float32 or
            x.numel() == 0 or
            any(t.device != x.device or t.dtype != x.dtype for t in fixed_tensors) or
            any(t.requires_grad for t in fixed_tensors)):
        return None
    if banked:
        if (curve_indices.device != x.device or
                curve_indices.dtype not in (torch.int32, torch.int64)):
            return None
    else:
        # The kernel argument is compiled away when BANKED is false.
        curve_indices = grid
    if has_pullback:
        if (pullback.device != x.device or pullback.dtype != x.dtype or
                pullback.requires_grad or pullback.numel() != 1):
            return None
    else:
        # The kernel argument is compiled away when HAS_PULLBACK is false.
        pullback = v_dd
    return _PiecewiseLinear.apply(
        x, grid, values, curve_indices, v_dd, scale, grid_spacing,
        pullback, bool(uniform), banked, has_pullback)
