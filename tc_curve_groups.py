"""Solve-local top-level plus histogram-shared remainder TC approximation."""
from dataclasses import dataclass

import torch


def validate_top_n(value, levels=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0
            or (levels is not None and value > levels)):
        raise ValueError('tc_curve_top_n must be an integer between 0 and the number of nonzero levels.')
    return value


@dataclass
class CurveGroups:
    corrections: object
    nominal: torch.Tensor
    mask: torch.Tensor | None
    histogram: torch.Tensor
    top_codes: torch.Tensor
    representative_codes: torch.Tensor
    floor: float = 0.
    training_draw: bool = False


@torch.no_grad()
def sample_groups(package, codes, top_n, *, generator=None, factor=None, sampling='histogram'):
    """Independent per convolution. Constants/masks are held through RHS replay.

    Zero-code weights are assigned to the remainder to preserve the shared
    path's QAT STE gradients, but never participate in histogram sampling.
    """
    from tc_empirical import EmpiricalLevels, empirical_bank_on
    validate_top_n(top_n, package.programmed_resistances.numel())
    if sampling not in ('histogram', 'uniform') or (top_n and sampling != 'histogram'):
        raise ValueError('Top-n groups require histogram sampling (uniform is supported only at n=0).')
    # Like the original shared sampler, sample on the package/generator device.
    # Plain dataclass tensors and Generators do not follow a wrapper's .to().
    weight_device = codes.device
    codes = codes.detach().to(device=package.means.device, dtype=torch.long)
    count = package.programmed_resistances.numel()
    histogram = torch.bincount(codes.flatten(), minlength=count+1)[1:]
    if top_n == 0:
        # One shared curve needs neither ranking nor a weight-sized group mask.
        if not torch.any(histogram):
            return None
        probabilities = (histogram.float() if sampling == 'histogram'
                         else torch.ones_like(histogram, dtype=torch.float))
        selected = torch.multinomial(probabilities, 1, generator=generator)
        top = codes.new_empty(0)
        mask = None
    else:
        ranked = torch.argsort(histogram, descending=True, stable=True)
        active = ranked[histogram[ranked] > 0]
        if not active.numel():
            return None
        top = active[:top_n]
        remaining = histogram.clone()
        remaining[top] = 0
        representative = (torch.multinomial(remaining.float(), 1, generator=generator)
                          if top.numel() < active.numel() else codes.new_empty(0))
        selected = torch.cat((top, representative))
        separate = codes.unsqueeze(0) == (top+1).reshape(-1, *([1]*codes.ndim))
        if representative.numel():
            mask = torch.cat((separate, (~separate.any(0)).unsqueeze(0)))
        else:
            mask = separate
            mask[-1] |= codes == 0  # No residual group; retain zero-code STE on the last curve.
        mask = mask.to(device=weight_device)
    nominal = package.programmed_resistances[selected]
    if package.empirical_bank is not None:
        bank = empirical_bank_on(package.empirical_bank, package.means.device, package.means.dtype)
        if isinstance(factor, str):
            allowed = bank['corner_indices'][factor]
            positions = torch.randint(allowed.shape[1], selected.shape,
                                      device=selected.device, generator=generator)
            indices = allowed[selected, positions]
        else:
            draws = bank['draws_per_code']
            indices = selected*draws + torch.randint(draws, selected.shape,
                device=selected.device, generator=generator)
        corrections = EmpiricalLevels(bank, indices)
    else:
        # Reuse Gaussian generation and the existing native-grid correction kernel.
        resistance = package.sample(selected+1, generator=generator, factor=factor)
        corrections = EmpiricalLevels.from_gaussian(package.v_grid, resistance)
    return CurveGroups(corrections, nominal, mask, histogram, top+1, selected+1,
                       package.floor_ohms if package.empirical_bank is None else 0.)


def group_convolution(owner, module, source, draw):
    """Common dense backend; samplers never participate in correction kernels."""
    from tc_empirical import EmpiricalLevels, grouped_convolution
    from tc_nonidealities import TCSharedCurve
    from tc_shared_correction import batched_empirical_correction, empirical_correction
    package = getattr(owner, '_tc_curve_package', None)
    grouped = getattr(owner, '_tc_grouped_convolution',
        isinstance(draw, CurveGroups) or getattr(owner, '_tc_conv_method', 'loop') == 'grouped')
    floor = getattr(package, 'floor_ohms', 0.)
    if isinstance(draw, TCSharedCurve):
        grid, table, slope, nominal = draw.prepare(package.v_grid, source, floor)
        constants = ((grid, table[:-1], slope, nominal),)
        count, mask = 1, None
    else:
        if isinstance(draw, CurveGroups):
            curves, nominal, mask, floor = draw.corrections, draw.nominal, draw.mask, draw.floor
        else:
            if torch.is_tensor(draw):
                key = (None if draw.is_inference() else draw._version, id(package))
                if getattr(draw, '_tc_prepared_key', None) != key:
                    from tc_nonidealities import positive_resistance
                    positive_resistance(draw, floor)
                    # Independent storage avoids a Tensor -> view -> Tensor cycle.
                    draw._tc_prepared = EmpiricalLevels.from_gaussian(package.v_grid, draw.clone())
                    draw._tc_prepared_key = key
                draw = draw._tc_prepared
            curves, nominal = draw, package.programmed_resistances
            if getattr(owner, '_tc_cache_level_routing', True):
                _, mask = curves.weight_routing(module.weight, owner._values_to_level_idx,
                                                (owner.q_hi, owner.weight_scale))
            else:
                codes = owner._values_to_level_idx(module.weight.detach())
                mask = codes.unsqueeze(0) == torch.arange(1,nominal.numel()+1,
                    device=codes.device).reshape(-1,*([1]*codes.ndim))
            if package.empirical_bank is not None: floor = 0.
        count = nominal.numel()
        constants = None
    fused = getattr(owner, '_tc_fused_correction', True)
    packed = None
    # One group uses the scalar kernel for both samplers. Packing a single
    # curve adds overhead without batching any work.
    batched = getattr(owner, '_tc_batched_correction', True)
    if fused and batched and count > 1 and grouped:
        packed = batched_empirical_correction(source, curves, nominal, owner.v_dd, floor=floor)
    if packed is None:
        if constants is None: constants = curves.prepare(source, nominal)
        corrected = (empirical_correction(source, grid, left, slope, r, owner.v_dd,
                     fused=fused, floor=floor) for grid,left,slope,r in constants)
        if count > 1 and not grouped:
            output = None
            for index, x in enumerate(corrected):
                value = _convolution(module, x, module.weight*mask[index])
                output = value if output is None else output+value
            return output
        corrected = list(corrected)
        packed = corrected[0] if len(corrected) == 1 else torch.cat(corrected, dim=1)
    if count == 1:
        weight = module.weight if mask is None else module.weight*mask[0]
        return _convolution(module, packed, weight)
    return grouped_convolution(module, packed, module.weight, None, count, mask=mask)


def _convolution(module, source, weight):
    from torch import nn
    from torch.nn import functional as F
    if isinstance(module, nn.ConvTranspose2d):
        return F.conv_transpose2d(source,weight,None,module.stride,module.padding,
                                 module.output_padding,module.groups,module.dilation)
    return F.conv2d(source,weight,None,module.stride,module.padding,module.dilation,module.groups)
