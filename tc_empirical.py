"""Code-aware TC empirical banks, using the existing native-grid loader."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import weakref

import torch

from tc_nonidealities import TCResistanceCurves
from utils import load_mc_res_curve_bank, mc45_corner_ids


def sampling_mode(args):
    """TC defaults to Gaussian; non-TC keeps its historical empirical default."""
    tc = getattr(args, 'tc_nonidealities', False) or getattr(args, 'tc_feedforward', False)
    mode = getattr(args, 'nonlinear_R_curve_sampling', None)
    if tc and getattr(args, '_exact_training_recovery', False) and not hasattr(args, '_tc_distribution_version'):
        # Pre-selector TC recovery recipes always trained with Gaussian curves.
        mode = 'multivariate_gaussian'
    if mode is None:
        mode = 'multivariate_gaussian' if tc else 'empirical_with_replacement'
    args.nonlinear_R_curve_sampling = mode
    if (tc and mode == 'empirical_with_replacement'
            and getattr(args, '_exact_training_recovery', False)
            and not hasattr(args, 'tc_curve_top_n')
            and getattr(args, 'tc_conv_method', 'loop') == 'shared'):
        # Before top-N, empirical "shared" actually executed the full per-code
        # loop. Preserve that arithmetic when continuing an old recovery file.
        import logging
        logging.warning('Legacy empirical recovery: preserving full-level loop (formerly named shared).')
        args.tc_conv_method = 'loop'
    if tc:
        args._tc_distribution_version = 1
    if tc and mode == 'empirical_with_replacement':
        path = getattr(args, 'tc_empirical_curve_bank', None)
        if not path or str(path).lower() == 'none':
            raise ValueError('Empirical TC requires tc_empirical_curve_bank.')
    return mode


def record_bank_identity(args, model):
    """Include loaded data provenance in ordinary/recovery checkpoint recipes."""
    for module in model.modules():
        package = getattr(module, '_tc_curve_package', None)
        if package is None:
            package = getattr(module, '_tc_resistance_package', None)
        if package is not None and package.empirical_bank is not None:
            args._tc_empirical_identity = package.empirical_bank.get('identity')
            return


@dataclass(frozen=True)
class EmpiricalLevels:
    bank: dict
    indices: torch.Tensor
    _prepared: dict = field(default_factory=dict, compare=False, repr=False)
    _routing: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_gaussian(cls, grid, resistance):
        """Adapt already sampled rows; sampling/RNG stay with the caller."""
        count = resistance.shape[0]
        bank = dict(v_grid=grid.expand(count, -1), R_left=resistance[:, :-1],
                    R_slope=resistance.diff(dim=1)/grid.diff(),
                    lengths=torch.full((count,), grid.numel(), device=grid.device),
                    fixed_length=grid.numel(), preselected=True)
        return cls(bank, torch.arange(count, device=grid.device))

    def weight_routing(self, weight, classify, signature):
        """Cache detached routing only, never a differentiable packed weight.

        A draw belongs to one FF/FB solve and is captured by checkpoint replay.
        Evaluation can retain it longer: identity/version checks also protect
        weight updates, replacement tensors and changed quantization settings.
        A weak reference avoids retaining the QAT autograd graph.
        """
        previous = self._routing
        # Tensor-valued quantization settings may mutate independently of the
        # weight. Keep their original uncached behavior rather than synchronizing
        # device scalars or guessing at their lifetime.
        version = (None if torch.is_inference_mode_enabled() or weight.is_inference()
                   or any(torch.is_tensor(v) for v in signature)
                   else weight._version)
        if (version is not None and previous
                and previous['weight']() is weight
                and previous['version'] == version
                and previous['signature'] == signature):
            return previous['codes'], previous['mask']
        codes = classify(weight.detach())
        mask = codes.unsqueeze(0) == torch.arange(
            1, self.indices.numel()+1, device=weight.device).reshape(
                -1, *([1]*weight.ndim))
        if version is not None:
            previous.clear()
            previous.update(weight=weakref.ref(weight), version=version,
                            signature=signature, codes=codes, mask=mask)
        return codes, mask

    def prepare(self, ref, programmed_resistances):
        """Prepare once per draw; all subsequent RHS/backward calls reuse it."""
        key = (ref.device, ref.dtype, programmed_resistances.data_ptr(),
               None if programmed_resistances.is_inference() else programmed_resistances._version)
        if key not in self._prepared:
            # Gather selected rows once, just as the packed kernel does. Indexing
            # the bank with individual CUDA scalar indices forces host readback.
            grid,left,slope,nominal,lengths = self.prepare_batched(ref,programmed_resistances)
            sizes = ([grid.shape[1]]*self.indices.numel() if lengths is None else
                     lengths.detach().cpu().tolist())
            self._prepared[key] = tuple((grid[k,:length].contiguous(),
                left[k,:length-1].contiguous(),slope[k,:length-1].contiguous(),nominal[k])
                for k,length in enumerate(sizes))
        return self._prepared[key]

    def prepare_batched(self, ref, programmed_resistances):
        """Selected native-grid rows for one batched correction, once per draw."""
        key = ('batched', ref.device, ref.dtype, programmed_resistances.data_ptr(),
               None if programmed_resistances.is_inference() else programmed_resistances._version)
        if key not in self._prepared:
            rows = tuple((self.bank[name] if self.bank.get('preselected') else
                          self.bank[name].index_select(0, self.indices)).to(ref).contiguous()
                         for name in ('v_grid', 'R_left', 'R_slope'))
            lengths = (None if self.bank.get('fixed_length') is not None else
                       self.bank['lengths'].index_select(0, self.indices).to(device=ref.device).contiguous())
            nominal = programmed_resistances.to(ref).contiguous()
            self._prepared[key] = (*rows, nominal, lengths)
        return self._prepared[key]


def grouped_convolution(module, packed_input, weight, codes, count, *, mask=None):
    """Existing Gaussian per-level grouping, shared with empirical TC."""
    from torch import nn
    from torch.nn import functional as F
    if mask is None:
        mask = codes.unsqueeze(0) == torch.arange(1,count+1,device=weight.device).reshape(-1,1,1,1,1)
    packed_weight = (weight.unsqueeze(0)*mask).reshape(count*weight.shape[0],*weight.shape[1:])
    if isinstance(module, nn.ConvTranspose2d):
        result = F.conv_transpose2d(packed_input,packed_weight,None,module.stride,module.padding,
            module.output_padding,count*module.groups,module.dilation)
    else:
        result = F.conv2d(packed_input,packed_weight,None,module.stride,module.padding,
            module.dilation,count*module.groups)
    return result.reshape(packed_input.shape[0],count,module.out_channels,*result.shape[-2:]).sum(1)


@dataclass(frozen=True)
class EmpiricalTCResistanceCurves(TCResistanceCurves):
    def sample_levels(self, *, generator=None, factor=None):
        bank = empirical_bank_on(self.empirical_bank, self.means.device, self.means.dtype)
        count = bank['draws_per_code']
        codes = torch.arange(self.levels.numel()-1, device=self.means.device)
        if isinstance(factor, str):
            allowed = bank['corner_indices'][factor]
            positions = torch.randint(allowed.shape[1], (codes.numel(),),
                                     device=codes.device, generator=generator)
            indices = allowed[codes, positions]
        else:
            indices = codes * count + torch.randint(count, (codes.numel(),),
                device=codes.device, generator=generator)
        return EmpiricalLevels(bank, indices)


def empirical_bank_on(bank, device, dtype):
    """One resident immutable bank per device/dtype, shared across modules."""
    key = (torch.device(device), dtype)
    cache = bank.setdefault('_device_cache', {})
    if key not in cache:
        cache[key] = {name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                      for name, value in bank.items() if torch.is_tensor(value)}
        cache[key].update(draws_per_code=bank['draws_per_code'],
            corner_indices={c: ix.to(device=device) for c, ix in bank.get('corner_indices', {}).items()})
        for name in ('fixed_length', 'preselected'):
            if name in bank: cache[key][name] = bank[name]
    return cache[key]


@lru_cache(maxsize=8)
def _load_directory(path, dtype, device):
    from data_utils import MC45CornerData
    folder = Path(path)
    marker = folder / 'TEST_ONLY_DUPLICATE.json'
    if marker.exists():
        import warnings
        warnings.warn('TEST-ONLY synthetic TC level bank: not valid for scientific accuracy results.',
                      RuntimeWarning, stacklevel=2)
    files = [folder / f'pseudo_res_{level}uS.csv' for level in range(5, 76, 5)]
    if any(not file.is_file() for file in files):
        raise ValueError('TC empirical bank requires files for all levels 5..75 uS.')
    banks, mappings, fingerprints = [], [], []
    for file in files:
        bank = load_mc_res_curve_bank(file, quantity='conductance', dtype=dtype, device=device)
        entries = MC45CornerData.coupler_file_corners(file, quantity='conductance')
        mapping = {f'{p.upper()}_V{v}_T{t}': entry['curve_indices']
                   for (p, v, t), entry in entries.items()}
        if set(mapping) != set(mc45_corner_ids()) or any(len(v) != 100 for v in mapping.values()):
            raise ValueError('Each empirical level must contain all 45 corners, 100 curves/corner.')
        banks.append(bank); mappings.append(mapping)
        fingerprints.append(hashlib.sha256(file.read_bytes()).hexdigest())
    count = banks[0]['v_grid'].shape[0]
    width = banks[0]['v_grid'].shape[1]
    if any(b['v_grid'].shape != (count, width) for b in banks):
        raise ValueError('Empirical levels must have equal curve counts and padded widths.')
    result = {key: torch.cat([b[key] for b in banks]) for key in banks[0]}
    result['draws_per_code'] = count
    result['native_layout'] = True
    lengths = torch.unique(result['lengths'])
    if lengths.numel() == 1:
        result['fixed_length'] = int(lengths.item())
    result['corner_indices'] = {
        corner: torch.tensor([m[corner] for m in mappings], dtype=torch.long, device=device)
                + torch.arange(15, device=device)[:, None] * count
        for corner in mc45_corner_ids()}
    result['identity'] = hashlib.sha256(''.join(fingerprints).encode()).hexdigest()
    return result


def is_npz_bank(path):
    """Legacy finite coupler banks do not include measured-pooling data."""
    return path is not None and Path(path).is_file() and Path(path).suffix == '.npz'


def prepare_empirical_curves(path, *, levels, R, R_max, dtype, device):
    if not math.isfinite(float(R)) or R <= 0 or not math.isfinite(float(R_max)) or R_max <= R:
        raise ValueError('Empirical TC requires finite positive R < R_max.')
    if levels.numel() != 16 or levels[0] != 0 or not torch.all(levels[1:] > levels[:-1]):
        raise ValueError('This empirical bank requires 15 increasing nonzero weight levels.')
    # Nominal hardware bins are 5..75 uS. Keep measured programming error intact.
    programmed = float(R) / levels[1:].to(device=device, dtype=dtype)
    # Retain the existing diagnostic NPZ format without fitting any Gaussian.
    if is_npz_bank(path):
        from tc_nonidealities import load_tc_empirical_bank
        base = EmpiricalTCResistanceCurves(
            torch.tensor([-.1, .1], dtype=dtype, device=device),
            levels.to(device=device, dtype=dtype), programmed, programmed,
            programmed[:, None].expand(-1, 2),
            torch.empty((0, 0), dtype=dtype, device=device), '', '')
        package = load_tc_empirical_bank(base, path)
        package.empirical_bank['fixed_length'] = package.empirical_bank['v_grid'].shape[1]
        package.empirical_bank['identity'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return package
    nominal = torch.arange(1, 16, device=device, dtype=dtype) * 5e-6
    if not torch.allclose(programmed.reciprocal(), nominal, rtol=.005, atol=0):
        raise ValueError('Empirical levels require the 5..75-uS mapping (R=13.3k, R_max=200k).')
    bank = _load_directory(str(Path(path).resolve()), dtype, torch.device(device))
    # These small placeholders provide code metadata, not Gaussian statistics.
    grid = torch.tensor([-.1, .1], dtype=dtype, device=device)
    return EmpiricalTCResistanceCurves(grid, levels.to(device=device, dtype=dtype),
        programmed, programmed, programmed[:, None].expand(-1, 2),
        torch.empty((0, 0), dtype=dtype, device=device), '', '',
        empirical_bank=bank)


def prepare_curves(mean_path, covariance_path, *, mode='multivariate_gaussian',
                   empirical_path=None, **kwargs):
    if mode == 'empirical_with_replacement':
        kwargs.pop('floor_ohms', None)  # Empirical measured curves bypass Gaussian guards.
        return prepare_empirical_curves(empirical_path, **kwargs)
    if mode != 'multivariate_gaussian':
        raise ValueError('Invalid TC curve distribution.')
    from tc_nonidealities import prepare_tc_resistance_curves
    return prepare_tc_resistance_curves(mean_path, covariance_path, **kwargs)


def pooling_options(args, ref):
    """Use a measured CSV-directory level; NPZ callers retain Gaussian pooling."""
    R = getattr(args, 'measured_pooling_nominal_R', None) or getattr(args, 'R', 13.3e3)
    conductance = 1e6 / float(R)
    level = round(conductance / 5) * 5
    if level not in range(5, 76, 5) or abs(conductance-level)/level > .005:
        raise ValueError('Empirical pooling nominal resistance must match a 5..75-uS level.')
    return dict(curve_path=str(Path(args.tc_empirical_curve_bank)/f'pseudo_res_{level}uS.csv'),
                nominal_R=R, tc_empirical=True)
