"""Relative physical integration-time variation, independent of circuit RNGs."""
import math
import os

import torch
from torch import nn


def nonnegative_std(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Compute-time standard deviations must be finite and nonnegative.')
    return value


def add_arguments(parser):
    for name in ('compute_time_fixed_std', 'compute_time_noise_std'):
        if '--' + name not in parser._option_string_actions:
            parser.add_argument('--' + name, type=nonnegative_std,
                                default=nonnegative_std(os.environ.get(name.upper(), '0')),
                                help='Relative timing STD: fixed per hardware realization or resampled per forward.')
    if '--timing_noise_scope' not in parser._option_string_actions:
        parser.add_argument('--timing_noise_scope', choices=('per_batch', 'per_sample'),
                            default=os.environ.get('TIMING_NOISE_SCOPE', 'per_batch'),
                            help='Per-sample timing noise uses equivalent drift/diffusion rescaling.')


class _TimeScaledRHS(nn.Module):
    """Autonomous physical dynamics on a common, nominal time interval."""
    def __init__(self, func, factor):
        super().__init__()
        self.func = func
        self.factor = factor
        self.tc_context = getattr(func, 'tc_context', None)

    def forward(self, t, y):
        out = self.func(t, y)
        if isinstance(out, tuple):
            return tuple(value * self.factor.reshape(-1, *([1] * (value.ndim - 1)))
                         for value in out)
        return out * self.factor.reshape(-1, *([1] * (out.ndim - 1)))


class ComputeTimeVariation(nn.Module):
    """Fixed layer variation plus batch-shared or independent sample jitter.

    CPU scalar RNGs never touch global or circuit RNGs. No state_dict keys
    are added, preserving legacy checkpoint compatibility. Exact recovery
    saves the fixed assignments and generators in training_recovery._runtime.
    """
    def __init__(self, fixed_std=0., noise_std=0., seed=4096, scope='per_batch'):
        super().__init__()
        self.fixed_std = nonnegative_std(fixed_std)
        self.noise_std = nonnegative_std(noise_std)
        self.seed = int(seed)
        if scope not in ('per_batch', 'per_sample'):
            raise ValueError('timing_noise_scope must be per_batch or per_sample.')
        self.scope = scope
        self.reset()

    def reset(self):
        self._compute_time_fixed_offsets = {}
        self._generators = {}
        self.last = {}

    def _normal(self, unit, kind, count=None):
        key = unit + ':' + kind
        if key not in self._generators:
            offset = sum((i + 1) * ord(c) for i, c in enumerate(key))
            self._generators[key] = torch.Generator().manual_seed((self.seed + offset) % (2**63))
        value = torch.randn(() if count is None else (count,),
                            generator=self._generators[key], dtype=torch.float64)
        return value.item() if count is None else value

    def factor(self, unit):
        if self.fixed_std == 0 and self.noise_std == 0:
            return 1.
        if unit not in self._compute_time_fixed_offsets:
            fixed = self.fixed_std * self._normal(unit, 'fixed') if self.fixed_std else 0.
            # Truncate only the unphysical tail, rather than clipping at zero.
            while fixed <= -1.:
                fixed = self.fixed_std * self._normal(unit, 'fixed')
            self._compute_time_fixed_offsets[unit] = fixed
        fixed = self._compute_time_fixed_offsets[unit]
        noise = self.noise_std * self._normal(unit, 'noise') if self.noise_std else 0.
        while 1. + fixed + noise <= 0:
            noise = self.noise_std * self._normal(unit, 'noise')
        factor = 1. + fixed + noise
        if not math.isfinite(factor):
            raise ValueError('Sampled compute-time factor is not finite.')
        self.last[unit] = dict(fixed_fraction=fixed, noise_fraction=noise, factor=factor)
        return factor

    def duration(self, nominal, unit):
        if self.fixed_std == 0 and self.noise_std == 0:
            return nominal
        actual = nominal * self.factor(unit)
        self.last[unit].update(nominal_seconds=float(nominal), actual_seconds=float(actual))
        return actual

    def options(self, nominal):
        if self.fixed_std == 0 and self.noise_std == 0:
            return nominal
        factor = self.factor('ode')
        start = nominal['t0']
        actual = dict(nominal)
        actual['t1'] = start + (nominal['t1'] - start) * factor
        actual['t_eval'] = [start + (t - start) * factor for t in nominal['t_eval']]
        if nominal.get('h') is not None:
            actual['h'] = nominal['h'] * factor
        self.last['ode'].update(nominal_seconds=float(nominal['t1'] - start),
                                actual_seconds=float(actual['t1'] - start))
        return actual

    def sample_factors(self, unit, ref, nominal):
        """One positive factor per image; the fixed offset remains layer-shared."""
        if unit not in self._compute_time_fixed_offsets:
            # Use the unchanged fixed stream without consuming timing-noise draws.
            fixed = self.fixed_std * self._normal(unit, 'fixed') if self.fixed_std else 0.
            while fixed <= -1.:
                fixed = self.fixed_std * self._normal(unit, 'fixed')
            self._compute_time_fixed_offsets[unit] = fixed
        fixed = self._compute_time_fixed_offsets[unit]
        noise = (self.noise_std * self._normal(unit, 'noise', ref.shape[0])
                 if self.noise_std else torch.zeros(ref.shape[0], dtype=torch.float64))
        invalid = 1. + fixed + noise <= 0
        while invalid.any():
            noise[invalid] = self.noise_std * self._normal(unit, 'noise', int(invalid.sum()))
            invalid = 1. + fixed + noise <= 0
        factors = 1. + fixed + noise
        if not torch.isfinite(factors).all():
            raise ValueError('Sampled compute-time factors are not finite.')
        self.last[unit] = dict(
            fixed_fraction=fixed, noise_fraction=noise.tolist(), factor=factors.tolist(),
            nominal_seconds=float(nominal), actual_seconds=(factors * float(nominal)).tolist())
        return factors.to(ref)

    @staticmethod
    def _scale_diffusion(eps, factor):
        if eps is None:
            return None
        if isinstance(eps, tuple):
            return tuple(ComputeTimeVariation._scale_diffusion(value, factor) for value in eps)
        return eps * factor

    def prepare_solve(self, func, y0, nominal):
        if self.scope == 'per_batch':
            return func, self.options(nominal)
        if getattr(func, 'energy_meter', None) is not None:
            raise ValueError('Per-sample timing is not supported by coupler energy metering.')
        ref = y0[0] if isinstance(y0, tuple) else y0
        factor = self.sample_factors('ode', ref, nominal['t1'] - nominal['t0'])
        options = dict(nominal)
        scale = factor.sqrt().reshape(-1, *([1] * (ref.ndim - 1)))
        options['eps'] = self._scale_diffusion(options.get('eps'), scale)
        return _TimeScaledRHS(func, factor), options

    def rescale_stage(self, drift, eps, unit, nominal):
        factor = self.sample_factors(unit, drift, nominal).reshape(
            -1, *([1] * (drift.ndim - 1)))
        return drift * factor, eps * factor.sqrt()


def configure(model, args, trial=0):
    """Install after physical conversion (and after inference shape discovery)."""
    fixed = nonnegative_std(getattr(args, 'compute_time_fixed_std', 0.))
    noise = nonnegative_std(getattr(args, 'compute_time_noise_std', 0.))
    scope = getattr(args, 'timing_noise_scope', 'per_batch')
    if scope not in ('per_batch', 'per_sample'):
        raise ValueError('timing_noise_scope must be per_batch or per_sample.')
    if fixed == 0 and noise == 0:
        # Also support explicitly disabling a previously configured model.
        for owner in list(model.modules()):
            if isinstance(getattr(owner, 'compute_time_variation', None), ComputeTimeVariation):
                del owner.compute_time_variation
        return 0  # No timing module, hooks, option copies, or RNG draws.
    base = next((getattr(args, key, None) for key in ('hardware_seed', 'data_seed', 'seed')
                 if getattr(args, key, None) is not None), 4096)
    owners = [m for m in model.modules()
              if getattr(m, '_tc_current_mode', False) or
              (hasattr(m, 'tc_method') and getattr(m, 'physical', False))]
    if not owners:
        if (getattr(args, 'physical_pretraining', False) or
                (hasattr(args, 'ode_wrapper') and args.ode_wrapper is None)):
            print('Compute-time variation inactive during unitless pretraining.', flush=True)
            return 0
        raise ValueError('Compute-time variation requires physical TC PCN or TC CNN integration.')
    for owner in owners:
        owner.compute_time_variation = ComputeTimeVariation(
            fixed, noise, int(base) + int(trial) + 1009 * int(owner.layer_idx), scope)
    print(f'Compute-time variation: fixed_std={fixed}, noise_std={noise}, '
          f'scope={scope}, blocks={len(owners)}, seed={int(base) + int(trial)}', flush=True)
    return len(owners)


def records(model):
    return {name: dict(module.last) for name, module in model.named_modules()
            if isinstance(module, ComputeTimeVariation)}
