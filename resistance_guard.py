"""Optional physical resistance floor for Gaussian curves; empirical banks bypass it."""
import math
import os
from functools import lru_cache

import torch


def resistance_floor(value):
    if value is None or str(value).lower() in ('none', ''):
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('resistance_floor_ohms must be finite and positive.')
    return value


def add_argument(parser):
    if '--resistance_floor_ohms' not in parser._option_string_actions:
        parser.add_argument('--resistance_floor_ohms', type=resistance_floor,
                            default=resistance_floor(os.environ.get('RESISTANCE_FLOOR_OHMS')),
                            help='Optional Gaussian resistance floor in ohms; empirical banks unchanged.')


@lru_cache(maxsize=64)
def _represented_floor(floor, dtype):
    represented = torch.tensor(floor, dtype=dtype)
    if not torch.isfinite(represented) or represented.item() <= 0:
        raise ValueError('Resistance floor is not representable in this dtype.')
    return floor


def guard_resistance(values, floor_ohms):
    """Apply an explicitly selected physical floor after conversion to ohms."""
    floor = resistance_floor(floor_ohms)
    if floor is None:
        return values
    floor = _represented_floor(floor, values.dtype)
    return values.clamp_min(floor)


def gaussian_resistance(sampled, quantity, value_scale, floor_ohms=None):
    """Keep the original native-unit positivity guard, then apply an Ohm floor."""
    sampled.clamp_(min=torch.finfo(sampled.dtype).eps * value_scale)
    if quantity == 'conductance':
        resistance = sampled.reciprocal()
    elif quantity == 'resistance':
        resistance = sampled
    else:
        raise ValueError('Gaussian curve quantity is invalid.')
    return guard_resistance(resistance, floor_ohms)


def saved_floor(config):
    """Read historical checkpoint metadata without retaining an old CLI alias."""
    tc = config.get('tc_nonidealities') or config.get('tc_feedforward')
    # The former TC-only argument existed in all parsers, including toggle.
    # Its saved value had no effect outside TC and must not become an override.
    value = config.get('resistance_floor_ohms',
                       config.get('tc_resistance_floor_ohms') if tc else None)
    value = resistance_floor(value)
    if tc:
        return 1e-6 if value is None else value
    return value
