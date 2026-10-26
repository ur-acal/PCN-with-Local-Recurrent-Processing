"""Data preparation for the opt-in true-continuous nonideality model.

This module does not install hooks or change any TC/toggle execution path.
Resistances and covariance are in ohms and ohms squared. The existing MC
loader's numerical normalization is undone once, not once per weight level.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch

from utils import load_mc_res_curve_gaussian, load_res_vs_vin


def load_tc_empirical_bank(package, path):
    """Attach NPZ curves [resistance code, draw, voltage], in absolute ohms.

    Required arrays: v_grid, programmed_resistances, curves. Code order must
    match the package; no mixing of programmed resistance levels is allowed.
    """
    from dataclasses import replace
    with np.load(path, allow_pickle=False) as data:
        grid = torch.as_tensor(data['v_grid'], device=package.means.device,
                               dtype=package.means.dtype)
        resistances = torch.as_tensor(data['programmed_resistances']).double()
        curves = torch.as_tensor(data['curves'], device=package.means.device,
                                 dtype=package.means.dtype)
    expected = package.programmed_resistances.detach().cpu().double()
    if resistances.shape != expected.shape or not torch.allclose(resistances, expected):
        raise ValueError('TC empirical bank programmed resistance codes do not match.')
    if (grid.ndim != 1 or grid.numel() < 2 or not torch.isfinite(grid).all()
            or not torch.all(grid[1:] > grid[:-1])):
        raise ValueError('TC empirical voltage grid must be finite and increasing.')
    if (curves.ndim != 3 or curves.shape[0] != expected.numel()
            or curves.shape[1] < 1 or curves.shape[2] != grid.numel()
            or not torch.isfinite(curves).all() or not torch.all(curves > 0)):
        raise ValueError('TC empirical curves must be positive finite [codes, draws, voltage].')
    n_draws = curves.shape[1]
    flat = curves.flatten(0, 1)
    bank = dict(v_grid=grid.expand(flat.shape[0], -1), R_left=flat[:, :-1],
                R_slope=(flat[:, 1:] - flat[:, :-1]) / (grid[1:] - grid[:-1]),
                lengths=torch.full((flat.shape[0],), grid.numel(), device=grid.device,
                                   dtype=torch.long), draws_per_code=n_draws)
    return replace(package, empirical_bank=bank)


class TCNoiseLifecycle:
    """Solve-local noise tape; accepted intervals, not RHS calls, advance it.

    Restarting on predefined grids replays the same Gaussian samples. Physical
    spin/curve realizations are captured separately in the owning RHS closure.
    """
    def __init__(self, coefficients, generators, fb_coefficients=None, fb_ref=None,
                 replay_cache=True):
        self.coefficients = coefficients
        self.generators = generators
        self.fb_coefficients, self.fb_ref = fb_coefficients, fb_ref
        self.tape = {}
        self.index = 0
        # Training may revisit old intervals during checkpoint recomputation.
        self.replay_cache = bool(replay_cache) or torch.is_grad_enabled()

    def restart(self):
        if not self.replay_cache and self.index:
            raise RuntimeError('Noise replay requires tc_noise_replay_cache=true.')
        self.index = 0

    def accepted(self):
        if not self.replay_cache:
            self.tape.clear()
        self.index += 1

    def _sample(self, key, ref, coefficients):
        cache_key = (self.index, key)
        if cache_key not in self.tape:
            value = torch.zeros_like(ref)
            for source, coefficient in zip(("sum", "coupler"), coefficients):
                if torch.count_nonzero(coefficient):
                    value = value + coefficient * torch.randn(
                        ref.shape, dtype=ref.dtype, device=ref.device,
                        generator=self.generators[(key, source)])
            self.tape[cache_key] = value
        return self.tape[cache_key]

    def normal(self, ref, branch):
        coeff = self.coefficients[branch]
        total = (coeff[0].square() + coeff[1].square()).sqrt()
        return self._sample(branch, ref, coeff) / total.clamp_min(torch.finfo(ref.dtype).tiny)

    def fb_current(self):
        if self.fb_coefficients is None:
            return 0.
        return self._sample("fb", self.fb_ref, self.fb_coefficients)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def positive_resistance(values, floor_ohms=1e-6):
    """Guard sampled/interpolated physical resistances before taking 1/R."""
    floor = _positive(floor_ohms, "floor_ohms")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Resistance values must be finite floating-point values.")
    represented = torch.tensor(floor, dtype=values.dtype)
    if not torch.isfinite(represented) or represented.item() <= 0:
        raise ValueError("Resistance floor is not representable in this dtype.")
    from resistance_guard import guard_resistance
    return guard_resistance(values, floor)


@dataclass(frozen=True)
class TCSharedCurve:
    """One solve-local relative distortion, selected independently per tensor."""
    code: torch.Tensor
    resistance: torch.Tensor
    nominal_R: torch.Tensor
    _prepared: dict = field(default_factory=dict, compare=False, repr=False)

    def prepare(self, grid, ref, floor):
        """Validate/cache constants once per draw/device/dtype, never per pixel RHS."""
        key = (grid.data_ptr(),
               *(None if t.is_inference() else t._version
                 for t in (grid,self.resistance,self.nominal_R)),
               ref.device, ref.dtype, float(floor))
        if key not in self._prepared:
            g, table, nominal = (t.to(ref).contiguous() for t in (grid,self.resistance,self.nominal_R))
            if g.ndim != 1 or table.shape != g.shape or nominal.numel() != 1:
                raise ValueError('TC shared correction requires one curve on its grid and scalar nominal R.')
            positive_resistance(table, floor)  # finite/dtype/floor validation only
            if not torch.isfinite(nominal).all() or not (nominal > 0).all():
                raise ValueError('TC nominal resistance must be finite and positive.')
            if g.numel() < 2 or not torch.isfinite(g).all() or not (g[1:] > g[:-1]).all():
                raise ValueError('TC voltage grid must be finite and strictly increasing.')
            slope = (table[1:]-table[:-1])/(g[1:]-g[:-1])
            if not torch.isfinite(slope).all():
                raise ValueError('TC resistance slopes must be finite.')
            self._prepared[key] = (g, table, slope, nominal)
        return self._prepared[key]


@dataclass(frozen=True)
class TCResistanceCurves:
    v_grid: torch.Tensor                 # [V]
    levels: torch.Tensor                 # [codes + 1], normalized conductance
    programmed_resistances: torch.Tensor # [codes], R / levels[1:]
    mean_column_resistances: torch.Tensor # [codes], nearest CSV column labels
    means: torch.Tensor                  # [codes, V], ohms
    factor: torch.Tensor                 # [V, V], ohms; shared by all codes
    mean_path: str
    covariance_path: str
    floor_ohms: float = 1e-6
    empirical_bank: object = None

    @property
    def covariance(self):
        """Unclipped Gaussian covariance, including the loader's tiny jitter.

        The positivity guard can alter tail statistics if it activates.
        """
        return self.factor @ self.factor.T

    def sample(self, codes, *, generator=None, chunk_size=65536, factor=None):
        """Independent draws per requested code occurrence, in bounded chunks.

        Training passes codes 1..15 once per tensor. Inference passes the
        code for each edge. Repeated codes receive independent draws here;
        sharing/caching lifetimes are the future caller's responsibility.
        Zero codes return +inf placeholders (open circuits), consume no RNG,
        and must never be interpolated as physical curves.
        """
        factor = self.factor if factor is None else factor
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        codes = torch.as_tensor(codes, device=self.means.device)
        if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("codes must contain integers.")
        if ((codes < 0) | (codes > self.means.shape[0])).any():
            raise ValueError("Code is outside the package's quantization range.")
        flat = codes.reshape(-1).long()
        result = self.means.new_full((flat.numel(), self.v_grid.numel()), float("inf"))
        active = torch.nonzero(flat, as_tuple=False).flatten()
        for start in range(0, active.numel(), chunk_size):
            positions = active[start:start + chunk_size]
            noise = torch.randn((positions.numel(), factor.shape[1]),
                                dtype=self.means.dtype, device=self.means.device,
                                generator=generator)
            sampled = self.means[flat[positions] - 1] + noise @ factor.T
            result[positions] = positive_resistance(sampled, self.floor_ohms)
        return result.reshape(*codes.shape, self.v_grid.numel())

    def sample_levels(self, *, generator=None, factor=None):
        """One batched independent draw for every nonzero code; no code loop."""
        codes = torch.arange(1, self.means.shape[0] + 1, device=self.means.device)
        return self.sample(codes, generator=generator, chunk_size=codes.numel(), factor=factor)

    @torch.no_grad()
    def sample_shared(self, codes, *, sampling="histogram", generator=None, factor=None):
        """Choose a Gaussian, then draw its curve; no spatial unrolling or STE.

        Uniform means uniform over all characterized nonzero codes, including
        codes absent from this tensor. An all-zero tensor consumes no RNG.
        """
        factor = self.factor if factor is None else factor
        if sampling not in ("histogram", "uniform"):
            raise ValueError("tc_curve_sampling must be histogram or uniform.")
        codes = codes.detach().to(device=self.means.device, dtype=torch.long)
        if not torch.any(codes):
            return None
        count = self.means.shape[0]
        if sampling == "histogram":
            probabilities = torch.bincount(codes.flatten(), minlength=count+1)[1:].to(self.means)
        else:
            probabilities = self.means.new_ones(count)
        index = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
        noise = torch.randn(factor.shape[1], device=self.means.device,
                            dtype=self.means.dtype, generator=generator)
        resistance = positive_resistance(self.means[index] + factor @ noise, self.floor_ohms)
        return TCSharedCurve(index+1, resistance, self.programmed_resistances[index])


def prepare_tc_corner_factors(covariance_path, v_grid, *, dtype, device):
    """Fit absolute resistance covariance per corner; leave code means alone."""
    from data_utils import MC45CornerData
    from utils import mc45_corner_ids
    entries = MC45CornerData.coupler_file_corners(covariance_path)
    entries = {f"{p.upper()}_V{v}_T{t}": entry
               for (p, v, t), entry in entries.items()}
    if set(entries) != set(mc45_corner_ids()) or any(
            len(entry['curve_indices']) != 100 for entry in entries.values()):
        raise ValueError("TC corner coupling requires 45 corners with 100 curves each.")
    target = v_grid.detach().cpu().double()
    factors = {}
    for corner in mc45_corner_ids():
        fit = load_mc_res_curve_gaussian(
            covariance_path, quantity='resistance',
            curve_indices=entries[corner]['curve_indices'], dtype=torch.float64)
        grid = fit['v_grid']
        # The installed grid may already be FP32; match nearest coordinates
        # rather than letting roundoff send searchsorted to the next voltage.
        indices = (grid[:, None] - target[None, :]).abs().argmin(0)
        if not torch.allclose(grid[indices], target, rtol=1e-6, atol=1e-9):
            raise ValueError("TC corner covariance grids must match the existing TC grid.")
        factor = fit['factor'] * fit['value_scale']
        if indices.numel() != grid.numel():
            rows = factor[indices]
            factor = torch.linalg.cholesky(rows @ rows.T)
        factor = factor.to(device=device, dtype=dtype)
        if not torch.isfinite(factor).all():
            raise ValueError("Nonfinite TC corner covariance factor.")
        factors[corner] = factor
    return factors


class TCCornerFactorBank:
    """Shared factors; migrate once per device/dtype, not once per layer."""

    def __init__(self, factors):
        ref = next(iter(factors.values()))
        self.factors = factors
        self._device_cache = {(ref.device, ref.dtype): factors}

    def get(self, corner, ref):
        key = (ref.device, ref.dtype)
        if key not in self._device_cache:
            self._device_cache[key] = {
                name: factor.to(device=ref.device, dtype=ref.dtype)
                for name, factor in self.factors.items()}
        return self._device_cache[key][corner]


def tc_training_corner_factor(module, default):
    """Select a resident factor in training; migrate only on first device use."""
    sampler = getattr(module, '_ft_corner_coupled_sampler', None)
    if module.training and sampler is not None and sampler.active_corner is not None:
        if getattr(module, '_tc_empirical_coupling', False):
            return sampler.active_corner
        return module._tc_corner_factors.get(sampler.active_corner, default)
    return default


def prepare_tc_resistance_curves(mean_path, covariance_path, *, levels,
                                 R=10e3, R_max=150e3, dtype=torch.float64,
                                 device="cpu", floor_ohms=1e-6):
    """Code-specific means plus one absolute covariance pooled over all MCs.

    Reuses existing CSV loading, MC grid alignment and full-covariance fit.
    Mean columns are selected with the existing nearest-resistance rule,
    after sorting by resistance, not by their position in the input CSV.
    Means are linearly interpolated on the MC covariance grid. Restrict to
    the common characterized band rather than extrapolate statistical data.
    Runtime R(V) interpolation is deliberately not implemented here.
    """
    floor = _positive(floor_ohms, "floor_ohms")
    R = _positive(R, "R")
    # The caller supplies ODEBlockPC._get_quant_magnitude_levels().
    # Keep quantization/remapping policy in its existing owner.
    levels = levels.detach().to(device="cpu", dtype=torch.float64)
    if (levels.ndim != 1 or levels.numel() < 2 or levels[0] != 0
            or not torch.isfinite(levels).all()
            or not (levels[1:] > levels[:-1]).all()):
        raise ValueError("levels must be a finite increasing grid starting at zero.")
    mean_path = str(Path(mean_path).resolve())
    covariance_path = str(Path(covariance_path).resolve())
    grid, labels, table = load_res_vs_vin(
        R=R, R_max=R_max, nonlinear_R_table=mean_path, dtype=torch.float64)
    if grid is None or grid.numel() < 2 or labels.numel() < 1:
        raise ValueError("Mean table must contain a voltage grid and resistance columns.")
    if not all(torch.isfinite(t).all() for t in (grid, labels, table)):
        raise ValueError("Mean table contains nonfinite data.")
    if (labels <= 0).any() or (table <= 0).any():
        raise ValueError("Mean resistance labels and values must be positive.")
    order = grid.argsort()
    grid, table = grid[order], table[order]
    order = labels.argsort()
    labels, table = labels[order], table[:, order]
    if not (grid[1:] > grid[:-1]).all() or not (labels[1:] > labels[:-1]).all():
        raise ValueError("Mean voltage grid and resistance labels must be unique.")
    programmed = float(R) / levels[1:]
    selected = (programmed[:, None] - labels[None, :]).abs().argmin(1)
    fit = load_mc_res_curve_gaussian(covariance_path, quantity="resistance",
                                    dtype=torch.float64, device="cpu")
    mc_grid = fit["v_grid"]
    common = (mc_grid >= grid[0]) & (mc_grid <= grid[-1])
    if common.count_nonzero() < 2:
        raise ValueError("Mean/covariance sources have no usable common voltage band.")
    target_grid = mc_grid[common]
    # Factor is normalized by one MC-wide scale, NOT by each level's mean.
    factor = fit["factor"] * fit["value_scale"]
    if not common.all():
        subfactor = factor[common]
        factor = torch.linalg.cholesky(subfactor @ subfactor.T)
    means = torch.from_numpy(np.stack([
        np.interp(target_grid.numpy(), grid.numpy(), table[:, column].numpy())
        for column in selected.tolist()
    ]))
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 for resistance covariance preparation.")
    def cast(t):
        converted = t.to(device=device, dtype=dtype)
        if not torch.isfinite(converted).all():
            raise ValueError("Prepared data is not finite in the requested dtype.")
        return converted

    return TCResistanceCurves(cast(target_grid), cast(levels), cast(programmed),
                              cast(labels[selected]), cast(means), cast(factor),
                              mean_path, covariance_path, floor)


@dataclass(frozen=True)
class CurrentASDIntegral:
    path: str
    lower_hz: float
    upper_hz: float
    reference_R: float
    variance_A2: float

    @property
    def std_A(self):
        return math.sqrt(self.variance_A2)

    def variance_at_resistance(self, resistance):
        """Selected effective power scaling R_ref/R over the whole CSV band."""
        return self.variance_A2 * self.reference_R / _positive(resistance, "resistance")


def integrate_current_asd(path, *, lower_hz=None, upper_hz=None, reference_R=50e3):
    """Integrate current ASD squared over frequency (Hz) to variance (A²).

    Column meanings are explicitly frequency and current ASD (A/sqrt(Hz)),
    regardless of the circuit export's VN header. Integrate piecewise-linear
    PSD with the trapezoid rule in linear frequency, not log frequency.
    No gamma, sqrt(dt), two-sided multiplier, or out-of-band extrapolation.
    """
    reference_R = _positive(reference_R, "reference_R")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 2 or data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError("ASD CSV must contain at least two finite frequency/ASD pairs.")
    data = data[np.argsort(data[:, 0])]
    freq, asd = data.T
    if (freq < 0).any() or (asd < 0).any() or not (np.diff(freq) > 0).all():
        raise ValueError("Frequencies must be unique/nonnegative and ASD nonnegative.")
    lower = float(freq[0] if lower_hz is None else lower_hz)
    upper = float(freq[-1] if upper_hz is None else upper_hz)
    if not math.isfinite(lower) or not math.isfinite(upper) or not freq[0] <= lower < upper <= freq[-1]:
        raise ValueError("Integration bounds must lie within the characterized band.")
    points = np.concatenate(([lower], freq[(freq > lower) & (freq < upper)], [upper]))
    with np.errstate(over="raise", invalid="raise"):
        psd = np.interp(points, freq, asd ** 2)
        variance = float(np.trapezoid(psd, points))
    if not math.isfinite(variance):
        raise ValueError("Integrated noise variance is not finite.")
    return CurrentASDIntegral(str(Path(path).resolve()), lower, upper, reference_R, variance)
