"""Passive inference-time coupler-energy measurement for TC PCNs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch


_RK_WEIGHTS = {
    "Euler": (1.0,),
    "ProjEuler": (1.0,),
    "RK2": (0.0, 1.0),
    "RK4": (1 / 6, 1 / 3, 1 / 3, 1 / 6),
    "ProjRK4": (1 / 6, 1 / 3, 1 / 3, 1 / 6),
    "Dopri5": (35 / 384, 0.0, 500 / 1113, 125 / 192,
               -2187 / 6784, 11 / 84, 0.0),
    "ProjDopri5": (35 / 384, 0.0, 500 / 1113, 125 / 192,
                   -2187 / 6784, 11 / 84, 0.0),
}

# Dormand--Prince dense-output matrix used by the solver itself.  For an
# accepted step of length h and endpoint fraction theta, the quadrature weights
# are P @ [theta, theta**2, theta**3, theta**4].  At theta=1 these reduce to the
# ordinary fifth-order RK weights above.  This lets the passive meter stop at
# the physical endpoint when the legacy solver accepts an overshooting step and
# interpolates the state back to t_end.
_DOPRI5_DENSE_P = (
    (1., -8048581381 / 2820520608, 8663915743 / 2820520608,
     -12715105075 / 11282082432),
    (0., 0., 0., 0.),
    (0., 131558114200 / 32700410799, -68118460800 / 10900136933,
     87487479700 / 32700410799),
    (0., -1754552775 / 470086768, 14199869525 / 1410260304,
     -10690763975 / 1880347072),
    (0., 127303824393 / 49829197408, -318862633887 / 49829197408,
     701980252875 / 199316789632),
    (0., -282668133 / 205662961, 2019193451 / 616988883,
     -1453857185 / 822651844),
    (0., 40617522 / 29380423, -110615467 / 29380423,
     69997945 / 29380423),
)


def _accepted_interval_weights(solver_name, fraction):
    """RK weights for the part of an accepted step inside physical time."""
    fraction = float(fraction)
    if not 0. <= fraction <= 1.:
        raise ValueError("Accepted-step fraction must lie in [0, 1].")
    if fraction == 1.:
        return _RK_WEIGHTS[solver_name]
    if solver_name not in ("Dopri5", "ProjDopri5"):
        raise ValueError(
            f"Partial accepted-step energy is not supported for {solver_name}.")
    powers = (fraction, fraction ** 2, fraction ** 3, fraction ** 4)
    return tuple(sum(coef * power for coef, power in zip(row, powers))
                 for row in _DOPRI5_DENSE_P)


def physical_coupler_sites(module):
    """Number of physically present sites, including programmed zeros."""
    if not hasattr(module, "mat") or module.mat.layout != torch.sparse_csr:
        raise TypeError(
            "TC coupler-energy measurement requires Validator-expanded MVMConv modules.")
    return int(module.mat.values().numel())


@dataclass
class _LayerTotals:
    energy_sum_J: float = 0.0
    samples: int = 0
    duration_sum_s: float = 0.0
    solves: int = 0
    ff_sites: int | None = None
    fb_sites: int | None = None


class TCCouplerEnergyMeter:
    """One-layer accepted-step RK quadrature; never part of solver state."""

    def __init__(self, layer, supply_voltage, totals):
        self.layer = int(layer)
        self.supply_voltage = float(supply_voltage)
        self.totals = totals
        self._solving = False
        self._step_active = False

    def start_solve(self, batch_size, expected_duration_s):
        if self._solving:
            raise RuntimeError("A coupler-energy solve is already active.")
        if torch.is_grad_enabled():
            raise RuntimeError("Coupler-energy measurement is inference-only.")
        self._solving = True
        self._batch_size = int(batch_size)
        self._expected_duration_s = float(abs(expected_duration_s))
        self._solve_energy = None
        self._solve_duration_s = 0.0

    def begin_step(self, dt, solver_name):
        if not self._solving or self._step_active:
            raise RuntimeError("Invalid coupler-energy step lifecycle.")
        try:
            self._weights = _RK_WEIGHTS[solver_name]
        except KeyError as exc:
            raise ValueError(
                f"Coupler-energy quadrature does not support {solver_name}.") from exc
        self._solver_name = solver_name
        self._dt_s = float(abs(dt))
        self._rhs_index = -1
        self._last_stage = None
        self._stage_power = []
        self._step_active = True

    def observe(self, stage, module):
        if not self._step_active:
            return
        if stage == "FB":
            self._rhs_index += 1
            self._last_stage = "FB"
        elif stage == "FF" and self._last_stage == "FB":
            self._last_stage = "FF"
        else:
            raise RuntimeError("Expected one FB observation followed by one FF observation.")
        if self._rhs_index >= len(self._weights):
            raise RuntimeError("More RHS observations than RK quadrature stages.")

        sites = physical_coupler_sites(module)
        attr = "fb_sites" if stage == "FB" else "ff_sites"
        previous = getattr(self.totals, attr)
        if previous is None:
            setattr(self.totals, attr, sites)
        elif previous != sites:
            raise RuntimeError("Physical coupler-site count changed between batches.")

        current = getattr(module, "_tc_last_coupler_current", None)
        module._tc_last_coupler_current = None
        if current is None:
            raise RuntimeError(
                "Coupler current was not recorded by the preceding expanded MVM.")
        with torch.no_grad():
            if stage == "FB":
                self._stage_power.append(self.supply_voltage * current)
            else:
                self._stage_power[-1] = (
                    self._stage_power[-1] + self.supply_voltage * current)

    def accept_step(self):
        if not self._step_active:
            raise RuntimeError("No coupler-energy step is active.")
        if self._rhs_index + 1 != len(self._weights) or self._last_stage != "FF":
            raise RuntimeError("Accepted RK step did not expose all FF/FB stages.")
        if len(self._stage_power) != len(self._weights):
            raise RuntimeError("Accepted RK step has incomplete power observations.")
        remaining = max(0., self._expected_duration_s - self._solve_duration_s)
        accepted_duration = min(self._dt_s, remaining)
        fraction = 0. if self._dt_s == 0. else accepted_duration / self._dt_s
        weights = _accepted_interval_weights(self._solver_name, fraction)
        step_energy = sum((power * weight for power, weight in
                           zip(self._stage_power, weights)),
                          torch.zeros_like(self._stage_power[0])) * self._dt_s
        self._solve_energy = (step_energy if self._solve_energy is None
                              else self._solve_energy + step_energy)
        self._solve_duration_s += accepted_duration
        self._step_active = False

    def cancel_step(self):
        self._step_active = False

    def finish_solve(self):
        if not self._solving or self._step_active:
            raise RuntimeError("Cannot finish an incomplete coupler-energy solve.")
        tolerance = max(1e-15, 5e-5 * self._expected_duration_s)
        if abs(self._solve_duration_s - self._expected_duration_s) > tolerance:
            raise RuntimeError(
                "Accepted physical duration does not match the layer integration interval: "
                f"{self._solve_duration_s} versus {self._expected_duration_s} s.")
        energy = (torch.zeros(self._batch_size) if self._solve_energy is None
                  else self._solve_energy)
        if float(energy.min()) < -max(1e-24, 1e-6 * float(energy.abs().max())):
            raise RuntimeError("RK power quadrature produced negative physical energy.")
        self.totals.energy_sum_J += float(energy.sum().cpu())
        self.totals.samples += self._batch_size
        self.totals.duration_sum_s += self._solve_duration_s
        self.totals.solves += 1
        self._solving = False

    def abort_solve(self):
        self._step_active = False
        self._solving = False


class TCCouplerEnergyStudy:
    def __init__(self, supply_voltage=1.3):
        supply_voltage = float(supply_voltage)
        if not math.isfinite(supply_voltage) or supply_voltage <= 0:
            raise ValueError("Coupler supply voltage must be positive and finite.")
        self.supply_voltage = supply_voltage
        self.layers = {}

    def meter(self, layer):
        layer = int(layer)
        totals = self.layers.setdefault(layer, _LayerTotals())
        return TCCouplerEnergyMeter(layer, self.supply_voltage, totals)

    def summary(self):
        if not self.layers or any(v.samples == 0 for v in self.layers.values()):
            raise RuntimeError("No complete coupler-energy measurements were recorded.")
        sample_counts = {v.samples for v in self.layers.values()}
        if len(sample_counts) != 1:
            raise RuntimeError("TC layers observed different numbers of inference samples.")
        samples = sample_counts.pop()
        layer_rows = []
        average_energy_J = 0.0
        coupler_site_time_s = 0.0
        total_sites = 0
        total_time_s = 0.0
        for layer, totals in sorted(self.layers.items()):
            sites = int(totals.ff_sites or 0) + int(totals.fb_sites or 0)
            duration = totals.duration_sum_s / totals.solves
            average_energy_J += totals.energy_sum_J / totals.samples
            coupler_site_time_s += sites * duration
            total_sites += sites
            total_time_s += duration
            layer_rows.append(dict(
                layer=layer, physical_time_s=duration,
                physical_coupler_sites=sites,
                ff_physical_coupler_sites=int(totals.ff_sites or 0),
                fb_physical_coupler_sites=int(totals.fb_sites or 0)))
        average_time = coupler_site_time_s / total_sites
        return dict(
            supply_voltage_V=self.supply_voltage,
            samples=samples,
            average_energy_per_sample_J=average_energy_J,
            average_coupler_power_W=average_energy_J / coupler_site_time_s,
            total_physical_coupler_sites=total_sites,
            average_physical_time_per_coupler_s=average_time,
            total_inference_time_s=total_time_s,
            coupler_site_time_s=coupler_site_time_s,
            layers=layer_rows)


def enable_tc_coupler_energy(model, supply_voltage=1.3):
    """Attach a passive meter to expanded one-state TC ODE blocks."""
    study = TCCouplerEnergyStudy(supply_voltage)
    if model.training:
        raise ValueError("Coupler-energy measurement requires model.eval().")
    for index, block in enumerate(model.PcConvs):
        if block.__class__.__name__ != "ODEXInitFFFB" or not getattr(
                block, "_tc_current_mode", False):
            raise TypeError(
                "This energy study requires one-state TC ODEXInitFFFB blocks.")
        # Validate expansion before entering the expensive dataset loop.
        physical_coupler_sites(block.FFconv)
        physical_coupler_sites(block.FBconv)
        for module in (block.FFconv, block.FBconv):
            if getattr(module, "_tc_curve_package", None) is None:
                raise TypeError(
                    "Coupler-energy measurement requires TC per-coupler R(V) curves.")
            module._tc_measure_coupler_energy = True
            module._tc_last_coupler_current = None
        block._tc_energy_meter = study.meter(getattr(block, "layer_idx", index))
    return study


def print_tc_coupler_energy(summary):
    print("TC_COUPLER_ENERGY "
          f"average_coupler_power_W={summary['average_coupler_power_W']:.12g} "
          f"average_energy_per_sample_J={summary['average_energy_per_sample_J']:.12g}",
          flush=True)
    print("TC_COUPLER_ENERGY "
          f"total_physical_coupler_sites={summary['total_physical_coupler_sites']} "
          f"average_physical_time_per_coupler_s={summary['average_physical_time_per_coupler_s']:.12g} "
          f"total_inference_time_s={summary['total_inference_time_s']:.12g}", flush=True)
    for row in summary["layers"]:
        print("TC_COUPLER_ENERGY "
              f"layer={row['layer']} physical_time_s={row['physical_time_s']:.12g} "
              f"physical_coupler_sites={row['physical_coupler_sites']}", flush=True)


def append_tc_coupler_energy(path, summary, trial):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(trial=int(trial), **summary)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
