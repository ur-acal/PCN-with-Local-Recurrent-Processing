"""Passive accepted-step coupler-energy accounting checks."""

import unittest

import torch

from tc_energy import TCCouplerEnergyStudy, physical_coupler_sites
from TorchDiffEqPack.odesolver import odesolve
from tc_nonidealities import TCNoiseLifecycle
from test_tc_inference import expanded
from test_tc_dense_training import make_block, synthetic_package


class TCEnergyTests(unittest.TestCase):
    def package(self):
        block = make_block()
        block.q_hi = 3
        block.weight_scale = 1.
        package = synthetic_package(block)
        return package.__class__(
            **{**package.__dict__, "factor": torch.zeros_like(package.factor)})

    def test_expanded_current_and_programmed_zero_site_count(self):
        module = expanded(self.package())
        source = torch.linspace(-.08, .08, 18, dtype=torch.float64).reshape(2, 1, 3, 3)
        module._tc_measure_coupler_energy = True
        module(source)
        actual = module._tc_last_coupler_current
        flat = source.reshape(2, -1).t()
        expected = torch.zeros(2, dtype=source.dtype)
        for edge, value in enumerate(module.mat.values()):
            if value != 0:
                resistance = module.R / value.abs()
                expected += flat[module.mat.col_indices()[edge]].abs() / resistance
        torch.testing.assert_close(actual, expected)
        self.assertEqual(physical_coupler_sites(module), module.mat.values().numel())
        self.assertGreater((module.mat.values() == 0).sum().item(), 0)

    def test_meter_reuses_forward_resistance_lookup(self):
        module = expanded(self.package())
        source = torch.full((2, 1, 3, 3), .05, dtype=torch.float64)
        calls = 0
        original = module._get_gaussian_curve_R_eff

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        module._get_gaussian_curve_R_eff = counted
        module._tc_measure_coupler_energy = True
        study = TCCouplerEnergyStudy(1.3)
        meter = study.meter(0)
        with torch.no_grad():
            meter.start_solve(2, 1e-9)
            meter.begin_step(1e-9, "Euler")
            module(source)
            calls_after_fb_forward = calls
            meter.observe("FB", module)
            self.assertEqual(calls, calls_after_fb_forward)
            module(source)
            calls_after_ff_forward = calls
            meter.observe("FF", module)
            self.assertEqual(calls, calls_after_ff_forward)
            meter.accept_step()
            meter.finish_solve()

    def test_dopri_quadrature_summary_and_simple_p_times_n_times_t_identity(self):
        module = expanded(self.package())
        study = TCCouplerEnergyStudy(1.3)
        meter = study.meter(0)
        batch = 2
        source = torch.full((batch, 1, 3, 3), .05, dtype=torch.float64)
        module._tc_measure_coupler_energy = True
        with torch.no_grad():
            meter.start_solve(batch, 2e-9)
            for dt in (0.75e-9, 1.25e-9):
                meter.begin_step(dt, "Dopri5")
                for _ in range(7):
                    module(source)
                    meter.observe("FB", module)
                    module(source)
                    meter.observe("FF", module)
                meter.accept_step()
            meter.finish_solve()
        summary = study.summary()
        self.assertEqual(summary["samples"], batch)
        self.assertGreater(summary["average_coupler_power_W"], 0)
        reconstructed = (summary["average_coupler_power_W"] *
                         summary["total_physical_coupler_sites"] *
                         summary["average_physical_time_per_coupler_s"])
        self.assertAlmostEqual(
            reconstructed, summary["average_energy_per_sample_J"], places=24)

    def test_dopri_overshoot_is_truncated_without_changing_solver_step(self):
        module = expanded(self.package())
        study = TCCouplerEnergyStudy(1.3)
        meter = study.meter(0)
        source = torch.full((2, 1, 3, 3), .05, dtype=torch.float64)
        module._tc_measure_coupler_energy = True
        module(source)
        one_stage_current = module._tc_last_coupler_current.clone()
        module._tc_last_coupler_current = None
        physical_duration = 1e-9
        overshooting_step = 1.075e-9
        with torch.no_grad():
            meter.start_solve(source.shape[0], physical_duration)
            meter.begin_step(overshooting_step, "Dopri5")
            for _ in range(7):
                module(source)
                meter.observe("FB", module)
                module(source)
                meter.observe("FF", module)
            meter.accept_step()
            meter.finish_solve()
        summary = study.summary()
        expected = float((2 * 1.3 * one_stage_current.mean() *
                          physical_duration).item())
        self.assertAlmostEqual(summary["average_energy_per_sample_J"],
                               expected, places=24)
        self.assertAlmostEqual(summary["layers"][0]["physical_time_s"],
                               physical_duration, places=18)

    def test_inference_only_guard(self):
        study = TCCouplerEnergyStudy()
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "inference-only"):
            study.meter(0).start_solve(1, 1e-9)

    def test_solver_observer_is_passive_and_counts_only_accepted_duration(self):
        class Observer:
            def __init__(self):
                self.duration = 0.
            def begin_step(self, dt, solver_name):
                self.pending = abs(float(dt))
                self.solver_name = solver_name
            def accept_step(self):
                self.duration += self.pending
            def cancel_step(self):
                self.pending = 0.
        class RHS(torch.nn.Module):
            def forward(self, t, y):
                return torch.cos(t).expand_as(y)

        y0 = torch.zeros(3)
        options = dict(method="dopri5", t0=0., t1=1., t_eval=[1.],
                       h=None, rtol=1e-6, atol=1e-6)
        def tc_rhs():
            rhs = RHS()
            zeros = torch.zeros_like(y0)
            rhs.tc_context = TCNoiseLifecycle(
                [(zeros, zeros)],
                {(0, "sum"): torch.Generator().manual_seed(1),
                 (0, "coupler"): torch.Generator().manual_seed(2)})
            return rhs
        reference = odesolve(tc_rhs(), y0, options)
        measured_rhs = tc_rhs()
        measured_rhs.energy_meter = Observer()
        measured = odesolve(measured_rhs, y0, options)
        torch.testing.assert_close(measured, reference)
        self.assertAlmostEqual(measured_rhs.energy_meter.duration, 1., places=6)


if __name__ == "__main__":
    unittest.main()
