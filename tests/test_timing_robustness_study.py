"""Analytic controls for the timing-study measurements, not production changes."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'timing_study', ROOT/'diagnostic_scripts/study_timing_robustness.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class TimingRobustnessStudyTests(unittest.TestCase):
    def test_pcn_loader_callback_restores_production_configuration(self):
        import compute_time_variation
        import validate_timing_noise_scope as loader
        configure = compute_time_variation.configure
        calls = []
        def check_study(*args):
            self.assertIs(compute_time_variation.configure, configure)
            calls.append(args)
        def fake_loader():
            with patch.object(compute_time_variation, 'configure', object()):
                loader.validate('model', None, 'output', 'pcn_s2_22l')
        with patch.object(study.sys, 'argv', ['study', '--model', 'pcn_s2_22l',
                                              '--output', '/tmp/not-created']), \
                patch.object(study, 'study', check_study), patch.object(loader, 'main', fake_loader):
            study.main()
        self.assertEqual(len(calls), 1)

    def test_physical_time_options_preserve_noise_and_step_count(self):
        options = dict(t0=2., t1=5., t_eval=[2., 5.], h=.3, eps=.5)
        actual = study.time_options(options, 1.2)
        self.assertAlmostEqual(actual['t1'], 5.6)
        self.assertAlmostEqual(actual['h'], .36)
        self.assertEqual(actual['eps'], .5)
        self.assertEqual(options['t1'], 5.)

    def test_endpoint_shapes_single_and_multiple_outputs(self):
        y, z = torch.ones(2, 3), torch.ones(2, 4)
        a, b = study.endpoints((torch.stack([y*0, y]), torch.stack([z*0, z])), (y,z))
        self.assertTrue(torch.equal(a, y))
        self.assertTrue(torch.equal(b, z))
        self.assertTrue(torch.equal(study.endpoints(y, y)[0], y))

    def test_raw_motion_and_clamp_blocking_are_separated(self):
        rail = .1
        state = torch.full((3, 2), rail)
        metrics = study.state_metrics(state, torch.ones_like(state), .1,
                                      lambda x: x.clamp(-rail, rail), rail)
        self.assertAlmostEqual(metrics['raw_motion_over_state'], 1.)
        self.assertEqual(metrics['projected_motion_over_state'], 0.)
        self.assertEqual(metrics['outward_drift_energy_fraction'], 1.)
        self.assertEqual(metrics['fraction_on_rail'], 1.)

    def test_linear_integrator_endpoint_response(self):
        from TorchDiffEqPack.odesolver import odesolve
        initial = torch.zeros(4, 2)
        drift = torch.full_like(initial, 2e7)
        options = dict(t0=0., t1=1e-9, t_eval=[1e-9], method='euler', h=1e-10,
                       proj_fn=lambda x: x.clamp(-.1,.1), eps=0.)
        for factor in (.8, 1., 1.2):
            result = odesolve(lambda t,y: drift, initial, study.time_options(options,factor))
            torch.testing.assert_close(result, factor*drift*1e-9, atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
