import argparse
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from compute_time_variation import ComputeTimeVariation, configure, add_arguments
from test_tc_dense_training import make_block, wrap
from test_tc_feedforward import block


class ComputeTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_validation_and_zero_is_exact_noop(self):
        for bad in (-.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ComputeTimeVariation(bad, 0)
        timing = ComputeTimeVariation()
        options = dict(t0=0., t1=2., t_eval=[0., 2.], h=.1)
        before = torch.get_rng_state().clone()
        self.assertIs(timing.options(options), options)
        nominal = torch.tensor(2.)
        self.assertIs(timing.duration(nominal, 'z'), nominal)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(timing._generators)
        model = block()
        keys = set(model.state_dict())
        self.assertEqual(configure(model, SimpleNamespace()), 0)
        self.assertFalse(hasattr(model, 'compute_time_variation'))
        self.assertEqual(keys, set(model.state_dict()))

    def test_fixed_noise_separate_and_positive(self):
        fixed = ComputeTimeVariation(.2, 0., 9)
        self.assertEqual(fixed.factor('z'), fixed.factor('z'))
        self.assertNotEqual(fixed.factor('z'), fixed.factor('y'))
        both = ComputeTimeVariation(.2, .2, 9)
        x, y = both.factor('z'), both.factor('z')
        self.assertNotEqual(x, y)
        self.assertEqual(fixed._compute_time_fixed_offsets['z'],
                         both._compute_time_fixed_offsets['z'])
        values = [ComputeTimeVariation(2., 2., seed).factor('ode') for seed in range(50)]
        self.assertTrue(all(v > 0 for v in values))
        before = torch.get_rng_state().clone()
        both.factor('y')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_per_sample_sampling_and_fixed_assignment(self):
        batch = ComputeTimeVariation(.2, .2, 4096)
        sample = ComputeTimeVariation(.2, .2, 4096, scope='per_sample')
        before = torch.get_rng_state().clone()
        ref = torch.zeros(8, 2, 3, 3)
        factors = sample.sample_factors('ode', ref, 2.)
        self.assertEqual(factors.shape, (8,))
        self.assertEqual(factors.unique().numel(), 8)
        batch.factor('ode')
        self.assertEqual(batch._compute_time_fixed_offsets, sample._compute_time_fixed_offsets)
        fixed = dict(sample._compute_time_fixed_offsets)
        again = sample.sample_factors('ode', ref[:3], 2.)
        self.assertEqual(again.shape, (3,))
        self.assertFalse(torch.equal(factors[:3], again))
        self.assertEqual(sample._compute_time_fixed_offsets, fixed)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_per_sample_rescales_both_state_derivatives_and_diffusion(self):
        timing = ComputeTimeVariation(.2, .2, scope='per_sample')
        y0 = (torch.ones(4, 2, 3, 3), torch.ones(4, 3, 3, 3))
        def rhs(t, states):
            return tuple(2. * state for state in states)
        rhs.tc_context = object()
        options = dict(t0=0., t1=2., t_eval=[2.], h=.2, eps=(.1, .2))
        func, actual = timing.prepare_solve(rhs, y0, options)
        factor = func.factor.reshape(4, 1, 1, 1)
        self.assertIs(func.tc_context, rhs.tc_context)
        for key in ('t0', 't1', 't_eval', 'h'):
            self.assertEqual(actual[key], options[key])
        for got, state, eps in zip(func(0., y0), y0, actual['eps']):
            torch.testing.assert_close(got, 2. * state * factor)
        torch.testing.assert_close(actual['eps'][0], .1 * factor.sqrt())
        torch.testing.assert_close(actual['eps'][1], .2 * factor.sqrt())
        self.assertEqual(options['eps'], (.1, .2))
        rhs.energy_meter = object()
        with self.assertRaisesRegex(ValueError, 'energy metering'):
            timing.prepare_solve(rhs, y0, options)

    def test_per_sample_physical_training_and_inference(self):
        for make in (lambda: wrap(make_block(False)), lambda: wrap(make_block(True)),
                     lambda: block(tc_method='euler', tc_num_steps=10)):
            model = make()
            configure(model, SimpleNamespace(compute_time_fixed_std=.2,
                      compute_time_noise_std=.2, timing_noise_scope='per_sample'))
            ref = next(model.parameters())
            x = torch.full((3, 2, 3, 3), .01, dtype=ref.dtype, requires_grad=True)
            out = model(x)
            out.sum().backward()
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(x[:1])).all())

    def test_recovery_rejects_changed_timing_scope(self):
        from pathlib import Path
        import tempfile
        from training_recovery import restore_latest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pth'
            torch.save(dict(training_recovery=dict(version=1, config={})), path)
            with self.assertRaisesRegex(ValueError, 'unchanged timing_noise_scope'):
                restore_latest(SimpleNamespace(recovery_checkpoint=path,
                    recovery_config=dict(timing_noise_scope='per_sample')))

    def test_options_do_not_accumulate_or_replace_unrelated_settings(self):
        projection = lambda x: x
        opts = dict(t0=1., t1=3., t_eval=[1., 2., 3.], h=.1,
                    rtol=1e-6, eps=.02, proj_fn=projection)
        timing = ComputeTimeVariation(.2, 0., 7)
        actual = timing.options(opts)
        factor = timing.last['ode']['factor']
        self.assertEqual(actual['t0'], 1.)
        self.assertEqual(actual['t1'], 1. + 2. * factor)
        self.assertEqual(actual['t_eval'], [1., 1.+factor, 1.+2.*factor])
        self.assertEqual(actual['h'], .1*factor)
        self.assertEqual(actual, timing.options(opts))
        self.assertEqual(opts['t1'], 3.)
        self.assertIs(actual['proj_fn'], projection)
        self.assertEqual(actual['eps'], opts['eps'])

    def test_zero_matches_one_two_state_and_cnn_outputs_and_gradients(self):
        for make in (lambda: wrap(make_block(False)), lambda: wrap(make_block(True)),
                     lambda: block(one_shot_conv=True)):
            reference = make()
            candidate = copy.deepcopy(reference)
            configure(candidate, SimpleNamespace(compute_time_fixed_std=0., compute_time_noise_std=0.))
            dtype = next(reference.parameters()).dtype
            x = torch.full((2, 2, 3, 3), .01, dtype=dtype, requires_grad=True)
            z = x.detach().clone().requires_grad_()
            a, b = reference(x), candidate(z)
            self.assertTrue(torch.equal(a, b))
            a.sum().backward()
            b.sum().backward()
            self.assertTrue(torch.equal(x.grad, z.grad))

    def test_trial_and_layer_seeds_are_independent_and_reproducible(self):
        args = SimpleNamespace(compute_time_fixed_std=.2, compute_time_noise_std=.1,
                               hardware_seed=4096)
        model = nn.Sequential(block(), block())
        model[1].layer_idx = 1
        configure(model, args)
        original = [m.compute_time_variation.factor('z') for m in model]
        self.assertNotEqual(*original)
        configure(model, args)
        self.assertEqual(original, [m.compute_time_variation.factor('z') for m in model])
        configure(model, args, trial=1)
        self.assertNotEqual(original, [m.compute_time_variation.factor('z') for m in model])
        configure(model, SimpleNamespace())
        self.assertTrue(all(not hasattr(m, 'compute_time_variation') for m in model))

    def test_exact_recovery_rejects_changed_timing_stds(self):
        from pathlib import Path
        import tempfile
        from training_recovery import restore_latest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.pth'
            saved = dict(compute_time_fixed_std=.1, compute_time_noise_std=.2)
            torch.save(dict(training_recovery=dict(version=1, config=saved)), path)
            for key in saved:
                current = dict(saved, **{key: 0.})
                with self.assertRaisesRegex(ValueError, 'unchanged compute_time_'):
                    restore_latest(SimpleNamespace(recovery_checkpoint=path, recovery_config=current))
            torch.save(dict(training_recovery=dict(version=1, config={})), path)
            with self.assertRaisesRegex(ValueError, 'unchanged compute_time_'):
                restore_latest(SimpleNamespace(recovery_checkpoint=path, recovery_config=saved))

    def test_pcn_one_and_two_state_qat_forward_backward_and_full_trajectory(self):
        for two in (False, True):
            with self.subTest(two=two):
                wrapper = wrap(make_block(two))
                model = wrapper.ode_block
                checkpoint_keys = set(wrapper.state_dict())
                configure(model, SimpleNamespace(compute_time_fixed_std=.1,
                          compute_time_noise_std=.2, data_seed=42))
                self.assertEqual(set(wrapper.state_dict()), checkpoint_keys)
                nominal = None
                options = []
                import ode_pc
                solve = ode_pc.aca_ode_solve
                def capture(fn, y0, opts, **kwargs):
                    options.append(opts)
                    return solve(fn, y0, opts, **kwargs)
                x = torch.full((2, 2, 3, 3), .01, requires_grad=True)
                with patch('ode_pc.aca_ode_solve', capture):
                    out = wrapper(x)
                    nominal = float(model.option_aca['t1'])
                    out.sum().backward()
                    wrapper(x)
                    model.forward_full_steps(x.detach() * .1)
                self.assertEqual(len(options), 3)
                self.assertNotEqual(float(options[0]['t1']), nominal)
                self.assertNotEqual(float(options[1]['t1']), float(options[0]['t1']))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(float(model.option_aca['t1']), nominal)

    def test_cnn_stages_and_one_shot_and_euler_duration(self):
        for one_shot in (False, True):
            model = block(one_shot_conv=one_shot, **({} if one_shot else
                          dict(tc_method='euler', tc_num_steps=10)))
            configure(model, SimpleNamespace(compute_time_fixed_std=.2,
                      compute_time_noise_std=.1, data_seed=42))
            x = torch.full((2, 2, 3, 3), .01, dtype=torch.float64, requires_grad=True)
            import physical_feedforward_tc
            solve = physical_feedforward_tc.odesolve
            options = []
            def capture(fn, y, opts):
                options.append(opts)
                return solve(fn, y, opts)
            with patch('physical_feedforward_tc.odesolve', capture):
                y = model(x)
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            state = model.compute_time_variation.last['z']
            self.assertAlmostEqual(state['actual_seconds'] / state['nominal_seconds'], state['factor'])
            if not one_shot:
                self.assertAlmostEqual(options[0]['h'] * 10, options[0]['t1'])
            self.assertNotEqual(model.compute_time_variation.factor('y'), state['factor'])

    def test_state1_timing_options_include_current_solve_diffusion(self):
        from ode_pc import ODEWrapper1State
        import ode_pc
        solve = ode_pc.aca_ode_solve
        for qat in (False, True):
            for training in (False, True):
                with self.subTest(qat=qat, training=training):
                    model = make_block()
                    noise = dict(enable_summing_current_noise=True,
                                 enable_coupler_noise=True,
                                 summing_current_p=6e-13, coupler_noise_p=6e-13)
                    wrapper = (wrap(model, **noise) if qat else ODEWrapper1State(
                        ode_block=model, tc_nonidealities=True,
                        R=1e4, R_max=150e3, C=49e-15, k=1e3, v_dd=.1,
                        state_bound=1., w_bits=5, thermal_noise=False,
                        is_first=True, is_last=True, **noise))
                    wrapper.train(training)
                    configure(model, SimpleNamespace(compute_time_fixed_std=.2,
                              compute_time_noise_std=.1, data_seed=42))
                    captured = []

                    def capture(fn, y0, options, **kwargs):
                        self.assertIsNotNone(options.get('eps'))
                        self.assertTrue(torch.any(torch.as_tensor(options['eps']) > 0))
                        torch.testing.assert_close(options['eps'], model.option_aca['eps'])
                        self.assertIsNotNone(fn.tc_context)
                        captured.append(options)
                        return solve(fn, y0, options, **kwargs)

                    x = torch.full((2, 2, 3, 3), .01, requires_grad=training)
                    with patch('ode_pc.aca_ode_solve', capture), torch.set_grad_enabled(training):
                        for _ in range(2):
                            output = wrapper(x)
                            self.assertTrue(torch.isfinite(output).all())
                            if training:
                                output.sum().backward()
                    self.assertEqual(len(captured), 2)
                    if training:
                        self.assertTrue(torch.isfinite(x.grad).all())

    def test_recovery_and_memory_probe_restore_fixed_and_dynamic_draws(self):
        from training_recovery import _runtime
        from checkpoint_memory_profiler import _capture_runtime, _restore_runtime
        model = block()
        configure(model, SimpleNamespace(compute_time_fixed_std=.2, compute_time_noise_std=.1))
        timing = model.compute_time_variation
        snapshot = _capture_runtime(model)
        expected = timing.factor('z')
        _restore_runtime(snapshot, 'cpu')
        self.assertEqual(timing.factor('z'), expected)
        runtime = copy.deepcopy(_runtime(model))
        expected = timing.factor('z')
        for name, attrs in runtime.items():
            for key, value in attrs.items():
                setattr(model.get_submodule(name), key, value)
        self.assertEqual(timing.factor('z'), expected)

    def test_two_cnn_convolutions_get_separate_durations(self):
        from physical_feedforward_tc import TCPhysicalCIFARBasicBlock
        model = TCPhysicalCIFARBasicBlock(
            nn.Conv2d(2, 2, 1, bias=False), nn.Conv2d(2, 2, 1, bias=False),
            tc_method='euler', tc_num_steps=10)
        configure(model, SimpleNamespace(compute_time_fixed_std=.2, compute_time_noise_std=.1))
        model(torch.full((2, 2, 3, 3), .01))
        timing = model.compute_time_variation.last
        self.assertEqual(set(timing), {'z', 'y'})
        self.assertNotEqual(timing['z']['factor'], timing['y']['factor'])
        for row in timing.values():
            self.assertAlmostEqual(row['actual_seconds']/row['nominal_seconds'], row['factor'])

    def test_cnn_diffusion_uses_perturbed_not_nominal_duration(self):
        model = block(one_shot_conv=True)
        configure(model, SimpleNamespace(compute_time_fixed_std=.2, compute_time_noise_std=0.))
        source = torch.full((2, 2, 3, 3), .01, dtype=torch.float64)
        eps = torch.full_like(source, .1)
        context = SimpleNamespace(normal=lambda state, index: torch.ones_like(state))
        with patch.object(model, '_noise_context', return_value=(context, eps)):
            actual = model._run_stage(model.conv1, source, 'z')
        duration = model.compute_time_variation.last['z']['actual_seconds']
        drift = model._current(model.conv1, source)/(model.R*model.C)
        expected = model.project_state(duration*drift + duration**.5*eps)
        torch.testing.assert_close(actual, expected)

    def test_parsers_environment_and_unitless_pretraining(self):
        with patch.dict('os.environ', COMPUTE_TIME_FIXED_STD='0.1', COMPUTE_TIME_NOISE_STD='0.2'):
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args([])
            self.assertEqual((args.compute_time_fixed_std, args.compute_time_noise_std), (.1, .2))
        with self.assertRaises(ValueError):
            configure(nn.Identity(), args)
        args.physical_pretraining = True
        self.assertEqual(configure(nn.Identity(), args), 0)


if __name__ == '__main__':
    unittest.main()
