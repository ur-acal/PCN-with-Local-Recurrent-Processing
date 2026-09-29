import os
import sys
import unittest
from unittest.mock import patch

import torch

from TorchDiffEqPack.odesolver import odesolve


class QuadraticRHS(torch.nn.Module):
    def forward(self, _t, y):
        return -0.2 * y + 0.01 * y.square()


class AdaptiveAcceptedStepReuseTests(unittest.TestCase):
    def test_every_training_parser_accepts_environment_and_cli_override(self):
        from mnist_train_eval.mnist_config import parse_args as mnist_args
        from train_ode_cifar import get_args as cifar_args
        from train_ode_imagenet import get_args as imagenet_args

        cases = (
            (cifar_args, ['train_ode_cifar.py']),
            (imagenet_args, ['train_ode_imagenet.py', '--imagenet_root', '/tmp']),
            (lambda: mnist_args(), ['mnist_train']),
        )
        for parser, argv in cases:
            with self.subTest(parser=argv[0]), \
                    patch.dict(os.environ, {'REUSE_ACCEPTED_STEP_TRAINING': 'true'}), \
                    patch.object(sys, 'argv', argv):
                self.assertTrue(parser().reuse_accepted_step_training)
            with self.subTest(parser=argv[0] + '_override'), \
                    patch.dict(os.environ, {'REUSE_ACCEPTED_STEP_TRAINING': 'true'}), \
                    patch.object(sys, 'argv', argv + [
                        '--reuse_accepted_step_training', 'false']):
                self.assertFalse(parser().reuse_accepted_step_training)

    def solve(self, method, reuse, *, safe=True, project=False, initial_step=0.1,
              regenerate=False):
        initial = torch.tensor([[0.2]], requires_grad=True)
        options = dict(
            method=method, t0=0.0, t1=0.2, h=initial_step, t_eval=[0.2],
            rtol=1e-5, atol=1e-7,
            reuse_accepted_step_training=reuse,
            accepted_step_reuse_safe=safe,
            regenerate_graph=regenerate,
        )
        if project:
            options['proj_fn'] = torch.nn.Hardtanh(-1.0, 1.0)
        solver = odesolve(QuadraticRHS(), initial, options, return_solver=True)
        original_adapt = solver.adapt_stepsize
        attempts = [0]

        def reject_first(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                return args[3] / 2, False, True
            return original_adapt(*args, **kwargs)

        with patch.object(solver, 'adapt_stepsize', side_effect=reject_first), \
                patch.object(solver, 'step', wraps=solver.step) as step:
            output, times = solver.integrate(
                initial, 0.0, t_eval=[0.2], return_steps=True)
        gradient = torch.autograd.grad(output.sum(), initial)[0]
        return output, times, gradient, step.call_count

    def test_every_enabled_solver_matches_replay_and_uses_fewer_steps(self):
        for method in ('rk12', 'rk23', 'dopri5'):
            with self.subTest(method=method):
                baseline = self.solve(method, False)
                reused = self.solve(method, True)
                torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
                torch.testing.assert_close(reused[1], baseline[1], rtol=0, atol=0)
                torch.testing.assert_close(reused[2], baseline[2], rtol=0, atol=0)
                self.assertLess(reused[3], baseline[3])

    def test_projected_dopri5_inherits_support(self):
        baseline = self.solve('dopri5', False, project=True)
        reused = self.solve('dopri5', True, project=True)
        torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(reused[2], baseline[2], rtol=0, atol=0)
        self.assertLess(reused[3], baseline[3])

    def solve_inference(self, *, reuse, project=False):
        initial = torch.tensor([[0.2]])
        options = dict(
            method='dopri5', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
            rtol=1e-5, atol=1e-7)
        if project:
            options['proj_fn'] = torch.nn.Hardtanh(-1.0, 1.0)
        solver = odesolve(
            QuadraticRHS(), initial, options, return_solver=True)
        if reuse is not None:
            solver.reuse_accepted_step_inference = reuse
        original_adapt = solver.adapt_stepsize
        attempts = [0]

        def reject_first(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                return args[3] / 2, False, True
            return original_adapt(*args, **kwargs)

        with torch.no_grad(), \
                patch.object(solver, 'adapt_stepsize', side_effect=reject_first), \
                patch.object(solver, 'step', wraps=solver.step) as step:
            output, times = solver.integrate(
                initial, 0.0, t_eval=[0.2], return_steps=True)
        return output, times, step.call_count

    def test_non_tc_dopri5_inference_reuses_without_changing_output(self):
        baseline = self.solve_inference(reuse=False)
        reused = self.solve_inference(reuse=None)
        torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(reused[1], baseline[1], rtol=0, atol=0)
        self.assertLess(reused[2], baseline[2])

    def test_non_tc_projected_dopri5_inference_reuses_without_changing_output(self):
        baseline = self.solve_inference(reuse=False, project=True)
        reused = self.solve_inference(reuse=None, project=True)
        torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(reused[1], baseline[1], rtol=0, atol=0)
        self.assertLess(reused[2], baseline[2])

    def test_non_tc_fresh_randomness_changes_seed_replay_in_inference(self):
        class FreshNoiseRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.draws = []

            def forward(self, _t, y):
                draw = torch.randn_like(y)
                self.draws.append(draw.clone())
                return -0.2 * y + draw * 1e-4

        def solve(reuse):
            torch.manual_seed(37)
            initial = torch.tensor([[0.2]])
            rhs = FreshNoiseRHS()
            solver = odesolve(rhs, initial, dict(
                method='dopri5', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                rtol=1e-3, atol=1e-3), return_solver=True)
            if reuse is not None:
                solver.reuse_accepted_step_inference = reuse
            original_adapt = solver.adapt_stepsize
            attempts = [0]

            def reject_first(*args, **kwargs):
                attempts[0] += 1
                if attempts[0] == 1:
                    return args[3] / 2, False, True
                return original_adapt(*args, **kwargs)

            with torch.no_grad(), patch.object(
                    solver, 'adapt_stepsize', side_effect=reject_first):
                output = solver.integrate(initial, 0.0, t_eval=[0.2])
            return output, len(rhs.draws)

        baseline = solve(False)
        reused = solve(None)
        self.assertFalse(torch.equal(reused[0], baseline[0]))
        self.assertLess(reused[1], baseline[1])

    def test_automatic_initial_step_preserves_legacy_first_step(self):
        baseline = self.solve('dopri5', False, initial_step=None)
        reused = self.solve('dopri5', True, initial_step=None)
        torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(reused[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(reused[2], baseline[2], rtol=0, atol=0)
        self.assertLess(reused[3], baseline[3])

    def test_explicit_safety_gate_keeps_replay(self):
        baseline = self.solve('dopri5', False, safe=False)
        gated = self.solve('dopri5', True, safe=False)
        torch.testing.assert_close(gated[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(gated[2], baseline[2], rtol=0, atol=0)
        self.assertEqual(gated[3], baseline[3])

    def test_regenerate_graph_keeps_search_replay(self):
        baseline = self.solve('dopri5', False, regenerate=True)
        gated = self.solve('dopri5', True, regenerate=True)
        torch.testing.assert_close(gated[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(gated[2], baseline[2], rtol=0, atol=0)
        self.assertEqual(gated[3], baseline[3])

    def test_fixed_grid_ignores_option(self):
        values = []
        for reuse in (False, True):
            initial = torch.tensor([[0.2]], requires_grad=True)
            output = odesolve(QuadraticRHS(), initial, dict(
                method='euler', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                reuse_accepted_step_training=reuse))
            values.append((output, torch.autograd.grad(output.sum(), initial)[0]))
        torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
        torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)

    def test_fresh_rhs_randomness_remains_fresh_for_retries_and_stages(self):
        class FreshNoiseRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.draws = []

            def forward(self, _t, y):
                draw = torch.randn_like(y)
                self.draws.append(draw.detach().clone())
                return -0.2 * y + draw * 1e-4

        step_counts = []
        for reuse in (False, True):
            torch.manual_seed(37)
            initial = torch.tensor([[0.2]], requires_grad=True)
            rhs = FreshNoiseRHS()
            solver = odesolve(rhs, initial, dict(
                method='rk12', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                rtol=1e-3, atol=1e-3,
                reuse_accepted_step_training=reuse), return_solver=True)
            original_adapt = solver.adapt_stepsize
            attempts = [0]

            def reject_first(*args, **kwargs):
                attempts[0] += 1
                if attempts[0] == 1:
                    return args[3] / 2, False, True
                return original_adapt(*args, **kwargs)

            with patch.object(solver, 'adapt_stepsize', side_effect=reject_first), \
                    patch.object(solver, 'step', wraps=solver.step) as step:
                output = solver.integrate(initial, 0.0, t_eval=[0.2])
            torch.autograd.grad(output.sum(), initial)
            self.assertEqual(len(rhs.draws), 2 * step.call_count)
            self.assertEqual(len({draw.item() for draw in rhs.draws}), len(rhs.draws))
            step_counts.append(step.call_count)
        self.assertLess(step_counts[1], step_counts[0])

    def test_sym12async_is_not_enabled(self):
        initial = torch.tensor([[0.2]], requires_grad=True)
        solver = odesolve(QuadraticRHS(), initial, dict(
            method='sym12async', t0=0.0, t1=0.2, h=0.1,
            rtol=1e-5, atol=1e-7,
            reuse_accepted_step_training=True), return_solver=True)
        self.assertFalse(getattr(solver, 'supports_accepted_step_reuse', False))

    def test_ode23s_is_not_enabled(self):
        baseline = self.solve('ode23s', False)
        gated = self.solve('ode23s', True)
        torch.testing.assert_close(gated[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(gated[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(gated[2], baseline[2], rtol=0, atol=0)
        self.assertEqual(gated[3], baseline[3])

    def test_endtime_entry_point_forwards_reuse_controls(self):
        from TorchDiffEqPack.odesolver_mem.odesolver_endtime import odesolve_endtime

        rhs = QuadraticRHS()
        rhs.tc_context = object()
        solver = odesolve_endtime(rhs, torch.tensor([[0.2]]), dict(
            method='rk12', t0=0.0, t1=0.2, h=0.1,
            reuse_accepted_step_training=True,
            accepted_step_reuse_safe=False), return_solver=True)
        self.assertTrue(solver.reuse_accepted_step_training)
        self.assertFalse(solver.accepted_step_reuse_safe)
        self.assertEqual(solver.noise_type, 'addi')

    def test_cached_switch_rhs_is_explicitly_gated(self):
        from pc_conv import PCConvReLU6
        from switch import (ODEXInitFFFBPixelSwitchEfficient,
                            PerturbODEXInitFFFB,
                            ODEXInitFFFBPixelSwitchParallel)

        def block(cls, **kwargs):
            pc = PCConvReLU6(
                inp_chan=2, out_chan=2, kernel_size=3, padding=1, cls=2,
                bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
            return cls(pc_conv=pc, noise_level=0.0, method='dopri5',
                       t_end=0.3, tol=1e-3, **kwargs)

        self.assertFalse(block(ODEXInitFFFBPixelSwitchParallel).
                         option_aca['accepted_step_reuse_safe'])
        self.assertTrue(block(ODEXInitFFFBPixelSwitchParallel,
                              use_cached_patches=False).
                        option_aca['accepted_step_reuse_safe'])
        self.assertFalse(block(ODEXInitFFFBPixelSwitchEfficient).
                         option_aca['accepted_step_reuse_safe'])
        self.assertTrue(block(PerturbODEXInitFFFB).
                        option_aca['accepted_step_reuse_safe'])

    def test_recovery_cli_and_environment_override_saved_config(self):
        from scripts.resume_local_ode_training import (
            apply_training_step_reuse_override)

        with patch.dict(os.environ, {}, clear=True):
            saved = {'reuse_accepted_step_training': True}
            apply_training_step_reuse_override(saved, None)
            self.assertTrue(saved['reuse_accepted_step_training'])
            old = {}
            apply_training_step_reuse_override(old, None)
            self.assertFalse(old['reuse_accepted_step_training'])
        with patch.dict(os.environ, {'REUSE_ACCEPTED_STEP_TRAINING': 'true'}):
            config = {'reuse_accepted_step_training': False}
            apply_training_step_reuse_override(config, None)
            self.assertTrue(config['reuse_accepted_step_training'])
            apply_training_step_reuse_override(config, False)
            self.assertFalse(config['reuse_accepted_step_training'])


if __name__ == '__main__':
    unittest.main()
