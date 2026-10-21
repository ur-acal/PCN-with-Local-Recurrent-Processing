import importlib
import logging
import os
import sys
import unittest
from unittest.mock import patch

import torch

from TorchDiffEqPack import odesolve
from TorchDiffEqPack.odesolver_mem.odesolver_endtime import odesolve_endtime


class ParameterizedRHS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(-0.2))

    def forward(self, _t, y):
        if isinstance(y, tuple):
            return tuple(self.rate * value + .01 * value.square() for value in y)
        return self.rate * y + .01 * y.square()


class AcceptedStepReuseTests(unittest.TestCase):
    def solve(self, method='dopri5', reuse=False, training=True, tuple_state=False,
              project=False, initial_step=.1, regenerate=False, safe=True,
              endtime=False, predefined=False, full_traj=False):
        rhs = ParameterizedRHS()
        initial = torch.tensor([[.2]], requires_grad=training)
        state = (initial, initial * 2) if tuple_state else initial
        options = dict(method=method, t0=0., t1=.2, h=initial_step, t_eval=[.2],
                       rtol=1e-5, atol=1e-7, regenerate_graph=regenerate,
                       reuse_accepted_step_training=reuse,
                       reuse_accepted_step_inference=reuse,
                       accepted_step_reuse_safe=safe)
        if project:
            options['proj_fn'] = torch.nn.Hardtanh(-1., 1.)
        entry = odesolve_endtime if endtime else odesolve
        with torch.set_grad_enabled(training):
            solver = entry(rhs, state, options, return_solver=True)
            adapt = getattr(solver, 'adapt_stepsize', None)
            calls = []

            def reject_first(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return args[3] / 2, False, True
                return adapt(*args, **kwargs)

            context = patch.object(solver, 'adapt_stepsize', side_effect=reject_first) if adapt else patch.object(solver, 'print_time', False)
            with context, patch.object(solver, 'step', wraps=solver.step) as step:
                output, times = solver.integrate(
                    state, 0., t_eval=[.2], return_steps=True, full_traj=full_traj,
                    predefine_steps=torch.tensor([.1, .2]) if predefined else None)
            outputs = output if isinstance(output, tuple) else (output,)
            gradients = torch.autograd.grad(sum(x.sum() for x in outputs), (initial, rhs.rate)) if training else ()
        return tuple(x.detach() for x in outputs), torch.as_tensor(times), gradients, step.call_count

    def assert_equivalent(self, baseline, reused, fewer=True):
        for actual, expected in zip(reused[0], baseline[0]):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(reused[1], baseline[1], atol=1e-7, rtol=0)
        for actual, expected in zip(reused[2], baseline[2]):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        if fewer:
            self.assertLess(reused[3], baseline[3])
        else:
            self.assertEqual(reused[3], baseline[3])

    def test_supported_solvers_training_and_inference_with_rejection(self):
        for method in ('rk12', 'rk23', 'dopri5'):
            for training in (False, True):
                with self.subTest(method=method, training=training):
                    self.assert_equivalent(self.solve(method, False, training), self.solve(method, True, training))

    def test_projection_tuple_automatic_step_and_full_trajectory(self):
        for options in (dict(project=True), dict(tuple_state=True),
                        dict(initial_step=None), dict(full_traj=True), dict(endtime=True)):
            with self.subTest(options=options):
                self.assert_equivalent(self.solve(reuse=False, **options), self.solve(reuse=True, **options))

    def test_fixed_predefined_and_fallback_paths(self):
        for options in (dict(method='euler'), dict(predefined=True),
                        dict(safe=False), dict(regenerate=True), dict(method='ode23s')):
            with self.subTest(options=options):
                self.assert_equivalent(self.solve(reuse=False, **options), self.solve(reuse=True, **options), fewer=False)
        module = importlib.import_module('TorchDiffEqPack.odesolver.adaptive_grid_solver')
        with patch.object(module, 'reload_state', True):
            self.assert_equivalent(self.solve(reuse=False, training=False),
                                   self.solve(reuse=True, training=False), fewer=False)

    def test_entrypoints_forward_capability_and_options(self):
        for entry in (odesolve, odesolve_endtime):
            rhs = ParameterizedRHS()
            rhs.accepted_step_reuse_safe = False
            solver = entry(rhs, torch.ones(1), dict(method='rk12', t0=0., t1=.2, h=.1,
                           reuse_accepted_step_training=True, reuse_accepted_step_inference=False), return_solver=True)
            self.assertTrue(solver.reuse_accepted_step_training)
            self.assertFalse(solver.reuse_accepted_step_inference)
            self.assertFalse(solver.accepted_step_reuse_safe)

    def test_fresh_random_rhs_preserves_fresh_draws(self):
        class RandomRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.draws = []

            def forward(self, _t, y):
                draw = torch.randn_like(y)
                self.draws.append(draw.detach())
                return -.2 * y + 1e-5 * draw

        counts = []
        for reuse in (False, True):
            torch.manual_seed(123)
            rhs = RandomRHS()
            initial = torch.ones(1, requires_grad=True)
            solver = odesolve(rhs, initial, dict(method='rk12', t0=0., t1=.2, h=.1,
                              rtol=1e-3, atol=1e-3, reuse_accepted_step_training=reuse), return_solver=True)
            with patch.object(solver, 'step', wraps=solver.step) as step:
                output = solver.integrate(initial, 0., t_eval=[.2])
            torch.autograd.grad(output.sum(), initial)
            self.assertEqual(len(rhs.draws), 2 * step.call_count)
            self.assertEqual(len(set(draw.item() for draw in rhs.draws)), len(rhs.draws))
            counts.append(step.call_count)
        self.assertLess(counts[1], counts[0])

    def test_training_cli_environment_override_and_block_forwarding(self):
        from train_ode_cifar import get_args as cifar_args
        from train_ode_imagenet import get_args as imagenet_args
        from ode_pc import ODEXInitFFFB, make_ode_block
        from pc_model import PCNetNoBatchNorm

        for parser, argv in ((cifar_args, ['train']), (imagenet_args, ['train', '--imagenet_root', '/tmp'])):
            with patch.dict(os.environ, REUSE_ACCEPTED_STEP_TRAINING='true'), patch.object(sys, 'argv', argv):
                self.assertTrue(parser().reuse_accepted_step_training)
            with patch.dict(os.environ, REUSE_ACCEPTED_STEP_TRAINING='true'), patch.object(sys, 'argv', argv + ['--reuse_accepted_step_training', 'false']):
                self.assertFalse(parser().reuse_accepted_step_training)
        model = PCNetNoBatchNorm(inp_channels=[3], out_channels=[4], max_pool=[0], cls=0)
        make_ode_block(model, ODEXInitFFFB, method='dopri5', t_end=.2, n_steps=5, reuse_accepted_step_training=True)
        self.assertTrue(model.PcConvs[0].option_aca['reuse_accepted_step_training'])


class DiagnosticLoggingTests(unittest.TestCase):
    def test_stats_are_skipped_when_disabled_and_preserved_when_enabled(self):
        from pc_model import PCNetNoBatchNorm
        model = PCNetNoBatchNorm(inp_channels=[3], out_channels=[4], max_pool=[0], cls=0)
        inputs = torch.randn(2, 3, 4, 4)
        logger = logging.getLogger('pc_model')
        old_level = logger.level
        try:
            logger.setLevel(logging.WARNING)
            with patch.object(torch.Tensor, 'median', side_effect=AssertionError('disabled diagnostic evaluated')):
                expected = model(inputs)
            with self.assertLogs(logger, level=logging.INFO) as messages:
                actual = model(inputs)
            self.assertTrue(any('Mean=' in message for message in messages.output))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        finally:
            logger.setLevel(old_level)


if __name__ == '__main__':
    unittest.main()
