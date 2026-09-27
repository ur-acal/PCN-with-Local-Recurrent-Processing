import os
import sys
import unittest
from unittest.mock import patch

import torch

from TorchDiffEqPack.odesolver import odesolve


class ParameterRHS(torch.nn.Module):
    def __init__(self, value=-0.2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.calls = 0

    def forward(self, _t, y):
        self.calls += 1
        return self.weight * y + 0.01 * y.square()


def solve(method, checkpoint_rhs, *, project=False, full_traj=False):
    initial = torch.tensor([[0.2]], requires_grad=True)
    rhs = ParameterRHS()
    options = dict(
        method=method, t0=0.0, t1=0.2, h=0.1, t_eval=[0.1, 0.2],
        rtol=1e-5, atol=1e-7,
        checkpoint_ode_rhs_training=checkpoint_rhs)
    if project:
        options['proj_fn'] = torch.nn.Hardtanh(-1.0, 1.0)
    output = odesolve(rhs, initial, options, full_traj=full_traj)
    calls_before_backward = rhs.calls
    def differentiable_sum(value):
        if torch.is_tensor(value):
            return value.sum() if value.requires_grad else 0
        return sum(differentiable_sum(item) for item in value)
    loss = differentiable_sum(output)
    gradients = torch.autograd.grad(loss, (initial, rhs.weight))
    return output, gradients, calls_before_backward, rhs.calls


class ODERHSCheckpointingTests(unittest.TestCase):
    def test_supported_solvers_match_outputs_and_gradients(self):
        for method in ('euler', 'rk2', 'rk4', 'rk12', 'rk23', 'dopri5'):
            with self.subTest(method=method):
                baseline = solve(method, False)
                checkpointed = solve(method, True)
                torch.testing.assert_close(checkpointed[0], baseline[0], rtol=0, atol=0)
                for actual, expected in zip(checkpointed[1], baseline[1]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(checkpointed[2], baseline[2])
                self.assertGreater(checkpointed[3], checkpointed[2])

    def test_projected_and_full_trajectory_paths(self):
        for project, full_traj in ((True, False), (False, True)):
            with self.subTest(project=project, full_traj=full_traj):
                baseline = solve('dopri5', False, project=project, full_traj=full_traj)
                checkpointed = solve('dopri5', True, project=project, full_traj=full_traj)
                torch.testing.assert_close(checkpointed[0], baseline[0], rtol=0, atol=0)
                for actual, expected in zip(checkpointed[1], baseline[1]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_forced_rejection_and_regenerated_graph_match(self):
        def run(enabled, regenerate):
            initial = torch.tensor([[0.2]], requires_grad=True)
            rhs = ParameterRHS()
            solver = odesolve(rhs, initial, dict(
                method='rk23', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                rtol=1e-5, atol=1e-7, regenerate_graph=regenerate,
                checkpoint_ode_rhs_training=enabled), return_solver=True)
            original_adapt = solver.adapt_stepsize
            attempts = [0]

            def reject_first(*args, **kwargs):
                attempts[0] += 1
                if attempts[0] == 1:
                    return args[3] / 2, False, True
                return original_adapt(*args, **kwargs)

            with patch.object(solver, 'adapt_stepsize', side_effect=reject_first):
                output = solver.integrate(initial, 0.0, t_eval=[0.2])
            gradients = torch.autograd.grad(output.sum(), (initial, rhs.weight))
            return output, gradients

        for regenerate in (False, True):
            with self.subTest(regenerate=regenerate):
                baseline, checkpointed = run(False, regenerate), run(True, regenerate)
                torch.testing.assert_close(checkpointed[0], baseline[0], rtol=0, atol=0)
                for actual, expected in zip(checkpointed[1], baseline[1]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tuple_state_matches(self):
        class TupleRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-0.2))

            def forward(self, _t, state):
                y, z = state
                return self.weight * y + z, self.weight * z - y

        results = []
        for enabled in (False, True):
            y = torch.tensor([[0.2]], requires_grad=True)
            z = torch.tensor([[0.1]], requires_grad=True)
            rhs = TupleRHS()
            output = odesolve(rhs, (y, z), dict(
                method='rk23', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                rtol=1e-5, atol=1e-7,
                checkpoint_ode_rhs_training=enabled))
            gradients = torch.autograd.grad(
                sum(value.sum() for value in output), (y, z, rhs.weight))
            results.append((output, gradients))
        for actual, expected in zip(results[1][0], results[0][0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tc_noise_index_is_replayed_and_restored(self):
        from tc_nonidealities import TCNoiseLifecycle

        class NoiseRHS(torch.nn.Module):
            def __init__(self, context):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-0.2))
                self.tc_context = context
                self.indices = []

            def forward(self, _t, y):
                self.indices.append(self.tc_context.index)
                return self.weight * y + self.tc_context.normal(y, 0) * 1e-3

        results = []
        for enabled in (False, True):
            generator = torch.Generator().manual_seed(17)
            coefficient = torch.tensor(1.0)
            context = TCNoiseLifecycle(
                [(coefficient, torch.tensor(0.0))],
                {(0, 'sum'): generator, (0, 'coupler'): generator})
            rhs = NoiseRHS(context)
            initial = torch.tensor([[0.2]], requires_grad=True)
            output = odesolve(rhs, initial, dict(
                method='euler', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                checkpoint_ode_rhs_training=enabled))
            gradients = torch.autograd.grad(output.sum(), (initial, rhs.weight))
            results.append((output, gradients, context.index, rhs.indices))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
        for actual, expected in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(results[1][2], 2)
        self.assertEqual(results[1][3][:2], [0, 1])
        self.assertEqual(results[1][3][2:], [1, 0])

    def test_no_grad_does_not_recompute(self):
        rhs = ParameterRHS()
        with torch.no_grad():
            odesolve(rhs, torch.tensor([[0.2]]), dict(
                method='rk2', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                checkpoint_ode_rhs_training=True))
        self.assertEqual(rhs.calls, 4)

        metered = ParameterRHS()
        metered.energy_meter = object()
        with torch.no_grad():
            solver = odesolve(metered, torch.tensor([[0.2]]), dict(
                method='ode23s', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True), return_solver=True)
        self.assertTrue(solver.checkpoint_ode_rhs_training)
        self.assertFalse(solver.checkpoint_ode_rhs_active)

    def test_global_rng_is_preserved_through_recomputation(self):
        class RandomRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-0.2))

            def forward(self, _t, y):
                return self.weight * y + torch.randn_like(y) * 1e-4

        results = []
        for enabled in (False, True):
            torch.manual_seed(31)
            initial = torch.tensor([[0.2]], requires_grad=True)
            rhs = RandomRHS()
            output = odesolve(rhs, initial, dict(
                method='rk4', t0=0.0, t1=0.2, h=0.1, t_eval=[0.2],
                checkpoint_ode_rhs_training=enabled))
            gradients = torch.autograd.grad(output.sum(), (initial, rhs.weight))
            results.append((output, gradients, torch.randn(1)))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
        for actual, expected in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(results[1][2], results[0][2], rtol=0, atol=0)

    def test_deferred_and_stateful_paths_fail_explicitly(self):
        initial = torch.tensor([[0.2]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'not implemented for ode23s'):
            odesolve(ParameterRHS(), initial, dict(
                method='ode23s', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        sym_rhs = ParameterRHS()
        with self.assertRaisesRegex(ValueError, 'not implemented for sym12async'):
            odesolve(sym_rhs, initial, dict(
                method='sym12async', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        self.assertEqual(sym_rhs.calls, 0)
        with self.assertRaisesRegex(ValueError, 'cached RHS'):
            odesolve(ParameterRHS(), initial, dict(
                method='dopri5', t0=0.0, t1=0.2, h=0.1,
                accepted_step_reuse_safe=False,
                checkpoint_ode_rhs_training=True))
        metered = ParameterRHS()
        metered.energy_meter = object()
        with self.assertRaisesRegex(ValueError, 'energy metering'):
            odesolve(metered, initial, dict(
                method='dopri5', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        class BatchNormRHS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.BatchNorm1d(2)

            def forward(self, _t, y):
                return self.norm(y)

        with self.assertRaisesRegex(ValueError, 'BatchNorm'):
            odesolve(BatchNormRHS(), torch.ones(2, 2, requires_grad=True), dict(
                method='rk2', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        closure_norm = torch.nn.BatchNorm1d(2)
        def closure_rhs(_t, y):
            return closure_norm(y)
        with self.assertRaisesRegex(ValueError, 'BatchNorm'):
            odesolve(closure_rhs, torch.ones(2, 2, requires_grad=True), dict(
                method='rk2', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        from TorchDiffEqPack.odesolver_mem.odesolver_endtime import odesolve_endtime
        with self.assertRaisesRegex(ValueError, 'adjoint solver'):
            odesolve_endtime(ParameterRHS(), initial, dict(
                method='dopri5', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))
        from TorchDiffEqPack.odesolver_mem.adjoint import odesolve_adjoint
        with self.assertRaisesRegex(ValueError, 'adjoint solver'):
            odesolve_adjoint(ParameterRHS(), initial, dict(
                method='dopri5', t0=0.0, t1=0.2, h=0.1,
                checkpoint_ode_rhs_training=True))

    def test_training_parsers_support_environment_and_cli_override(self):
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
                    patch.dict(os.environ, {'CHECKPOINT_ODE_RHS_TRAINING': 'true'}), \
                    patch.object(sys, 'argv', argv):
                self.assertTrue(parser().checkpoint_ode_rhs_training)
            with self.subTest(parser=argv[0] + '_override'), \
                    patch.dict(os.environ, {'CHECKPOINT_ODE_RHS_TRAINING': 'true'}), \
                    patch.object(sys, 'argv', argv + [
                        '--checkpoint_ode_rhs_training', 'false']):
                self.assertFalse(parser().checkpoint_ode_rhs_training)

    def test_block_and_derived_options_inherit_setting(self):
        from ode_pc import S2NoMinusZChargeZ
        from pc_conv import PCConvReLU6

        pc = PCConvReLU6(
            inp_chan=2, out_chan=2, kernel_size=3, padding=1, cls=2,
            bypass=False, tie_weights=False, tie_bp=False, layer_idx=0)
        block = S2NoMinusZChargeZ(
            pc_conv=pc, noise_level=0.0, method='dopri5', t_end=0.3,
            tol=1e-3, checkpoint_ode_rhs_training=True)
        self.assertTrue(block.option_aca['checkpoint_ode_rhs_training'])
        self.assertTrue(block.option_init['checkpoint_ode_rhs_training'])

    def test_recovery_override_precedence(self):
        from scripts.resume_local_ode_training import apply_rhs_checkpoint_override

        with patch.dict(os.environ, {}, clear=True):
            saved = {'checkpoint_ode_rhs_training': True}
            apply_rhs_checkpoint_override(saved, None)
            self.assertTrue(saved['checkpoint_ode_rhs_training'])
            old = {}
            apply_rhs_checkpoint_override(old, None)
            self.assertFalse(old['checkpoint_ode_rhs_training'])
        with patch.dict(os.environ, {'CHECKPOINT_ODE_RHS_TRAINING': 'true'}):
            config = {'checkpoint_ode_rhs_training': False}
            apply_rhs_checkpoint_override(config, None)
            self.assertTrue(config['checkpoint_ode_rhs_training'])
            apply_rhs_checkpoint_override(config, False)
            self.assertFalse(config['checkpoint_ode_rhs_training'])


if __name__ == '__main__':
    unittest.main()
