import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

from TorchDiffEqPack.odesolver import odesolve
from checkpoint_memory_profiler import (
    _TrainerSnapshot,
    _ProfiledFirstEpochLoader,
    apply_checkpoint_ode_rhs_portion,
    checkpoint_layer_count,
    find_minimum_safe_checkpoint_layers,
    parse_checkpoint_ode_rhs_portion,
    set_checkpoint_ode_rhs_layer_count,
)


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
    def test_profiled_batches_continue_original_worker_iterator(self):
        class Loader:
            def __init__(self):
                self.epochs = 0

            def __len__(self):
                return 4

            def __iter__(self):
                self.epochs += 1
                yield from range(4)

        loader = Loader()
        iterator = iter(loader)
        batches = [next(iterator), next(iterator)]
        replay = _ProfiledFirstEpochLoader(loader, batches, iterator)
        self.assertEqual(list(replay), [0, 1, 2, 3])
        self.assertEqual(list(replay), [0, 1, 2, 3])
        self.assertEqual(loader.epochs, 2)

    def test_profiler_snapshot_restores_lazy_crd_optimizer_group(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loader_generator = torch.Generator().manual_seed(17)
        trainer = types.SimpleNamespace(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            warmup_scheduler=None,
            grad_scaler=None,
            scaler=None,
            _feature_kd_loss=None,
            _crd_loss=None,
            _crd_initialized=False,
            train_dataloader=types.SimpleNamespace(
                generator=loader_generator),
        )
        snapshot = _TrainerSnapshot(trainer)
        crd = torch.nn.Linear(2, 2)
        trainer._crd_loss = crd
        trainer._crd_initialized = True
        optimizer.add_param_group({"params": crd.parameters()})
        optimizer.state[next(crd.parameters())]["momentum_buffer"] = torch.ones(2, 2)
        torch.rand(1, generator=loader_generator)

        snapshot.restore(trainer)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIsNone(trainer._crd_loss)
        self.assertFalse(trainer._crd_initialized)
        restored = torch.rand(1, generator=loader_generator)
        expected = torch.rand(1, generator=torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(restored, expected))

    def test_non_ode_pcn_layers_are_not_selected(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.PcConvs = torch.nn.ModuleList([torch.nn.Conv2d(1, 1, 1)])

        model = Model()
        metadata = apply_checkpoint_ode_rhs_portion(model, True, 1.0)
        self.assertEqual(metadata["total_layers"], 0)
        self.assertFalse(hasattr(
            model.PcConvs[0], "checkpoint_ode_rhs_training"))

    def test_checkpoint_portion_parser_and_rounding(self):
        self.assertEqual(parse_checkpoint_ode_rhs_portion('auto'), 'auto')
        self.assertEqual(parse_checkpoint_ode_rhs_portion('0.5'), 0.5)
        self.assertEqual(checkpoint_layer_count(0.25, 6), 2)
        self.assertEqual(checkpoint_layer_count(0.5, 5), 3)
        self.assertEqual(checkpoint_layer_count(0.75, 6), 5)
        self.assertEqual(checkpoint_layer_count('auto', 6), 6)
        for invalid in (-0.1, 1.1, 'invalid', float('nan')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_checkpoint_ode_rhs_portion(invalid)

    def test_automatic_layer_search_advances_toward_safe_prefix(self):
        attempted = []

        def safe(count):
            attempted.append(count)
            return count >= 13

        self.assertEqual(find_minimum_safe_checkpoint_layers(19, safe), 13)
        self.assertEqual(attempted[0], 19)
        self.assertIn(9, attempted)
        with self.assertRaisesRegex(RuntimeError, 'Full ODE RHS'):
            find_minimum_safe_checkpoint_layers(4, lambda _count: False)

    def test_runtime_layer_selection_updates_wrapper_snapshots(self):
        class Block(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.layer_idx = index
                self.checkpoint_ode_rhs_training = True
                self.option_init = {'checkpoint_ode_rhs_training': True}
                self.option_patch = {'checkpoint_ode_rhs_training': True}
                self.option_aca = {'checkpoint_ode_rhs_training': True}

            def forward(self, value):
                return value

        class Wrapper:
            def __init__(self, block):
                self.ode_block = block
                self.orig_option_init = dict(block.option_init)
                self.orig_option_patch = dict(block.option_patch)
                self.orig_option_aca = dict(block.option_aca)

            def prehook(self, _module, _inputs):
                return None

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.PcConvs = torch.nn.ModuleList([Block(i) for i in range(4)])
                self.wrappers = []
                for block in self.PcConvs:
                    wrapper = Wrapper(block)
                    block.register_forward_pre_hook(wrapper.prehook)
                    self.wrappers.append(wrapper)

        model = Model()
        metadata = set_checkpoint_ode_rhs_layer_count(model, 2)
        self.assertEqual(metadata['selected_indices'], [0, 1])
        for index, (block, wrapper) in enumerate(
                zip(model.PcConvs, model.wrappers)):
            expected = index < 2
            self.assertEqual(block.checkpoint_ode_rhs_training, expected)
            for name in ('option_init', 'option_patch', 'option_aca'):
                self.assertEqual(
                    getattr(block, name)['checkpoint_ode_rhs_training'], expected)
                self.assertEqual(
                    getattr(wrapper, 'orig_' + name)[
                        'checkpoint_ode_rhs_training'], expected)

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

    def test_training_parsers_support_checkpoint_portion(self):
        from mnist_train_eval.mnist_config import parse_args as mnist_args
        from train_ode_cifar import get_args as cifar_args
        from train_ode_imagenet import get_args as imagenet_args

        cases = (
            (cifar_args, ['train_ode_cifar.py']),
            (imagenet_args, ['train_ode_imagenet.py', '--imagenet_root', '/tmp']),
            (lambda: mnist_args(), ['mnist_train']),
        )
        for parser, argv in cases:
            with self.subTest(parser=argv[0]), patch.dict(
                    os.environ, {'CHECKPOINT_ODE_RHS_PORTION': 'auto'}), \
                    patch.object(sys, 'argv', argv):
                self.assertEqual(parser().checkpoint_ode_rhs_portion, 'auto')
            with self.subTest(parser=argv[0] + '_override'), patch.dict(
                    os.environ, {'CHECKPOINT_ODE_RHS_PORTION': 'auto'}), \
                    patch.object(sys, 'argv', argv + [
                        '--checkpoint_ode_rhs_portion', '0.5']):
                self.assertEqual(parser().checkpoint_ode_rhs_portion, 0.5)

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

    def test_make_ode_block_applies_rounded_checkpoint_prefix(self):
        from ode_pc import ODEXInitFFFB, make_ode_block
        from pc_conv import PCConvReLU6
        from pc_model import PCNetNoBatchNorm

        model = PCNetNoBatchNorm(
            inp_channels=[2, 2, 2, 2], out_channels=[2, 2, 2, 2],
            max_pool=[False] * 4, num_classes=2,
            pc_conv_layer=PCConvReLU6, first_bn=False, kernel_size=3,
            stride=1, dropout=0.0)
        make_ode_block(
            model, ode_block=ODEXInitFFFB, method='dopri5', t_end=0.2,
            tol=1e-3, n_steps=2, checkpoint_ode_rhs_training=True,
            checkpoint_ode_rhs_portion=0.5)
        self.assertEqual(
            [block.option_aca['checkpoint_ode_rhs_training']
             for block in model.PcConvs],
            [True, True, False, False])

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
