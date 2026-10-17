"""Projected Euler clamps once, after the deterministic and stochastic update."""
import unittest
from unittest.mock import patch

import torch

from TorchDiffEqPack.odesolver import odesolve
from TorchDiffEqPack.odesolver.fixed_grid_solver import ProjEuler
from tc_nonidealities import TCNoiseLifecycle


class ProjectedEulerOrderingTests(unittest.TestCase):
    def test_noise_can_cancel_deterministic_overshoot_before_projection(self):
        for tuple_input in (False, True):
            state = torch.zeros(1, dtype=torch.float64)
            y0 = (state, state.clone()) if tuple_input else state

            def rhs(t, y):
                return (torch.ones_like(y[0]) * 1.2,
                        torch.ones_like(y[1]) * -1.2) if tuple_input else torch.ones_like(y) * 1.2

            context = TCNoiseLifecycle([], {})
            rhs.tc_context = context
            options = dict(t0=0., t1=1., t_eval=[1.], method='euler', h=1.,
                           eps=(.1, .1) if tuple_input else .1,
                           noise_type='addi', proj_fn=lambda x: x.clamp(-1., 1.))
            solver = odesolve(rhs, y0, options, return_solver=True)
            self.assertIsInstance(solver, ProjEuler)
            with patch.object(context, 'normal', side_effect=lambda ref, branch:
                              torch.full_like(ref, -1. if branch == 0 else 1.)):
                result = solver.integrate(y0, t0=0., t_eval=[1.])
            values = result if tuple_input else (result,)
            for value, expected in zip(values, (1., -1.)):
                torch.testing.assert_close(value, torch.full_like(value, expected))
            self.assertEqual(context.index, 1)

    def test_deterministic_integrated_projection_is_preserved(self):
        state = torch.zeros(1, dtype=torch.float64)
        result = odesolve(lambda t, y: torch.ones_like(y) * 1.2, state,
                         dict(t0=0., t1=1., t_eval=[1.], method='euler', h=.1,
                              proj_fn=lambda x: x.clamp(-1., 1.)))
        torch.testing.assert_close(result, torch.ones_like(result))


if __name__ == '__main__':
    unittest.main()
