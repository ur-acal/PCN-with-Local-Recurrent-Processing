"""Fixed TC grids must not create an extra sample for endpoint roundoff."""
import unittest
from types import SimpleNamespace
import torch
from TorchDiffEqPack.odesolver.fixed_grid_solver import Euler


class TCEulerGridTests(unittest.TestCase):
    def grid(self, duration, h, tc=True):
        solver = Euler(lambda t, y: y, torch.tensor(0.), (torch.zeros(1),),
                       t1=torch.as_tensor(duration), h=h)
        if tc:
            solver.tc_context = SimpleNamespace()
        solver.integrate_predefined_grids = lambda *a, **kw: kw['predefine_steps']
        return solver.integrate((torch.zeros(1),), torch.tensor(0.)), solver.t1

    def test_roundoff_endpoint_snaps_without_sixth_step(self):
        t = torch.tensor(2.8133706475585996e-9)
        h = torch.nextafter(t / 5, torch.tensor(0.))
        steps, end = self.grid(t, h)
        self.assertEqual(len(steps), 5)
        self.assertEqual(steps[-1], end)

    def test_genuine_partial_interval_is_not_snapped(self):
        steps, end = self.grid(1., .3)
        self.assertLess(steps[-1], end)

    def test_legacy_grid_is_unchanged(self):
        t = torch.tensor(2.8133706475585996e-9)
        h = torch.nextafter(t / 5, torch.tensor(0.))
        steps, end = self.grid(t, h, tc=False)
        self.assertLess(steps[-1], end)


if __name__ == '__main__':
    unittest.main()
