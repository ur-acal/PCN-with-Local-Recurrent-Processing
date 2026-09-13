import unittest
from unittest.mock import patch

import torch
from torch import nn

from switch import (
    ODEXInitFFFBPixelSwitchStrang,
    ODEXInitFFFBPixelSwitchYoshida4,
    SWITCH_CLASSES,
)


class _ToyStrang(ODEXInitFFFBPixelSwitchStrang):
    def __init__(self):
        nn.Module.__init__(self)
        self.n_iters = 1
        self.option_aca = {'t0': torch.tensor(0.), 't1': torch.tensor(1.),
                           't_eval': [torch.tensor(0.), torch.tensor(1.)],
                           'h': None, 'method': 'euler'}
        self.integration_time = torch.tensor([0., 1.])
        self.visits = []
        self.intervals = []

    def init_y(self, x):
        return x.clone()

    def _make_fixed_pixel_ode_fn(self, x, active_h, active_w):
        # Record explicit order and add the source state's left-neighbor value at the
        # active site, making sequential-state visibility directly testable.
        def fn(t, y):
            w = y.shape[-1]
            self.visits.append((active_h, active_w, y.detach().clone()))
            result = torch.zeros_like(y)
            result[..., active_w] = y[..., (active_w - 1) % w]
            return result
        return fn

    def _build_interval_option_aca(self, t0, t1, ref_tensor):
        self.intervals.append((float(t0), float(t1)))
        return super()._build_interval_option_aca(t0, t1, ref_tensor)


class _ToyYoshida(ODEXInitFFFBPixelSwitchYoshida4):
    def __init__(self):
        nn.Module.__init__(self)
        self.n_iters = 1
        self.scale_RHS = False
        self.option_aca = {'t0': torch.tensor(0.), 't1': torch.tensor(1.),
                           't_eval': [torch.tensor(0.), torch.tensor(1.)],
                           'h': None, 'method': 'euler'}
        self.integration_time = torch.tensor([0., 1.])
        self.intervals = []

    def init_y(self, x):
        return x.clone()

    def _build_interval_option_aca(self, t0, t1, ref_tensor):
        self.intervals.append((float(t0), float(t1)))
        return super()._build_interval_option_aca(t0, t1, ref_tensor)


def _one_euler_step(fn, y, option):
    t0, t1 = option['t0'], option['t1']
    return torch.stack((y, y + (t1 - t0) * fn(t0, y)))


class SwitchStrangTests(unittest.TestCase):
    def test_registered(self):
        self.assertIs(SWITCH_CLASSES['ODEXInitFFFBPixelSwitchStrang'],
                      ODEXInitFFFBPixelSwitchStrang)

    def test_forward_reverse_order_half_duration_and_fresh_reads(self):
        block = _ToyStrang()
        x = torch.tensor([[[[1., 2., 4.]]]])
        with patch('switch.aca_ode_solve', side_effect=_one_euler_step):
            out = block._run_explicit_pixel_switch(x)
        self.assertEqual([(h, w) for h, w, _ in block.visits],
                         [(0, 0), (0, 1), (0, 2), (0, 2), (0, 1), (0, 0)])
        # Every Strang half-flow uses direct local duration T / (2*n_iters).
        self.assertEqual(len(block.visits), 6)
        self.assertTrue(all(abs((t1 - t0) - 0.5) < 1e-6
                            for t0, t1 in block.intervals))
        self.assertAlmostEqual(sum(t1 - t0 for t0, t1 in block.intervals),
                               3.0, places=6)
        # Pixel 1 sees pixel 0 after pixel 0's first update, not the old x.
        self.assertNotEqual(block.visits[1][2][..., 0].item(), x[..., 0].item())
        torch.testing.assert_close(out, torch.tensor([[[[6.75, 5.0, 7.5]]]]))
        self.assertTrue(torch.isfinite(out).all())

    def test_fixed_pixel_uses_installed_rhs_factory(self):
        """A hardware wrapper may replace _make_ode_fn; Strang must retain it."""
        block = _ToyStrang()
        calls = []

        def installed_factory(x):
            calls.append(block._strang_active_pixel)

            def fn(t, y):
                return 7.0 * torch.ones_like(y)
            return fn

        block._make_ode_fn = installed_factory
        fn = ODEXInitFFFBPixelSwitchStrang._make_fixed_pixel_ode_fn(
            block, torch.zeros(1), 2, 3)
        self.assertEqual(calls, [(2, 3)])
        torch.testing.assert_close(
            fn(torch.tensor(0.0), torch.zeros(1)), torch.full((1,), 7.0))


class SwitchYoshidaTests(unittest.TestCase):
    def test_registered_and_coefficients(self):
        self.assertIs(SWITCH_CLASSES['ODEXInitFFFBPixelSwitchYoshida4'],
                      ODEXInitFFFBPixelSwitchYoshida4)
        a = ODEXInitFFFBPixelSwitchYoshida4.YOSHIDA_A
        b = ODEXInitFFFBPixelSwitchYoshida4.YOSHIDA_B
        self.assertAlmostEqual(2.0 * a + b, 1.0, places=14)
        self.assertAlmostEqual(2.0 * a ** 3 + b ** 3, 0.0, places=13)

    def test_composes_validated_strang_with_positive_solver_intervals(self):
        block = _ToyYoshida()
        factory_pixels = []

        # This stands in for the wrapper-installed physical RHS. If Yoshida
        # bypasses self._make_ode_fn, the test cannot produce the expected 7.
        def installed_factory(x):
            factory_pixels.append(block._strang_active_pixel)

            def wrapped_rhs(t, y):
                return 7.0 * torch.ones_like(y)
            return wrapped_rhs

        block._make_ode_fn = installed_factory
        x = torch.zeros(1, 1, 1, 1)
        with patch('switch.aca_ode_solve', side_effect=_one_euler_step):
            out = block._run_explicit_pixel_switch(x)

        a = block.YOSHIDA_A
        b = block.YOSHIDA_B
        expected_intervals = [a / 2, a / 2, abs(b) / 2, abs(b) / 2,
                              a / 2, a / 2]
        self.assertEqual(factory_pixels, [(0, 0)] * 6)
        self.assertTrue(all(t0 == 0.0 and t1 > t0
                            for t0, t1 in block.intervals))
        for (_, t1), expected in zip(block.intervals, expected_intervals):
            self.assertAlmostEqual(t1, expected, places=6)
        # Two half-flows per S2 give 7*(2a+b) = 7.
        torch.testing.assert_close(out, torch.full_like(out, 7.0))
        self.assertFalse(hasattr(block, '_strang_active_pixel'))

    def test_wrapper_transformation_precedes_positive_and_negative_sign(self):
        block = _ToyYoshida()
        calls = []

        def installed_factory(x):
            calls.append(block._strang_active_pixel)

            def wrapped_rhs(t, y):
                # Represents all wrapper transformations already applied.
                return 13.0 * torch.ones_like(y)
            return wrapped_rhs

        block._make_ode_fn = installed_factory
        positive = block._make_signed_fixed_pixel_ode_fn(
            torch.zeros(1), 2, 3, 1.0)
        self.assertEqual(calls[-1], (2, 3))
        torch.testing.assert_close(
            positive(torch.tensor(0.), torch.zeros(1)),
            torch.full((1,), 13.0))
        del block._strang_active_pixel

        negative = block._make_signed_fixed_pixel_ode_fn(
            torch.zeros(1), 4, 5, -1.0)
        self.assertEqual(calls[-1], (4, 5))
        torch.testing.assert_close(
            negative(torch.tensor(0.), torch.zeros(1)),
            torch.full((1,), -13.0))
        del block._strang_active_pixel


if __name__ == '__main__':
    unittest.main()
