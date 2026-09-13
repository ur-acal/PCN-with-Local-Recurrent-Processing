"""Data-only TC tests; no training jobs, datasets, or GPU are required."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ode_pc import ODEBlockPC
from tc_nonidealities import (
    positive_resistance,
    prepare_tc_resistance_curves, integrate_current_asd,
)
from utils import load_mc_res_curve_gaussian
from validation import MVMConv


ROOT = Path(__file__).resolve().parents[1]


def mapping(q_hi=15, weight_scale=1.):
    # Exercise the real existing methods without constructing a model.
    block = ODEBlockPC.__new__(ODEBlockPC)
    torch.nn.Module.__init__(block)
    block.q_hi, block.weight_scale = q_hi, weight_scale
    return block


class TCDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.mean_path = Path(self.temp.name) / "means.csv"
        self.mc_path = Path(self.temp.name) / "mc.csv"
        grid = np.array([-.1, 0., .1])
        # Deliberately unsorted labels and voltages: key 3 is 10k, not 30k.
        means = np.array([[30000, 10000, 15000]]) + np.array([[-100], [0], [100]])
        np.savetxt(self.mean_path, np.column_stack((grid, means))[::-1], delimiter=",",
                   header="Vin,30000,10000,15000", comments="")
        deviations = np.array([[120, 20, 30], [0, 160, 40], [20, 30, 100]], dtype=float)
        self.samples = np.concatenate((deviations, -deviations)) + [20000, 21000, 22000]
        columns = np.column_stack([v for row in self.samples for v in (grid, row)])
        np.savetxt(self.mc_path, columns, delimiter=",",
                   header=",".join(f"v{i},r{i}" for i in range(6)), comments="")

    def package(self, **kwargs):
        return prepare_tc_resistance_curves(self.mean_path, self.mc_path,
            levels=mapping(3)._get_quant_magnitude_levels(torch.empty(0, dtype=torch.float64)),
            R=1e4, R_max=3e4, **kwargs)

    def test_pooling_path_and_nominal_resistance_overrides(self):
        from types import SimpleNamespace
        from tc_cli import pooling_options
        wrappers = [SimpleNamespace(ode_block=SimpleNamespace(
            FFconv=torch.nn.Conv2d(1, 1, 1).double()))]
        args = SimpleNamespace(measured_pooling_curve_path=str(self.mean_path),
                               measured_pooling_nominal_R=10000.,
                               tc_covariance_table=str(self.mc_path))
        first = pooling_options(args, wrappers)
        args.measured_pooling_nominal_R = 30000.
        second = pooling_options(args, wrappers)
        torch.testing.assert_close(second['curve_gaussian']['mean'] -
                                   first['curve_gaussian']['mean'], torch.full((3,), 20000., dtype=torch.float64))
        alternate = Path(self.temp.name) / 'alternate.csv'
        data = np.loadtxt(self.mean_path, delimiter=',', skiprows=1)
        data[:, 1:] += 1234.
        np.savetxt(alternate, data, delimiter=',', header='Vin,30000,10000,15000', comments='')
        args.measured_pooling_curve_path = str(alternate)
        third = pooling_options(args, wrappers)
        torch.testing.assert_close(third['curve_gaussian']['mean'] -
                                   second['curve_gaussian']['mean'], torch.full((3,), 1234., dtype=torch.float64))
        torch.testing.assert_close(first['curve_gaussian']['factor'], third['curve_gaussian']['factor'])

    def test_code_mapping_matches_existing_dense_and_unrolled_rules(self):
        for R_max in (None, 150e3, 80e3):
            with self.subTest(R_max=R_max):
                scale = 1. if R_max is None else max(1., 150e3 / R_max)
                block = mapping(15, scale)
                levels = block._get_quant_magnitude_levels(torch.empty(0, dtype=torch.float64))
                weights = torch.cat((levels, -levels)).reshape(2, -1)
                torch.testing.assert_close(block._values_to_level_idx(weights),
                                           torch.arange(16).expand(2, -1))
                stub = type("ExistingMVM", (), {})()
                actual = MVMConv._get_quant_magnitude_levels(stub, weights, 15, scale)
                torch.testing.assert_close(levels, actual)
        block = mapping()
        levels = block._get_quant_magnitude_levels(torch.empty(0, dtype=torch.float64))
        self.assertEqual(levels[0], 0)
        self.assertEqual(1e4 / levels[15], 1e4)
        self.assertEqual(1e4 / levels[1], 150e3)
        self.assertEqual(block._values_to_level_idx(torch.tensor([.5 / 15])).item(), 0)

    def test_mean_selection_and_absolute_covariance(self):
        package = self.package()
        torch.testing.assert_close(package.programmed_resistances,
                                   torch.tensor([30000., 15000., 10000.], dtype=torch.float64))
        torch.testing.assert_close(package.mean_column_resistances, package.programmed_resistances)
        torch.testing.assert_close(package.means[:, 1], package.programmed_resistances)
        expected = torch.from_numpy(np.cov(self.samples, rowvar=False, ddof=1))
        torch.testing.assert_close(package.covariance, expected, rtol=1e-10, atol=1e-7)
        # The per-code mean must not scale the common absolute covariance.
        means = np.loadtxt(self.mean_path, delimiter=",", skiprows=1)
        means[:, 1:] *= 2
        np.savetxt(self.mean_path, means, delimiter=",", header="Vin,30000,10000,15000", comments="")
        changed = self.package()
        torch.testing.assert_close(changed.means, package.means * 2)
        torch.testing.assert_close(changed.factor, package.factor)

    def test_batched_draws_are_independent_reproducible_and_match_formula(self):
        package = self.package()
        generator = torch.Generator().manual_seed(23)
        expected = package.means + torch.randn((3, 3), dtype=torch.float64,
                                               generator=generator) @ package.factor.T
        actual = package.sample_levels(generator=torch.Generator().manual_seed(23))
        torch.testing.assert_close(actual, expected)
        residual = actual - package.means
        self.assertFalse(torch.equal(residual[0], residual[1]))
        codes = torch.tensor([1, 1, 2, 3])
        draws = package.sample(codes, generator=torch.Generator().manual_seed(17), chunk_size=2)
        self.assertFalse(torch.equal(draws[0], draws[1]))
        torch.testing.assert_close(draws, package.sample(codes,
            generator=torch.Generator().manual_seed(17), chunk_size=2))

    def test_empirical_covariance_same_across_levels(self):
        package = self.package()
        codes = torch.arange(1, 4).repeat_interleave(20000)
        draws = package.sample(codes, generator=torch.Generator().manual_seed(7))
        for level in range(3):
            residual = draws[level * 20000:(level + 1) * 20000] - package.means[level]
            torch.testing.assert_close(residual.mean(0), torch.zeros(3, dtype=torch.float64),
                                       atol=2., rtol=0)
            torch.testing.assert_close(torch.cov(residual.T), package.covariance,
                                       atol=180., rtol=.05)

    def test_zero_codes_do_not_sample_and_guards_are_positive(self):
        package = self.package()
        g = torch.Generator().manual_seed(43)
        before = g.get_state().clone()
        self.assertTrue(torch.isinf(package.sample(torch.zeros(5, dtype=torch.long), generator=g)).all())
        self.assertTrue(torch.equal(before, g.get_state()))
        with_zero = package.sample(torch.tensor([0, 1, 0, 3]), generator=g)
        without_zero = package.sample(torch.tensor([1, 3]), generator=torch.Generator().manual_seed(43))
        torch.testing.assert_close(with_zero[[1, 3]], without_zero)
        extreme = replace(package, means=-torch.ones_like(package.means),
                          factor=torch.zeros_like(package.factor), floor_ohms=.001)
        self.assertTrue((extreme.sample_levels() == .001).all())
        self.assertEqual(package.sample(torch.tensor([], dtype=torch.long)).shape, (0, 3))
        guarded = positive_resistance(torch.tensor([-1., 0., 2.]), .01)
        torch.testing.assert_close(guarded, torch.tensor([.01, .01, 2.]))
        with self.assertRaises(ValueError):
            positive_resistance(torch.tensor([float("nan")]))

    def test_common_grid_and_interpolated_means(self):
        grid = np.array([-.05, 0., .05])
        table = grid[:, None] * 1000 + [30000., 10000., 15000.]
        np.savetxt(self.mean_path, np.column_stack((grid, table)), delimiter=",",
                   header="Vin,30000,10000,15000", comments="")
        # Only one MC voltage in this band: do not invent extrapolated data.
        with self.assertRaisesRegex(ValueError, "common voltage band"):
            self.package()
        grid = np.array([-.1, .05])
        table = grid[:, None] * 1000 + [30000., 10000., 15000.]
        np.savetxt(self.mean_path, np.column_stack((grid, table)), delimiter=",",
                   header="Vin,30000,10000,15000", comments="")
        package = self.package()
        torch.testing.assert_close(package.v_grid, torch.tensor([-.1, 0.], dtype=torch.float64))
        torch.testing.assert_close(package.means[:, 1], package.programmed_resistances)
        expected = torch.from_numpy(np.cov(self.samples, rowvar=False)[:2, :2])
        torch.testing.assert_close(package.covariance, expected, atol=1e-7, rtol=1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            prepare_tc_resistance_curves(self.mean_path, self.mc_path,
                levels=torch.tensor([0., 1.]), R=0)
        package = self.package()
        for codes in ([1.2], [-1], [4]):
            with self.assertRaises(ValueError):
                package.sample(codes)
        with self.assertRaises(ValueError):
            package.sample([1], chunk_size=0)

    def test_reference_asd_constant_spectrum_and_resistance_scaling(self):
        path = Path(self.temp.name) / "asd.csv"
        np.savetxt(path, [[1000, 2e-12], [10, 2e-12], [100, 2e-12]], delimiter=",",
                   header="VN() X,VN() Y", comments="")
        integral = integrate_current_asd(path)
        self.assertAlmostEqual(integral.variance_A2 / (4e-24 * 990), 1.)
        self.assertAlmostEqual(integral.std_A ** 2 / integral.variance_A2, 1.)
        self.assertAlmostEqual(integral.variance_at_resistance(1e4) / integral.variance_A2, 5.)
        sub = integrate_current_asd(path, lower_hz=50, upper_hz=300)
        self.assertAlmostEqual(sub.variance_A2 / (4e-24 * 250), 1.)
        for kwargs in ({"lower_hz": 0}, {"upper_hz": 1001}, {"lower_hz": 1000}):
            with self.assertRaises(ValueError):
                integrate_current_asd(path, **kwargs)

    def test_asd_squares_before_integrating_and_validates(self):
        path = Path(self.temp.name) / "asd.csv"
        np.savetxt(path, [[1, 2], [3, 4]], delimiter=",", header="f,asd", comments="")
        self.assertEqual(integrate_current_asd(path).variance_A2, 20.)
        self.assertEqual(integrate_current_asd(path, lower_hz=2).variance_A2, 13.)
        for data in ([[1, -2], [2, 3]], [[1, 2], [1, 3]], [[1, 2], [2, float("nan")]]):
            np.savetxt(path, data, delimiter=",", header="f,asd", comments="")
            with self.assertRaises(ValueError):
                integrate_current_asd(path)

    def test_real_characterization_sources(self):
        mean_path = ROOT / "hardware_data/res_vs_vin_10k_150k.csv"
        mc_path = ROOT / "hardware_data/mc_45_corners/CU_4500_r_vs_vin.csv"
        package = prepare_tc_resistance_curves(mean_path, mc_path,
            levels=mapping()._get_quant_magnitude_levels(torch.empty(0, dtype=torch.float64)))
        fit = load_mc_res_curve_gaussian(mc_path, quantity="resistance", dtype=torch.float64)
        self.assertEqual(package.means.shape, (15, fit["v_grid"].numel()))
        torch.testing.assert_close(package.factor, fit["factor"] * fit["value_scale"])
        raw = np.loadtxt(mc_path, delimiter=",", skiprows=1)
        self.assertEqual(raw.shape[1] // 2, 4500)
        # Check absolute-ohm covariance directly, allowing loader regularization.
        aligned = []
        for i in range(4500):
            grid, curve = raw[:, 2 * i], raw[:, 2 * i + 1]
            order = np.argsort(grid)
            aligned.append(np.interp(package.v_grid.numpy(), grid[order], curve[order]))
        expected = torch.from_numpy(np.cov(aligned, rowvar=False))
        torch.testing.assert_close(package.covariance, expected, rtol=1e-7, atol=1e-4)
        self.assertTrue((package.sample_levels(generator=torch.Generator().manual_seed(99)) > 0).all())
        integral = integrate_current_asd(ROOT / "hardware_data/coupler_asd_vs_freq.csv")
        self.assertEqual((integral.lower_hz, integral.upper_hz), (1e3, 1e9))
        raw = np.loadtxt(integral.path, delimiter=",", skiprows=1)
        self.assertAlmostEqual(integral.variance_A2 / np.trapezoid(raw[:, 1] ** 2, raw[:, 0]), 1.)
