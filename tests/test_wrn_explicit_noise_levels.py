import unittest
from types import SimpleNamespace

from baseline.run_wrn_bn_recalibration_experiment import levels_for_spec


class ExplicitNoiseLevelTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "mismatch_type": "additive",
            "compare": {"levels": [0.0, 0.01, 0.02, 0.1]},
        }

    def test_csv_uses_legacy_comparison_grid(self):
        args = SimpleNamespace(
            mode="full",
            noise_levels="csv",
            preflight_additive_level=0.05,
            preflight_multiplicative_level=0.2,
        )
        self.assertEqual(levels_for_spec(args, self.spec), [0.0, 0.01, 0.02, 0.1])

    def test_explicit_levels_are_not_filtered_by_legacy_grid(self):
        args = SimpleNamespace(
            mode="full",
            noise_levels="0,0.25,0.5,0.75,1.0,1.25,1.5",
            preflight_additive_level=0.05,
            preflight_multiplicative_level=0.2,
        )
        self.assertEqual(
            levels_for_spec(args, self.spec),
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        )


if __name__ == "__main__":
    unittest.main()

