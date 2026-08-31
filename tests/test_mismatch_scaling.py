import copy
import unittest

import torch
import torch.nn as nn

from baseline.run_baseline import FixedMismatchHelper
from mismatch_utils import additive_mismatch_scale
from ode_pc import ODEBlockPC
from pc_model import PCNet


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.fc = nn.Linear(1, 2, bias=True)


class MismatchScalingTests(unittest.TestCase):
    def test_additive_mismatch_scale_known_values(self):
        tensor = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        self.assertAlmostEqual(additive_mismatch_scale(tensor, "max_abs").item(), 4.0)
        self.assertAlmostEqual(additive_mismatch_scale(tensor, "rms").item(), (30.0 / 4.0) ** 0.5, places=6)

    def test_additive_mismatch_scale_zero_and_invalid_mode(self):
        self.assertEqual(additive_mismatch_scale(torch.zeros(5), "rms").item(), 0.0)
        with self.assertRaisesRegex(ValueError, "Unsupported additive scale mode"):
            additive_mismatch_scale(torch.ones(1), "unknown")

    def test_baseline_additive_formula_is_exact(self):
        clean = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        for mode in ("max_abs", "rms"):
            with self.subTest(mode=mode):
                helper = FixedMismatchHelper(nn.Linear(4, 1, bias=False), 0.2, "additive", additive_scale_mode=mode)
                torch.manual_seed(7)
                expected = clean + 0.2 * torch.randn_like(clean) * additive_mismatch_scale(clean, mode)
                actual = clean.clone()
                torch.manual_seed(7)
                helper._apply_noise_(actual)
                self.assertTrue(torch.equal(actual, expected))

    def test_baseline_multiplicative_formula_is_unchanged(self):
        clean = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        helper = FixedMismatchHelper(nn.Linear(4, 1, bias=False), 0.2, "multiplicative", additive_scale_mode="rms")
        torch.manual_seed(11)
        expected = clean * (1.0 + 0.2 * torch.randn_like(clean))
        actual = clean.clone()
        torch.manual_seed(11)
        helper._apply_noise_(actual)
        self.assertTrue(torch.equal(actual, expected))

    def test_baseline_seed_restoration_and_selection_are_preserved(self):
        model = TinyBaseline()
        clean_state = copy.deepcopy(model.state_dict())
        helper = FixedMismatchHelper(
            model, 0.1, "additive", noise_to_norm=False, seed=23,
            exclude_param_names={"conv.bias"}, additive_scale_mode="rms",
        )
        helper.snapshot_clean_state()
        first = helper.add_noise()
        first_state = copy.deepcopy(model.state_dict())
        helper.restore_clean_state()
        second = helper.add_noise()

        for name, value in first_state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertTrue(torch.equal(model.conv.bias, clean_state["conv.bias"]))
        self.assertTrue(torch.equal(model.bn.weight, clean_state["bn.weight"]))
        self.assertTrue(torch.equal(model.bn.bias, clean_state["bn.bias"]))
        self.assertFalse(torch.equal(model.fc.weight, clean_state["fc.weight"]))
        self.assertEqual({r.name for r in first.applied_records}, {r.name for r in second.applied_records})

        helper.restore_clean_state()
        for name, value in clean_state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_all_zero_selected_tensor_is_a_documented_no_change(self):
        model = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(model.weight)
        helper = FixedMismatchHelper(model, 0.4, "additive", seed=3, additive_scale_mode="rms")
        helper.snapshot_clean_state()
        summary = helper.add_noise()
        self.assertTrue(summary.applied_records[0].all_zero)
        self.assertEqual(torch.count_nonzero(model.weight).item(), 0)

    def test_ode_additive_formula_uses_shared_scale(self):
        clean = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        for mode in ("max_abs", "rms"):
            with self.subTest(mode=mode):
                block = ODEBlockPC.__new__(ODEBlockPC)
                nn.Module.__init__(block)
                block.noise_level = 0.2
                block.mismatch_type = "add"
                block.additive_scale_mode = mode
                torch.manual_seed(17)
                expected = clean + torch.randn_like(clean) * (0.2 * additive_mismatch_scale(clean, mode))
                actual = clean.clone()
                torch.manual_seed(17)
                block._apply_noise(actual)
                self.assertTrue(torch.equal(actual, expected))

    def test_pcn_classifier_weight_and_bias_use_additive_formula(self):
        for mode in ("max_abs", "rms"):
            with self.subTest(mode=mode):
                net = PCNet.__new__(PCNet)
                nn.Module.__init__(net)
                net.device = torch.device("cpu")
                net.noise_level = 0.2
                net.clean_params = {}
                net.PcConvs = nn.ModuleList()
                net.linear = nn.Linear(3, 2, bias=True)
                with torch.no_grad():
                    net.linear.weight.copy_(torch.tensor([[-4.0, -1.0, 2.0], [3.0, 1.0, -2.0]]))
                    net.linear.bias.copy_(torch.tensor([-3.0, 1.0]))
                clean_weight = net.linear.weight.detach().clone()
                clean_bias = net.linear.bias.detach().clone()

                torch.manual_seed(29)
                expected_weight = clean_weight + torch.randn_like(clean_weight) * net.noise_level * additive_mismatch_scale(clean_weight, mode)
                expected_bias = clean_bias + torch.randn_like(clean_bias) * net.noise_level * additive_mismatch_scale(clean_bias, mode)
                torch.manual_seed(29)
                net.add_noise(noise_to_linear=True, mismatch_type="add", additive_scale_mode=mode)

                self.assertTrue(torch.equal(net.linear.weight, expected_weight))
                self.assertTrue(torch.equal(net.linear.bias, expected_bias))
                net.recover_params()
                self.assertTrue(torch.equal(net.linear.weight, clean_weight))
                self.assertTrue(torch.equal(net.linear.bias, clean_bias))

    def test_pcn_classifier_multiplicative_formula_is_unchanged(self):
        net = PCNet.__new__(PCNet)
        nn.Module.__init__(net)
        net.device = torch.device("cpu")
        net.noise_level = 0.2
        clean = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        torch.manual_seed(31)
        expected = clean * (1.0 + torch.randn_like(clean) * net.noise_level)
        actual = clean.clone()
        torch.manual_seed(31)
        net._apply_noise(actual, mismatch_type="mul", additive_scale_mode="rms")
        self.assertTrue(torch.equal(actual, expected))

    def test_pcn_can_exclude_conv_bias_but_noise_conv_weight_and_classifier(self):
        net = PCNet.__new__(PCNet)
        nn.Module.__init__(net)
        net.device = torch.device("cpu")
        net.noise_level = 0.2
        net.clean_params = {}
        net.PcConvs = nn.ModuleList()
        net.first_conv = nn.Conv2d(1, 2, 1, bias=True)
        net.linear = nn.Linear(2, 3, bias=True)
        clean = copy.deepcopy(net.state_dict())

        torch.manual_seed(37)
        net.add_noise(
            noise_to_linear=True,
            noise_to_conv_bias=False,
            mismatch_type="add",
            additive_scale_mode="rms",
        )

        self.assertFalse(torch.equal(net.first_conv.weight, clean["first_conv.weight"]))
        self.assertTrue(torch.equal(net.first_conv.bias, clean["first_conv.bias"]))
        self.assertFalse(torch.equal(net.linear.weight, clean["linear.weight"]))
        self.assertFalse(torch.equal(net.linear.bias, clean["linear.bias"]))
        net.recover_params()
        for name, value in clean.items():
            self.assertTrue(torch.equal(net.state_dict()[name], value), name)




class ReportIntegrityTests(unittest.TestCase):
    def test_rms_aggregate_without_reference_keeps_pcn_fields_empty(self):
        from baseline.run_wrn_bn_recalibration_experiment import aggregate_rows
        rows = []
        for trial, accuracy in enumerate((70.0, 72.0)):
            rows.append({
                "dataset": "cifar10", "architecture": "WRN_16_2",
                "mismatch_type": "additive", "mismatch_level": 0.1,
                "additive_scale_mode": "rms",
                "frozen_bn_accuracy": accuracy - 1,
                "recalibrated_bn_accuracy": accuracy,
                "pcn_node_best_mean_accuracy": "",
                "pcn_node_best_std_accuracy": "",
                "pcn_node_best_column": "",
                "mismatch_parameter_policy": "existing", "trial": trial,
            })
        aggregate = aggregate_rows(rows)
        self.assertEqual(aggregate[0]["pcn_node_mean_accuracy"], "")
        self.assertEqual(aggregate[0]["remaining_pcn_gap"], "")
        self.assertEqual(aggregate[0]["additive_scale_mode"], "rms")

    def test_final_report_rejects_stale_rms_pcn_reference(self):
        from baseline.generate_combined_rms_mismatch_report import validate_rms_sources
        original = {("cifar10", "WRN_16_2", "additive", 0.1): {}}
        row = {"dataset": "cifar10", "architecture": "WRN_16_2",
               "mismatch_level": "0.1", "pcn_reference_scale_mode": "rms",
               "pcn_node_mean_accuracy": "80.0"}
        source = {("cifar10", "WRN_16_2", 0.1): row}
        pcn = {("cifar10", "WRN_16_2"): {0.1: [81.0, 83.0]}}
        with self.assertRaisesRegex(ValueError, "Stale PCN reference"):
            validate_rms_sources([("synthetic", source)], original, pcn)
        row["pcn_node_mean_accuracy"] = "82.0"
        validate_rms_sources([("synthetic", source)], original, pcn)

    def test_final_report_rejects_duplicate_keys(self):
        from baseline.generate_combined_rms_mismatch_report import keyed
        row = {"dataset": "cifar10", "architecture": "WRN_16_2",
               "mismatch_type": "additive", "mismatch_level": "0.1"}
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            keyed([row, dict(row)], "synthetic.csv")


if __name__ == "__main__":
    unittest.main()
