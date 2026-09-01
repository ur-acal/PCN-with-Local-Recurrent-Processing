import copy
import unittest

import torch
import torch.nn as nn

from baseline.run_baseline import FixedMismatchHelper
from baseline.cifar_resnet import WideResNetCIFAR
from mismatch_utils import (
    additive_mismatch_scale,
    apply_pcn_ff_gain,
    apply_wrn_ff_gain,
)
from baseline.run_wrn_bn_recalibration_experiment import aggregate_rows
from ode_pc import ODEBlockPC
from pc_model import PCNet


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.fc = nn.Linear(1, 2, bias=True)


class TinyPCNFFLayout(nn.Module):
    def __init__(self, depth, widen_factor):
        super().__init__()
        repeats = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        channels = [(3, widths[0])]
        for stage in range(3):
            in_channels = widths[stage]
            out_channels = widths[stage + 1]
            channels.append((in_channels, out_channels))
            channels.extend((out_channels, out_channels) for _ in range(repeats - 1))
        self.PcConvs = nn.ModuleList()
        for in_channels, out_channels in channels:
            block = nn.Module()
            block.FFconv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.PcConvs.append(block)


class MismatchScalingTests(unittest.TestCase):
    def test_wrn_aggregate_preserves_ff_gain(self):
        row = {
            "dataset": "cifar100",
            "architecture": "WRN_28_2",
            "mismatch_type": "additive",
            "mismatch_level": 0.0,
            "additive_scale_mode": "max_abs",
            "ff_gain": 0.93,
            "frozen_bn_accuracy": 77.0,
            "recalibrated_bn_accuracy": 75.0,
            "pcn_node_best_mean_accuracy": "",
            "pcn_node_best_std_accuracy": "",
            "pcn_node_best_column": "",
            "mismatch_parameter_policy": "existing",
        }
        self.assertEqual(aggregate_rows([row])[0]["ff_gain"], 0.93)

    def test_additive_mismatch_scale_known_values(self):
        tensor = torch.tensor([-4.0, -1.0, 2.0, 3.0])
        self.assertAlmostEqual(additive_mismatch_scale(tensor, "max_abs").item(), 4.0)
        self.assertAlmostEqual(additive_mismatch_scale(tensor, "rms").item(), (30.0 / 4.0) ** 0.5, places=6)

    def test_additive_mismatch_scale_zero_and_invalid_mode(self):
        self.assertEqual(additive_mismatch_scale(torch.zeros(5), "rms").item(), 0.0)
        with self.assertRaisesRegex(ValueError, "Unsupported additive scale mode"):
            additive_mismatch_scale(torch.ones(1), "unknown")

    def test_max_sqrt_uses_per_output_filter_maximum(self):
        conv = nn.Conv2d(1, 2, 1, bias=False)
        weight = torch.tensor([[[[4.0]]], [[[-9.0]]]])
        scale = additive_mismatch_scale(weight, "max_sqrt", module=conv)
        self.assertTrue(torch.equal(scale, torch.tensor([[[[2.0]]], [[[3.0]]]])))

        transpose = nn.ConvTranspose2d(2, 2, 1, bias=False)
        transpose_weight = torch.tensor([[[[1.0]], [[9.0]]], [[[4.0]], [[1.0]]]])
        transpose_scale = additive_mismatch_scale(
            transpose_weight, "max_sqrt", module=transpose
        )
        self.assertTrue(torch.equal(transpose_scale, torch.tensor([[[[2.0]], [[3.0]]]])))

    def test_max_sqrt_requires_filter_weight_module(self):
        with self.assertRaisesRegex(ValueError, "max_sqrt requires"):
            additive_mismatch_scale(torch.ones(2), "max_sqrt")

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

    def test_baseline_max_sqrt_formula_and_selection(self):
        model = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 4.0], [9.0, 1.0]]))
            model.bias.copy_(torch.tensor([2.0, 3.0]))
        clean = copy.deepcopy(model.state_dict())
        helper = FixedMismatchHelper(
            model, 0.2, "additive", seed=13, additive_scale_mode="max_sqrt"
        )
        helper.snapshot_clean_state()
        helper.add_noise()
        self.assertFalse(torch.equal(model.weight, clean["weight"]))
        self.assertTrue(torch.equal(model.bias, clean["bias"]))

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

    def test_ode_max_sqrt_formula_uses_module_filter_axis(self):
        module = nn.Conv2d(1, 2, 1, bias=False)
        clean = torch.tensor([[[[4.0]]], [[[-9.0]]]])
        block = ODEBlockPC.__new__(ODEBlockPC)
        nn.Module.__init__(block)
        block.noise_level = 0.2
        block.mismatch_type = "add"
        block.additive_scale_mode = "max_sqrt"
        torch.manual_seed(19)
        expected = clean + torch.randn_like(clean) * (
            0.2 * additive_mismatch_scale(clean, "max_sqrt", module=module)
        )
        actual = clean.clone()
        torch.manual_seed(19)
        block._apply_noise(actual, module=module)
        self.assertTrue(torch.equal(actual, expected))

    def test_pcn_and_wrn_ff_gain_select_matching_weights(self):
        for depth in (16, 28):
            for widen_factor in (2, 4):
                with self.subTest(depth=depth, widen_factor=widen_factor):
                    pcn = TinyPCNFFLayout(depth, widen_factor)
                    wrn = WideResNetCIFAR(depth=depth, widen_factor=widen_factor)
                    pcn_before = {name: value.detach().clone() for name, value in pcn.named_parameters()}
                    wrn_before = {name: value.detach().clone() for name, value in wrn.named_parameters()}
                    pcn_records = apply_pcn_ff_gain(pcn, 0.9)
                    wrn_records = apply_wrn_ff_gain(wrn, 0.9)

                    self.assertEqual(len(pcn_records), len(wrn_records))
                    self.assertEqual(
                        sorted(record[1] for record in pcn_records),
                        sorted(record[1] for record in wrn_records),
                    )
                    self.assertEqual(
                        sum(record[2] for record in pcn_records),
                        sum(record[2] for record in wrn_records),
                    )
                    for name, _, _ in pcn_records:
                        self.assertTrue(torch.equal(dict(pcn.named_parameters())[name], pcn_before[name] * 0.9))
                    for name, _, _ in wrn_records:
                        self.assertTrue(torch.equal(dict(wrn.named_parameters())[name], wrn_before[name] * 0.9))

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

    def test_pcn_max_sqrt_noises_classifier_weight_but_not_bias(self):
        net = PCNet.__new__(PCNet)
        nn.Module.__init__(net)
        net.device = torch.device("cpu")
        net.noise_level = 0.2
        net.clean_params = {}
        net.PcConvs = nn.ModuleList()
        net.linear = nn.Linear(3, 2, bias=True)
        clean = copy.deepcopy(net.state_dict())

        torch.manual_seed(41)
        net.add_noise(
            noise_to_linear=True,
            mismatch_type="add",
            additive_scale_mode="max_sqrt",
        )

        self.assertFalse(torch.equal(net.linear.weight, clean["linear.weight"]))
        self.assertTrue(torch.equal(net.linear.bias, clean["linear.bias"]))

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
