import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baseline.cifar_resnet  # noqa: F401,E402
from baseline.baseline_cifar_configs import build_model, get_baseline_config  # noqa: E402
from baseline.add_wrn_nobn_to_presentation_table import add_markdown_column  # noqa: E402
from baseline.run_baseline import (  # noqa: E402
    FixedMismatchHelper,
    convolution_bias_parameter_names,
)
from baseline.run_wrn_nobn_mismatch_experiment import validate_model  # noqa: E402


PAIRS = (
    ("wrn_16_2_cifar", "wrn_16_2_cifar_nobn"),
    ("wrn_16_4_cifar", "wrn_16_4_cifar_nobn"),
    ("wrn_28_2_cifar", "wrn_28_2_cifar_nobn"),
    ("wrn_28_4_cifar", "wrn_28_4_cifar_nobn"),
)

NO_BIAS_PAIRS = tuple(
    (bn_name, nobn_name + "_no_bias") for bn_name, nobn_name in PAIRS
)


def make_model(name):
    cfg = get_baseline_config(name, case="custom_noresize")
    return build_model(name, cfg, num_classes=10)


class WRNNoBatchNormTests(unittest.TestCase):
    def test_variants_have_no_bn_and_all_convolutions_have_bias(self):
        for _, nobn_name in PAIRS:
            with self.subTest(model=nobn_name):
                model = make_model(nobn_name)
                self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
                convolutions = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                self.assertTrue(convolutions)
                self.assertTrue(all(m.bias is not None for m in convolutions))

    def test_conv_and_linear_weight_topology_matches_bn_counterpart(self):
        for bn_name, nobn_name in PAIRS:
            with self.subTest(model=nobn_name):
                bn_model, nobn_model = make_model(bn_name), make_model(nobn_name)
                selected = lambda model: {
                    name: tuple(parameter.shape)
                    for name, parameter in model.named_parameters()
                    if name.endswith("weight") and parameter.ndim >= 2
                }
                self.assertEqual(selected(bn_model), selected(nobn_model))

    def test_existing_wrn_defaults_still_use_bn_and_biasless_convs(self):
        model = make_model("wrn_16_2_cifar")
        self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
        self.assertTrue(all(m.bias is None for m in model.modules() if isinstance(m, nn.Conv2d)))

    def test_no_bn_no_bias_variants_are_pure_bn_removal_ablations(self):
        for bn_name, no_bias_name in NO_BIAS_PAIRS:
            with self.subTest(model=no_bias_name):
                bn_model = make_model(bn_name)
                model = make_model(no_bias_name)
                self.assertFalse(
                    any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
                )
                self.assertTrue(
                    all(m.bias is None for m in model.modules() if isinstance(m, nn.Conv2d))
                )

                def conv_topology(candidate):
                    return {
                        name: (
                            tuple(module.weight.shape), module.stride,
                            module.padding, module.groups,
                        )
                        for name, module in candidate.named_modules()
                        if isinstance(module, nn.Conv2d)
                    }

                self.assertEqual(conv_topology(model), conv_topology(bn_model))

                model.train()
                loss = model(torch.randn(2, 3, 32, 32)).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(
                    all(
                        module.weight.grad is not None
                        and torch.isfinite(module.weight.grad).all()
                        for module in model.modules()
                        if isinstance(module, (nn.Conv2d, nn.Linear))
                    )
                )

    def test_existing_bn_free_variants_keep_convolution_biases(self):
        for _, nobn_name in PAIRS:
            with self.subTest(model=nobn_name):
                model = make_model(nobn_name)
                self.assertTrue(
                    all(m.bias is not None for m in model.modules() if isinstance(m, nn.Conv2d))
                )

    def test_bn_free_evaluator_accepts_biasless_convolutions(self):
        model = make_model("wrn_16_2_cifar_nobn_no_bias")
        self.assertEqual(validate_model(model), set())

    def test_conv_biases_are_skipped_but_weights_and_classifier_are_noised(self):
        torch.manual_seed(1)
        model = make_model("wrn_16_2_cifar_nobn")
        excluded = convolution_bias_parameter_names(model)
        clean = {name: p.detach().clone() for name, p in model.named_parameters()}
        helper = FixedMismatchHelper(
            model, .1, "multiplicative", False, True, seed=77,
            exclude_param_names=excluded,
        )
        helper.snapshot_clean_state()
        summary = helper.add_noise()
        applied = {record.name for record in summary.applied_records}
        skipped = {record.name for record in summary.skipped_records}
        self.assertTrue(excluded <= skipped)
        self.assertFalse(applied & excluded)
        self.assertIn("conv1.weight", applied)
        self.assertIn("fc.weight", applied)
        self.assertIn("fc.bias", applied)
        for name in excluded:
            self.assertTrue(torch.equal(dict(model.named_parameters())[name], clean[name]))

    def test_markdown_sidecar_adds_bn_free_column(self):
        markdown = """# Table

****************************************************************************
old notation
****************************************************************************

## cifar10 WRN_16_2 multiplicative

| mismatch level | trials | PCN/NODE acc | WRN folded recal-BN acc | folded BN recovery | remaining PCN gap folded recal |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 10 | 90.00 | 80.00 | 5.00 | 10.00 |
"""
        row_key = ("cifar10", "WRN_16_2", "multiplicative", .1)
        output = add_markdown_column(markdown, {row_key: {"wrn_nobn_mean_accuracy": "85.25"}})
        self.assertIn("WRN BN-free acc", output)
        self.assertIn("| 0.10 | 10 | 90.00 | 80.00 | 85.25 | 5.00 | 10.00 |", output)

    def test_additive_mismatch_handles_exactly_zero_tensor(self):
        model = make_model("wrn_16_2_cifar_nobn")
        self.assertEqual(torch.count_nonzero(model.fc.bias).item(), 0)
        helper = FixedMismatchHelper(model, .2, "additive", False, True, seed=13)
        helper.snapshot_clean_state()
        summary = helper.add_noise()
        record = next(r for r in summary.applied_records if r.name == "fc.bias")
        self.assertTrue(record.all_zero)
        self.assertFalse(record.changed)

    def test_same_seed_reproduces_fixed_noisy_state(self):
        model_a = make_model("wrn_16_2_cifar_nobn")
        model_b = copy.deepcopy(model_a)
        excluded = convolution_bias_parameter_names(model_a)
        for model in (model_a, model_b):
            helper = FixedMismatchHelper(
                model, .2, "additive", False, True, seed=991,
                exclude_param_names=excluded,
            )
            helper.snapshot_clean_state()
            helper.add_noise()
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(parameter_a, parameter_b))


if __name__ == "__main__":
    unittest.main()
