import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline.run_baseline as rb
import weight_range_audit as audit
from tests.test_bn_recalibration import LabeledTensorDataset, TinyBNNet, _inputs, _labels


class FakeODEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.FFconv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.FBconv = nn.ConvTranspose2d(3, 3, 3, padding=1, bias=False)
        self.bypass = nn.Conv2d(3, 3, 1, bias=False)
        self.tie_weights = False
        self.tie_bp = False
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, 3, 1, 1))])

    def forward(self, x):
        y = self.FFconv(x)
        for _ in range(5):
            y = self.FFconv(self.FBconv(y))
        return y + self.bypass(x)


class FakePCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.PcConvs = nn.ModuleList([FakeODEBlock(), FakeODEBlock()])
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        for block in self.PcConvs:
            x = block(x)
        return self.linear(x.mean(dim=(2, 3)))


class WeightRangeAuditTests(unittest.TestCase):
    def test_max_abs_rms_and_kappa_are_correct(self):
        x = torch.tensor([3.0, 4.0])
        stats = audit.tensor_range_stats(x)
        self.assertEqual(stats.max_abs, 4.0)
        self.assertAlmostEqual(stats.rms, math.sqrt((9.0 + 16.0) / 2.0))
        self.assertAlmostEqual(stats.kappa, 4.0 / math.sqrt(12.5))

    def test_kappa_is_scale_invariant_for_nonzero_scalar(self):
        x = torch.tensor([-1.0, 2.0, -3.0, 6.0])
        base = audit.tensor_range_stats(x).kappa
        self.assertAlmostEqual(audit.tensor_range_stats(7.25 * x).kappa, base)
        self.assertAlmostEqual(audit.tensor_range_stats(-0.5 * x).kappa, base)

    def test_repeating_and_reshaping_identical_distribution_preserves_kappa(self):
        x = torch.tensor([1.0, -2.0, 4.0, -8.0])
        y = x.repeat(4).reshape(2, 2, 4)
        self.assertAlmostEqual(audit.tensor_range_stats(x).kappa, audit.tensor_range_stats(y).kappa)

    def test_all_zero_records_nan_kappa(self):
        stats = audit.tensor_range_stats(torch.zeros(5, 3))
        self.assertEqual(stats.max_abs, 0.0)
        self.assertEqual(stats.rms, 0.0)
        self.assertTrue(stats.all_zero)
        self.assertTrue(math.isnan(stats.kappa))

    def test_wrn_audit_does_not_change_state_rng_logits_predictions_or_accuracy(self):
        torch.manual_seed(123)
        model = TinyBNNet(dropout=False).eval()
        x = _inputs(seed=456, n=8)
        y = _labels(len(x))
        loader = DataLoader(LabeledTensorDataset(x, y), batch_size=4, shuffle=False)
        helper = rb.FixedMismatchHelper(model, noise_sigma=0.1, noise_type="additive", noise_to_norm=False)
        helper.snapshot_clean_state()

        before_params = rb._snapshot_hashes(model.named_parameters())
        before_buffers = rb._snapshot_hashes(model.named_buffers())
        with torch.no_grad():
            logits_before = model(x)
        preds_before = logits_before.argmax(dim=1)
        acc_before = rb.evaluate_once(model, loader, torch.device("cpu"), max_batches=2)
        rng_before = torch.random.get_rng_state().clone()

        primary, nonprimary = audit.collect_wrn_weight_range_audit_rows(
            model,
            dataset="cifar10",
            model_name="tiny_wrn_like",
            should_noise_param=helper._should_noise_param,
            param_to_module=helper._param_to_module,
        )
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

        with torch.no_grad():
            logits_after = model(x)
        preds_after = logits_after.argmax(dim=1)
        acc_after = rb.evaluate_once(model, loader, torch.device("cpu"), max_batches=2)

        self.assertEqual(before_params, rb._snapshot_hashes(model.named_parameters()))
        self.assertEqual(before_buffers, rb._snapshot_hashes(model.named_buffers()))
        self.assertTrue(torch.equal(logits_before, logits_after))
        self.assertTrue(torch.equal(preds_before, preds_after))
        self.assertEqual(acc_before, acc_after)
        self.assertEqual([r["tensor_name"] for r in primary], ["conv.weight", "fc.weight"])
        self.assertEqual([r["tensor_name"] for r in nonprimary], ["fc.bias"])

    def test_shared_pcn_tensors_are_not_duplicated_by_solver_calls(self):
        torch.manual_seed(9)
        model = FakePCN().eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            for _ in range(3):
                model(x)

        primary, nonprimary = audit.collect_pcn_ode_weight_range_audit_rows(
            model, dataset="cifar10", model_name="fake_pcn"
        )
        names = [r["tensor_name"] for r in primary]
        self.assertEqual(names.count("PcConvs.0.FFconv.weight"), 1)
        self.assertEqual(names.count("PcConvs.0.FBconv.weight"), 1)
        self.assertEqual(names.count("PcConvs.1.FFconv.weight"), 1)
        self.assertEqual(names.count("PcConvs.1.FBconv.weight"), 1)
        self.assertEqual(names.count("linear.weight"), 1)
        self.assertEqual([r["tensor_name"] for r in nonprimary], ["linear.bias"])

    def test_csv_writer_creates_separate_nonprimary_table(self):
        rows = [
            {
                "dataset": "cifar10",
                "model_condition": "WRN",
                "model_name": "m",
                "semantic_order_index": 1,
                "tensor_name": "conv.weight",
                "module_type": "Conv2d",
                "tensor_shape": "1x1x1x1",
                "num_elements": 1,
                "max_abs": 1.0,
                "rms": 1.0,
                "kappa": 1.0,
                "all_zero": False,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "audit.csv")
            audit.save_audit_csv(path, rows, rows)
            self.assertTrue(Path(path).exists())
            self.assertTrue(Path(audit.nonprimary_csv_path(path)).exists())


if __name__ == "__main__":
    unittest.main()
