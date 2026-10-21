import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline.activation_diagnostics import (
    RelativeError, cache_clean, evaluate_cached, make_helper,
    run_diagnostics, selected_parameters, summarize, units_for,
)
from baseline.cifar_resnet import WideResNetCIFAR
from baseline.cifar_wrn_pool_after_add import WideResNetPoolAfterAddCIFAR
from baseline.run_baseline import BNRecalibrationConfig


class ActivationDiagnosticTests(unittest.TestCase):
    def test_shared_cache_reuse_and_identity_guard(self):
        model = self.model()
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'shared'
            run_diagnostics(model, 'wrn', loader, None, None, Path(tmp) / 'first', 'cpu',
                            claims=(), cache_directory=cache, cache_signature={'identity': 1})
            with patch('baseline.activation_diagnostics.cache_clean', side_effect=AssertionError('cache rebuilt')):
                run_diagnostics(model, 'wrn', loader, None, None, Path(tmp) / 'second', 'cpu',
                                claims=(), cache_directory=cache, cache_signature={'identity': 1})
                with self.assertRaises(ValueError):
                    run_diagnostics(model, 'wrn', loader, None, None, Path(tmp) / 'bad', 'cpu',
                                    claims=(), cache_directory=cache, cache_signature={'identity': 2})

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_metrics_batch_invariance_and_zero_flags(self):
        clean = torch.tensor([[1., 2.], [0., 0.], [1e-20, 0.], [2., 0.]])
        noisy = clean + torch.tensor([[0.1, 0.2], [2., 0.], [1e-20, 0.], [1., 0.]])
        one, split = RelativeError(), RelativeError()
        one.update(noisy, clean, torch.arange(4))
        split.update(noisy[:3], clean[:3], torch.arange(3))
        split.update(noisy[3:], clean[3:], torch.tensor([3]))
        for key, value in one.result().items():
            self.assertAlmostEqual(value, split.result()[key])
        self.assertEqual(one.zeros, [dict(sample_id=1, absolute_l2=2.)])
        self.assertEqual(one.valid, 3)
        zero = RelativeError()
        zero.update(torch.ones(1, 2), torch.zeros(1, 2), [0])
        self.assertIsNone(zero.result()["dataset_relative_l2"])
        self.assertIsNone(zero.result()["mean_image_relative_l2"])

    def model(self):
        torch.manual_seed(9)
        return WideResNetCIFAR(depth=10, widen_factor=1, num_classes=3,
                              final_dropout_rate=0.25).eval()

    def test_selection_includes_shortcuts_linear_bias_and_restores_bn(self):
        model = self.model()
        all_names = selected_parameters(model)
        self.assertIn("conv1.weight", all_names)
        self.assertIn("fc.bias", all_names)
        self.assertIn("layer2.0.shortcut.weight", all_names)
        target = selected_parameters(model, "layer2.0")
        self.assertNotIn("fc.weight", target)
        self.assertTrue(all(n.startswith("layer2.0.") for n in target))
        before = {n: v.clone() for n, v in model.state_dict().items()}
        helper = make_helper(model, target, "max_additive")
        helper.noise_sigma, helper.seed = 0.08, 123
        helper.add_noise()
        for n, v in model.state_dict().items():
            self.assertEqual(torch.equal(v, before[n]), n not in target)
        model.bn.running_mean.add_(3)
        helper.restore_clean_state()
        for n, v in model.state_dict().items():
            torch.testing.assert_close(v, before[n], rtol=0, atol=0)

    def test_fixed_noise_formula_and_no_resampling(self):
        for condition in ("max_additive", "multiplicative"):
            model = self.model()
            with torch.no_grad():
                model.fc.bias.fill_(0.2)
            selected = selected_parameters(model)
            helper = make_helper(model, selected, condition)
            helper.noise_sigma, helper.seed = 0.08, 123
            generator = torch.Generator().manual_seed(123)
            expected = {}
            for name, p in model.named_parameters():
                if name in selected:
                    noise = torch.randn(p.shape, generator=generator)
                    expected[name] = (p.detach() * (1 + 0.08 * noise)
                        if condition == "multiplicative"
                        else p.detach() + 0.08 * noise * p.detach().abs().max())
            helper.add_noise()
            for name, p in model.named_parameters():
                if name in expected:
                    torch.testing.assert_close(p, expected[name], rtol=0, atol=0)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            with torch.no_grad():
                x = torch.randn(2, 3, 8, 8)
                a, b = model(x.clone()), model(x.clone())
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)

    def test_capture_before_relu_and_clean_isolated_identity(self):
        model = self.model()
        data = DataLoader(TensorDataset(torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2])),
                          batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            paths, summary = cache_clean(model, "wrn", data, Path(tmp) / "clean", "cpu")
            self.assertEqual(summary["samples"], 3)
            final = torch.load(paths[0].parent / "final_feature.pt", weights_only=True)
            last = torch.load(paths[0].parent / "layer3.0.pt", weights_only=True)
            torch.testing.assert_close(final["output"], last["output"], rtol=0, atol=0)
            for name, _ in units_for(model, "wrn"):
                metrics, _, _ = evaluate_cached(model, "wrn", paths, "cpu", name)
                self.assertEqual(metrics[name].result()["dataset_relative_l2"], 0)

    def test_pool_after_add_records_pre_pool_boundary(self):
        model = WideResNetPoolAfterAddCIFAR(
            depth=10, widen_factor=1, num_classes=3, pool_type="max").eval()
        data = DataLoader(TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])),
                          batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = cache_clean(model, "wrn", data, Path(tmp) / "clean", "cpu")
            transition = torch.load(paths[0].parent / "layer2.0.pt", weights_only=True)
            self.assertEqual(transition["output"].shape[-2:], (8, 8))
            metrics, _, _ = evaluate_cached(model, "wrn", paths, "cpu", "layer2.0")
            self.assertEqual(metrics["layer2.0"].result()["dataset_relative_l2"], 0)

    def test_pipeline_paired_bn_restoration_and_reports(self):
        model = self.model()
        before = {n: v.clone() for n, v in model.state_dict().items()}
        x = torch.randn(3, 3, 8, 8)
        loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 2])), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BNRecalibrationConfig(enabled=True, diagnostics_dir=str(Path(tmp) / "bn"))
            rows = run_diagnostics(model, "wrn", loader, [x], cfg, tmp, "cpu",
                levels={"max_additive": [0.08]}, trials=2)
            summarize(rows, tmp)
            self.assertEqual({r["bn_policy"] for r in rows}, {"frozen", "recalibrated"})
            self.assertEqual({r["target"] for r in rows if r["claim"] == 1},
                             {n for n, _ in units_for(model, "wrn")})
            self.assertTrue(all(r["samples"] == 3 for r in rows))
            result = json.loads((Path(tmp) / "claim2/network/max_additive/level_0.08/"
                                 "trial_0/recalibrated/summary.json").read_text())
            self.assertEqual(result["noisy_correct"] - result["clean_correct"],
                             result["wrong_to_correct"] - result["correct_to_wrong"])
            clean_logits = torch.load(Path(tmp) / 'clean_logits.pt', weights_only=True)
            self.assertEqual(clean_logits['logits'].shape, (3, 3))
            for trial in range(2):
                for policy in ('frozen', 'recalibrated'):
                    path = Path(tmp) / f'claim2/network/max_additive/level_0.08/trial_{trial}/{policy}'
                    noisy = torch.load(path / 'noisy_logits.pt', weights_only=True)
                    summary = json.loads((path / 'summary.json').read_text())
                    self.assertTrue(torch.equal(noisy['sample_ids'], clean_logits['sample_ids']))
                    self.assertTrue(torch.equal(noisy['labels'], clean_logits['labels']))
                    self.assertEqual(noisy['logits'].shape, (3, 3))
                    self.assertEqual(int((noisy['logits'].argmax(1) == noisy['labels']).sum()),
                                     summary['noisy_correct'])
            self.assertTrue((Path(tmp) / "aggregate.csv").exists())
            with self.assertRaises(FileExistsError):
                run_diagnostics(model, "wrn", loader, [x], cfg, tmp, "cpu")
        for n, v in model.state_dict().items():
            torch.testing.assert_close(v, before[n], rtol=0, atol=0)

    def test_pcn_major_verifications(self):
        from pc_model import PCNetNoBatchNorm
        from pc_conv import PCConvNoisy
        from ode_pc import ODEXInitFFFB, make_ode_block
        model = PCNetNoBatchNorm(inp_channels=[3, 4], out_channels=[4, 4],
            max_pool=[0, 1], num_classes=3, dropout=0.25, first_bn=False,
            bias=False, cls=2, bypass=False, pc_conv_layer=PCConvNoisy,
            noise_level=0.0)
        model = make_ode_block(model, ode_block=ODEXInitFFFB, noise_level=0,
                               method="dopri5", t_end=1.75, tol=1e-4).eval()
        self.assertIn("linear.bias", selected_parameters(model))
        self.assertFalse(any("b0" in n for n in selected_parameters(model)))
        data = DataLoader(TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])),
                          batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_diagnostics(model, "pcn", data, None, None, tmp, "cpu",
                levels={"multiplicative": [0.30], "max_additive": [0.08]}, trials=1)
            self.assertEqual({r["bn_policy"] for r in rows}, {"no_bn"})
            self.assertEqual({r["claim"] for r in rows}, {1, 2})
            last = torch.load(Path(tmp) / "clean/batch_00000/PcConvs.1.pt", weights_only=True)
            final = torch.load(Path(tmp) / "clean/batch_00000/final_feature.pt", weights_only=True)
            self.assertEqual(last["output"].shape[-1], 8)
            self.assertEqual(final["output"].shape[-1], 4)
            logits = torch.load(Path(tmp) / 'claim2/network/multiplicative/level_0.3/'
                                'trial_0/no_bn/noisy_logits.pt', weights_only=True)
            self.assertEqual(logits['logits'].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
