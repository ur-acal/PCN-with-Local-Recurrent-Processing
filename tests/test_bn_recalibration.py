
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline.run_baseline as rb


class TensorOnlyDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


class LabeledTensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TinyBNNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5) if dropout else nn.Identity()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TinyNoBNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x).mean(dim=(2, 3))
        return self.fc(x)


def _model(seed=0, dropout=False):
    torch.manual_seed(seed)
    model = TinyBNNet(dropout=dropout)
    model.eval()
    return model


def _inputs(seed=10, n=8):
    torch.manual_seed(seed)
    return torch.randn(n, 3, 8, 8)


def _labels(n=8):
    return torch.arange(n) % 2


def _helper(model, sigma, noise_type="multiplicative", seed=123):
    helper = rb.FixedMismatchHelper(
        model=model,
        noise_sigma=sigma,
        noise_type=noise_type,
        noise_to_norm=False,
        include_buffers=True,
        seed=seed,
    )
    helper.snapshot_clean_state()
    helper.restore_clean_state()
    if sigma > 0:
        helper.add_noise()
    return helper


def _recalibrate(model, x, tmp_path, noise_level=0.1):
    cfg = rb.BNRecalibrationConfig(
        enabled=True,
        num_samples=str(len(x)),
        batch_size=4,
        subset_seed=5,
        num_workers=0,
        diagnostics_dir=str(tmp_path),
    )
    loader = DataLoader(TensorOnlyDataset(x), batch_size=4, shuffle=False)
    return rb.recalibrate_batchnorm_statistics(
        model=model,
        calibration_loader=loader,
        device=torch.device("cpu"),
        recal_cfg=cfg,
        model_arch="tiny",
        noise_level=noise_level,
        trial=0,
    )


class BNRecalibrationTests(unittest.TestCase):
    def test_recalibration_disabled_reproduces_logits_predictions_and_accuracy(self):
        x = _inputs()
        y = _labels(len(x))
        model_a = _model(seed=1)
        model_b = copy.deepcopy(model_a)
        _helper(model_a, sigma=0.1, seed=77)
        _helper(model_b, sigma=0.1, seed=77)

        with torch.no_grad():
            logits_a = model_a(x)
            logits_b = model_b(x)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(logits_a.argmax(dim=1), logits_b.argmax(dim=1)))

        loader = DataLoader(LabeledTensorDataset(x, y), batch_size=4, shuffle=False)
        acc_a = rb.evaluate_once(model_a, loader, torch.device("cpu"), max_batches=2)
        acc_b = rb.evaluate_once(model_b, loader, torch.device("cpu"), max_batches=2)
        self.assertEqual(acc_a, acc_b)

    def test_same_mismatch_seed_produces_same_fixed_noisy_weights(self):
        model_a = _model(seed=2)
        model_b = _model(seed=2)
        _helper(model_a, sigma=0.2, seed=1234)
        _helper(model_b, sigma=0.2, seed=1234)
        self.assertEqual(rb._snapshot_hashes(model_a.named_parameters()), rb._snapshot_hashes(model_b.named_parameters()))

    def test_recalibration_does_not_resample_or_alter_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=3)
            _helper(model, sigma=0.15, seed=222)
            before = rb._snapshot_hashes(model.named_parameters())
            report = _recalibrate(model, _inputs(n=12), tmp_path, noise_level=0.15)
            after = rb._snapshot_hashes(model.named_parameters())
            self.assertEqual(before, after)
            self.assertEqual(report.fixed_noisy_param_hash, rb.combined_hash(model.named_parameters()))

    def test_learned_parameters_and_bn_affine_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=4)
            _helper(model, sigma=0.1, seed=333)
            before_params = rb._snapshot_hashes(model.named_parameters())
            before_bn_affine = rb._snapshot_hashes(rb.bn_affine_tensors(model))
            _recalibrate(model, _inputs(n=16), tmp_path)
            self.assertEqual(before_params, rb._snapshot_hashes(model.named_parameters()))
            self.assertEqual(before_bn_affine, rb._snapshot_hashes(rb.bn_affine_tensors(model)))

    def test_only_allowed_bn_buffers_change(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=5)
            _helper(model, sigma=0.1, seed=444)
            before_buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}
            report = _recalibrate(model, _inputs(n=16), tmp_path)
            changed = []
            for name, buf in model.named_buffers():
                if not torch.equal(buf, before_buffers[name]):
                    changed.append(name)
                    self.assertIn(name, report.changed_bn_buffers)
                    self.assertTrue(name.endswith(("running_mean", "running_var", "num_batches_tracked")))
            self.assertEqual(set(changed), set(report.changed_bn_buffers))

    def test_dropout_and_non_bn_modules_remain_eval(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=6, dropout=True)
            _helper(model, sigma=0.1, seed=555)
            _recalibrate(model, _inputs(n=16), tmp_path)
            self.assertFalse(model.drop.training)
            for name, module in model.named_modules():
                if name and not isinstance(module, rb.BN_TYPES):
                    self.assertFalse(module.training)

    def test_calibration_subset_seed_is_deterministic(self):
        class FakeCIFAR(Dataset):
            def __init__(self, root, train, download, transform):
                self.transform = transform
                self.x = torch.zeros(20, 3, 8, 8)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], idx

        with mock.patch.object(rb.datasets, "CIFAR10", FakeCIFAR), mock.patch.object(
            rb, "_build_eval_transform", lambda model, dataset_name, cfg: (lambda x: x)
        ):
            args = SimpleNamespace(dataset="cifar10", data_dir="/tmp", pin_memory=False)
            cfg = {"timm_input_size": (3, 32, 32)}
            recal_cfg = rb.BNRecalibrationConfig(enabled=True, num_samples="7", batch_size=2, subset_seed=99, num_workers=0)
            loader_a = rb.build_bn_calibration_loader(_model(), args, cfg, recal_cfg)
            loader_b = rb.build_bn_calibration_loader(_model(), args, cfg, recal_cfg)
            self.assertEqual(loader_a.dataset.base_dataset.indices, loader_b.dataset.base_dataset.indices)

    def test_sigma_zero_recalibration_works(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=7)
            _helper(model, sigma=0.0, seed=777)
            before = rb._snapshot_hashes(model.named_parameters())
            report = _recalibrate(model, _inputs(n=16), tmp_path, noise_level=0.0)
            self.assertEqual(before, rb._snapshot_hashes(model.named_parameters()))
            self.assertEqual(report.num_bn_modules, 1)

    def test_clean_checkpoint_can_be_recalibrated_as_sanity_control(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            model = _model(seed=8)
            before = rb._snapshot_hashes(model.named_parameters())
            report = _recalibrate(model, _inputs(n=16), tmp_path, noise_level=0.0)
            self.assertEqual(before, rb._snapshot_hashes(model.named_parameters()))
            self.assertEqual(report.num_calibration_samples, 16)

    def test_no_bn_model_is_documented_noop(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            torch.manual_seed(9)
            model = TinyNoBNNet().eval()
            cfg = rb.BNRecalibrationConfig(enabled=True, diagnostics_dir=str(tmp_path), num_workers=0)
            loader = DataLoader(TensorOnlyDataset(_inputs(n=4)), batch_size=2)
            report = rb.recalibrate_batchnorm_statistics(
                model=model,
                calibration_loader=loader,
                device=torch.device("cpu"),
                recal_cfg=cfg,
                model_arch="tiny_no_bn",
                noise_level=0.1,
                trial=0,
            )
            self.assertEqual(report.num_bn_modules, 0)
            self.assertEqual(report.changed_bn_buffers, [])
            self.assertTrue(Path(report.diagnostics_path).exists())


if __name__ == "__main__":
    unittest.main()
