import math
import unittest

import torch
from torch import nn

from diagnostic_scripts.queue_feedforward_propagated_snr import metric, paired_snr


class ToyNoisyBlock(nn.Module):
    def __init__(self, seed, coupler_scale):
        super().__init__()
        self.enable_coupler_noise = True
        self._generators = {
            'sum': torch.Generator().manual_seed(seed),
            'coupler': torch.Generator().manual_seed(seed + 1)}
        self.scale = coupler_scale
        self.bn = nn.BatchNorm2d(2)
        self.bn.running_mean.fill_(0.4)
        self.bn.running_var.fill_(2)
        self.bn.bias.data.fill_(0.3)

    def forward(self, x):
        out = x + 0.01 * torch.randn(x.shape, generator=self._generators['sum'])
        if self.enable_coupler_noise:
            out = out + self.scale * torch.randn(x.shape, generator=self._generators['coupler'])
        return self.bn(out)


class PropagatedSNRTests(unittest.TestCase):
    def test_zero_coupler_has_no_propagated_error(self):
        model = nn.Sequential(ToyNoisyBlock(1, 0), ToyNoisyBlock(2, 0)).eval()
        rows, summary, reference, noisy = paired_snr(model, torch.ones(3, 2, 4, 4))
        self.assertTrue(torch.equal(reference, noisy))
        self.assertTrue(all(math.isinf(row['after_bn_snr_db']) for row in rows))
        self.assertTrue(summary['reference_repeat_bit_identical'])
        self.assertTrue(all(block.enable_coupler_noise for block in model))

    def test_upstream_noise_reaches_noise_free_downstream_block(self):
        model = nn.Sequential(ToyNoisyBlock(1, 0.2), ToyNoisyBlock(2, 0)).eval()
        rows, summary, _, _ = paired_snr(model, torch.ones(3, 2, 4, 4))
        self.assertGreater(rows[1]['before_bn_error_rms'], 0)
        self.assertAlmostEqual(rows[0]['after_bn_error_rms'],
                               rows[1]['before_bn_error_rms'], places=6)
        self.assertNotEqual(rows[0]['before_bn_snr_db'], rows[0]['after_bn_snr_db'])
        self.assertTrue(summary['frozen_bn_unchanged'])

    def test_frozen_bn_required(self):
        with self.assertRaises(ValueError):
            paired_snr(nn.Sequential(ToyNoisyBlock(1, 0.2)), torch.ones(3, 2, 4, 4))

    def test_metric(self):
        values = torch.ones(2, 2)
        self.assertAlmostEqual(metric(values, values + 0.1)['snr_db'], 20, places=5)

    def test_metric_rejects_nonfinite_values_and_power_overflow(self):
        for value in (math.nan, math.inf, -math.inf, 1e308):
            for bad_reference in (False, True):
                with self.subTest(value=value, bad_reference=bad_reference):
                    good = torch.ones(2, 2, dtype=torch.float64)
                    bad = torch.full_like(good, value)
                    with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                        metric(bad if bad_reference else good,
                               good if bad_reference else bad)

    def test_later_nonfinite_activations_or_logits_fail_and_restore_hooks(self):
        class Nonfinite(nn.Module):
            def forward(self, x):
                return x * math.nan

        for before_bn in (False, True):
            with self.subTest(before_bn=before_bn):
                first, second = ToyNoisyBlock(1, 0.2), ToyNoisyBlock(2, 0.2)
                first.enable_coupler_noise = False
                tail = ([Nonfinite(), second] if before_bn else
                        [second, Nonfinite()])
                model = nn.Sequential(first, *tail).eval()
                with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                    paired_snr(model, torch.ones(3, 2, 4, 4))
                self.assertFalse(first.enable_coupler_noise)
                self.assertTrue(second.enable_coupler_noise)
                self.assertFalse(first.bn._forward_hooks)
                self.assertFalse(second.bn._forward_hooks)

    def test_small_repeat_error_is_reported_not_rejected(self):
        class Jitter(ToyNoisyBlock):
            count = 0

            def forward(self, x):
                self.count += 1
                return super().forward(x) + self.count * 1e-7

        model = nn.Sequential(Jitter(1, 0.2)).eval()
        _, summary, _, _ = paired_snr(model, torch.ones(3, 2, 4, 4))
        self.assertFalse(summary['reference_repeat_bit_identical'])
        self.assertGreater(summary['reference_repeat_logits']['max_abs_error'], 0)
        self.assertTrue(summary['numerical_floor_below_one_percent'])

    def test_large_repeat_error_is_marked_unreliable(self):
        class Jitter(ToyNoisyBlock):
            count = 0

            def forward(self, x):
                self.count += 1
                return super().forward(x) + self.count

        model = nn.Sequential(Jitter(1, 0.001)).eval()
        _, summary, _, _ = paired_snr(model, torch.ones(3, 2, 4, 4))
        self.assertFalse(summary['numerical_floor_below_one_percent'])

    def test_actual_tc_euler_named_streams_and_bn(self):
        from physical_feedforward_tc import TCPhysicalCIFARBasicBlock
        torch.manual_seed(4)
        model = TCPhysicalCIFARBasicBlock(
            nn.Conv2d(2, 2, 1, bias=False), nn.Conv2d(2, 2, 1, bias=False),
            norm1=nn.BatchNorm2d(2), norm2=nn.BatchNorm2d(2),
            act1=nn.ReLU(), act2=nn.ReLU(),
            tc_method='euler', tc_num_steps=10,
            enable_summing_current_noise=True, enable_coupler_noise=True,
            enable_spin_variation=True).eval()
        rows, summary, _, _ = paired_snr(model, torch.ones(3, 2, 4, 4) * 0.01)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[-1]['after_bn_error_rms'], 0)
        self.assertTrue(summary['reference_repeat_bit_identical'])
        self.assertTrue(model.enable_summing_current_noise)
        self.assertTrue(model.enable_spin_variation)


if __name__ == '__main__':
    unittest.main()
