import math
import unittest

from trainer_timm import SuddenCollapseMonitor


class SuddenCollapseMonitorTests(unittest.TestCase):
    def test_early_near_chance_values_do_not_stop_unarmed_run(self):
        monitor = SuddenCollapseMonitor(100, ema_alpha=1.0)
        for epoch in range(1, 6):
            self.assertIsNone(monitor.observe(epoch, math.log(100), 0.01))
        self.assertFalse(monitor.armed)

    def test_accuracy_collapse_stops_after_learning(self):
        monitor = SuddenCollapseMonitor(100, ema_alpha=1.0)
        self.assertIsNone(monitor.observe(1, 2.0, 0.30))
        self.assertTrue(monitor.armed)
        event = monitor.observe(2, 4.5, 0.02)
        self.assertEqual(event["reason"], "validation_accuracy_returned_near_chance")

    def test_large_loss_return_to_random_stops_after_learning(self):
        monitor = SuddenCollapseMonitor(10, ema_alpha=1.0)
        self.assertIsNone(monitor.observe(1, 1.0))
        self.assertTrue(monitor.armed)
        event = monitor.observe(2, 2.2)
        self.assertEqual(event["reason"], "smoothed_loss_returned_near_random")

    def test_nonfinite_loss_always_stops(self):
        monitor = SuddenCollapseMonitor(10)
        event = monitor.observe(1, float("nan"))
        self.assertEqual(event["reason"], "nonfinite_train_loss")


if __name__ == "__main__":
    unittest.main()
