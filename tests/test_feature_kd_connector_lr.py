"""Auxiliary KD groups must not inherit the temporary warmup LR."""
from types import SimpleNamespace

import unittest
import torch

from trainer_timm import TrainerCiFarTimmStyle, TrainerCiFarTimmStyleFeatureKD


class Auxiliary(torch.nn.Linear):
    def trainable_parameters(self):
        return self.parameters()


def check_schedule(test, schedule, warmup, lr):
    model = torch.nn.Linear(2, 2)
    owner = SimpleNamespace(
        optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=0.001),
        _feature_kd_loss=Auxiliary(2, 3), timm_sched=schedule,
        T_0=None, num_epochs=300, min_lr=1e-6,
        warmup_epoch=warmup, warmup_lr=1e-5,
        lr_reduce_on="80,122", decay_rate=0.1,
    )
    # Mirror the real base-trainer -> feature-KD constructor ordering.
    TrainerCiFarTimmStyle._build_timm_scheduler(owner)
    TrainerCiFarTimmStyleFeatureKD._add_feature_kd_to_optimizer(owner)
    TrainerCiFarTimmStyle._build_timm_scheduler(owner)
    test.assertEqual(owner.scheduler.base_values, [lr, lr])
    for epoch in [0, 1, 5, 10, 100, 200]:
        owner.scheduler.step(epoch)
        groups = owner.optimizer.param_groups
        test.assertAlmostEqual(groups[1]["lr"], groups[0]["lr"])
        test.assertEqual(groups[1]["weight_decay"], groups[0]["weight_decay"])
        test.assertEqual(groups[1]["momentum"], groups[0]["momentum"])


class ConnectorLearningRateTest(unittest.TestCase):
    def test_scheduler_initialization(self):
        for schedule in ["cosine", "multistep"]:
            for warmup in [0, 5]:
                for lr in [0.005, 0.01]:
                    with self.subTest(schedule=schedule, warmup=warmup, lr=lr):
                        check_schedule(self, schedule, warmup, lr)

    def test_without_scheduler_initial_lr_retains_existing_fallback(self):
        owner = SimpleNamespace(
            optimizer=torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.02),
            _feature_kd_loss=Auxiliary(2, 3),
        )
        TrainerCiFarTimmStyleFeatureKD._add_feature_kd_to_optimizer(owner)
        self.assertEqual(owner.optimizer.param_groups[-1]["lr"], 0.02)


if __name__ == "__main__":
    unittest.main()
