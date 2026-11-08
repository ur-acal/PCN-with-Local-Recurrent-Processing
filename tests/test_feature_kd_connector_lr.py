"""Auxiliary KD groups must not inherit the temporary warmup LR."""
from types import SimpleNamespace

import unittest
from unittest import mock
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainer_timm import (TrainerCiFarTimmStyle, TrainerCiFarTimmStyleFeatureKD,
                          TrainerCiFarTimmStyleSRRL)


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


@pytest.mark.parametrize('steps', [1, 2, 4])
@pytest.mark.parametrize('kd', [False, True])
@pytest.mark.parametrize('max_norm', [None, .2])
def test_plain_accumulation_matches_large_batch_updates(steps, kd, max_norm):
    """Exercise the production loop, including momentum, decay and clipping."""
    torch.manual_seed(81)
    inputs, labels = torch.randn(12, 3), torch.randint(2, (12,))
    initial = nn.Linear(3, 2).state_dict()
    trainer = TrainerCiFarTimmStyle.__new__(TrainerCiFarTimmStyle)
    trainer.model = nn.Linear(3, 2)
    trainer.model.load_state_dict(initial)
    trainer.grad_accum_steps = steps
    trainer.device = torch.device('cpu')
    trainer.orig_t_inp = False
    trainer.mixup_fn = trainer.noisy_model = trainer.teacher_model = None
    trainer._kd_enabled = kd
    trainer.max_norm = max_norm
    trainer.distill_alpha, trainer.distill_temperature = .3, 2.
    if kd:
        trainer.teacher_model = nn.Linear(3, 2).requires_grad_(False)
        trainer.teacher_forward_for_distillation = lambda x: (trainer.teacher_model(x).detach(), None)
    trainer.train_loss_fn = nn.CrossEntropyLoss()
    trainer.train_dataloader = DataLoader(
        TensorDataset(inputs, labels), batch_size=2,
        generator=torch.Generator().manual_seed(70))
    trainer._student_forward = lambda x: (trainer.model(x), None)
    trainer.optimizer = torch.optim.SGD(
        trainer.model.parameters(), lr=.1, momentum=.9, weight_decay=.01)
    reference = nn.Linear(3, 2)
    reference.load_state_dict(initial)
    optimizer = torch.optim.SGD(reference.parameters(), lr=.1, momentum=.9, weight_decay=.01)
    for epoch in range(2):
        with mock.patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
            trainer.train_one_epoch(epoch)
            assert step.call_count == (6 + steps - 1) // steps
        for offset in range(0, len(inputs), 2 * steps):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                reference(inputs[offset:offset + 2 * steps]), labels[offset:offset + 2 * steps])
            if kd:
                logits = reference(inputs[offset:offset + 2 * steps])
                target = trainer.teacher_model(inputs[offset:offset + 2 * steps]).detach()
                distill = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / 2., dim=1),
                    nn.functional.softmax(target / 2., dim=1), reduction='batchmean') * 4.
                loss = .7 * loss + .3 * distill
            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
            optimizer.step()
        for actual, expected in zip(trainer.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def _srrl_accumulation_trainer(mode, steps):
    from test_toggle_physical_head import base_model, physical
    from test_tc_physical_head import build_tc
    from ode_pc import (make_ode_block, wrap_ode_block, S2DTCFFFB,
                        S2DTCPulseFFFB, S2DTCQATWrapper2State)
    from distillation.srrl import SRRLLoss
    torch.manual_seed(55)
    model = base_model(measured_activation_scope='all')
    if mode == 'toggle':
        model, _ = physical(model, measured=True)
    elif mode == 'tc':
        model, _ = build_tc(state=2, model=model)
    else:
        make_ode_block(
            model, ode_block=S2DTCPulseFFFB if mode == 'pulse4' else S2DTCFFFB,
            method='rk2', t_end=1.75, n_steps=5, tol=1e-4,
            s2_dtc_substeps=4 if mode == 'pulse4' else 1,
            toggle_timing_mode='fixed', toggle_y_time=10e-9, z_over_y_time=1,
            toggle_timing_R=67e3, toggle_timing_C=500e-15,
            checkpoint_ode_rhs_training=(mode == 'pulse4'))
        if mode != 'pretrain':
            model, _ = wrap_ode_block(
                model, ode_wrapper=S2DTCQATWrapper2State, R=67e3, C=500e-15,
                v_dd=.6, one_over_q=6, w_bits=5, weight_quant_factor_bits=2,
                enable_measured_activation=False)
    trainer = TrainerCiFarTimmStyleSRRL.__new__(TrainerCiFarTimmStyleSRRL)
    trainer.model = model
    trainer.grad_accum_steps = steps
    trainer.device = torch.device('cpu')
    trainer.orig_t_inp = False
    trainer.mixup_fn = trainer.noisy_model = None
    trainer._paired_mixup_active = trainer._kd_enabled = False
    trainer._feature_kd_requested = True
    trainer.max_norm = .2
    trainer.feature_kd_beta = 1.
    trainer.train_loss_fn = nn.CrossEntropyLoss()
    trainer._feature_kd_loss = SRRLLoss(4, 4)
    trainer.teacher_model = nn.Conv2d(3, 4, 1)
    trainer._teacher_extractor = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
    trainer._teacher_extractor.classify_feature = trainer._teacher_extractor.forward
    for module in (trainer.teacher_model, trainer._teacher_extractor):
        module.requires_grad_(False)
    def teacher_forward(x):
        with torch.no_grad():
            features = trainer.teacher_model(x)
            return trainer._teacher_extractor(features), [features]
    trainer.teacher_forward_for_distillation = teacher_forward
    trainer._prepare_student_inputs = lambda x: x
    trainer.optimizer = torch.optim.SGD(
        list(model.parameters()) + list(trainer._feature_kd_loss.parameters()),
        lr=.001, momentum=.9, weight_decay=.001)
    return trainer


@pytest.mark.parametrize('mode', ['pretrain', 'toggle', 's2', 'pulse4', 'tc'])
@pytest.mark.parametrize('steps', [1, 2])
def test_real_pcn_srrl_accumulation_matches_manual_microbatches(mode, steps):
    """Real PCN/SRRL graphs; the final one-minibatch window must not be halved."""
    from pc_model import logits_for_loss
    torch.set_num_threads(1)
    trainer = _srrl_accumulation_trainer(mode, steps)
    reference = _srrl_accumulation_trainer(mode, steps)
    inputs, labels = torch.rand(6, 3, 4, 4), torch.tensor([0, 1, 2, 0, 1, 2])
    trainer.train_dataloader = DataLoader(
        TensorDataset(inputs, labels), batch_size=2,
        generator=torch.Generator().manual_seed(70))
    for epoch in range(2):
        torch.manual_seed(141 + epoch)
        with mock.patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as update:
            trainer.train_one_epoch(epoch)
            assert update.call_count == (3 + steps - 1) // steps
        torch.manual_seed(141 + epoch)
        reference.model.train()
        reference._feature_kd_loss.train()
        for start in range(0, 3, steps):
            reference.optimizer.zero_grad()
            window = min(steps, 3 - start)
            for index in range(start, start + window):
                x, y = inputs[index * 2:index * 2 + 2], labels[index * 2:index * 2 + 2]
                teacher_logits, teacher_features = reference.teacher_forward_for_distillation(x)
                outputs, features = reference._student_forward_feature_kd(x)
                kd_loss, _ = reference._compute_feature_kd_loss(features, teacher_features, teacher_logits)
                loss = nn.functional.cross_entropy(logits_for_loss(outputs, reference.model), y) + kd_loss
                (loss if window == 1 else loss / window).backward()
            nn.utils.clip_grad_norm_(list(reference.model.parameters()) +
                                    list(reference._feature_kd_loss.parameters()), .2)
            reference.optimizer.step()
        for actual, expected in zip(
                list(trainer.model.state_dict().values()) + list(trainer._feature_kd_loss.state_dict().values()),
                list(reference.model.state_dict().values()) + list(reference._feature_kd_loss.state_dict().values())):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_accumulation_fails_before_trainer_initialization():
    with pytest.raises(ValueError, match='grad_accum_steps must be positive'):
        TrainerCiFarTimmStyle(grad_accum_steps=0)


if __name__ == "__main__":
    unittest.main()
