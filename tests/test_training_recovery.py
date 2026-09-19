import random
import ast
from pathlib import Path
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from training_recovery import HISTORY, latest_path, save_latest, restore_latest, remove_latest


class IdentityQuantizer(nn.Module):
    def forward(self, weight):
        return weight


def make_trainer(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.2), nn.Linear(4, 2))
    parametrize.register_parametrization(model[0], 'weight', IdentityQuantizer())
    model._noise_generators = {'cpu': torch.Generator().manual_seed(91)}
    aux = nn.Linear(2, 2)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': aux.parameters()}],
                                lr=.1, momentum=.9)
    return SimpleNamespace(model=model, _feature_kd_loss=aux, optimizer=optimizer,
                           scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, .8),
                           device='cpu', save_path=str(tmp_path), model_name='toy')


def step(trainer):
    trainer.optimizer.zero_grad()
    x = torch.randn(8, 3) + random.random() + np.random.rand()
    x += torch.rand(8, 3, generator=trainer.model._noise_generators['cpu'])
    trainer._feature_kd_loss(trainer.model(x)).square().mean().backward()
    trainer.optimizer.step()
    trainer.scheduler.step()


def test_exact_qat_aux_optimizer_rng_continuation(tmp_path):
    trainer = make_trainer(tmp_path)
    step(trainer)
    history = dict.fromkeys(HISTORY, None)
    history.update(val_acc=.2, train_loss_list=[1.], val_acc_list=[.2])
    before = torch.get_rng_state().clone()
    weight = trainer.model[0].weight.detach().clone()
    save_latest(trainer, 1, history)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(weight, trainer.model[0].weight)
    assert parametrize.is_parametrized(trainer.model[0])
    step(trainer)
    expected = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    resumed = make_trainer(tmp_path)
    resumed.recovery_checkpoint = latest_path(trainer)
    assert restore_latest(resumed) == (1, history)
    step(resumed)
    for key, value in expected.items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert resumed.optimizer.param_groups[0]['lr'] == trainer.optimizer.param_groups[0]['lr']
    for key, value in trainer._feature_kd_loss.state_dict().items():
        assert torch.equal(value, resumed._feature_kd_loss.state_dict()[key])
    remove_latest(resumed)
    assert not __import__('os').path.exists(latest_path(resumed))


def test_no_automatic_discovery_and_atomic_failure(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    history = dict.fromkeys(HISTORY, None)
    save_latest(trainer, 1, history)
    assert restore_latest(trainer) is None
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        save_latest(trainer, 2, history)
    assert torch.load(latest_path(trainer), weights_only=False)['epoch'] == 1


def test_save_latest_accepts_loader_without_generator_attribute(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train_dataloader = object()
    trainer.val_dataloader = object()
    history = dict.fromkeys(HISTORY, None)
    save_latest(trainer, 1, history)
    checkpoint = torch.load(latest_path(trainer), weights_only=False)
    assert checkpoint['training_recovery']['loader_generators'] == {}


@pytest.mark.parametrize('filename,classname', [('trainer.py', 'TrainerCiFar'),
                                               ('trainer_timm.py', 'TrainerCiFarTimmStyle')])
def test_real_train_loop_resume_and_final_cleanup(tmp_path, filename, classname):
    # Execute the actual loop without importing sensor-data dependencies.
    tree = ast.parse((Path(__file__).resolve().parents[1] / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    namespace = dict(restore_latest=restore_latest, save_latest=save_latest, remove_latest=remove_latest)
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'), namespace)
    trainer = make_trainer(tmp_path)
    trainer.num_epochs = 3
    trainer.eval_every = 2
    trainer.skip_eval_epochs = 0
    trainer.final_eval_only = False
    trainer.dataset_name = 'cifar100'
    trainer.train_dataloader = trainer.val_dataloader = None
    trainer.evaluate = lambda _: (.5, .8, None, None)
    seen = []
    def train_epoch(epoch):
        seen.append(epoch)
        if epoch == 1:
            raise RuntimeError('interruption')
        return 1.
    trainer.train_one_epoch = train_epoch
    saved = []
    def save_final(acc, epoch, suffix):
        saved.append(suffix)
        assert Path(latest_path(trainer)).exists()
        return 'existing-checkpoint-path'
    trainer._save_model_ckpt = save_final
    with pytest.raises(RuntimeError, match='interruption'):
        namespace['train'](trainer)
    assert not saved  # No final-completion artifact on interruption.
    trainer.recovery_checkpoint = latest_path(trainer)
    trainer.train_one_epoch = lambda epoch: seen.append(epoch) or 1.
    namespace['train'](trainer)
    assert seen == [0, 1, 1, 2]
    assert saved == ['_best_ckpt.pth', '_last_ckpt.pth']
    assert not Path(latest_path(trainer)).exists()


@pytest.mark.parametrize('filename,classname', [('trainer.py', 'TrainerCiFar'),
                                               ('trainer_timm.py', 'TrainerCiFarTimmStyle')])
def test_final_only_policy_uses_health_checks_then_one_final_eval(
        tmp_path, filename, classname):
    tree = ast.parse((Path(__file__).resolve().parents[1] / filename).read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == classname)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'train')
    namespace = dict(restore_latest=restore_latest, save_latest=save_latest,
                     remove_latest=remove_latest)
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'),
         namespace)
    trainer = make_trainer(tmp_path)
    trainer.num_epochs = 3
    trainer.eval_every = 1
    trainer.skip_eval_epochs = 0
    trainer.final_eval_only = True
    trainer.health_check_epochs = {2}
    trainer.health_check_batches = 4
    trainer.dataset_name = 'cifar100'
    trainer.train_dataloader = trainer.val_dataloader = SimpleNamespace(
        generator=None)
    trainer.train_one_epoch = lambda epoch: float(epoch)
    health = []
    trainer._evaluate_training_health = lambda: health.append(2) or (.2, .5)
    final_evaluations = []
    trainer.evaluate = lambda loader: final_evaluations.append(loader) or (
        .6, .9, None, None)
    saved = []
    trainer._save_model_ckpt = lambda acc, epoch, suffix: (
        saved.append((suffix, acc, epoch)) or 'last-path')

    namespace['train'](trainer)

    assert health == [2]
    assert final_evaluations == [trainer.val_dataloader]
    assert saved == [('_last_ckpt.pth', .6, 3)]
    assert not Path(latest_path(trainer)).exists()


def test_real_training_health_check_executes_and_restores_rng():
    """Exercise the actual method in trainer.py, including its module globals."""
    import trainer as trainer_module

    data_generator = torch.Generator().manual_seed(73)
    loader = DataLoader(
        TensorDataset(torch.randn(12, 4), torch.arange(12) % 3),
        batch_size=2, shuffle=True, generator=data_generator, num_workers=0)
    dummy = SimpleNamespace(
        model=nn.Linear(4, 3), train_dataloader=loader,
        health_check_seed=4096,
        health_check_batches=2,
        device=torch.device('cpu'), dataset_name='cifar10', orig_t_inp=False,
        loss_fn=nn.CrossEntropyLoss(),
        _prepare_student_inputs=lambda inputs: inputs,
        reset_spin_variation_for_inference=lambda: None,
    )
    dummy.evaluate = types.MethodType(trainer_module.TrainerCiFar.evaluate, dummy)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator_state = data_generator.get_state().clone()

    health_acc, health_top5 = (
        trainer_module.TrainerCiFar._evaluate_training_health(dummy))
    assert 0 <= health_acc <= 1
    assert health_top5 is None
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(data_generator.get_state(), generator_state)
