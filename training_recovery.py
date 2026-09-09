"""Explicit, epoch-boundary recovery; never used for checkpoint discovery."""
import os
import random
import tempfile

import numpy as np
import torch


HISTORY = ('train_loss_list', 'val_acc_list', 'best_acc', 'val_acc',
           'best_epoch', 'best_top5', 'val_top5', 'best_model_path')


def latest_path(trainer):
    return os.path.join(trainer.save_path, trainer.model_name,
                        trainer.model_name + '_latest_ckpt.pth')


def _runtime(model):
    # These are non-buffer hardware RNGs/assignments. Deterministic convolution
    # caches and transient per-forward tensors need not be serialized.
    names = {'_generator', '_gaussian_curve_samples', '_spin_factor_y',
             '_spin_factor_z', '_nonlinear_R_training_curve_indices'}
    return {name: {**{key: value for key, value in vars(module).items()
                     if key in names or key.endswith('_generators')
                     or isinstance(value, torch.Generator)},
                   **{key: module._buffers[key]
                      for key in module._non_persistent_buffers_set}}
            for name, module in model.named_modules()}


def save_latest(trainer, epoch, history):
    aux = {}
    for name in ('_feature_kd_loss', '_crd_loss'):
        module = getattr(trainer, name, None)
        if module is not None:
            aux[name] = module.state_dict()
    components = {}
    for name in ('optimizer', 'scheduler', 'warmup_scheduler', 'scaler'):
        component = getattr(trainer, name, None)
        if component is not None:
            components[name] = component.state_dict()
    state = dict(net=trainer.model.state_dict(), epoch=epoch,
                 acc=history['val_acc'], net_type=type(trainer.model).__name__)
    for name in ('dataset_name', 'img_type', 'timm_model_name', 'pretrained',
                 'use_model_data_config', 'timm_input_size', 'timm_mean',
                 'timm_std', 'interpolation'):
        if hasattr(trainer, name):
            state[name] = getattr(trainer, name)
    if hasattr(trainer.model, 'init_args'):
        state['init_args'] = trainer.model.init_args
    if any('parametrizations.weight.original' in key for key in state['net']):
        state['checkpoint_weight_format'] = 'full_param'
    state['training_recovery'] = dict(
        version=1, history={key: history[key] for key in HISTORY},
        components=components, aux=aux, runtime=_runtime(trainer.model),
        config=getattr(trainer, 'recovery_config', {}),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None),
        loader_generators={name: loader.generator.get_state()
                           for name in ('train_dataloader', 'val_dataloader')
                           if (loader := getattr(trainer, name, None)) is not None
                           and loader.generator is not None})
    path = latest_path(trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.latest-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_latest(trainer):
    path = getattr(trainer, 'recovery_checkpoint', None)
    if not path:
        return None
    # RNG state tensors must remain on CPU, including pickled Generator states.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    recovery = checkpoint.get('training_recovery')
    if recovery is None:
        return None  # Existing weights-only resume/FT behavior.
    if recovery['version'] != 1:
        raise ValueError('Unsupported training recovery version')
    # Recovery is continuation, not conversion into another physical stage.
    current = getattr(trainer, 'recovery_config', {})
    for key in ('physical_pretraining', 'physical_feedforward', 'physical_level',
                'toggle_timing_mode', 'toggle_y_time', 'z_over_y_time',
                'scale_train_recipe', 'one_over_q', 'v_dd', 'num_epochs',
                'override', 'distill_method', 'input_quant_bits',
                'center_student_input', 'timm_aug_level'):
        if key in current and key in recovery['config'] and current[key] != recovery['config'][key]:
            raise ValueError(f'Full recovery requires unchanged {key}; use an ordinary checkpoint for a new stage')
    for name in ('train_dataloader', 'val_dataloader'):
        if getattr(getattr(trainer, name, None), 'persistent_workers', False):
            raise ValueError('Exact epoch recovery requires persistent_workers=False')
    trainer.model.load_state_dict(checkpoint['net'], strict=True)
    for name, state in recovery['aux'].items():
        module = getattr(trainer, name, None)
        if name == '_crd_loss' and module is None:
            dims = [state[f'embed_{side}.linear.weight'].shape[1] for side in ('s', 't')]
            trainer._ensure_crd_initialized(*(torch.empty(1, dim, device=trainer.device)
                                              for dim in dims))
            module = trainer._crd_loss
        if module is None:
            raise ValueError(f'Resume requires the saved distillation module: {name}')
        module.load_state_dict(state)
    for name, state in recovery['components'].items():
        component = getattr(trainer, name, None)
        if component is None:
            raise ValueError(f'Resume requires {name}')
        component.load_state_dict(state)
    for name, attributes in recovery['runtime'].items():
        module = trainer.model.get_submodule(name)
        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(trainer.device)
            if isinstance(value, dict):
                return {key: to_device(item) for key, item in value.items()}
            return value
        for key, value in attributes.items():
            setattr(module, key, to_device(value))
    for name, state in recovery['loader_generators'].items():
        getattr(trainer, name).generator.set_state(state.cpu())
    rng = recovery['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    print(f'Resuming completed epoch {checkpoint["epoch"]} from {path}')
    return checkpoint['epoch'], recovery['history']


def remove_latest(trainer):
    """Called only AFTER the existing final checkpoint writer succeeds."""
    path = latest_path(trainer)
    if os.path.exists(path):
        os.unlink(path)
