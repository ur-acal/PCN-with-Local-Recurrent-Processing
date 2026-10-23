"""Opt-in CNN FT coupling, using the same current-ReLU bank as MC45."""
import ast
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from baseline.cifar_resnet import WideResNetCIFAR
from baseline.train_baseline_cifar import parse_args
from data_utils import MC45CornerData
from ft_corner_sampling import _activation_corner_curve_indices
from measured_activation import (
    CubicBSplineActivation, MEASURED_ACTIVATION_TYPES,
    configure_feedforward_measured_activation,
    configure_measured_activation_corner_mode,
    feedforward_measured_activation_factory,
)
from measured_pooling import MeasuredAvgPool2d, configure_feedforward_measured_pooling
from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_wrappers

ROOT = Path(__file__).resolve().parents[1]
RELU = ROOT / 'hardware_data/mc_45_corners/relu_monteCarlo'
COUPLER = ROOT / 'hardware_data/mc_45_corners/coupler_full_range'


def test_current_relu_mapping_matches_every_mc45_evaluation_file():
    _, curves, _ = CubicBSplineActivation._load_csv(RELU)
    names = tuple(curves)
    mapping = _activation_corner_curve_indices(names)
    catalog = MC45CornerData(ROOT / 'hardware_data/mc_45_corners')
    assert len(mapping) == 45
    for corner in catalog.corners:
        prefix = Path(corner['relu']['path']).stem.upper() + '_'
        assert len(mapping[corner['id']]) == 100
        assert {names[i] for i in mapping[corner['id']]} == {
            prefix + f'MC{i}' for i in range(1, 101)}


def test_real_wrn_forward_backward_couples_all_three_banks():
    torch.manual_seed(73)
    model = convert_wide_resnet_to_physical(
        WideResNetCIFAR(depth=10, base_width=2, widen_factor=1,
                       in_chans=4, num_classes=3),
        physical=True, physical_level=2, R=50e3, C=500e-15,
        v_dd=0.5, one_over_q=20, toggle_timing_mode='fixed',
        toggle_y_time=15e-9, enable_spin_variation=False,
        enable_summing_current_noise=False, enable_coupler_noise=False)
    factory = feedforward_measured_activation_factory(
        RELU, .5, corner='RELU_TTG4_MC1', curve_sharing='per_spin',
        fuse_measured_activation=False)
    configure_feedforward_measured_activation(model, factory)
    configure_measured_activation_corner_mode(model, 'random_per_forward', 'per_spin')
    wrappers = list(iter_physical_wrappers(model))
    package = wrappers[0].configure_nonlinear_R_training(COUPLER)
    for wrapper in wrappers[1:]:
        wrapper.install_nonlinear_R_training_package(package)
    configure_feedforward_measured_pooling(
        model, enable_nonideality=True, curve_path=COUPLER,
        quantity='conductance', nominal_R=50e3, training_curve_mode='exact_curve')

    # Execute the actual trainer's opt-in setup block, not a parallel test implementation.
    import baseline.train_baseline_cifar as train
    tree = ast.parse(Path(train.__file__).read_text())
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Attribute)
              and n.test.attr == 'ft_corner_coupled_sampling']
    assert len(blocks) == 1
    with patch.object(sys, 'argv', ['train', '--model_name', 'wrn_flexible_cifar_avgpool_shortcut',
                                  '--dataset', 'cifar100', '--ft_corner_coupled_sampling', 'true',
                                  '--activation_corner_mode', 'random_per_forward']):
        args = parse_args()
    exec(compile(ast.Module(body=blocks, type_ignores=[]), str(train.__file__), 'exec'),
         dict(model=model, args=args, activation_enabled=True, logging=logging))
    sampler = model._ft_corner_coupled_sampler
    before_keys = tuple(model.state_dict())
    for corner in ('TT_V1_T1', 'SS_V2_T2'):
        sampler.corners = (corner,)
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(torch.rand(2, 4, 8, 8))
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        assert model._last_ft_corner_coupled_corner == corner
        for wrapper in wrappers:
            block = wrapper.block
            allowed = set(block._nonlinear_R_training_pkg['ft_corner_curve_indices'][corner])
            assert block._nonlinear_R_training_curve_indices
            assert set(block._nonlinear_R_training_curve_indices.values()) <= allowed
        activations = [m for m in model.modules() if isinstance(m, MEASURED_ACTIVATION_TYPES)]
        for activation in activations:
            assert activation._sampled_curve_indices is not None
            assert set(activation._sampled_curve_indices.flatten().tolist()) <= set(
                activation._ft_corner_curve_indices[corner])
        pools = [m for m in model.modules() if isinstance(m, MeasuredAvgPool2d)]
        assert pools
        for pool in pools:
            assert pool._curve_assignments
            for assignment in pool._curve_assignments.values():
                assert set(assignment.flatten().tolist()) <= set(pool._ft_corner_curve_indices[corner])
    model.eval()
    with torch.no_grad():
        model(torch.rand(2, 4, 8, 8))
    assert sampler.active_corner is None
    assert model._last_ft_corner_coupled_corner is None
    assert tuple(model.state_dict()) == before_keys


@pytest.mark.parametrize('coupled, override', [(False, False), (True, False), (True, True)])
def test_ft_shell_forwards_flag_and_keeps_legacy_bank_default(tmp_path, coupled, override):
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    env = dict(PATH=str(tmp_path) + ':' + os.environ['PATH'],
               MODEL_NAME='wrn_flexible_cifar_avgpool_shortcut', MODEL_CKPT='unused',
               ACTIVATION_CORNER_MODE='random_per_forward', ACTIVATION_CURVE_SHARING='per_spin',
               FT_CORNER_COUPLED_SAMPLING=str(coupled).lower())
    if override:
        env.update(ACTIVATION_CURVE_PATH='explicit/bank', ACTIVATION_CORNER='EXPLICIT_MC1')
    result = subprocess.run(['bash', 'launch_scripts/run_feedforward_physical_ft.sh'],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    line = next(l for l in result.stdout.splitlines() if l.startswith('Running:'))
    tokens = shlex.split(line)[3:]
    with patch.object(sys, 'argv', ['train', *tokens]):
        args = parse_args()
    assert args.ft_corner_coupled_sampling == coupled
    assert args.activation_curve_sharing == 'per_spin'
    if override:
        assert args.activation_curve_path == 'explicit/bank'
        assert args.activation_corner == 'EXPLICIT_MC1'
        return
    assert args.activation_curve_path.endswith(
        'mc_45_corners/relu_monteCarlo' if coupled else 'relu_current_0p2uA_all.csv')
    assert args.activation_corner == ('RELU_TTG4_MC1' if coupled else 'TT')


def test_coupling_off_by_default_and_rejected_for_pretraining():
    from baseline.train_baseline_cifar import main
    base = ['train', '--model_name', 'wrn_flexible_cifar_avgpool_shortcut', '--dataset', 'cifar100']
    with patch.object(sys, 'argv', base):
        assert not parse_args().ft_corner_coupled_sampling
    with patch.object(sys, 'argv', [*base, '--ft_corner_coupled_sampling', 'true',
                                  '--physical_pretraining', 'true']):
        with pytest.raises(ValueError, match='Level-2 fine-tuning'):
            main()


@pytest.mark.parametrize('coupled,override', [(False, False), (True, False), (True, True)])
def test_slurm_to_ft_launcher_preserves_coupled_bank_defaults(coupled, override):
    # Exercise the submitter's real exported environment and the real FT builder;
    # intercept only submission/Python so no training or scheduler job is started.
    env = dict(PATH=os.environ['PATH'], REPO_ROOT=str(ROOT),
               MODEL_NAME='wrn_flexible_cifar_avgpool_shortcut', MODEL_CKPT='unused',
               mode='ft_and_eval', ACTIVATION_CORNER_MODE='random_per_forward',
               ACTIVATION_CURVE_SHARING='per_spin',
               FT_CORNER_COUPLED_SAMPLING=str(coupled).lower())
    if override:
        env.update(ACTIVATION_CURVE_PATH='explicit/bank', ACTIVATION_CORNER='EXPLICIT_MC1')
    command = '''
mkdir() { :; }
sbatch() {
  python() { :; }; export -f python
  bash ./launch_scripts/run_feedforward_physical_ft.sh
}
source ./launch_scripts/slurm_search_feedforward_config.sh
'''
    out = subprocess.check_output(['bash', '-c', command], cwd=ROOT, env=env, text=True)
    line = next(l for l in out.splitlines() if l.startswith('Running:'))
    with patch.object(sys, 'argv', ['train', *shlex.split(line)[3:]]):
        args = parse_args()
    assert args.ft_corner_coupled_sampling == coupled
    assert args.activation_curve_sharing == 'per_spin'
    assert args.activation_curve_path == (
        'explicit/bank' if override else
        './hardware_data/mc_45_corners/relu_monteCarlo' if coupled else
        './hardware_data/relu_current_0p2uA_all.csv')
    assert args.activation_corner == (
        'EXPLICIT_MC1' if override else 'RELU_TTG4_MC1' if coupled else 'TT')
