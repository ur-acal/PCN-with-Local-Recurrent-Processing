"""Regression coverage for diagnostic failures found in pre-commit review."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('noise_level', [0., .1, {'FFconv': .05, 'FBconv': .1}])
def test_parameter_noise_uses_actual_device_and_preserves_rng(noise_level):
    import torch
    from pc_model import PCNet
    # A CPU parameter with a stale CUDA hint requires no CUDA allocation.
    model = SimpleNamespace(noise_level=noise_level, device=torch.device('cuda'))
    parameter = torch.ones(5)
    level = max(noise_level.values()) if isinstance(noise_level, dict) else noise_level
    torch.manual_seed(19)
    expected = 1 + torch.randn_like(parameter) * level
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    PCNet._apply_noise(model, parameter)
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize('zero_gradient', [True, False])
def test_gradient_probe_reports_zero_and_nonzero_incoming_energy(monkeypatch, zero_gradient):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'diagnostic_scripts'))
    from probe_srrl_curve_gradients import GradientProbe, PiecewiseLinearActivation
    curve = Path(__file__).resolve().parents[1] / 'hardware_data/mc_45_corners/1022_RELU_Voltage/tt_25_1.csv'
    activation = PiecewiseLinearActivation(str(curve), v_dd=.6, corner='MC18',
        normalize_positive_endpoint=False, fuse_measured_activation=False)
    activation.set_coordinate_pullback_scale(.1)
    model = torch.nn.Module()
    model.final_activation = activation
    probe = GradientProbe(model, {'0906': activation, '1022': activation})
    x = torch.tensor([.1, .2], requires_grad=True)
    (activation(x) * (0 if zero_gradient else 1)).sum().backward()
    row = probe.finish()['final_activation']
    normalized = ['rms_gain_0906', 'rms_gain_1022',
        'negative_input_gradient_energy_fraction', 'upper_input_gradient_energy_fraction',
        'gradient_energy_on_near_zero_slope_0906', 'gradient_energy_on_near_zero_slope_1022']
    if zero_gradient:
        assert row['incoming_gradient_energy'] == 0
        assert row['transmitted_energy_0906'] == row['transmitted_energy_1022'] == 0
        assert all(row[key] is None for key in normalized)
    else:
        assert row['incoming_gradient_energy'] == 2
        assert all(row[key] is not None for key in normalized)
        assert row['rms_gain_0906'] == row['rms_gain_1022']
    assert row['near_zero_slope_fraction_0906'] == row['near_zero_slope_fraction_1022']
    json.dumps(row, allow_nan=False)
    assert not activation._forward_hooks


@pytest.mark.parametrize('constant', [True, False])
def test_feature_geometry_handles_collapsed_and_noncollapsed_features(monkeypatch, constant):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'diagnostic_scripts'))
    from probe_srrl_feature_geometry import geometry
    x = torch.ones(4, 3) if constant else torch.tensor([[1., 0.], [-1., 0.]])
    row = geometry(x)
    assert row['dimensions'] == x.shape[1]
    if constant:
        assert row['variance'] == row['participation_rank'] == row['entropy_rank'] == 0
        assert row['leading_variance_fraction'] is None
    else:
        assert row['variance'] == pytest.approx(1.)
        assert row['participation_rank'] == row['entropy_rank'] == row['leading_variance_fraction'] == 1
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize('timing,rk2', [(False,False), (True,False), (True,True)])
def test_selective_parent_runs_only_requested_reference_worker(tmp_path, monkeypatch, timing, rk2):
    from diagnostic_scripts import benchmark_toggle_input_correction as diagnostic
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(diagnostic.subprocess, 'run', run)
    def unexpected(*args):
        pytest.fail('Selective mode must not build an ordinary-sweep report')
    monkeypatch.setattr(diagnostic, 's2_report', unexpected)
    argv = ['benchmark', '--s2-dtc', '--batch', str(tmp_path/'batch.pt'),
        '--output', str(tmp_path/'out'), '--scene', 'nonlinear',
        '--representation', 'dense', '--reset-z-per-cycle',
        '--selective-reference', str(tmp_path/'reference')]
    if timing:
        argv.append('--selective-timing')
    if rk2:
        argv.append('--selective-timing-rk2')
    monkeypatch.setattr(sys, 'argv', argv)
    diagnostic.s2_main()
    assert len(commands) == 1
    command = commands[0]
    for key, expected in (('--s2-worker', 'numerical'), ('--dynamics', 'pulse'),
                          ('--scene', 'nonlinear'), ('--representation', 'dense'),
                          ('--selective-reference', str(tmp_path/'reference'))):
        assert command[command.index(key)+1] == expected
    assert '--reset-z-per-cycle' in command
    assert ('--selective-timing' in command) is timing
    assert ('--selective-timing-rk2' in command) is rk2


@pytest.mark.parametrize('options', [
    ['--selective-timing'], ['--selective-timing-rk2'],
    ['--selective-reference', 'ref', '--reset-study'],
    ['--selective-reference', 'ref', '--s2-worker', 'benchmark']])
def test_selective_options_reject_incompatible_modes_before_launch(tmp_path, monkeypatch, options):
    from diagnostic_scripts import benchmark_toggle_input_correction as diagnostic
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--s2-dtc',
        '--batch', str(tmp_path/'batch.pt'), '--output', str(tmp_path/'out'), *options])
    with pytest.raises(SystemExit):
        diagnostic.s2_main()
    assert not (tmp_path/'out').exists()


def test_slice_reference_identity_covers_transient_curves_and_hardware(tmp_path):
    import torch
    from torch import nn
    from diagnostic_scripts.benchmark_toggle_input_correction import (
        digest, s2_slice_reference_identity, s2_validate_slice_reference)
    model = nn.Module()
    block = nn.Linear(2,2)
    block.R, block.toggle_y_time = 67e3, 1e-8
    block._nonlinear_R_training_curve_indices = {'FFconv': 3, 'FBconv': 4}
    block._nonlinear_R_training_pkg = dict(R=67e3, R_left=torch.ones(2,2))
    model.PcConvs, model.linear = nn.ModuleList([block]), nn.Linear(2,2)
    curve = tmp_path/'curves.csv'
    curve.write_text('native curve content')
    settings = dict(R=67e3, cycle_ns=10)
    def config():
        return dict(checkpoint_sha256='ckpt', batch_hash='batch',
            model_hash=digest(model.state_dict()), microbatch=16, settings=settings.copy(),
            hardware_identity=s2_slice_reference_identity(model, settings, {'coupler': curve}))
    previous = config()
    s2_validate_slice_reference(previous, config())
    block._nonlinear_R_training_curve_indices['FBconv'] = 5
    assert previous['model_hash'] == config()['model_hash']
    with pytest.raises(ValueError, match='hardware_identity'):
        s2_validate_slice_reference(previous, config())
    block._nonlinear_R_training_curve_indices['FBconv'] = 4
    for change, undo in (
        (lambda: setattr(block, 'R', 68e3), lambda: setattr(block, 'R', 67e3)),
        (lambda: block._nonlinear_R_training_pkg['R_left'].mul_(2),
         lambda: block._nonlinear_R_training_pkg['R_left'].div_(2)),
        (lambda: curve.write_text('changed native curves'),
         lambda: curve.write_text('native curve content'))):
        change()
        with pytest.raises(ValueError, match='hardware_identity'):
            s2_validate_slice_reference(previous, config())
        undo()
    missing = dict(previous)
    del missing['hardware_identity']
    with pytest.raises(ValueError, match='lacks hardware_identity'):
        s2_validate_slice_reference(missing, config())
    settings['cycle_ns'] = 20
    with pytest.raises(ValueError, match='settings'):
        s2_validate_slice_reference(previous, config())


@pytest.mark.parametrize('saved,current', [(None, False), (False, False),
                                         (True, True), (None, True), (True, False)])
def test_s2_cached_reference_requires_matching_reset(saved, current):
    from diagnostic_scripts.benchmark_toggle_input_correction import s2_validate_reference
    row = dict(initial_hash='weights', batch_hash='images', assignments='curves')
    if saved is not None:
        row['reset_z_per_cycle'] = saved
    if bool(saved) != current:
        with pytest.raises(ValueError, match='reset_z_per_cycle'):
            s2_validate_reference(row, 'weights', 'images', 'curves', current)
    else:
        s2_validate_reference(row, 'weights', 'images', 'curves', current)
    with pytest.raises(ValueError, match='weights/batch/curve'):
        s2_validate_reference(row, 'different', 'images', 'curves', current)


@pytest.mark.parametrize('selection', ['latest', 'full_param_last'])
def test_merge_training_suite_forwards_source_selection(tmp_path, monkeypatch, selection):
    from diagnostic_scripts import benchmark_toggle_input_correction as diagnostic
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index('--output') + 1])
        output.mkdir()
        result = dict(initial_model_hash='model', initial_auxiliary_hash='aux',
                      batch_hash='batch', median_seconds=1.)
        (output/'result.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(diagnostic.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['benchmark_toggle_input_correction.py',
        '--s2-pulse-ft-cache', '--merge-training-suite', '--source-root', str(tmp_path),
        '--ft-args', str(tmp_path/'argv.json'), '--batch-cache', str(tmp_path/'batch.pt'),
        '--output', str(tmp_path/'suite'), '--source-selection', selection])
    diagnostic.s2_pulse_ft_cache_diagnostic()
    assert len(commands) == 6
    assert all(command[command.index('--source-selection') + 1] == selection
               for command in commands)


@pytest.mark.parametrize('divisor', [1, 2])
def test_collapse_diagnostic_records_actual_accumulation_backward(tmp_path, monkeypatch, divisor):
    import torch
    from torch import nn
    from diagnostic_scripts import replay_c10_cascode_pt_collapse as diagnostic
    cls = diagnostic.TrainerCiFarTimmStyleFeatureKD
    owner = cls.__new__(cls)
    owner.model = nn.Linear(2, 2)
    owner._feature_kd_loss = nn.Linear(2, 2)
    params = list(owner.model.parameters()) + list(owner._feature_kd_loss.parameters())
    owner.optimizer = torch.optim.SGD(params, lr=.01)
    owner.grad_accum_steps, owner.max_norm = divisor, None
    recorder = diagnostic.Recorder(tmp_path)
    recorder.owner, recorder.active = owner, True
    recorder.epoch, recorder.batch = 1, 1
    # Restore every process-global hook installed by the real diagnostic.
    for obj, name in ((cls, 'train_one_epoch'), (cls, '_unpack_train_batch'),
                      (torch.Tensor, 'backward'), (diagnostic.trainer_base, 'save_latest'),
                      (diagnostic.training_recovery, 'save_latest')):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    diagnostic.install(recorder, 30)
    def training_step(self):
        outputs = self.model(torch.ones(1, 2))
        student_features = [outputs]
        ce_loss = outputs.square().mean()
        recorder.stat = self._feature_kd_loss(student_features[-1]).square().mean()
        recorder.pred = outputs.mean().square()
        loss = ce_loss + recorder.stat + recorder.pred
        expected = torch.autograd.grad(loss/divisor, params, retain_graph=True)
        self._backward_step_accum(loss, params, False, divisor)
        return expected, float(loss.detach())
    try:
        expected, objective = training_step(owner)
        assert recorder.pending is not None
        assert recorder.pending['objective'] == pytest.approx(objective)
        assert set(recorder.pending['term_gradient_norms']) == {'ce', 'stat', 'pred'}
        for param, grad in zip(params, expected):
            torch.testing.assert_close(param.grad, grad)
    finally:
        recorder.rows.close()


def toggle_results():
    names = ('logits', 'loss_logits', 'input_gradients', 'quantized_input_gradients',
             'model_gradients', 'auxiliary_gradients')
    metric = dict(max_abs=0., relative_l2=0., strict_elementwise=True, finite=True,
                  keys_match=True, none_status_match=True)
    row = dict(accepted=True, prediction_changes=0, assignments_match=True,
               end_rng_match=True, metrics={key: dict(metric) for key in names})
    parity = dict(rows=[copy.deepcopy(row) for _ in range(3)], cudnn_allow_tf32=False,
        matmul_allow_tf32=False, cudnn_deterministic=True, checkpoint_selected_layers=16)
    benchmark = dict(status='ok', median_seconds=1., peak_allocated_bytes=1024,
        peak_reserved_bytes=2048, source_sha256='same', initial_weights_hash='same',
        initial_auxiliary_hash='same', batch_hash='same')
    return dict(strict=copy.deepcopy(parity), production=copy.deepcopy(parity),
        benchmark_off=dict(benchmark, fusion='off'), benchmark_on=dict(benchmark, fusion='on'))


@pytest.mark.parametrize('failure', [None, 'strict', 'production'])
def test_toggle_report_and_exit_follow_recorded_validation(tmp_path, monkeypatch, failure):
    from diagnostic_scripts import benchmark_toggle_input_correction as diagnostic
    results = toggle_results()
    if failure:
        results[failure]['rows'][2]['accepted'] = False
        results[failure]['rows'][2]['metrics']['model_gradients'].update(
            max_abs=1., relative_l2=1., strict_elementwise=False, finite=False)
    (tmp_path/'summary.json').write_text(json.dumps(results))
    monkeypatch.setattr(sys, 'argv', ['benchmark_toggle_input_correction.py',
        '--report-only', '--output', str(tmp_path)])
    if failure:
        with pytest.raises(SystemExit, match=failure):
            diagnostic.main()
    else:
        diagnostic.main()
    report = (tmp_path/'summary.md').read_text()
    assert f"Deterministic FP32 {'fails' if failure == 'strict' else 'passes'}." in report
    assert f"Production-precision parity {'fails' if failure == 'production' else 'passes'}" in report
    if failure:
        assert 'All compared tensors finite=False' in report
    # A failed ON comparison is not evidence of OFF/OFF variability.
    assert 'Production OFF/OFF also fails' not in report


def test_toggle_coordinator_rejects_failed_records_even_when_workers_exit_zero(tmp_path, monkeypatch):
    from diagnostic_scripts import benchmark_toggle_input_correction as diagnostic
    results = toggle_results()
    results['strict']['rows'][2]['accepted'] = False
    monkeypatch.setattr(diagnostic, 'capture', lambda output: {})
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **kw: '')
    def run(command, **kw):
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        kind = command[command.index('--worker')+1]
        if kind == 'parity':
            record = results[command[command.index('--precision')+1]]
        else:
            record = results['benchmark_'+command[command.index('--fused')+1]]
        (output/'result.json').write_text(json.dumps(record))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(diagnostic.subprocess, 'run', run)
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['benchmark_toggle_input_correction.py', '--output', str(output)])
    with pytest.raises(SystemExit, match='strict'):
        diagnostic.main()
    assert (output/'summary.json').is_file() and (output/'summary.md').is_file()
    assert 'Deterministic FP32 fails.' in (output/'summary.md').read_text()


@pytest.mark.parametrize('measurements', [1, 3, 5, 7])
@pytest.mark.parametrize('legacy', [False, True])
def test_checkpoint_sweep_uses_saved_measurement_count(tmp_path, monkeypatch, measurements, legacy):
    from diagnostic_scripts import benchmark_tc_empirical as diagnostic
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    monkeypatch.setenv('BENCHMARK_MEASURED_STEPS', '99')
    checkpoint = tmp_path/'saved_ckpt_runs/experiment/student/student_last_ckpt.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    output = tmp_path/'reports'
    output.mkdir()
    record = dict(status='ok', steps=[{} for _ in range(measurements)],
                  checkpoint_selected_layers=22, checkpoint_total_layers=22)
    if not legacy:
        record.update(warmups=2, measurements=measurements)
    (output/'state1_checkpoint22.json').write_text(json.dumps(record))
    steps = [dict(peak_allocated_bytes=9999 if i == 0 else 100,
                  peak_reserved_bytes=19999 if i == 0 else 200)
             for i in range(2+measurements)]
    log = '\n'.join('BENCHMARK_STEP '+str(i)+' '+repr(row) for i, row in enumerate(steps))
    (output/'state1_checkpoint22.log').write_text(log)
    diagnostic.checkpoint_sweep(SimpleNamespace(resume=True, checkpoint_layers=[22], batch_size=128),
                               output, {}, ['experiment'])
    saved = json.loads((output/'checkpoint_summary.json').read_text())[0]
    assert saved['peak_allocated_including_warmup_bytes'] == 9999
    assert saved['peak_reserved_including_warmup_bytes'] == 19999


def test_checkpoint_sweep_still_rejects_truncated_step_log(tmp_path, monkeypatch):
    from diagnostic_scripts import benchmark_tc_empirical as diagnostic
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    checkpoint = tmp_path/'saved_ckpt_runs/experiment/student/student_last_ckpt.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    output = tmp_path/'reports'
    output.mkdir()
    (output/'state1_checkpoint22.json').write_text(json.dumps(
        dict(status='ok', steps=[{}, {}, {}], checkpoint_selected_layers=22, checkpoint_total_layers=22)))
    (output/'state1_checkpoint22.log').write_text(
        "BENCHMARK_STEP 0 {'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200}\n")
    with pytest.raises(ValueError, match='expected 5 logged steps, found 1'):
        diagnostic.checkpoint_sweep(SimpleNamespace(resume=True, checkpoint_layers=[22], batch_size=128),
                                   output, {}, ['experiment'])
    assert not (output/'checkpoint_summary.json').exists()


def test_download_audit_retains_empty_partial_and_complete_corners(tmp_path, monkeypatch):
    from diagnostic_scripts import audit_1022_download as diagnostic
    monkeypatch.setattr(diagnostic, '__file__', str(tmp_path/'diagnostic_scripts/audit_1022_download.py'))
    monkeypatch.setattr(sys, 'argv', ['audit_1022_download.py'])
    (tmp_path/'results').mkdir()
    logs = tmp_path/'logs/slurm_jobs'
    logs.mkdir(parents=True)
    for job in [*range(2196406, 2196410), *range(2196950, 2196958)]:
        (logs/f'slurm_{job}.out').write_text(
            'Running SS_V1_T1 (2 trials)\nRuntimeError: OOM\n'
            'Running TT_V1_T1 (2 trials)\ntrial 0: 70.0%\n'
            'Running FF_V1_T1 (2 trials)\ntrial 0: 80.0%\ntrial 1: 82.0%\n')
    diagnostic.main()
    output = tmp_path/'results/1022_download_audit'
    summary = json.loads((output/'log_summary.json').read_text())
    empty, partial, complete = summary['2196406']['evaluations']
    assert empty['mean_percent'] is None and empty['completed_trials'] == 0
    assert empty['status'] == partial['status'] == 'incomplete'
    assert partial['mean_percent'] == 70. and partial['completed_trials'] == 1
    assert complete['status'] == 'complete' and complete['mean_percent'] == 81.
    assert 'RuntimeError: OOM' in (output/'2196406_evidence.txt').read_text()
