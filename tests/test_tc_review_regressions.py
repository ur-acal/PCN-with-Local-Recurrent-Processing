"""Regression coverage for diagnostic failures found in pre-commit review."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


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
