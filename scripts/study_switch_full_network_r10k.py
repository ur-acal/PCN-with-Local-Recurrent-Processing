"""R10k/Rmax150k/ENOB=None matched 324-case study with compact reporting."""
import gc
import json
import shutil
import time

import torch

# Install the validated overflow-only norm correction without editing production
# numerical sources. Replace that revision's output, builder and preparation.
import study_switch_full_network_normsafe as recovery
import report_switch_r10k as reporting
from switch_study_r10k_model import build

study = recovery.study
study.OUT = reporting.OUT
study.build = build
SOURCE_INPUTS = recovery.ORIGINAL_OUT
BASE_PREPARE = recovery.original_prepare
BASE_RUN_CASE = study.run_case
BASE_CAPTURE = study.capture_forward
EXTRA_SOURCES = ('scripts/study_switch_full_network_r10k.py',
                 'scripts/switch_study_r10k_model.py', 'scripts/report_switch_r10k.py',
                 'scripts/study_switch_full_network_normsafe.py',
                 'TorchDiffEqPack/misc.py', 'TorchDiffEqPack/utils.py', 'pc_conv.py')


def verify_inputs(manifest):
    for index, expected in enumerate(manifest['batch_sha256']):
        assert study.digest(study.OUT/'data'/f'batch_{index}.pt') == expected, 'Frozen input changed'


def capture_forward(net, wrappers, x, reference=None):
    logits, traces = BASE_CAPTURE(net, wrappers, x, reference)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite output logits')
    for trace in traces:
        for value in trace.values():
            tensors = value.values() if isinstance(value, dict) else (value,)
            if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise FloatingPointError('Nonfinite layer trace or error')
    if reference is not None:
        assert not bool(traces[0]['input']['difference_sq'].any()), 'Case received different first-layer input'
    return logits, traces


def prepare(batches):
    assert batches == 7, 'This study requires exactly seven frozen batches'
    old_manifest = json.loads((SOURCE_INPUTS/'manifest.json').read_text())
    path = study.OUT/'manifest.json'
    if not path.exists():
        # Inputs only: references and switched cases must be recomputed with the
        # new physical settings. Never copy old reference or case outputs.
        shutil.copytree(SOURCE_INPUTS/'data', study.OUT/'data', dirs_exist_ok=True)
        manifest = dict(old_manifest)
        manifest.pop('converted_parameter_sha256', None)
        manifest.pop('numerical_control', None)
        manifest.update(R=10000., R_max=150000., enob=None,
                        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        model_builder='scripts/switch_study_r10k_model.py',
                        input_origin=str(SOURCE_INPUTS/'data'),
                        input_origin_manifest_sha256=study.digest(SOURCE_INPUTS/'manifest.json'),
                        reference_policy='recompute all references under R10k/Rmax150k/ENOB=None',
                        numerical_control='overflow-only float64 RMS reduction; float32 states; invalid initial-step guard',
                        mean_layer_error='arithmetic mean of 16 per-layer pooled output relative L2 errors',
                        config_count=324)
        manifest['source_sha256'] = {name: study.digest(study.ROOT/name)
                                     for name in set(old_manifest['source_sha256']) | set(EXTRA_SOURCES)}
        verify_inputs(manifest)
        study.atomic_json(path, manifest)
    manifest = json.loads(path.read_text())
    assert (manifest['R'], manifest['R_max'], manifest['enob']) == (10000.,150000.,None)
    assert manifest['batch_sha256'] == old_manifest['batch_sha256']
    verify_inputs(manifest)
    prior_weights = manifest.get('converted_parameter_sha256')
    manifest = BASE_PREPARE(batches)
    if prior_weights is not None:
        assert prior_weights == manifest['converted_parameter_sha256'], 'Converted parameters changed on resume'
    references = {str(p.relative_to(study.OUT)): study.digest(p)
                  for p in study.OUT.glob('reference_*/*.pt')}
    assert len(references) == 21
    if 'reference_sha256' in manifest:
        assert manifest['reference_sha256'] == references
    manifest['reference_sha256'] = references
    study.atomic_json(path, manifest)
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def run_case(config):
    manifest = json.loads((study.OUT/'manifest.json').read_text())
    verify_inputs(manifest)
    result = BASE_RUN_CASE(config)
    # Extra numerical evidence is separate from the base case artifact schema.
    study.atomic_json(study.OUT/'cases'/study.case_id(config)/'execution.json',
                      {'driver_sha256': study.digest(__file__),
                       'input_batch_sha256': manifest['batch_sha256'],
                       'R': 10000., 'R_max': 150000., 'enob': None,
                       'worker_cumulative_norm_fallbacks': recovery.fallback_count})
    return result


def report(status, failures):
    study.OUT.mkdir(parents=True, exist_ok=True)
    count = len(list((study.OUT/'cases').glob('*/summary.json')))
    study.atomic_json(study.OUT/'status.json',
                      {'status': status, 'completed': count, 'expected': 324,
                       'failed_cases': failures, 'updated_unix': time.time()})
    reporting.render()


study.prepare = prepare
study.run_case = run_case
study.report = report
study.capture_forward = capture_forward


if __name__ == '__main__':
    study.main()
