"""Isolated matched-study revision: recover nonfinite float32 RMS reductions.

No production source is patched on disk. Finite legacy norm results are returned
unchanged. Frozen inputs/references belong to the original study; switched cases
are all rerun here, never merged with the original revision's case results.
"""
import json
import shutil
import sys

import torch

import study_switch_full_network as study
import TorchDiffEqPack.odesolver.adaptive_grid_solver as adaptive

ORIGINAL_OUT = study.OUT
study.OUT = study.ROOT / 'results/switch_full_network_matched_normsafe'
LEGACY_NORM = adaptive.norm
fallback_count = 0


def norm_with_overflow_fallback(values):
    global fallback_count
    result = LEGACY_NORM(values)
    if bool(torch.isfinite(result)):
        return result
    tensors = (values,) if torch.is_tensor(values) else values
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError('Nonfinite solver norm input')
    precise = LEGACY_NORM(tuple(value.double() for value in tensors))
    result = precise.to(result.dtype)
    if not bool(torch.isfinite(result)):
        raise FloatingPointError('Solver RMS exceeds original scalar dtype')
    fallback_count += 1
    return result


adaptive.norm = norm_with_overflow_fallback
original_initial_step = adaptive.AdaptiveGridSolver.select_initial_step_scipy


def guarded_initial_step(self, *args, **kwargs):
    step = original_initial_step(self, *args, **kwargs)
    if not 0 < float(step) < float('inf'):
        raise FloatingPointError(f'Invalid initial solver step: {step}')
    return step


adaptive.AdaptiveGridSolver.select_initial_step_scipy = guarded_initial_step
original_prepare = study.prepare


def prepare(batches):
    # Invoked under the original supervisor's output-directory lock.
    manifest_path = study.OUT / 'manifest.json'
    control = {'kind': 'nonfinite RMS fallback only; float64 reduction, float32 states',
               'driver_sha256': study.digest(__file__),
               'original_manifest_sha256': study.digest(ORIGINAL_OUT / 'manifest.json'),
               'reference_origin': str(ORIGINAL_OUT),
               'case_policy': 'rerun all 324 switched cases; no legacy case reuse'}
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['numerical_control'] == control
    else:
        manifest = json.loads((ORIGINAL_OUT / 'manifest.json').read_text())
        manifest['numerical_control'] = control
        # Copy immutable evidence, so later changes to original results cannot
        # alter this revision. Original source/checkpoint/input checks still run.
        for folder in ('data', 'reference_1e-06', 'reference_1e-07', 'reference_1e-08'):
            shutil.copytree(ORIGINAL_OUT / folder, study.OUT / folder, dirs_exist_ok=True)
        study.atomic_json(manifest_path, manifest)
    return original_prepare(batches)


study.prepare = prepare


if __name__ == '__main__':
    study.main()
