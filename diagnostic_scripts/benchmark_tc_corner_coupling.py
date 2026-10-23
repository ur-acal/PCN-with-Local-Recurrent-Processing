"""Small end-to-end TC training check; not a full-dataset accuracy benchmark.

OFF is compared against affected runtime methods loaded literally from git HEAD.
Timing excludes profiler; kernel counts use a separate profiled training batch.
"""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from test_tc_corner_coupling import make_model


@contextmanager
def prechange_methods():
    import tc_nonidealities, ode_pc, physical_feedforward_tc, measured_pooling
    selections = (
        (tc_nonidealities, 'TCResistanceCurves', ('sample', 'sample_levels', 'sample_shared')),
        (ode_pc, 'ODEBlockPC', ('_tc_curves_for_solve',)),
        (physical_feedforward_tc, 'TCPhysicalBasicBlock', ('_current',)),
        (measured_pooling, 'MeasuredAvgPool2d', ('_sample_gaussian_curves',)),
    )
    saved = []
    try:
        for module, cls_name, names in selections:
            source = subprocess.check_output(
                ['git', 'show', f'HEAD:{Path(module.__file__).name}'], cwd=ROOT, text=True)
            cls_ast = next(n for n in ast.parse(source).body
                           if isinstance(n, ast.ClassDef) and n.name == cls_name)
            cls = getattr(module, cls_name)
            for name in names:
                node = next(n for n in cls_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
                namespace = dict(vars(module))
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<git-HEAD>', 'exec'), namespace)
                saved.append((cls, name, getattr(cls, name)))
                setattr(cls, name, namespace[name])
        yield
    finally:
        for cls, name, method in saved:
            setattr(cls, name, method)


def batch(model, x):
    model.zero_grad(set_to_none=True)
    y = model(x)
    y.square().mean().backward()
    return y


def snapshot(kind):
    model = make_model(kind, coupled=False, checkpoint=True, device='cuda')
    x = torch.full((8, 3 if kind == 'cnn' else 2,
                    8 if kind == 'cnn' else 4, 8 if kind == 'cnn' else 4), .015, device='cuda')
    y = batch(model, x)
    return y.detach().cpu(), [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    results = []
    for kind in ('state1', 'state2', 'cnn'):
        current = snapshot(kind)
        with prechange_methods():
            old = snapshot(kind)
        torch.testing.assert_close(current[0], old[0], rtol=0, atol=0)
        assert len(current[1]) == len(old[1])
        for a, b in zip(current[1], old[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Check GPU checkpoint replay against ordinary autograd, with matched RNG.
        replay = []
        for checkpoint in (False, True):
            model = make_model(kind, checkpoint=checkpoint, device='cuda')
            x = torch.full((8, 3 if kind == 'cnn' else 2,
                8 if kind == 'cnn' else 4, 8 if kind == 'cnn' else 4), .015, device='cuda')
            y = batch(model, x)
            replay.append((y.detach().cpu(), [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None]))
        torch.testing.assert_close(replay[0][0], replay[1][0], rtol=0, atol=0)
        for a, b in zip(replay[0][1], replay[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        models = {enabled: make_model(kind, coupled=enabled, checkpoint=True, device='cuda')
                  for enabled in (False, True)}
        x = torch.full((8, 3 if kind == 'cnn' else 2,
            8 if kind == 'cnn' else 4, 8 if kind == 'cnn' else 4), .015, device='cuda')
        for model in models.values():
            for _ in range(5):
                batch(model, x)
        torch.cuda.synchronize()
        elapsed = {False: [], True: []}
        for i in range(30):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                start = time.perf_counter()
                batch(models[enabled], x)
                torch.cuda.synchronize()
                elapsed[enabled].append(time.perf_counter() - start)
        for enabled, model in models.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                batch(model, x)
                torch.cuda.synchronize()
            launches = sum(e.count for e in profile.key_averages() if 'cudaLaunchKernel' in e.key)
            row = dict(model=kind, coupled=enabled, batch_size=8,
                median_ms=statistics.median(elapsed[enabled])*1000, cuda_kernel_launches=launches,
                off_vs_prechange_logits_max_diff=0., off_vs_prechange_gradients_max_diff=0.,
                checkpoint_logits_max_diff=0., checkpoint_gradients_max_diff=0.)
            results.append(row)
            print(json.dumps(row), flush=True)
    output = ROOT / 'results/tc_corner_coupling_validation'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'benchmark.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
