"""Validate norm overflow recovery and saved full-network controls on CUDA."""
import json
import logging

import torch
import study_switch_full_network_normsafe as recovery

s = recovery.study


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    normal = (torch.tensor([1., 2., 3.], device='cuda'),)
    assert torch.equal(recovery.LEGACY_NORM(normal), recovery.norm_with_overflow_fallback(normal))
    large = (torch.full((128, 24, 16, 16), 3e17, device='cuda'),)
    assert not torch.isfinite(recovery.LEGACY_NORM(large))
    assert torch.allclose(recovery.norm_with_overflow_fallback(large), large[0][0, 0, 0, 0])
    records = []
    data = torch.load(recovery.ORIGINAL_OUT / 'data/batch_0.pt', weights_only=True)
    for method, cls, tol, folder in (
        ('Full', s.ODEXInitFFFB, 1e-6, 'reference_1e-06'),
        ('Full', s.ODEXInitFFFB, 1e-7, 'reference_1e-07'),
        ('Full', s.ODEXInitFFFB, 1e-8, 'reference_1e-08'),
        ('Jacobi', s.Jacobi, 1e-6, 'cases/Jacobi_m1_n5_tol1e-06'),
        ('Jacobi', s.Jacobi, 1e-7, None),
    ):
        s.seed()
        net, wraps = s.build(cls, 5)
        for block in net.PcConvs:
            block.option_aca.update(rtol=tol, atol=tol)
        start_fallbacks = recovery.fallback_count
        torch.cuda.reset_peak_memory_stats()
        s.seed()
        logits, traces = s.capture_forward(net, wraps, data['input'].cuda())
        assert torch.isfinite(logits).all()
        row = {'method': method, 'tol': tol, 'batch': 0,
               'fallback_count': recovery.fallback_count - start_fallbacks,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        if folder:
            old = torch.load(recovery.ORIGINAL_OUT / folder / 'batch_0.pt', weights_only=True)
            row['logits_bitwise_equal'] = torch.equal(logits.cpu(), old['logits'])
            assert row['logits_bitwise_equal'], row
            if method == 'Full':
                row['all_layer_traces_bitwise_equal'] = all(
                    torch.equal(t[k], o[k]) for t, o in zip(traces, old['layers'])
                    for k in ('input', 'pre_quant', 'output', 'rail_fraction'))
                assert row['all_layer_traces_bitwise_equal'], row
        records.append(row)
        print(json.dumps(row), flush=True)
        del net, wraps, logits, traces
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    target = s.ROOT / 'results/switch_full_network_oom_diagnosis/validation.json'
    s.atomic_json(target, {'driver_sha256': s.digest(recovery.__file__),
                           'validation_sha256': s.digest(__file__), 'controls': records})


if __name__ == '__main__':
    main()
