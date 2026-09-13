"""Validate requested wrapper settings and exact frozen-input repeatability."""
import gc
import logging

import torch
import study_switch_full_network_r10k as current


@torch.no_grad()
def main():
    s = current.study
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    path = current.SOURCE_INPUTS/'data/batch_0.pt'
    data = torch.load(path,weights_only=True)
    x = data['input'].cuda()
    s.seed()
    net, wraps = s.build(s.ODEXInitFFFB,1)
    weight_hash = s.weight_digest(net)
    assert len(wraps)==16
    for w in wraps:
        assert (w.R,w.R_max,w.enob)==(10000,150000,None)
        probe = torch.tensor([-.2,.0123456,.2],device='cuda')
        assert w._quantize_output(probe) is probe
    s.seed()
    first, traces = s.capture_forward(net,wraps,x)
    s.seed()
    second, repeated = s.capture_forward(net,wraps,x)
    assert torch.equal(first,second), 'Reference is not repeatable'
    assert all(torch.equal(a[k],b[k]) for a,b in zip(traces,repeated)
               for k in ('input','pre_quant','output','rail_fraction'))
    reference_accuracy = float((first.cpu().argmax(1)==data['labels']).float().mean())
    del net,wraps,second,repeated
    gc.collect()
    torch.cuda.empty_cache()
    s.seed()
    net,wraps = s.build(s.Jacobi,5)
    assert s.weight_digest(net)==weight_hash
    for b in net.PcConvs:
        b.option_aca.update(rtol=1e-7,atol=1e-7)
    before = current.recovery.fallback_count
    s.seed()
    logits,errors = s.capture_forward(net,wraps,x,traces)
    assert torch.isfinite(logits).all()
    assert s.global_relative(errors[0]['input'])==0., 'Input to first layer changed'
    target = s.OUT/'validation'
    target.mkdir(parents=True,exist_ok=True)
    s.atomic_json(target/'smoke.json',{
        'driver_sha256':s.digest(current.__file__), 'model_builder_sha256':s.digest(s.ROOT/'scripts/switch_study_r10k_model.py'),
        'input_batch0_sha256':s.digest(path), 'R':10000.,'R_max':150000.,'enob':None,
        'all_16_output_quantizers_identity':True,
        'reference_logits_and_all_layer_traces_bitwise_repeatable':True,
        'converted_weights_equal_between_reference_and_jacobi':True,
        'jacobi_first_layer_input_error':0.,'jacobi_finite':True,
        'jacobi_norm_fallbacks':current.recovery.fallback_count-before,
        'reference_batch0_accuracy_only':reference_accuracy,
        'scope':'128-image smoke control, not the seven-batch study result'})
    print('R10k / Rmax150k / ENOB=None smoke validation passed',flush=True)


if __name__=='__main__':
    main()
