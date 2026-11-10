"""Read an actual corner COMMAND and audit a shortened expanded forward.

Production checkpoints/caches are read-only: cache writes are forbidden, result
outputs go under --output-dir, and no production modules are edited.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corner-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    cli = parser.parse_args()
    os.chdir(ROOT)
    out = cli.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with cli.corner_log.open() as stream:
        command = shlex.split(stream.readline().removeprefix('COMMAND: '))
    argv = command[command.index('ode_inference.py') + 1:]

    def set_arg(key, value):
        flag = '--' + key
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
        else:
            argv.extend([flag, str(value)])

    def get_arg(key):
        return argv[argv.index('--' + key) + 1]

    for key, value in dict(noisy_trials=1, test_bs=2, max_eval_batches=1,
                           hw_val_path=out / 'hw_validation',
                           ablation_single_case='true',
                           ablation_case_name='cascode_runtime_audit').items():
        set_arg(key, value)
    (out / 'argv.json').write_text(json.dumps(argv, indent=2))
    import torch
    import torch.nn.functional as F
    import validation
    import ode_pc
    import ode_inference
    from utils import load_mc_res_curve_bank
    from measured_pooling import MeasuredAvgPool2d
    from measured_activation import MEASURED_ACTIVATION_TYPES
    from final_linear import DigitalLinear

    report = dict(source_log=str(cli.corner_log), curve_path=get_arg('nonlinear_R_table'),
                  mvm={}, pooling={}, activation={}, effects={}, cache_writes=0)
    armed = False
    expected = load_mc_res_curve_bank(report['curve_path'])

    def rms(value):
        return float(value.detach().float().square().mean().sqrt().cpu())

    def forbid_write(*args, **kwargs):
        report['cache_writes'] += 1
        raise RuntimeError('Audit refuses to create or overwrite production caches')

    validation._save_expanded_weight_cache = forbid_write
    original_edge = validation.MVMConv._forward_pulse_per_edge

    def edge(self, x, weights, nominal_R):
        result = original_edge(self, x, weights, nominal_R)
        if armed:
            item = report['mvm'][self._audit_name]
            item['calls'] = item.get('calls', 0) + 1
            if 'nonlinear_minus_linear_rms' not in item:
                ideal = torch.sparse.mm(weights, x)
                item['nonlinear_minus_linear_rms'] = rms(result - ideal)
                item['nominal_R'] = float(nominal_R)
        return result

    validation.MVMConv._forward_pulse_per_edge = edge

    def wrap_effect(cls, name, stage_index, extract):
        original = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if armed:
                stage = args[stage_index]
                key = self._audit_name + ':' + name + ':' + stage
                item = report['effects'].setdefault(key, {'calls': 0})
                item['calls'] += 1
                if 'rms' not in item:
                    item['rms'] = rms(extract(self, args, result))
            return result
        setattr(cls, name, wrapped)

    wrap_effect(ode_pc.ToggleAveragedPhysicalFFFB, '_apply_spin_variation', 0,
                lambda self, args, result: result - args[1])
    wrap_effect(ode_pc.ToggleAveragedPhysicalFFFB, '_brownian_increment', 2,
                lambda self, args, result: result)
    wrap_effect(ode_pc.TogglePulseFFFB, '_coupler_brownian_increment', 2,
                lambda self, args, result: result)
    wrap_effect(ode_pc.TogglePulseFFFB, '_sample_dtc_window', 1,
                lambda self, args, result: result['end_offset'])

    original_head = DigitalLinear.forward

    def head(self, inputs):
        result = original_head(self, inputs)
        if armed:
            cfg = self.head_config
            saved = cfg['final_adc_noise_lsb']
            try:
                cfg['final_adc_noise_lsb'] = 0
                quiet = original_head(self, inputs)
            finally:
                cfg['final_adc_noise_lsb'] = saved
            report['digital_head'] = dict(config=dict(cfg), physical=self.physical,
                                          adc_effect_rms=rms(result - quiet),
                                          bias_present=self.bias is not None)
        return result

    DigitalLinear.forward = head
    original_validator = ode_inference.Validator

    class AuditValidator(original_validator):
        def __init__(self, *args, **kwargs):
            nonlocal armed
            super().__init__(*args, **kwargs)
            report['cache_fingerprint'] = self.expanded_weight_cache_fingerprint
            report['layers'] = len(self.model.PcConvs)
            for index, layer in enumerate(self.model.PcConvs):
                layer._audit_name = str(index)
                for side in ('FFconv', 'FBconv'):
                    module = getattr(layer, side)
                    assert isinstance(module, validation.MVMConv), (index, side)
                    name = f'{index}.{side}'
                    module._audit_name = name
                    assert module._toggle_pulse_edges
                    assert module.nonlinear_R_curve_sharing == 'per_coupler'
                    assert module.nonlinear_R_curve_sampling == 'empirical_with_replacement'
                    assignment = module.nonlinear_R_curve_assignment
                    assert assignment.numel() == module.mat.values().numel()
                    for key, tensor in expected.items():
                        actual = getattr(module, 'nonlinear_R_curve_bank_' + key)
                        assert torch.equal(actual.cpu(), tensor), (name, key)
                    report['mvm'][name] = dict(
                        edges=assignment.numel(), curve_min=int(assignment.min()),
                        curve_max=int(assignment.max()), bank_matches=True)
            for name, module in self.model.named_modules():
                if isinstance(module, MeasuredAvgPool2d):
                    assert module.enable_nonideality
                    assert module.nominal_R == 67000
                    for key, tensor in expected.items():
                        assert torch.equal(getattr(module, key).cpu(), tensor), (name, key)
                    report['pooling'][name] = dict(bank_matches=True, nominal_R=module.nominal_R)

                    def pool_hook(mod, inputs, result, name=name):
                        if armed:
                            item = report['pooling'][name]
                            item['calls'] = item.get('calls', 0) + 1
                            kernel, stride, _, _ = mod._pool_geometry(inputs[0])
                            ideal = F.avg_pool2d(inputs[0], kernel, stride)
                            item['nonideal_minus_ideal_rms'] = rms(result - ideal)
                    module.register_forward_hook(pool_hook)
                if isinstance(module, MEASURED_ACTIVATION_TYPES):
                    assert '1022_RELU_Voltage' in module.curve_path
                    assert module.curve_sharing == 'per_spin'
                    report['activation'][name] = dict(path=module.curve_path,
                                                      sharing=module.curve_sharing)

                    def activation_hook(mod, inputs, result, name=name):
                        if armed:
                            item = report['activation'][name]
                            item['calls'] = item.get('calls', 0) + 1
                            item['assigned_spins'] = mod._sampled_curve_indices.numel()
                    module.register_forward_hook(activation_hook)

            def output_hook(mod, inputs, result):
                if armed:
                    report['output_finite'] = bool(torch.isfinite(result).all())
                    report['predictions'] = result.argmax(1).tolist()
            self.model.register_forward_hook(output_hook)
            armed = True

    ode_inference.Validator = AuditValidator
    sys.argv = ['ode_inference.py'] + argv
    try:
        ode_inference.run_ode_inference()
        assert len(report['mvm']) == 32
        assert all(item.get('calls', 0) > 0 for item in report['mvm'].values())
        assert all(item.get('calls', 0) > 0 for item in report['pooling'].values())
        assert all(item.get('calls', 0) > 0 for item in report['activation'].values())
        assert report['digital_head']['adc_effect_rms'] > 0
        assert report['output_finite']
        report['completed'] = True
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
