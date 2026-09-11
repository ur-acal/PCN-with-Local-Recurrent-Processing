"""Paired PCN frozen/recalibrated BN evaluation using the WRN calibration protocol."""

import csv
import hashlib
import io
import json
import math
import pickle
import statistics
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from baseline.run_baseline import (
    BNRecalibrationConfig, all_parameter_tensors, combined_hash,
    build_bn_calibration_loader, recalibrate_batchnorm_statistics,
)
from baseline.run_wrn_bn_recalibration_experiment import evaluate_with_predictions
from inference_utils import get_test_data, load_and_prepare_model
from pc_model import PC_CONV_CLASS
from ode_pc import ODEBLOCK_CLASSES


def evaluate_pair(model, test_loader, calibration_loader, device, cfg, model_name, level, trial):
    before = combined_hash(all_parameter_tensors(model))
    frozen, _, _ = evaluate_with_predictions(model, test_loader, device)
    report = recalibrate_batchnorm_statistics(
        model, calibration_loader, device, cfg, model_name, level, trial,
    )
    recalibrated, _, _ = evaluate_with_predictions(model, test_loader, device)
    if combined_hash(all_parameter_tensors(model)) != before:
        raise AssertionError("Paired evaluation changed learned weights")
    return frozen, recalibrated, report


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


@torch.no_grad()
def run_paired_bn_evaluation(args):
    from ode_inference import get_t_end, seed_mismatch_trial

    if (args.task not in {'cifar10', 'cifar100'} or args.img_type != 'rgb'
            or args.test_expanded or args.thermal_noise or args.ode_wrapper is not None
            or args.test_only or args.hw_validate or args.weight_range_audit_only
            or args.max_eval_batches is not None or args.return_init != '0'
            or args.ff_gain != 1.0 or args.noise_to_conv_bias or args.conv_only
            or args.sweep_eps or args.t_end_sf != 1.0 or args.shuffle_test):
        raise ValueError("Paired BN mode requires full RGB CIFAR evaluation, unwrapped ODE, gain 1, and no conv-bias noise")
    if not args.output_pickle or args.seed is None or args.noisy_trials < 1:
        raise ValueError("Paired BN mode requires output_pickle, an explicit seed, and positive noisy_trials")
    if args.pc_conv != 'PCConvNoisy' or args.ode_block is None:
        raise ValueError("Paired BN mode requires PCConvNoisy and an explicit ODE block")
    if args.calibration_num_samples < 1 or args.calibration_batch_size < 1 or args.calibration_num_workers < 0:
        raise ValueError("Invalid calibration sample, batch or worker count")
    levels = [float(x) for x in args.noise_level_list.split(',')]
    if len(set(levels)) != len(levels) or any(not math.isfinite(x) or x < 0 for x in levels):
        raise ValueError("Noise levels must be unique, finite and nonnegative")
    output = Path(args.output_pickle).parent
    marker = output / 'paired_complete.json'
    # A failed rerun must not leave an old completion marker.
    marker.unlink(missing_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)
    checkpoint = Path(args.model_dir) / args.model_name / f'{args.model_name}_{args.ckpt}_ckpt.pth'
    data_dir = args.data_dir or '../data'
    test_loader = get_test_data(args.test_bs, img_type='rgb', task=args.task, data_root=data_dir)
    cfg = BNRecalibrationConfig(
        enabled=True, num_samples=str(args.calibration_num_samples),
        batch_size=args.calibration_batch_size, subset_seed=args.calibration_subset_seed,
        num_workers=args.calibration_num_workers, diagnostics_dir=str(output / 'bn_diagnostics'),
    )
    t_end = get_t_end(args)
    ode_params = dict(ode_block=ODEBLOCK_CLASSES[args.ode_block], t_end=t_end,
                      method=args.method, tol=args.tol, n_steps=args.n_steps,
                      ts_scale=args.ts_scale, mismatch_type=args.mismatch_type,
                      additive_scale_mode=args.additive_scale_mode)
    calibration_loader = None
    results = {'frozen': {}, 'recalibrated': {}}
    rows = []
    for level in levels:
        for mode in results:
            results[mode][level] = []
        for trial in range(1 if level == 0 else args.noisy_trials):
            # Reconstruct clean weights and BN buffers every trial. Apply mismatch only below.
            model = load_and_prepare_model(
                str(checkpoint), device=device, pc_conv_layer=PC_CONV_CLASS[args.pc_conv],
                noise_level=0.0, noise_to_bn=False, noise_to_linear=True,
                noise_to_conv_bias=False, fuse_bn=False, ode_params=ode_params,
            )
            if not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()):
                raise ValueError("Paired BN evaluation requires a checkpoint with BatchNorm")
            if calibration_loader is None:
                calibration_loader = build_bn_calibration_loader(
                    model, SimpleNamespace(dataset=args.task, data_dir=data_dir, pin_memory=True),
                    {'timm_input_size': (3, 32, 32)}, cfg,
                )
            clean_bn = {n: v.detach().clone() for n, v in model.state_dict().items()
                        if n.startswith(('BNs.', 'BNend.'))}
            seed = seed_mismatch_trial(args.seed, args.model_index, level, trial)
            model.noise_level = level
            for block in model.PcConvs:
                block.noise_level = level
            if level > 0:
                model.add_noise(noise_to_bn=args.noise_to_bn, noise_to_linear=True,
                                noise_to_conv_bias=False, mismatch_type=args.mismatch_type,
                                additive_scale_mode=args.additive_scale_mode)
            if not args.noise_to_bn:
                assert all(torch.equal(v, model.state_dict()[n]) for n, v in clean_bn.items())
            frozen, recal, report = evaluate_pair(
                model, test_loader, calibration_loader, device, cfg, args.model_name, level, trial,
            )
            results['frozen'][level].append(frozen)
            results['recalibrated'][level].append(recal)
            rows.append(dict(level=level, trial=trial, seed=seed, frozen_accuracy=frozen,
                             recalibrated_accuracy=recal, noise_to_bn=args.noise_to_bn,
                             **asdict(report)))
            write_atomic(output / 'trials.json', json.dumps(rows, indent=2).encode())
            print(f'level={level:g} trial={trial} frozen={frozen:.2f} recalibrated={recal:.2f}', flush=True)
            del model, clean_bn
    for mode, values in results.items():
        write_atomic(output / mode / 'result.pkl', pickle.dumps({t_end: {
            'noise_acc_spec': {'Johnson': values}, 't': (t_end, t_end, t_end, t_end),
        }}))
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['level', 'num_trials', 'frozen_mean_accuracy', 'frozen_std_accuracy',
                     'recalibrated_mean_accuracy', 'recalibrated_std_accuracy'])
    for level in levels:
        a, b = results['frozen'][level], results['recalibrated'][level]
        writer.writerow([level, len(a), statistics.mean(a), statistics.pstdev(a), statistics.mean(b), statistics.pstdev(b)])
    write_atomic(output / 'full_aggregate.csv', table.getvalue().encode())
    write_atomic(marker, json.dumps(dict(
        status='complete', model_name=args.model_name, checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        args=vars(args), calibration=asdict(cfg), t_end=t_end,
    ), indent=2).encode())
