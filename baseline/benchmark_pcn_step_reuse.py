"""Compare trained FP PCN logits and one-batch costs with adaptive-step reuse off/on."""
import argparse
import copy
import hashlib
import json
import logging
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from inference_utils import get_test_data, load_and_prepare_model
from ode_pc import ODEXInitFFFB
from pc_conv import PCConvNoisy


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def max_difference(a, b):
    return float((a.detach().cpu() - b.detach().cpu()).abs().max())


def benchmark(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to report CPU timing as a GPU benchmark')
    logging.disable(logging.WARNING)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)
    inputs, labels = next(iter(get_test_data(args.batch_size, 'rgb', args.dataset, data_root=str(args.data_dir))))
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        model = load_and_prepare_model(
            str(args.checkpoint), device=device, pc_conv_layer=PCConvNoisy,
            noise_level=0., fuse_bn=False, noise_to_bn=False, noise_to_linear=False,
            noise_to_conv_bias=False, dropout=.25,
            ode_params=dict(ode_block=ODEXInitFFFB, method='dopri5', t_end=1.75, tol=1e-4, n_steps=15))
    # Historical init_args omit dropout, and the loader filters block kwargs.
    model.dropout = .25
    if args.mismatch_level:
        seed_all(args.seed)
        model.noise_level = args.mismatch_level
        for block in model.PcConvs:
            block.noise_level = args.mismatch_level
            block.mismatch_type = args.mismatch_type
        with torch.no_grad():
            model.add_noise(noise_to_bn=False, noise_to_linear=True, noise_to_conv_bias=False,
                            mismatch_type=args.mismatch_type, additive_scale_mode='max_abs')
    snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.001)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())

    def reset(reuse, training):
        model.load_state_dict(snapshot)
        optimizer.load_state_dict(initial_optimizer)
        optimizer.zero_grad(set_to_none=True)
        model.train(training)
        for module in model.modules():
            options = getattr(module, 'option_aca', None)
            if options is not None:
                options['reuse_accepted_step_training'] = reuse
                options['reuse_accepted_step_inference'] = reuse
        seed_all(args.seed)
        synchronize(device)

    def batch(training):
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            if training:
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
        return logits.detach()

    result = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  model_class=type(model).__name__, dataset=args.dataset, batch_size=len(labels),
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  pytorch=torch.__version__, solver='dopri5', t_end=1.75, solver_tolerance=1e-4,
                  final_dropout=model.dropout,
                  logit_tolerance=args.tolerance, seed=args.seed, input_source='first CIFAR test batch, normalized',
                  mismatch_type=args.mismatch_type, mismatch_level=args.mismatch_level,
                  timing_scope='resident-data single batch; training includes forward, cross-entropy, backward, SGD update; reset/load excluded',
                  warmup=args.warmup, repeats=args.repeats, cpu_threads=args.threads, comparisons={})
    for training in (True, False):
        mode = 'training' if training else 'inference'
        correctness = []
        for reuse in (False, True):
            reset(reuse, training)
            calls = [0]

            def count_rhs(_module, _inputs):
                calls[0] += 1

            hooks = [block.FFconv.register_forward_pre_hook(count_rhs) for block in model.PcConvs]
            logits = batch(training)
            synchronize(device)
            for hook in hooks:
                hook.remove()
            correctness.append(dict(logits=logits.cpu(), rhs_calls=calls[0]))
        diff = max_difference(correctness[0]['logits'], correctness[1]['logits'])
        if not np.isfinite(diff) or diff > args.tolerance:
            raise AssertionError(f'{mode}: max logit difference {diff} exceeds {args.tolerance}')
        times = {False: [], True: []}
        memory = {False: [], True: []}
        # Alternating order reduces drift from temperature and clock changes.
        for repeat in range(args.warmup + args.repeats):
            for reuse in ((False, True) if repeat % 2 == 0 else (True, False)):
                reset(reuse, training)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                output = batch(training)
                synchronize(device)
                elapsed = time.perf_counter() - start
                if repeat >= args.warmup:
                    times[reuse].append(elapsed)
                    memory[reuse].append(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
                del output
        rows = {}
        for reuse in (False, True):
            rows['on' if reuse else 'off'] = dict(
                median_seconds=statistics.median(times[reuse]), seconds=times[reuse],
                peak_allocated_bytes=max(memory[reuse]) if device.type == 'cuda' else None,
                rhs_calls=correctness[int(reuse)]['rhs_calls'])
        result['comparisons'][mode] = dict(max_absolute_logit_difference=diff, passed=True,
                                          speedup=rows['off']['median_seconds'] / rows['on']['median_seconds'], **rows)
        print(json.dumps({mode: result['comparisons'][mode]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--mismatch-type', choices=['mul', 'add'], default='mul')
    parser.add_argument('--mismatch-level', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.batch_size < 1 or args.tolerance <= 0 or args.mismatch_level < 0:
        parser.error('Invalid repeat, warmup, batch size, tolerance or mismatch level')
    benchmark(args)


if __name__ == '__main__':
    main()
