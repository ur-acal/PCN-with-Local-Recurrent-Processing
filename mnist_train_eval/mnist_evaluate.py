"""Fresh unrolled hardware realization per trial; shared MNIST test preprocessing."""
import gc
import json
import logging
import statistics

import torch
from tqdm import tqdm

from mnist_train_eval.mnist_config import parse_args, seed_all, device_for, model_name
from mnist_train_eval.mnist_data import mnist_loader
from mnist_train_eval.mnist_train import (
    build_cnn, check_checkpoint, pcn_wrapper_options, configure_pcn_pooling)


def cpu_unroll_convolution(input_shape, kernel, stride=1, padding=0):
    """Reuse the exact trimmed builder without a GPU scalar sync for each edge."""
    from physical_feedforward_tc import TCPhysicalBasicBlock
    matrix, first, second = TCPhysicalBasicBlock.unroll_convolution(
        input_shape, kernel.detach().cpu(), stride=stride, padding=padding)
    return matrix.to(kernel.device), first, second


@torch.no_grad()
def build_pcn_inference(args, device, seed):
    from inference_utils import load_and_prepare_model
    from pc_conv import PCConvReLU6Noisy
    from ode_pc import ODEXInitFFFB, S2NoisyIYAsXZAs0
    saved = {}
    model = load_and_prepare_model(
        str(args.checkpoint), device, pc_conv_layer=PCConvReLU6Noisy,
        data_parallel=False, noise_to_bn=False, noise_to_linear=False,
        fuse_bn=False, conv_only=True, noise_level=0., weight=None,
        ode_params=dict(ode_block=ODEXInitFFFB if args.tc_state == 1 else S2NoisyIYAsXZAs0,
            method='dopri5', t_end=args.t_end, tol=args.tol,
            n_steps=5, offset_eps=0., sde_noise_type='add'),
        ode_wrapper_params=pcn_wrapper_options(args, inference=True, seed=seed),
        wrappers=saved)
    configure_pcn_pooling(model, saved['wrappers'], args, seed)
    return model, saved['wrappers']


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv, evaluation=True)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    logging.basicConfig(level=logging.WARNING)
    device = device_for(args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    check_checkpoint(args, checkpoint, 'ft')
    if args.family == 'pcn' and checkpoint.get('checkpoint_weight_format') != 'flattened_quantized':
        raise ValueError('PCN evaluation uses the ordinary baked last checkpoint with QATTester, not full_param.')
    loader = mnist_loader(args.data_dir, train=False, batch_size=args.test_batch_size,
        num_workers=args.num_workers, seed=args.seed, download=args.download,
        limit_samples=args.limit_test_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (model_name(args) + '_tc_trials.json')
    records = []
    for trial in range(args.n_trials):
        seed = args.seed + trial
        seed_all(seed)
        print(f'Trial {trial + 1}/{args.n_trials}: preparing unrolled hardware (seed={seed})', flush=True)
        if args.family == 'cnn':
            from feedforward_validation import FeedForwardCNNValidator
            model, wrappers = build_cnn(args, device, inference=True, checkpoint=checkpoint, seed=seed)
            model.eval()
            for wrapper in wrappers:
                wrapper.block.unroll_convolution = cpu_unroll_convolution
            # Do not retain the last physical block into the next trial.
            del wrapper
            validator = FeedForwardCNNValidator(model, str(args.expanded_weight_dir / model_name(args)),
                device, loader, str(args.output_dir))
        else:
            from validation import Validator
            from tc_cli import reset_after_probe
            model, wrappers = build_pcn_inference(args, device, seed)
            model.eval()
            validator = Validator(model, str(args.expanded_weight_dir / model_name(args)),
                device, loader, str(args.output_dir), wrapper=wrappers)
            reset_after_probe(model)
        # Validators install new physical modules after the initial eval() call.
        # Put those replacements in evaluation mode as well; the fused TC
        # per-edge inference path deliberately rejects training-mode modules.
        model.eval()
        # Coupler assignments survive the dry run. Reset transient sampled
        # spin/pooling/activation values without replacing physical couplers.
        if args.family == 'cnn':
            for module in model.modules():
                for name in ('reset_spin_variation', 'reset_measured_pooling'):
                    reset = getattr(module, name, None)
                    if callable(reset):
                        reset()
            del module, reset
        print(f'Trial {trial + 1}/{args.n_trials}: evaluating {len(loader.dataset)} samples', flush=True)
        correct, total = 0, 0
        pbar = tqdm(enumerate(loader), total=len(loader), disable=False)
        for _, (inputs, labels) in pbar:
            outputs = model(inputs.to(device))
            if not torch.isfinite(outputs).all():
                raise RuntimeError(f'Nonfinite logits in trial {trial + 1}')
            correct += (outputs.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
            pbar.set_postfix(acc=f'{100.0 * correct / total:.2f}%')
        record = dict(trial=trial + 1, hardware_seed=seed, samples=total, accuracy=correct / total)
        records.append(record)
        path.write_text(json.dumps(dict(config=vars(args), trials=records,
            mean_accuracy=statistics.mean(r['accuracy'] for r in records)), indent=2, default=str) + '\n')
        print(f'Trial {trial + 1}/{args.n_trials}: {correct}/{total} = {correct / total:.4%}', flush=True)
        del validator, model, wrappers
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'Results: {path}', flush=True)
    return records


if __name__ == '__main__':
    main()
