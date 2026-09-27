"""MNIST pretrain/FT entry; only assembly and wiring of existing computation."""
import json
import logging

import timm
import torch

import baseline.cifar_resnet  # Registers the MNIST assemblies beside CIFAR CNNs.
from mnist_train_eval.mnist_config import parse_args, model_name, seed_all, device_for


def noise_options(args, seed):
    return dict(enable_spin_variation=args.enable_spin_variation,
        sigma_spin=args.sigma_spin, spin_variation_mean=args.spin_variation_mean,
        spin_variation_seed=seed,
        enable_summing_current_noise=args.enable_summing_current_noise,
        summing_current_p=args.summing_current_p, summing_noise_seed=seed,
        enable_coupler_noise=args.enable_coupler_noise,
        coupler_noise_p=args.coupler_noise_p, coupler_noise_seed=seed,
        enable_slow_summing_current=False, enable_slow_coupler_noise=False)


def pcn_wrapper_options(args, *, inference=False, seed=None):
    from ode_pc import QATWrapper1State, QATWrapper2State, QATTester1State, QATTester2State
    seed = args.seed if seed is None else seed
    wrappers = (QATTester1State, QATTester2State) if inference else (QATWrapper1State, QATWrapper2State)
    return dict(ode_wrapper=wrappers[args.tc_state - 1],
        R=args.R, R_max=args.R_max, C=args.C, k=1e3, v_dd=args.v_dd,
        one_over_q=args.one_over_q, w_bits=5, weight_quant_factor_bits=None, enob=None,
        tie_cap=False, thermal_noise=False, offset_eps=0.,
        tc_nonidealities=True, tc_covariance_table=args.tc_covariance_table,
        tc_conv_method=args.tc_conv_method, tc_curve_sampling=args.tc_curve_sampling,
        tc_noise_reference_R=args.tc_noise_reference_R,
        tc_asd_reference_p=args.tc_asd_reference_p, tc_fb_asd_path=args.tc_fb_asd_path,
        nonlinear_R=args.enable_nonlinear_R, nonlinear_R_table=args.nonlinear_R_table,
        nonlinear_R_train_mode='none', nonlinear_R_curve_seed=seed,
        nonlinear_R_curve_sharing='per_coupler' if inference else 'shared',
        enable_measured_activation=args.enable_measured_activation,
        activation_curve_path=args.activation_curve_path, activation_corner=args.activation_corner,
        activation_curve_sharing='per_spin' if inference else 'per_model', activation_curve_seed=seed,
        activation_interpolation='piecewise_linear', activation_normalize_positive_endpoint=False,
        **noise_options(args, seed))


def configure_pcn_pooling(model, wrappers, args, seed):
    if args.enable_measured_pooling:
        from tc_cli import pooling_options
        from measured_pooling import configure_measured_pooling
        configure_measured_pooling(model, wrappers, enable_nonideality=True,
            **pooling_options(args, wrappers), seed=seed, training_curve_mode='exact_curve')


def check_checkpoint(args, checkpoint, stage):
    saved = checkpoint.get('mnist_config')
    if saved is None:
        raise ValueError('Expected a checkpoint produced by mnist_train')
    for key in ('family', 'variant', 'tc_state', 'dropout', 't_end'):
        if key == 'tc_state' and args.family == 'cnn':
            continue
        if saved[key] != getattr(args, key):
            raise ValueError(f'Checkpoint {key}={saved[key]} differs from requested {getattr(args, key)}')
    if saved['stage'] != stage:
        raise ValueError(f'Expected a {stage} checkpoint, got {saved["stage"]}')


def build_cnn(args, device, *, inference=False, checkpoint=None, seed=None):
    from physical_feedforward import convert_wide_resnet_to_physical, iter_physical_wrappers
    from measured_activation import feedforward_measured_activation_factory, configure_feedforward_measured_activation
    seed = args.seed if seed is None else seed
    physical = args.stage != 'pretrain'
    # Match the existing CIFAR entry: install QAT on CPU before moving the
    # complete model (including the newly registered quantizer buffers).
    model = timm.create_model(model_name(args), pretrained=False,
        in_chans=1, num_classes=10, final_dropout_rate=args.dropout)
    model = convert_wide_resnet_to_physical(model, physical_level=2,
        physical=physical, qat=False, R=args.R, C=args.C, v_dd=args.v_dd,
        one_over_q=args.one_over_q, w_bits=5, weight_quant_factor_bits=None,
        noise_level=0., enob=None, toggle_timing_mode='derived',
        tc_options=dict(one_shot_conv=args.one_shot_conv, tc_method='dopri5',
            tc_tol=args.tol, tc_noise_reference_R=args.tc_noise_reference_R,
            tc_covariance_table=args.tc_covariance_table, tc_curve_sampling=args.tc_curve_sampling,
            reuse_accepted_step_training=args.reuse_accepted_step_training,
            checkpoint_ode_rhs_training=args.checkpoint_ode_rhs_training),
        **(noise_options(args, seed) if physical else {}))
    wrappers = list(iter_physical_wrappers(model))
    if inference:
        if checkpoint.get('checkpoint_weight_format') == 'full_param':
            for wrapper in wrappers:
                wrapper.enable_qat_()
        model.load_state_dict(checkpoint['net'], strict=True)
        if checkpoint.get('checkpoint_weight_format') != 'full_param':
            for wrapper in wrappers:
                wrapper.block._uses_quantized_weight_scale = True
                for name, module in (('conv1', wrapper.block.conv1), ('conv2', wrapper.block.conv2)):
                    if module is not None:
                        wrapper.block.clean_params[name].copy_(module.weight.detach())
    elif checkpoint is not None:
        model.load_state_dict(checkpoint['net'], strict=True)
    if not physical:
        return model.to(device), wrappers
    if not inference:
        for wrapper in wrappers:
            wrapper.enable_qat_()
    if args.enable_measured_activation:
        configure_feedforward_measured_activation(model,
            feedforward_measured_activation_factory(args.activation_curve_path, args.v_dd,
                corner=args.activation_corner, curve_sharing='per_spin' if inference else 'per_model',
                curve_seed=seed, normalize_positive_endpoint=False, interpolation='piecewise_linear'))
    if args.enable_nonlinear_R:
        method = wrappers[0].configure_nonlinear_R_inference if inference else wrappers[0].configure_nonlinear_R_training
        package = method(args.nonlinear_R_table, curve_seed=seed)
        for wrapper in wrappers[1:]:
            installer = wrapper.install_nonlinear_R_inference_package if inference else wrapper.install_nonlinear_R_training_package
            installer(package)
    if args.enable_measured_pooling:
        from tc_feedforward_cli import configure_pooling
        from copy import copy
        options = copy(args)
        options.nonlinear_R_curve_seed = seed
        configure_pooling(model, options)
    return model.to(device), wrappers


def build_pcn(args, device, checkpoint=None):
    from pc_model import PCNetNoBatchNorm
    from pc_conv import PCConvReLU6
    from ode_pc import make_ode_block, wrap_ode_block, ODEXInitFFFB, S2NoisyIYAsXZAs0
    deep = args.variant == 'deep'
    model = PCNetNoBatchNorm(
        inp_channels=[1, 32, 32] if deep else [1, 32],
        out_channels=[32, 32, 64] if deep else [32, 64],
        max_pool=[False, True, False] if deep else [True, False],
        num_classes=10, pc_conv_layer=PCConvReLU6, avg_pooling=True,
        dropout=args.dropout, first_bn=False, stride=1, kernel_size=3,
        bias=False, tie_weights=False, tie_bp=False, bypass=False).to(device)
    block_cls = ODEXInitFFFB if args.tc_state == 1 else S2NoisyIYAsXZAs0
    model = make_ode_block(model, ode_block=block_cls, noise_level=0., method='dopri5',
        t_end=args.t_end, tol=args.tol, n_steps=5, offset_eps=0., sde_noise_type='add',
        reuse_accepted_step_training=args.reuse_accepted_step_training,
        checkpoint_ode_rhs_training=args.checkpoint_ode_rhs_training)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['net'], strict=True)
    wrappers = []
    if args.stage == 'ft':
        model, wrappers = wrap_ode_block(model, **pcn_wrapper_options(args))
        configure_pcn_pooling(model, wrappers, args, args.seed)
    return model, wrappers


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    logging.basicConfig(level=logging.WARNING)
    device = device_for(args)
    seed_all(args.seed)
    checkpoint = None
    if args.stage == 'ft':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        check_checkpoint(args, checkpoint, 'pretrain')
    model, wrappers = (build_cnn if args.family == 'cnn' else build_pcn)(args, device, checkpoint=checkpoint)
    from mnist_train_eval.mnist_trainer import MNISTTrainer
    name = model_name(args) + '_' + args.stage
    print(f'Model: {name}; parameters={sum(p.numel() for p in model.parameters())}', flush=True)
    trainer = MNISTTrainer(model, args, name)
    trainer.train()


if __name__ == '__main__':
    main()
