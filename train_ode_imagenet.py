from typing import List
import os
import argparse
import logging

import torch
import torch.nn as nn

from ode_pc import (
    ODEBLOCK_CLASSES,
    make_ode_block,
    ODEWrapper_CLASSES,
    wrap_ode_block,
    QUANTIZER_CLASSES,
)
from switch import SWITCH_CLASSES

from pc_conv import PCConv, PartialTiedPCConv
from pc_model import PCNet, PCN_CLASSES, PC_CONV_CLASS
from inference_utils import load_and_prepare_model

from train_ode_cifar import (
    str2bool,
    get_model_name,
    build_teacher_model,
    evaluate_teacher,
)

from baseline.baseline_cifar_configs import CASE_DEFAULTS

from trainer_imagenet import (
    TrainerImageNetTimmStyle,
    TrainerImageNetTimmStyleSRRL,
    TrainerImageNetTimmStyleMGD,
)

ODEBLOCK_CLASSES.update(SWITCH_CLASSES)


def get_args():
    p = argparse.ArgumentParser(description="Train PCNet / ODE-PCNet on ImageNet with timm-style trainer")
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_ckpt")
    # ------------------------------------------------------------------
    # Dataset / trainer args
    # ------------------------------------------------------------------
    p.add_argument("--imagenet_root", type=str, required=True,
                   help="Path to ImageNet root containing train/ and val/ class subfolders.")
    p.add_argument("--save_path", type=str, default=model_save_path)
    p.add_argument("--skip_eval_epochs", type=int, default=0)
    p.add_argument("--img_type", type=str, default="rgb", choices=["rgb"])
    p.add_argument("--dataset", type=str, default="imagenet",
                   choices=["imagenet", "imagenet1k", "ilsvrc2012"])
    p.add_argument("--task", type=str, default="imagenet",
                   choices=["imagenet", "imagenet1k", "ilsvrc2012"])
    p.add_argument("--timm_trainer", type=str2bool, default=True,
                   help="Must be true for ImageNet. The non-timm TrainerCiFar path is CIFAR-only.")
    p.add_argument("--timm_sched", type=str, default="cosine", choices=["multistep", "cosine"])

    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--optim", type=str, choices=["SGD", "Adam"], default="SGD")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--lr_reduce_on", type=str, default="30,60,90")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--max_g_norm", type=float, default=None)
    p.add_argument("--warmup_epoch", type=int, default=0)
    p.add_argument("--cosine_t0", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=1)

    p.add_argument("--model_name", type=str, default=None,
                   help="Resume from a checkpoint. None means training from scratch.")
    p.add_argument("--ckpt", type=str, default="best")

    p.add_argument("--mem_frac", type=float, default=1.0)
    p.add_argument("--test_only", type=str2bool, default=False)

    # Kept only for compatibility with get_model_name() from train_ode_cifar.py.
    p.add_argument("--rggb_to_rgb", type=str2bool, default=False)

    # ------------------------------------------------------------------
    # PCNet / PCConv args
    # ------------------------------------------------------------------
    p.add_argument("--inp_channels", type=int, nargs="+",
                   default=[3, 64, 64, 128, 128, 256, 256, 512],
                   help="list of input-channel sizes")
    p.add_argument("--out_channels", type=int, nargs="+",
                   default=[64, 64, 128, 128, 256, 256, 512, 512],
                   help="list of output-channel sizes")
    p.add_argument("--stride", type=int, nargs="+", default=1)
    p.add_argument("--kernel_size", type=int, nargs="+", default=3)
    p.add_argument("--first_ksz", type=int, default=7)
    p.add_argument("--first_stride", type=int, default=2)
    p.add_argument("--first_pad", type=int, default=3)
    p.add_argument("--max_pool", type=int, nargs="+",
                   default=[True, False, True, False, True, False, False, False])
    p.add_argument("--avg_pooling", type=str2bool, default=True)
    p.add_argument("--separable", type=str, nargs="+", default=None)
    p.add_argument("--patch_dim", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=1000)

    # ------------------------------------------------------------------
    # ODE hyper-params
    # ------------------------------------------------------------------
    p.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()), default="ODEBlockPC")
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--n_steps", type=float, default=10)
    p.add_argument("--t_end", type=float, default=1.0)
    p.add_argument("--offset_eps", type=float, default=None)
    p.add_argument("--patch_node", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_stride", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_cycle", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--patch_pad", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--fold_scalar", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--n_iters", type=int, default=2)

    # ------------------------------------------------------------------
    # Quantization-aware training / wrapper args
    # ------------------------------------------------------------------
    p.add_argument("--ode_wrapper", type=str, choices=list(ODEWrapper_CLASSES.keys()) + [None], default=None)
    p.add_argument("--qat_cls", type=str, choices=list(QUANTIZER_CLASSES.keys()) + [None], default=None)
    p.add_argument("--R", type=float, default=1e5)
    p.add_argument("--R_max", type=lambda s: None if s.lower() in {"none", ""} else float(s), default=None)
    p.add_argument("--C", type=float, default=49e-15)
    p.add_argument("--v_dd", type=float, default=1.0)
    p.add_argument("--enob", type=lambda s: None if s.lower() in {"none", ""} else int(s), default=None)
    p.add_argument("--w_bits", type=int, default=8)
    p.add_argument("--tie_cap", type=str2bool, default=False)
    p.add_argument("--one_over_q", type=float, default=10)

    # ------------------------------------------------------------------
    # Noise-inject / SDE args
    # ------------------------------------------------------------------
    p.add_argument("--noise_level", default=None, type=float)
    p.add_argument("--noise_type", default="mul", type=str, choices=["mul", "add"])
    p.add_argument("--sde_noise_type", type=str, default="mul", choices=["mul", "add"])

    # ------------------------------------------------------------------
    # Distillation args
    # ------------------------------------------------------------------
    p.add_argument("--teacher_ckpt", type=str, default=None)
    p.add_argument("--teacher_arch", type=str, default=None,
                   help="efficientnet-b4, efficientnet_v2_s, etc.")
    p.add_argument("--teacher_arch_source", type=str, default="auto",
                   choices=["auto", "torchvision", "hankyul2"])
    p.add_argument("--teacher_input_size", type=int, default=224)
    p.add_argument("--teacher_center_crop", type=str2bool, default=True)

    p.add_argument(
        "--distill_method",
        type=str,
        default="none",
        choices=[
            "none",
            "kd",
            "srrl", "kd_srrl", "kd+srrl",
            "mgd", "kd_mgd", "kd+mgd",
            "reviewkd", "kd_reviewkd", "kd+reviewkd",
        ],
    )
    p.add_argument("--distill_alpha", type=float, default=0.0)
    p.add_argument("--distill_temperature", type=float, default=1.0)

    # Kept for compatibility with TrainerCiFar signature / feature-KD base.
    p.add_argument("--contrast_method", type=str, default="memory", choices=["memory", "moco"])
    p.add_argument("--neg_sample", type=str, default="index", choices=["index", "label"])
    p.add_argument("--orig_t_inp", type=str2bool, default=False)
    p.add_argument("--crd_feat_dim", type=int, default=128)
    p.add_argument("--crd_k", type=int, default=16384)
    p.add_argument("--crd_temperature", type=float, default=0.07)
    p.add_argument("--crd_momentum", type=float, default=0.5)
    p.add_argument("--crd_beta", type=float, default=0.8)

    # ------------------------------------------------------------------
    # PCConv hyper-params
    # ------------------------------------------------------------------
    p.add_argument("--padding", type=int, nargs="+", default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--tie_weights", type=str2bool, default=False)
    p.add_argument("--tie_bp", type=str2bool, default=False)
    p.add_argument("--bypass", type=str2bool, default=False)
    p.add_argument("--relu_bp", type=str2bool, default=False)
    p.add_argument("--use_pc", type=str2bool, default=True)
    p.add_argument("--pcn", type=str, choices=list(PCN_CLASSES.keys()) + [None], default=None)
    p.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys()) + [None], default=None)
    p.add_argument("--tie_method", type=str, choices=["kernel_random", "random", None], default=None)
    p.add_argument("--tie_frac", type=float, default=1.0)

    return p.parse_args()


def _get_feature_kd_trainer(args):
    method_lower = getattr(args, "distill_method", "none").lower()

    if "srrl" in method_lower:
        return TrainerImageNetTimmStyleSRRL

    if "mgd" in method_lower:
        return TrainerImageNetTimmStyleMGD

    if "reviewkd" in method_lower:
        raise ImportError(
            "TrainerImageNetTimmStyleReviewKD is not available. "
            "Add the ReviewKD ImageNet subclass in trainer_imagenet.py first."
        )

    return TrainerImageNetTimmStyle


def main():
    args = get_args()

    if not args.timm_trainer:
        raise ValueError("ImageNet path requires --timm_trainer true. TrainerCiFar is CIFAR-only.")

    args.dataset = "imagenet"
    args.task = "imagenet"
    args.img_type = "rgb"
    args.num_classes = 1000

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)

    loss_fn = nn.CrossEntropyLoss()

    model_args = {
        "inp_channels": args.inp_channels,
        "out_channels": args.out_channels,
        "max_pool": args.max_pool,
        "num_classes": args.num_classes,
        "kernel_size": args.kernel_size if not (
            isinstance(args.kernel_size, List) and len(args.kernel_size) == 1
        ) else args.kernel_size[0],
        "stride": args.stride if not (
            isinstance(args.stride, List) and len(args.stride) == 1
        ) else args.stride[0],
        "padding": args.padding if args.patch_dim is None else "same",
        "cls": 0,
        "bias": args.bias,
        "lr": 0.0,
        "tie_weights": args.tie_weights,
        "tie_bp": args.tie_bp,
        "relu_between": True,
        "bypass": args.bypass,
        "relu_bp": args.relu_bp,
        "use_pc": args.use_pc,
        "first_bn": False,
        "dropout": args.dropout,
        "patch_dim": args.patch_dim,
        "separable": args.separable,
        "avg_pooling": args.avg_pooling,
        "first_ksz": args.first_ksz,
        "first_stride": args.first_stride,
        "first_pad": args.first_pad,
    }

    if args.pc_conv and "Noisy" in args.pc_conv:
        default_noise = 0.0 if args.noise_level is None else args.noise_level
        model_args["noise_level"] = float(default_noise)

    # Select PCConv module.
    pc_conv_mod = PCConv
    if args.pc_conv is not None:
        pc_conv_mod = PC_CONV_CLASS.get(args.pc_conv, PCConv)
    elif args.tie_method is not None:
        pc_conv_mod = PartialTiedPCConv
        model_args.update({"tie_method": args.tie_method, "tie_frac": args.tie_frac})

    model_args.update({"pc_conv_layer": pc_conv_mod})
    logging.warning("----- Using PC Conv layer: %s -----", pc_conv_mod.__name__)

    # Select PCNet model.
    pcn_model = PCN_CLASSES.get(args.pcn, PCNet)
    logging.warning("----- Using PCN model: %s -----", pcn_model.__name__)

    # Build or load model.
    if args.model_name is None:
        model = pcn_model(**model_args)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = None
    else:
        ckpt_path = os.path.join(
            args.save_path,
            args.model_name,
            args.model_name + "_{}_ckpt.pth".format(args.ckpt),
        )
        noisy_params = {"noise_level": 0.0, "weight": None}
        model = load_and_prepare_model(
            model_path=ckpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_struct=pcn_model,
            pc_conv_layer=pc_conv_mod,
            data_parallel=False,
            noise_to_bn=False,
            noise_to_linear=False,
            fuse_bn=False,
            conv_only=False,
            ode_params=None,
            **noisy_params,
        )
        model.dropout = args.dropout

    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)

    logging.warning("input channels: %s", getattr(model, "ics", None))
    logging.warning("output channels: %s", getattr(model, "ocs", None))
    logging.warning("max pooling: %s", getattr(model, "max_pool", None))
    logging.warning("pooling layer: %s", getattr(model, "max_pool2d", None))
    logging.warning("dropout rate: %s", getattr(model, "dropout", None))
    logging.warning("Total number of parameters: %s", total_params / 1e6)
    logging.warning("Model name: %s", model_name)

    logging.info("----- Printing out model parameter names: -----")
    for name, param in model.named_parameters():
        logging.info("name: %s, shape: %s, param count: %s", name, param.shape, param.numel())

    # Convert PCConv blocks to ODE blocks.
    ode_kw = [
        "offset_eps",
        "sde_noise_type",
        "patch_node",
        "patch_stride",
        "patch_cycle",
        "patch_pad",
        "fold_scalar",
        "n_iters",
    ]
    ode_kwargs = {}
    for _name, _val in vars(args).items():
        if _name in ode_kw and _val is not None:
            ode_kwargs[_name] = _val

    ode_block = ODEBLOCK_CLASSES[args.ode_block]

    student_in_channels = None
    if hasattr(model, "ics") and getattr(model, "ics"):
        try:
            student_in_channels = int(model.ics[0])
        except (TypeError, ValueError, IndexError):
            student_in_channels = None

    model = make_ode_block(
        pc_net=model,
        ode_block=ode_block,
        noise_level=0.0,
        method=args.method,
        t_end=args.t_end,
        tol=args.tol,
        n_steps=args.n_steps,
        **ode_kwargs,
    )

    logging.warning("PcConv converted to ODEBlock: %s", ode_block.__name__)
    logging.warning("t_end: %s", args.t_end)
    logging.warning("method: %s", args.method)
    logging.warning("tol: %s", args.tol)

    if args.model_name is not None:
        logging.warning("Before training, loaded model is set to train mode.")
        model.train()

    # Wrap blocks for QAT / analog simulation.
    if args.ode_wrapper is not None:
        wrapper_params = {
            "ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper],
            "calib_path": None,
            "R": args.R,
            "R_max": args.R_max,
            "C": args.C,
            "v_dd": args.v_dd,
            "w_bits": args.w_bits,
            "enob": args.enob,
            "qat_cls": QUANTIZER_CLASSES[args.qat_cls],
            "tie_cap": args.tie_cap,
            "one_over_q": args.one_over_q,
        }
        model, _ = wrap_ode_block(model, **wrapper_params)
        logging.warning("ODEBlock in network wrapped, ode_wrapper_params=%s", wrapper_params)

    logging.warning("Training task: ImageNet-1K")
    logging.warning(
        "lr reduce on: %s, max grad norm: %s, cosine annealing T0: %s",
        args.lr_reduce_on,
        args.max_g_norm,
        args.cosine_t0,
    )

    # Teacher / distillation.
    method_lower = args.distill_method.lower()
    needs_teacher = (
        ("kd" in method_lower and args.distill_alpha > 0.0)
        or ("srrl" in method_lower)
        or ("mgd" in method_lower)
        or ("reviewkd" in method_lower)
    )

    teacher_model = None
    if needs_teacher:
        if not args.teacher_ckpt:
            raise ValueError(f"distill_method={args.distill_method} requires --teacher_ckpt.")
        teacher_model = build_teacher_model(
            args,
            student_in_channels=student_in_channels,
            orig_t_inp=args.orig_t_inp,
        )
        logging.warning("Training with %s.", args.distill_method)
    elif args.teacher_ckpt:
        logging.warning("teacher_ckpt provided but distillation disabled by configuration; ignoring teacher.")

    timm_trainer_cls = _get_feature_kd_trainer(args)

    cfg = CASE_DEFAULTS["imagenet1k_scratch"].copy()

    # Runtime overrides, same style as train_ode_cifar.py.
    cfg["lr"] = args.learning_rate
    cfg["num_epochs"] = args.num_epochs
    cfg["warmup_epoch"] = args.warmup_epoch
    cfg["weight_decay"] = args.weight_decay
    cfg["batch_size"] = args.batch_size
    cfg["test_batch_size"] = args.batch_size

    # Scheduler choice from args.
    cfg["timm_sched"] = args.timm_sched
    cfg["lr_reduce_on"] = args.lr_reduce_on

    if args.timm_sched == "cosine":
        cfg["skip_eval_epochs"] = min(max(cfg["skip_eval_epochs"], 0.5 * args.num_epochs), args.num_epochs * 0.8)

    trainer_kwargs = dict(
        # Trainer base args.
        model=model,
        model_name=model_name,
        save_path=args.save_path,
        batch_size=cfg["batch_size"],
        optim_type="sgd",  # ignored by TrainerImageNetTimmStyle._get_optimizer
        weight_decay=cfg["weight_decay"],
        loss_fn=loss_fn,
        learning_rate=cfg["lr"],
        num_epochs=cfg["num_epochs"] if not args.test_only else 2,
        warmup_epoch=cfg["warmup_epoch"],
        lr_reduce_on=cfg.get("lr_reduce_on", "30,60,90"),
        T0=args.cosine_t0,
        test_bs=cfg["test_batch_size"],
        max_norm=cfg.get("max_norm", None),
        aug=False,
        eval_every=args.eval_every if not args.test_only else 2,
        img_type="rgb",
        dataset_name="imagenet",

        # ImageNet-specific args.
        imagenet_root=args.imagenet_root,
        num_classes=args.num_classes,

        # Distillation args.
        distill_method=args.distill_method,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        teacher_model=teacher_model,
        orig_t_inp=args.orig_t_inp,
        teacher_input_size=args.teacher_input_size,
        teacher_center_crop=args.teacher_center_crop,

        # TrainerImageNetTimmStyle / timm-style args from config.
        timm_opt=cfg["timm_opt"],
        momentum=cfg["momentum"],
        timm_sched=cfg["timm_sched"],
        min_lr=cfg["min_lr"],
        warmup_lr=cfg["warmup_lr"],
        timm_aug=True,
        use_model_data_config=cfg.get("use_model_data_config", True),
        timm_input_size=cfg["timm_input_size"],
        timm_train_scale=cfg["timm_train_scale"],
        timm_train_ratio=cfg["timm_train_ratio"],
        hflip=cfg["hflip"],
        vflip=cfg.get("vflip", 0.0),
        color_jitter=cfg["color_jitter"],
        auto_augment=cfg["auto_augment"],
        re_prob=cfg["re_prob"],
        label_smoothing=cfg["label_smoothing"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],
        num_workers=cfg.get("num_workers", 8),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True),

        # PCNet is custom, not timm-created.
        is_timm_model=False,
        skip_eval_epochs=cfg.get("skip_eval_epochs", 0) if (not args.test_only and args.cosine_t0 is None) else 0,

        # AMP and gradient accumulation related args.
        amp_enabled=cfg.get("amp_enabled", False),
        amp_dtype=cfg.get("amp_dtype", "bf16"),
        grad_accum_steps=cfg.get("grad_accum_steps", 1),

        # Mismatch-aware training.
        noise_level=args.noise_level,
        noise_type=args.noise_type,

        # Kept for compatibility with feature-KD base.
        contrast_method=args.contrast_method,
        neg_sample=args.neg_sample,
        crd_feat_dim=args.crd_feat_dim,
        crd_k=args.crd_k,
        crd_temperature=args.crd_temperature,
        crd_momentum=args.crd_momentum,
        crd_beta=args.crd_beta,
    )

    trainer = timm_trainer_cls(**trainer_kwargs)

    if args.model_name is not None and hasattr(trainer, "load_feature_kd_from_ckpt"):
        logging.warning("Loading distillation method's auxiliary module for FT.")
        trainer.load_feature_kd_from_ckpt(ckpt_path=ckpt_path)

    if teacher_model is not None:
        evaluate_teacher(teacher_model, trainer)

    if args.test_only:
        logging.warning("Test only. Run 2 epochs.")

    trainer.train()


if __name__ == "__main__":
    main()