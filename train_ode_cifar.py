import torch
import os
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.quantization as quantization

from ode_pc import ODEBLOCK_CLASSES, make_ode_block
from pc_conv import PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from trainer import TrainerCiFar


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    p = argparse.ArgumentParser(description="Train PCNet on CIFAR with neural ode")
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_ckpt")
    # TrainerCiFar args
    p.add_argument("--save_path",     type=str,   default=model_save_path)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--optim",         type=str,   choices=["SGD", "Adam"], default="SGD",
                   help="optimizer")
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--lr_reduce_on", type=str, default="80,122,150,225,262")
    p.add_argument("--num_epochs",    type=int,   default=300)
    p.add_argument("--max_g_norm", type=float, default=None)
    p.add_argument("--warmup_epoch",  type=int,   default=0)
    # PCNet / PCConv args
    p.add_argument("--inp_channels",  type=int, nargs="+", default=[3,  64, 64, 128, 128, 256, 256, 512],
                   help="list of input-channel sizes, e.g. 3 16 32")
    p.add_argument("--out_channels",  type=int, nargs="+", default=[64, 64, 128, 128, 256, 256, 512, 512],
                   help="list of output-channel sizes")
    p.add_argument("--max_pool",      type=int, nargs="+",
                   default=[False, False, True, False, True, False, False, False])
    p.add_argument("--num_classes",   type=int, default=10)
    # ODE hyper-params
    p.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()),
                        default="ODEBlockPC")
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--tol", type=float, default=1e-3, help="ODE solver tolerance")
    p.add_argument("--n_steps", type=float, default=10, help="ODE solver number of steps")
    p.add_argument("--t_end", type=float, default=1.0, help="Stop time of the solver")
    # PCConv hyper-params
    p.add_argument("--kernel_size",   type=int, default=3)
    p.add_argument("--stride",        type=int, default=1)
    p.add_argument("--padding",       type=int, default=1)
    p.add_argument("--dropout",       type=float, default=0.0)
    p.add_argument("--bias",          action="store_true")
    p.add_argument("--tie_weights",   type=str2bool, default=False)
    p.add_argument("--tie_bp",        type=str2bool, default=False)
    p.add_argument("--bypass",        type=str2bool, default=False)
    p.add_argument("--relu_bp", type=str2bool, default=False)
    p.add_argument("--use_pc", type=str2bool, default=True)
    p.add_argument("--pcn", type=str, choices=list(PCN_CLASSES.keys()) + [None],
                   default=None)
    p.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys()) + [None],
                        default=None)
    p.add_argument("--tie_method", type=str, choices=["kernel_random", "random", None], default=None,
                   help="method used to select positions in the kernel to tie between FF/FB")
    p.add_argument("--tie_frac", type=float, default=1.0,
                   help="fraction to tie weights of FF/FB")
    p.add_argument("--test_only", type=str2bool, default=False)
    # Quantization Aware Training arguments
    p.add_argument("--qat", type=str2bool, default=False,
                   help="Enable Quantization Aware Training")
    p.add_argument("--qat_backend", type=str, choices=["fbgemm", "qnnpack"], default="fbgemm",
                   help="Quantization backend (fbgemm for x86, qnnpack for ARM)")
    p.add_argument("--qat_start_epoch", type=int, default=0,
                   help="Epoch to start quantization aware training")
    # Quick testing arguments
    p.add_argument("--subset_fraction", type=float, default=1.0,
                   help="Fraction of dataset to use (0.0-1.0). Use smaller values for quick testing")
    return p.parse_args()

def _constr_model_name(args, rep=1):
    name_dict = {str(True): "with", str(False): "no"}
    model_name = 'NODE_PPCN'
    if args.pcn is not None and args.pcn != "PCNet":
        model_name = args.pcn
    if args.pc_conv is not None:
        model_name += "_" + args.pc_conv
    model_name += '_{}'.format(args.ode_block) \
                  + '_{}Solver'.format(args.method) + '_{}TEnd'.format(str(args.t_end)) + '_{}Tol_'.format(str(args.tol)) \
                  + str(args.weight_decay) + 'WD_' \
                  + name_dict[str(args.tie_bp)] + 'BPtied_' \
                  + name_dict[str(args.bypass)] + 'BP_' \
                  + str(args.batch_size) + 'BS_' + str(args.learning_rate) + 'LR_' \
                  + str(args.dropout) + 'Dropout_' + str(len(args.inp_channels)) + "Layers"

    if args.tie_method is not None:
        model_name += "_" + args.tie_method + "TieMethod_" + str(args.tie_frac) + "TieFrac"
    model_name = model_name + "_" + str(rep) + 'REP'
    return model_name

def get_model_name(args):
    rep = 1
    model_name = _constr_model_name(args, rep)
    model_dir = os.path.join(args.save_path, model_name)
    while os.path.exists(model_dir):
        rep += 1
        model_name = _constr_model_name(args, rep)
        model_dir = os.path.join(args.save_path, model_name)
    return model_name

def main():
    args = get_args()
    if args.test_only:
        logging.basicConfig(level=logging.INFO)

    # map optimizer name -> class
    # optim_type = getattr(optim, args.optim)
    loss_fn = nn.CrossEntropyLoss()

    model_args = {
        "inp_channels": args.inp_channels,
        "out_channels": args.out_channels,
        "max_pool": args.max_pool,
        "num_classes": args.num_classes,
        "kernel_size": args.kernel_size,
        "stride": args.stride,
        "padding": args.padding,
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
    }

    # Select PCConv Module to use
    pc_conv_mod = PCConv
    if args.pc_conv is not None:
        pc_conv_mod = PC_CONV_CLASS.get(args.pc_conv, PCConv)
    elif args.tie_method is not None:
        pc_conv_mod = PartialTiedPCConv
        model_args.update({"tie_method": args.tie_method, "tie_frac": args.tie_frac})
    model_args.update({"pc_conv_layer": pc_conv_mod})
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv_mod.__name__))

    # Select PCNet model to use
    pcn_model = PCN_CLASSES.get(args.pcn, PCNet)
    logging.warning("----- Using PCN model: {} -----".format(pcn_model.__name__))

    # build model
    model = pcn_model(**model_args)

    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)
    logging.warning("input channels: {}".format(model.ics))
    logging.warning("output channels: {}".format(model.ocs))
    logging.warning("max pooling: {}".format(model.max_pool))
    logging.warning("dropout rate: {}".format(model.dropout))
    logging.warning("Total number of parameters: {}".format(total_params / 1e6))
    logging.warning("Model name: {}".format(model_name))
    logging.info("----- Printing out model parameter names: -----")
    for name, param in model.named_parameters():
        logging.info("name: {}, shape: {}, param count: {}".format(name, param.shape, param.numel()))

    # convert block to Neural ode
    ode_block = ODEBLOCK_CLASSES[args.ode_block]
    model = make_ode_block(
        pc_net=model, ode_block=ode_block, noise_level=0.0, method=args.method, t_end=args.t_end,
        tol=args.tol, n_steps=args.n_steps)
    logging.warning("PcConv converted to ODEBlock: {}".format(ode_block.__name__))
    logging.warning("t_end: {}".format(args.t_end))
    logging.warning("method: {}".format(args.method))
    logging.warning("tol: {}".format(args.tol))

    # Quantization Aware Training setup
    if args.qat:
        logging.warning("Enabling Quantization Aware Training (QAT)")
        logging.warning("QAT backend: {}".format(args.qat_backend))
        logging.warning("QAT start epoch: {}".format(args.qat_start_epoch))

        # Set quantization backend
        torch.backends.quantized.engine = args.qat_backend

        # Update model name to include QAT info
        model_name = model_name + "_QAT_INT8_{}".format(args.qat_backend)

    # Get trainer
    logging.warning("lr reduce on: {}, max grad norm: {}".format(args.lr_reduce_on, args.max_g_norm))
    trainer = TrainerCiFar(
        model         = model,
        model_name    = model_name,
        save_path     = args.save_path,
        batch_size    = args.batch_size,
        optim_type    = args.optim,
        weight_decay  = args.weight_decay,
        loss_fn       = loss_fn,
        learning_rate = args.learning_rate,
        num_epochs    = args.num_epochs,
        warmup_epoch  = args.warmup_epoch,
        lr_reduce_on  = args.lr_reduce_on,
        test_bs       = args.batch_size,
        max_norm      = args.max_g_norm,
        qat           = args.qat,
        qat_backend   = args.qat_backend,
        qat_start_epoch = args.qat_start_epoch,
        subset_fraction = args.subset_fraction,
    )

    if args.test_only:
        _ = model(next(iter(trainer.train_dataloader))[0].to(trainer.device))
        logging.info("Test model forward only. Exit without training the model.")
        logging.info("Result: {}".format(_))
        exit(0)

    trainer.train()

if __name__ == "__main__":
    main()
