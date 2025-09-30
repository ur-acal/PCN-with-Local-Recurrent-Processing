from typing import List
import torch
import os
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ode_pc import ODEBLOCK_CLASSES, make_ode_block, ODEWrapper_CLASSES, wrap_ode_block, QUANTIZER_CLASSES
from pc_conv import PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from trainer import TrainerCiFar
from inference_utils import load_and_prepare_model, test_once


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    p = argparse.ArgumentParser(description="Train PCNet on CIFAR with neural ode")
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_ckpt")
    # TrainerCiFar args
    p.add_argument("--save_path",     type=str,   default=model_save_path)
    p.add_argument("--img_type", type=str, default="rgb")
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--optim",         type=str,   choices=["SGD", "Adam"], default="SGD",
                   help="optimizer")
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--lr_reduce_on", type=str, default="80,122,150,225,262")
    p.add_argument("--num_epochs",    type=int,   default=300)
    p.add_argument("--max_g_norm", type=float, default=None)
    p.add_argument("--warmup_epoch",  type=int,   default=0)
    p.add_argument("--cosine_t0", type=int, default=None,
                   help="T0 of cosine annealing schedule; if None, using default reduce on epoch scheduler")
    p.add_argument("--aug", type=str2bool, default=False)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--model_name", type=str, default=None,
                   help="Resume from a checkpoint. None means training from scratch")
    # PCNet / PCConv args
    p.add_argument("--inp_channels",  type=int, nargs="+", default=[3,  64, 64, 128, 128, 256, 256, 512],
                   help="list of input-channel sizes, e.g. 3 16 32")
    p.add_argument("--out_channels",  type=int, nargs="+", default=[64, 64, 128, 128, 256, 256, 512, 512],
                   help="list of output-channel sizes")
    p.add_argument("--stride", type=int, nargs="+",
                   default=1)
    p.add_argument("--kernel_size", type=int, nargs="+",
                   default=3)
    p.add_argument("--first_ksz", type=int, default=3)
    p.add_argument("--first_stride", type=int, default=1)
    p.add_argument("--first_pad", type=str, choices=["same", "valid"], default="valid")
    p.add_argument("--max_pool",      type=int, nargs="+",
                   default=[False, False, True, False, True, False, False, False])
    p.add_argument("--avg_pooling", type=str2bool, default=False)
    p.add_argument("--separable", type=str, nargs="+", default=None)
    p.add_argument("--patch_dim", type=int, default=None)
    p.add_argument("--num_classes",   type=int, default=10)
    # ODE hyper-params
    p.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()),
                        default="ODEBlockPC")
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--tol", type=float, default=1e-3, help="ODE solver tolerance")
    p.add_argument("--n_steps", type=float, default=10, help="ODE solver number of steps")
    p.add_argument("--t_end", type=float, default=1.0, help="Stop time of the solver")
    p.add_argument("--offset_eps", type=float, default=None, help="Noise level of the offset")
    # Quantization-aware training related args
    p.add_argument("--ode_wrapper", type=str, choices=list(ODEWrapper_CLASSES.keys()) + [None],
                        default=None)
    p.add_argument("--qat_cls", type=str, choices=list(QUANTIZER_CLASSES.keys()) + [None],
                   default=None)
    p.add_argument("--R", type=float, default=1e5, help="Resistance")
    p.add_argument("--C", type=float, default=49e-15, help="Capacitance")
    p.add_argument("--v_dd", type=float, default=1.0, help="V_DD")
    p.add_argument("--w_bits", type=int, default=8, help="weight quantized bits")
    # PCConv hyper-params
    # p.add_argument("--kernel_size",   type=int, default=3)
    # p.add_argument("--stride",        type=int, default=1)
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
    p.add_argument("--mem_frac", type=float, default=1.0)
    p.add_argument("--test_only", type=str2bool, default=False)
    return p.parse_args()

def _constr_model_name(args, rep=1):
    name_dict = {str(True): "with", str(False): "no"}
    model_name = 'NODE_PPCN'
    if args.pcn is not None and args.pcn != "PCNet":
        model_name = args.pcn
    if args.pc_conv is not None:
        model_name += "_" + args.pc_conv
    if args.offset_eps is not None:
        model_name += "_{}eps".format(args.offset_eps)
    model_name += '_{}'.format(args.ode_block) \
                  + '_{}Solver'.format(args.method) + '_{}TEnd'.format(str(args.t_end)) + '_{}Tol_'.format(str(args.tol)) \
                  + str(args.weight_decay) + 'WD_' \
                  + str(args.batch_size) + 'BS_'
    if args.cosine_t0 is not None:
        model_name += "{}CosLR{}T0_".format(str(args.learning_rate), args.cosine_t0)
    else:
        model_name += str(args.learning_rate) + 'LR_'
    model_name += str(args.dropout) + 'Dropout_' + str(len(args.inp_channels)) + "Layers_" \
                  + str(len([_ for _ in args.max_pool if _])) + "Pool"

    if args.tie_method is not None:
        model_name += "_" + args.tie_method + "TieMethod_" + str(args.tie_frac) + "TieFrac"
    if args.img_type != "rgb":
        model_name += "_" + args.img_type
    model_name = model_name + "_" + str(rep) + 'REP'
    if args.model_name is not None:
        if args.ode_wrapper is None:
            ft_prefix = "ft"
        elif args.qat_cls is None or args.qat_cls == "SymQuantizeWeight":
            ft_prefix = "QAT{}b".format(args.w_bits)
        else:
            ft_prefix = "QAT{}b{}".format(args.w_bits, args.qat_cls)
        eps_val = args.model_name.split("_")[2]
        ode_blk = args.model_name.split("_")[3]
        orig_rep = args.model_name.split("_")[-1]
        model_name = ft_prefix + args.model_name.split(orig_rep)[0].replace(
            eps_val, "{}eps".format(args.offset_eps)).replace(
            ode_blk, "{}".format(args.ode_block)) + str(rep) + 'REP'
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

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device=0)

    # map optimizer name -> class
    # optim_type = getattr(optim, args.optim)
    loss_fn = nn.CrossEntropyLoss()

    model_args = {
        "inp_channels": args.inp_channels,
        "out_channels": args.out_channels,
        "max_pool": args.max_pool,
        "num_classes": args.num_classes,
        "kernel_size": args.kernel_size,
        "stride": args.stride if not (isinstance(args.stride, List) and len(args.stride) == 1) else args.stride[0],
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
    if args.model_name is None:
        model = pcn_model(**model_args)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ckpt_path = os.path.join(args.save_path, args.model_name, args.model_name + "_best_ckpt.pth")
        noisy_params = {"noise_level": 0.0, "weight": None}
        model = load_and_prepare_model(model_path=ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu",
                                       model_struct=pcn_model,
                                       pc_conv_layer=pc_conv_mod, data_parallel=False,
                                       noise_to_bn=False, noise_to_linear=False,
                                       fuse_bn=False, conv_only=False, ode_params=None,
                                       **noisy_params)
        model.dropout = args.dropout

    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)
    logging.warning("input channels: {}".format(model.ics))
    logging.warning("output channels: {}".format(model.ocs))
    logging.warning("max pooling: {}".format(model.max_pool))
    logging.warning("pooling layer: {}".format(model.max_pool2d))
    logging.warning("dropout rate: {}".format(model.dropout))
    logging.warning("Total number of parameters: {}".format(total_params / 1e6))
    logging.warning("Model name: {}".format(model_name))
    logging.info("----- Printing out model parameter names: -----")
    for name, param in model.named_parameters():
        logging.info("name: {}, shape: {}, param count: {}".format(name, param.shape, param.numel()))

    # convert block to Neural ode
    ode_kw, ode_kwargs = ["offset_eps"], {}
    for _name, _val in vars(args).items():
        if _name in ode_kw and _val is not None:
            ode_kwargs[_name] = _val
    ode_block = ODEBLOCK_CLASSES[args.ode_block]
    model = make_ode_block(
        pc_net=model, ode_block=ode_block, noise_level=0.0, method=args.method, t_end=args.t_end,
        tol=args.tol, n_steps=args.n_steps, **ode_kwargs)
    logging.warning("PcConv converted to ODEBlock: {}".format(ode_block.__name__))
    logging.warning("t_end: {}".format(args.t_end))
    logging.warning("method: {}".format(args.method))
    logging.warning("tol: {}".format(args.tol))

    # sanity check if resume training
    if args.model_name is not None:
        logging.warning("Before training, evaluate the accuracy of the loaded model")
        # test_once(model, device='cuda' if torch.cuda.is_available() else 'cpu', model_name=args.model_name)
        model.train()

    # wrap blocks for QAT
    if args.ode_wrapper is not None:
        wrapper_params = {"ode_wrapper": ODEWrapper_CLASSES[args.ode_wrapper], "calib_path": None,
                          "R": args.R, "C": args.C, "v_dd": args.v_dd, "w_bits": args.w_bits,
                          "qat_cls": QUANTIZER_CLASSES[args.qat_cls]}
        model = wrap_ode_block(model, **wrapper_params)
        logging.warning("ODEBlock in network wrapped, ode_wrapper_params={}".format(wrapper_params))

    # Get trainer
    logging.warning("lr reduce on: {}, max grad norm: {}, cosine annealing T0: {}".format(
        args.lr_reduce_on, args.max_g_norm, args.cosine_t0))
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
        aug           = args.aug,
        T0            = args.cosine_t0,
        eval_every    = args.eval_every,
        img_type      = args.img_type,
    )

    if args.test_only:
        _ = model(next(iter(trainer.train_dataloader))[0].to(trainer.device))
        logging.info("Test model forward only. Exit without training the model.")
        logging.info("Result: {}".format(_))
        exit(0)

    trainer.train()

if __name__ == "__main__":
    main()
