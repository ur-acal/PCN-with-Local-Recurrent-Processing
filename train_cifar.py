import torch
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConv, PCConvNoisy
from pc_model import PCNet
from trainer import TrainerCiFar

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    p = argparse.ArgumentParser(description="Train PCNet on CIFAR")
    model_save_path = "/home/rongzeng/_workspce_old/repos/pcn/PCN-with-Local-Recurrent-Processing/saved_ckpt"
    # TrainerCiFar args
    p.add_argument("--save_path",     type=str,   default=model_save_path)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--optim",         type=str,   choices=["SGD","Adam"], default="Adam",
                   help="optimizer")
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--num_epochs",    type=int,   default=300)
    # PCNet / PCConv args
    p.add_argument("--inp_channels",  type=int, nargs="+", default=[3,  64, 64, 128, 128, 256, 256, 512],
                   help="list of input-channel sizes, e.g. 3 16 32")
    p.add_argument("--out_channels",  type=int, nargs="+", default=[64, 64, 128, 128, 256, 256, 512, 512],
                   help="list of output-channel sizes")
    p.add_argument("--max_pool",      type=int, nargs="+",
                   default=[False, False, True, False, True, False, False, False])
    p.add_argument("--num_classes",   type=int, default=10)
    # PCConv hyper-params
    p.add_argument("--kernel_size",   type=int, default=3)
    p.add_argument("--stride",        type=int, default=1)
    p.add_argument("--padding",       type=int, default=1)
    p.add_argument("--cls",           type=int, default=5)
    p.add_argument("--bias",          action="store_true")
    p.add_argument("--lr_pc",       type=float, default=0.01,
                   help="PC layer recurrent learning rate")
    p.add_argument("--tie_weights",   type=str2bool, default=False)
    p.add_argument("--tie_bp",        type=str2bool, default=False)
    p.add_argument("--relu_between",  type=str2bool, default=False)
    p.add_argument("--bypass",        type=str2bool, default=False)
    return p.parse_args()

def _constr_model_name(args, rep=1):
    name_dict = {str(True): "with", str(False): "no"}
    model_name = 'PPCN' + '_' + str(args.cls) + 'CLS_' + str(args.weight_decay) + 'WD_' \
                 + name_dict[str(args.tie_weights)] + 'Tied_' + name_dict[str(args.tie_bp)] + 'BP_tied_' \
                 + name_dict[str(args.bypass)] + 'BP' + name_dict[str(args.relu_between)] + 'Relu_' \
                 + str(rep) + 'REP'
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

    # map optimizer name -> class
    optim_type = getattr(optim, args.optim)
    loss_fn = nn.CrossEntropyLoss()

    # build model
    model = PCNet(
        inp_channels  = args.inp_channels,
        out_channels  = args.out_channels,
        max_pool      = args.max_pool,
        num_classes   = args.num_classes,
        pc_conv_layer = PCConv,
        kernel_size   = args.kernel_size,
        stride        = args.stride,
        padding       = args.padding,
        cls           = args.cls,
        bias          = args.bias,
        lr            = args.lr_pc,
        tie_weights   = args.tie_weights,
        tie_bp        = args.tie_bp,
        relu_between  = args.relu_between,
        bypass        = args.bypass,
    )

    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)
    print("Total number of parameters: {}".format(total_params))
    print("Model name: {}".format(model_name))

    trainer = TrainerCiFar(
        model         = model,
        model_name    = model_name,
        save_path     = args.save_path,
        batch_size    = args.batch_size,
        optim_type    = optim_type,
        weight_decay  = args.weight_decay,
        loss_fn       = loss_fn,
        learning_rate = args.learning_rate,
        num_epochs    = args.num_epochs,
    )
    trainer.train()

if __name__ == "__main__":
    main()
