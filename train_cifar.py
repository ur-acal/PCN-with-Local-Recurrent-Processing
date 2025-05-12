import torch
import os
import argparse
import logging
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
    p.add_argument("--optim",         type=str,   choices=["SGD", "Adam"], default="Adam",
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
    p.add_argument("--cls",           type=int, default=30)
    p.add_argument("--bias",          action="store_true")
    p.add_argument("--lr_pc",       type=float, default=0.01,
                   help="PC layer recurrent learning rate")
    p.add_argument("--tie_weights",   type=str2bool, default=False)
    p.add_argument("--tie_bp",        type=str2bool, default=False)
    p.add_argument("--relu_between",  type=str2bool, default=False)
    p.add_argument("--bypass",        type=str2bool, default=False)
    p.add_argument("--relu_bp", type=str2bool, default=False)
    p.add_argument("--use_pc", type=str2bool, default=True)
    p.add_argument("--test_only", type=str2bool, default=False)
    return p.parse_args()

def _constr_model_name(args, rep=1):
    name_dict = {str(True): "with", str(False): "no"}
    model_name = 'PPCN' + '_' + str(args.cls) + 'CLS_' + str(args.lr_pc) + 'LRPC_'+ str(args.weight_decay) + 'WD_' \
                 + name_dict[str(args.tie_weights)] + 'Tied_' + name_dict[str(args.tie_bp)] + 'BPtied_' \
                 + name_dict[str(args.relu_between)] + 'Relu_'+ name_dict[str(args.bypass)] + 'BP_' \
                 + name_dict[str(args.relu_bp)] + 'ReluBP_' + name_dict[str(args.use_pc)] + 'PC_' \
                 + str(len(args.inp_channels)) + "Layers_" + str(rep) + 'REP'
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
        relu_bp       = args.relu_bp,
        use_pc        = args.use_pc,
    )

    total_params = sum(p.numel() for p in model.parameters())
    model_name = get_model_name(args)
    logging.warning("input channels: {}".format(model.ics))
    logging.warning("output channels: {}".format(model.ocs))
    logging.warning("max pooling: {}".format(model.max_pool))
    logging.warning("Total number of parameters: {}".format(total_params / 1e6))
    logging.warning("Model name: {}".format(model_name))
    logging.info("----- Printing out model parameter names: -----")
    for name, param in model.named_parameters():
        logging.info("name: {}, shape: {}, param count: {}".format(name, param.shape, param.numel()))

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
    )

    if args.test_only:
        _ = model(next(iter(trainer.train_dataloader))[0].to(trainer.device))
        logging.info("Test model forward only. Exit without training the model.")
        exit(0)

    trainer.train()

if __name__ == "__main__":
    main()
