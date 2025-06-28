import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import inspect
import sys
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from pc_model import PCNet, PCN_CLASSES
from pc_conv import PCConv, PCConvNoisy
from ode_pc import make_ode_block

import logging
log = logging.getLogger(__name__)
log.propagate = False

# attach a handler that only prints the message
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


def get_train_data(bs=2048):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader


def get_test_data(test_bs=2048):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=2)
    return test_loader

def collect_init_args(module_class):
    all_args = set()
    for cls in module_class.__mro__:
        if cls is object:
            continue
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            continue

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            all_args.add(name)
    return all_args


def filter_args(module_class, arg_dict):
    if module_class is None:
        return
    arg_set = collect_init_args(module_class)
    remove_args = set()
    for mod_arg in arg_dict:
        if mod_arg not in arg_set:
            log.info(f"{mod_arg} need to be removed for {module_class.__name__}")
            remove_args.add(mod_arg)
    for mod_arg in remove_args:
        arg_dict.pop(mod_arg)


def load_and_prepare_model(model_path, device, model_struct=PCNet, pc_conv_layer=PCConvNoisy,
                           data_parallel=False, noise_to_bn=False, noise_to_linear=False, fuse_bn=True,
                           conv_only=False, ode_params=None, **kwargs):
    checkpoint_weight = torch.load(model_path, map_location=device)  # weights_only=False
    model_args = checkpoint_weight["init_args"]["model_args"]
    mod_args = checkpoint_weight["init_args"]["kwargs"]

    # Get model used
    sd_model_struct = checkpoint_weight.get("net_type", None)
    model_struct = PCN_CLASSES[sd_model_struct] if sd_model_struct else model_struct
    log.warning("----- Using :{} model -----".format(model_struct.__name__))

    # filter out unused init arguments
    filter_args(model_struct, model_args)
    if pc_conv_layer is not None:
        filter_args(pc_conv_layer, mod_args)
        filter_args(pc_conv_layer, kwargs)
    else:
        log.info("pc_conv_layer not specified")

    # check if we should use pc_conv_layer defined in the model loaded or the one passed in
    if "pc_conv_layer" in collect_init_args(model_struct) and pc_conv_layer is not None:
        model_args["pc_conv_layer"] = pc_conv_layer

    init_kwargs = {**model_args, **mod_args}
    init_kwargs.update(kwargs) # overwritten the loaded args with passed in kwargs (if overlapping keys exist)

    net_ = model_struct(**init_kwargs)
    net_ = net_.to(device)
    if data_parallel:
        net_ = nn.DataParallel(net_)
        net_.load_state_dict(checkpoint_weight['net'])
        net_ = net_.module
    else:
        net_.load_state_dict(checkpoint_weight['net'])

    if conv_only:
        log.warning("Replacing all transposed conv with conv")
        replace_transpose_conv(net_)

    # No batch normalization for each convolutional layers in PcConv layers
    # If there is any, then need to change the _init_noise function

    # Add noise
    if hasattr(net_, "add_noise"):
        clean_params = {_name: _p.clone() for _name, _p in net_.named_parameters()}
        clean_buffs = {_name: _buf.clone() for _name, _buf in net_.named_buffers()}
        net_.add_noise(noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear)

        noise_level = net_.noise_level
        #############################################################################
        # ODE related
        if isinstance(ode_params, dict):
            net_ = make_ode_block(net_, noise_level=noise_level, **ode_params)
            logging.warning("PcConv converted to ODEBlock, ode_params={}".format(ode_params))
        #############################################################################
        if noise_level > 0.0:
            for _name, _p in net_.named_parameters():
                if noise_to_bn and "bn" in _name.lower() and "pc" not in _name.lower():
                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[_name])
                elif noise_to_linear and "linear" in _name.lower() and "pc" not in _name.lower():
                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[_name])

                if isinstance(ode_params, dict):
                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[_name])

                logging.info("Noise check, name: {}, is equal: {}".format(_name, torch.allclose(_p, clean_params[_name])))

            if noise_to_bn:
                # adding noise to running mean and variance of batch norm
                for _name, _buf in net_.named_buffers():
                    if _name.endswith(('running_mean', 'running_var')):
                        assert torch.allclose(_buf, torch.zeros_like(_buf)) or not torch.allclose(_buf, clean_buffs[_name])
            log.warning("----- Noise added, sanity check passed -----")
    log.warning("----- Model loaded -----")
    return net_


def replace_transpose_conv(module: nn.Module):
    for _name, _child in module.named_children():
        if isinstance(_child, nn.ConvTranspose2d):
            conv2d_fb = nn.Conv2d(in_channels=_child.in_channels, out_channels=_child.out_channels,
                                  kernel_size=_child.kernel_size, stride=_child.stride, padding=_child.padding,
                                  bias=_child.bias)
            conv2d_fb.weight.data = _child.weight.data.permute([1,0,2,3]).flip([2,3])
            setattr(module, "FBconv", conv2d_fb)
        replace_transpose_conv(_child)

if __name__ == '__main__':
    pass
