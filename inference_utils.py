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

from cifar_test import PredNetBpD
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES
from pc_conv import PCConv, PCConvNoisy, PartialTiedPCConv
from bn_fuse import fuse_bn_recursively

import logging
log = logging.getLogger(__name__)
log.propagate = False

# attach a handler that only prints the message
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


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
                           data_parallel=False, noise_to_bn=False, noise_to_linear=False, fuse_bn=True, **kwargs):
    checkpoint_weight = torch.load(model_path, map_location=device)  # weights_only=False
    model_args = checkpoint_weight["init_args"]["model_args"]
    mod_args = checkpoint_weight["init_args"]["kwargs"]

    # Get model used
    sd_model_struct = checkpoint_weight.get("net_type", None)
    model_struct = PCN_CLASSES[sd_model_struct] if sd_model_struct else model_struct
    log.warning("----- Using :{} model -----".format(model_struct.__name__))

    # filter out unused init arguments
    filter_args(model_struct, model_args)
    filter_args(pc_conv_layer, mod_args) if pc_conv_layer is not None else log.info("pc_conv_layer not specified")

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

    # No batch normalization for each convolutional layers in PcConv layers
    # If there is any, then need to change the _init_noise function
    if fuse_bn:
        net_.eval()
        net_ = fuse_bn_recursively(net_)

    # Add noise
    if hasattr(net_, "add_noise"):
        clean_params = {_name: _p.clone() for _name, _p in net_.named_parameters()}
        clean_buffs = {_name: _buf.clone() for _name, _buf in net_.named_buffers()}
        net_.add_noise(noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear)
        noise_level = net_.noise_level
        if noise_level > 0.0:
            for _name, _p in net_.named_parameters():
                if noise_to_bn and "bn" in _name.lower() and "pc" not in _name.lower():
                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[_name])
                elif noise_to_linear and "linear" in _name.lower() and "pc" not in _name.lower():
                    assert torch.allclose(_p, torch.zeros_like(_p)) or not torch.allclose(_p, clean_params[_name])

            if noise_to_bn:
                # adding noise to running mean and variance of batch norm
                for _name, _buf in net_.named_buffers():
                    if _name.endswith(('running_mean', 'running_var')):
                        assert torch.allclose(_buf, torch.zeros_like(_buf)) or not torch.allclose(_buf, clean_buffs[_name])
            log.warning("----- Noise added, sanity check passed -----")
    log.warning("----- Model loaded -----")
    return net_


def expand_and_save_weights(sample_imgs, model_path, device="cpu", model_struct=PCNet, pc_conv_layer=PCConvNoisy,
                            data_parallel=True, weight_dir="expanded_weights", model_suffix=".pt", model_name=None,
                            noise_to_bn=False, noise_to_linear=False, **kwargs):
    log.warning("----- Start to expand and save weights -----")
    # Load model
    net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel,
                                  noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear, **kwargs)
    # Save expanded weights
    if model_name is not None:
        weight_path = os.path.join(weight_dir, model_name)
    else:
        weight_path = os.path.join(weight_dir, model_path.split('/')[-1].split(model_suffix)[0])
    if os.path.isdir(weight_path) and any(os.scandir(weight_path)):
        log.warning("When running expand_and_save_weights, found expanded weights under {}".format(weight_path))
        return
    os.makedirs(weight_path, exist_ok=True)
    net_.eval()
    net_.save_expanded_weights(sample_imgs.to(device), weight_path)
    log.warning("----- weights expanded and saved -----")


def plot_layer_pcn_loss(sample_imgs, model_path, device="cpu", model_struct=PCNet, pc_conv_layer=PCConvNoisy,
                        data_parallel=True, loss_plot_dir="loss_plot", model_suffix=".t7", model_name=None,
                        noise_to_bn=False, noise_to_linear=False, **kwargs):
    log.warning("----- Start to plot layer PCN loss -----")
    noise_level = kwargs.get("noise_level", 0.0)
    if model_name is not None:
        loss_plot_dir = os.path.join(loss_plot_dir, model_name, "noise_level_{}".format(noise_level))
    else:
        loss_plot_dir = os.path.join(
            loss_plot_dir, model_path.split('/')[-1].split(model_suffix)[0], "noise_level_{}".format(noise_level))
    os.makedirs(loss_plot_dir, exist_ok=True)
    kwargs.update({"plot_path": loss_plot_dir})
    net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel,
                                  noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear, **kwargs)
    net_.eval()
    _ = net_(sample_imgs.to(device))
    log.warning("----- Loss is plotted -----")


def run_lr_cls_experiment(model_path, test_loader, noise_level_list, device="cpu", model_struct=PCNet,
                          pc_conv_layer=PCConvNoisy, data_parallel=True, noisy_trials=10, model_name=None,
                          noise_to_bn=False, noise_to_linear=False, fuse_bn=True,
                          scale_factor=None, plot_path="loss_plot/lr_cls_acc", **kwargs):
    cycles, lr_pc = 5.0, 1.0 # default setting
    if isinstance(model_name, str):
        cycles = float(model_name.split("CLS")[0].split("_")[-1])
        lr_pc = float(model_name.split("LRPC")[0].split("_")[-1])
    log.warning("----- Model cycles: {}, LR PC: {} -----".format(cycles, lr_pc))
    scale_factor = scale_factor if scale_factor is not None else [1, 2, 4, 8, 10, 16, 20, 32, 50]
    sf_acc_dict, cls_list = {}, []
    for _sf in scale_factor:
        cur_cls, cur_lr_pc = int(cycles * _sf), lr_pc / _sf
        assert np.allclose(cur_cls * cur_lr_pc, cycles * lr_pc)
        log.warning("----- Current cycles: {}, LR PC: {} -----".format(cur_cls, cur_lr_pc))
        params_ = deepcopy(kwargs)
        params_.update({"cls": cur_cls, "lr": cur_lr_pc})
        cur_noise_acc = run_noise_experiment(model_path, test_loader, noise_level_list=noise_level_list,
                                 model_struct=model_struct, pc_conv_layer=pc_conv_layer, data_parallel=data_parallel,
                                 device=device, noisy_trials=noisy_trials, model_name=model_name,
                                 noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear,
                                 fuse_bn=fuse_bn, **params_)
        sf_acc_dict[cur_cls] = [cur_noise_acc[_nl] for _nl in noise_level_list]
        cls_list.append(cur_cls)
    noise_acc_dict = {}
    log.warning("-------- Final Result --------")
    log.warning("-------- Model name: {} --------".format(model_name))
    for i, _noise_level in enumerate(noise_level_list):
        noise_acc_dict[_noise_level] = (cls_list, [sf_acc_dict[_][i] for _ in cls_list])
        log.warning("Noise level: {}, cycles list: {}, Acc: {}".format(_noise_level, cls_list,
                                                          ["{:.2f}%".format(_) for _ in noise_acc_dict[_noise_level][1]]))
    plot_acc_diff_cls(noise_acc_dict, plot_path, model_name, cycles, lr_pc)
    log.warning("-------- Scaling cycles and lr experiment finished --------")


def plot_acc_diff_cls(noise_acc_dict, plot_path, model_name, cycles, lr_pc):
    save_dir = os.path.join(plot_path, model_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(8, 4))

    noise_level_list = list(noise_acc_dict.keys())
    cls_list = noise_acc_dict[noise_level_list[0]][0]

    for cur_nl in sorted(noise_level_list):
        cur_tup = noise_acc_dict[cur_nl]
        label = f"pvt noise level = {cur_nl}"
        ax.plot(cur_tup[0], cur_tup[1], label=label)

    ax.set_xlabel("Number of cycles")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)

    # place legend to the right
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        prop={'family': 'Times New Roman'}
    )
    title = "Model trained with cycles = {}, lr pc = {}".format(cycles, lr_pc)
    ax.set_title(title)

    # make room on the right for the legend
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    fig.savefig(os.path.join(str(save_dir), "cls_vs_acc_noiseList_{}_clsList_{}.pdf".format(noise_level_list, cls_list)),
                format="pdf", bbox_inches="tight")
    with open(os.path.join(str(save_dir), "noise_acc_dict.pkl"), "wb") as fp:
        pickle.dump(noise_acc_dict, fp)


def run_noise_experiment(model_path, test_loader, noise_level_list, device="cpu", model_struct=PCNet,
                         pc_conv_layer=PCConvNoisy, data_parallel=True, noisy_trials=10, model_name=None,
                         noise_to_bn=False, noise_to_linear=False, fuse_bn=True, **kwargs):
    noise_acc = {}
    for noise_level in noise_level_list:
        trials = noisy_trials if noise_level > 0 else 1
        acc_list = []
        for t in range(trials):
            # reinitialize net with different noise during each trial
            params_ = deepcopy(kwargs)
            params_.update({"noise_level": noise_level})
            net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel,
                                          noise_to_bn=noise_to_bn, noise_to_linear=noise_to_linear,
                                          fuse_bn=fuse_bn, **params_)
            net_.eval()
            total = 0
            correct = 0

            for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), disable=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    output_tensor = net_(inputs)

                # Get the predicted class
                _, predicted = torch.max(output_tensor, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Calculate the accuracy
            accuracy = 100 * correct / total
            acc_list.append(accuracy)
            log.warning(f'Test Accuracy at noise level {noise_level}: {accuracy:.2f}%')
        avg_acc = sum(acc_list) / len(acc_list)
        noise_acc[noise_level] = avg_acc
        log.warning("Average test acc over {} trials is {}".format(trials, avg_acc))
    log.warning("-------- Final Result --------")
    log.warning("-------- Model name: {} --------".format(model_name))
    for _nl, _acc in noise_acc.items():
        log.warning("Noise level: {}, Acc:{:.2f}%".format(_nl, _acc))
    log.warning("-------- Noisy experiment finished --------")
    return noise_acc


if __name__ == '__main__':
    batch_size = 4096 * 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.warning(f'Using device: {device}')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    num_samples = len(test_set)

    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # set different model initialization parameters here
    model_path_ = "checkpoint/PredNetBpD_5_30CLS_FalseNes_0.001WD_FalseTIED_4REP_best_ckpt.t7"
    weight_dir_ = "expanded_weights"
    loss_plot_dir_ = "loss_plot"
    expanded_weight_path = os.path.join(weight_dir_, model_path_.split('/')[-1].split(".t7")[0])

    # save and expand models
    expand_and_save = False
    if expand_and_save:
        model_params = {"num_classes": 10, "cls": 30, "lr": 1e-2, "noise_level": 0, "solver": 'LD',
                        "layer_number": [0, 1, 2, 3, 4], "num_iterations": 30, "train_weight": False}
        expand_and_save_weights(next(iter(test_loader))[0], model_path=model_path_, device=device,
                                weight_dir=weight_dir_, **model_params)

    # plot noise level 0
    test_no_noise = False
    if test_no_noise:
        model_params = {"num_classes": 10, "cls": 30, "lr": 1e-2, "noise_level": 0, "solver": 'LD',
                        "layer_number": [0, 1, 2, 3, 4], "num_iterations": 30, "train_weight": False,
                        "pcn_weight_type": "fb", "use_relu": False,
                        "pc_weight": expanded_weight_path}
        plot_layer_pcn_loss(next(iter(test_loader))[0], model_path=model_path_, device=device,
                            loss_plot_dir=loss_plot_dir_, **model_params)
        model_params.update({"solver": "SGD"})
        plot_layer_pcn_loss(next(iter(test_loader))[0], model_path=model_path_, device=device,
                            loss_plot_dir=loss_plot_dir_, **model_params)

    # noise experiments
    noise_experiment = True
    if noise_experiment:
        model_params = {"num_classes": 10, "cls": 30, "lr": 1e-2, "noise_level": None, "solver": 'LD',
                        "layer_number": [0, 1, 2, 3, 4], "num_iterations": 30, "train_weight": False,
                        "pcn_weight_type": "ff", "use_relu": True,
                        "noise_to_ff": False, "noise_to_bp": False,
                        "pc_weight": expanded_weight_path, "plot_path": None}
        noise_level_list_ = [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40]
        run_noise_experiment(model_path_, test_loader, noise_level_list=noise_level_list_,
                             device=device, noisy_trials=5, **model_params)
