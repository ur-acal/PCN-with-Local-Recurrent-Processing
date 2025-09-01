import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import pickle
import argparse
import logging

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy
from typing import List

from pc_conv import PCConvNoisy, PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from ode_pc import make_ode_block, is_adaptive, ODEBLOCK_CLASSES
from inference_utils import load_and_prepare_model, replace_transpose_conv
from data_utils import ToPackedRGGB, RawImgDataset

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from cross_sim.dnn_inference_params import dnn_inference_params
from cross_sim.cross_bar_params import base_params_args
from cross_sim.test_analog_model import test_analog_model, get_exp_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCConvNoisy tests with custom noise parameters"
    )
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing the saved model checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Identifier or filename of the model to load")
    parser.add_argument("--fuse_bn", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True, help="Fuse batch norm into conv")
    parser.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys())+[None],
                   default=None)
    parser.add_argument("--ode_block", type=str, choices=list(ODEBLOCK_CLASSES.keys()) + [None],
                        default=None)
    # input calibration arguments
    parser.add_argument("--calib_samples", type=int, default=None)
    parser.add_argument("--calib_type", type=str, default="min_max", choices=["min_max", "perc_hi_lo"])
    parser.add_argument("--calib_perc", type=float, default=0.99999)
    parser.add_argument("--sym_quant", type=lambda v: v.lower() in ('yes','true','t','1'), default=False)
    parser.add_argument("--calib_path", type=str, default=None)
    parser.add_argument("--calib_only", type=lambda v: v.lower() in ('yes','true','t','1'), default=False)
    # cross-sim params
    parser.add_argument("--prop_error", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True)
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--input_bits", type=int, default=8)
    parser.add_argument("--adc_bits", type=int, default=0)
    parser.add_argument("--bias_rows", type=int, default=0)
    parser.add_argument("--inp_min", type=int, default=None)
    parser.add_argument("--inp_max", type=int, default=None)
    parser.add_argument("--test_only", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=False)
    return parser.parse_args()

def get_calib_loader(bs=128, n_samples=None, img_type="rgb"):
    if img_type in {"rgb", "rggb"}:
        if img_type == "rgb":
            # Todo: Should we keep the random crop here?
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                ToPackedRGGB(return_orig=False),
                transforms.RandomCrop(16, padding=2),
                transforms.RandomHorizontalFlip(),
            ])
        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
        ])
        train_set = RawImgDataset(root=os.path.join("../cifar-10-data", img_type), train=True, transform=transform_train)
    if n_samples is not None:
        perm = torch.randperm(len(train_set))
        train_set = Subset(train_set, perm[:n_samples])
    calib_dataloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=2)
    return calib_dataloader

def get_leaf_mods(model: nn.Module) -> List[nn.Module]:
    leaf_mod = []
    for _, _child in model.named_children():
        if len(list(_child.children())) == 0 and len(list(_child.parameters())) > 0:
            leaf_mod.append(_child)
        leaf_mod.extend(get_leaf_mods(_child))
    return leaf_mod

def calibrate_input(model: nn.Module, device, model_name,
                    calib_bs=128, calib_samples=None,
                    percentile=0.99999, symmetric=False, save_to=None, img_type="rgb"):
    # if saved, just loading the result directly
    if save_to is not None:
        save_to = os.path.join(save_to, "{}_{}_{}.pkl".format(
            model_name, calib_samples, str(percentile).replace(".", "p")))
        if os.path.exists(save_to):
            logging.warning("Calibration result already exists. Return existing result.")
            with open(save_to, "rb") as fp:
                calib_res = pickle.load(fp)
            return calib_res

    model.eval()
    leaf_mod = get_leaf_mods(model)

    stats = {}
    hooks = []

    def _make_hook(mod):
        mod_idx = id(mod)
        def _hook(_, inp):
            x = inp[0].detach()
            rec = stats.setdefault(mod_idx, {
                "min": torch.inf,
                "max": -torch.inf,
                "q_lo": torch.inf,
                "q_hi": -torch.inf,
            })
            rec["min"] = min(rec["min"], x.min().cpu().item())
            rec["max"] = max(rec["max"], x.max().cpu().item())
            # Todo: To get the true percentile, we need to accumulate all samples across batches
            rec["q_lo"] = min(rec["q_lo"], x.quantile(1 - percentile).cpu().item())
            rec["q_hi"] = max(rec["q_hi"], x.quantile(percentile).cpu().item())
            if symmetric:
                max_abs = max(abs(rec["min"]), abs(rec["max"]))
                rec["min"] = -max_abs
                rec["max"] = max_abs
            stats[mod_idx] = rec
        return _hook

    for _mod in leaf_mod:
        hooks.append(_mod.register_forward_pre_hook(_make_hook(_mod)))

    calib_loader = get_calib_loader(bs=calib_bs, n_samples=calib_samples, img_type=img_type)
    for _batch in calib_loader:
        _inp, _ = _batch
        _inp = _inp.to(device)
        __ = model(_inp)

    for _h in hooks:
        _h.remove()

    min_max, perc_hi_lo = [], []
    for _mod in leaf_mod:
        if id(_mod) not in stats:
            logging.info("mod {} not used".format(_mod))
            continue
        _rec = stats[id(_mod)]
    # for _, _rec in stats.items():
        min_max.append([_rec["min"], _rec["max"]])
        perc_hi_lo.append([_rec["q_lo"], _rec["q_hi"]])
        logging.info("mod: {}, min_max: {}, perc {}: {}".format(
            _mod, [_rec["min"], _rec["max"]], percentile, [_rec["q_lo"], _rec["q_hi"]]))

    calib_res = {"min_max": np.array(min_max), "perc_hi_lo": np.array(perc_hi_lo)}
    if save_to is not None:
        with open(save_to, "wb") as fp:
            pickle.dump(calib_res, fp)
            logging.warning("Calibration result saved to: {}".format(save_to))

    return calib_res


def cross_sim_inference(args, Nruns=10, noise_level=0.0, proportional_error=True, digital_bias=False,
                        ideal=False, weight_bits=8, input_bits=8, adc_bits=0, bias_rows=0):
    # args = parse_args()
    if args.test_only or args.calib_only:
        # set level in the very beginning before calling logging.warning, otherwise the line below will not work
        logging.basicConfig(level=logging.INFO)

    # Get pc_conv_layer to use
    pc_conv = PC_CONV_CLASS.get(args.pc_conv, PCConv)
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv.__name__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_best_ckpt.pth")

    #######################################################################################
    # Noise is added through cross-sim api
    #######################################################################################
    noisy_params = {"noise_level": 0.0}

    # Get noise-free model
    if "TEnd" in args.model_name and "Solver" in args.model_name:
        t_end = float(args.model_name.split("TEnd")[0].split("_")[-1])
        ode_params = {"ode_block": ODEBLOCK_CLASSES[args.ode_block], "t_end": t_end, "method": "dopri5",
                      "tol": 1e-3, "ts_scale": 1}
        net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                      pc_conv_layer=pc_conv, data_parallel=False,
                                      noise_to_bn=False, noise_to_linear=False,
                                      fuse_bn=False, conv_only=True, ode_params=ode_params,
                                      **noisy_params)
    else:
        net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                      pc_conv_layer=pc_conv, data_parallel=False,
                                      noise_to_bn=False, noise_to_linear=False,
                                      fuse_bn=False, conv_only=True, **noisy_params)

    # Create a list of CrossSimParameters objects
    n_layers = len(convertible_modules(net_))
    params_list = [CrossSimParameters()] * n_layers

    # Params arguments common to all layers
    params_args = deepcopy(base_params_args)
    params_args.update({
        'ideal' : ideal,
        'weight_bits' : weight_bits,
        'digital_bias' : digital_bias,
        'alpha_error' : noise_level,
        'proportional_error' : proportional_error,
        'input_bits' : input_bits,
        'adc_bits' : adc_bits,
        'useGPU' : True if device == 'cuda' else False,
    })

    ### Load input limits
    # Todo: Support input range calibration for more models
    if args.inp_min is not None and args.inp_max is not None:
        input_ranges = torch.stack(
            [torch.tensor([-2.64, 2.64])] + [torch.tensor([args.inp_min, args.inp_max])] * (n_layers - 1)
        ).numpy()
    else:
        # Todo: Currently set calib_bs = calib_samples
        input_ranges = calibrate_input(model=net_, device=device, model_name=args.model_name,
                                       calib_bs=args.calib_samples,
                                       calib_samples=args.calib_samples, percentile=args.calib_perc,
                                       symmetric=args.sym_quant, save_to=args.calib_path)[args.calib_type]
        if args.calib_only:
            logging.warning("Calibrating inputs only, exit now.")
            exit(0)

    ### Load ADC limits
    # Todo: Skipped for now

    ### Set the parameters
    for k in range(n_layers):
        print("layer: {}, input_range_k: {}".format(k, input_ranges[k]))
        params_args_k = params_args.copy()
        params_args_k['positiveInputsOnly'] = input_ranges[k][0] >= 0
        params_args_k['input_range'] = input_ranges[k]
        # params_args_k['adc_range'] = adc_ranges[k] # Todo: Skipped for now
        params_list[k] = dnn_inference_params(**params_args_k)

    #### Convert PyTorch layers to analog layers
    print("----- Start to convert from torch -----")
    analog_net = from_torch(net_, params_list, fuse_batchnorm=True, bias_rows=bias_rows)
    print("----- Successfully converted from torch -----")

    #### Load and transform CIFAR-10 dataset
    batch_size = 64
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.2470, 0.2435, 0.2616])
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    mean_acc, std_acc, acc_list = test_analog_model(analog_model=analog_net, Nruns=Nruns, N=len(test_dataset),
                                                    device=device, batch_size=batch_size, data_loader=test_dataloader)

    return mean_acc, std_acc, acc_list


def run_cross_sim_inference():
    args = parse_args()

    ###############################
    ## Configurations
    # prop_error_ = True
    n_weight_bits_ = 8
    n_input_bits_ = 8
    n_adc_bits_ = 0
    # n_bias_rows_ = 0
    ###############################

    logging.warning("Running test with cross-sim")
    if args.test_only or args.calib_only:
        _ = cross_sim_inference(args, Nruns=1, noise_level=0.2, proportional_error=args.prop_error, digital_bias=False,
                                ideal=False, weight_bits=args.weight_bits, input_bits=args.input_bits,
                                adc_bits=args.adc_bits, bias_rows=args.bias_rows)
        exit(0)

    n_trials = 20
    noise_level_list_ = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                         0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20,
                         .25, .30, .35, .40]
    # noise_level_list_ = [0, 0.4]

    acc_log_dict, acc_list_dict = {}, {}
    for _nl in noise_level_list_:
        _n_trials = n_trials if _nl > 0.0 else 1
        _mean, _std, _acc = cross_sim_inference(args, Nruns=_n_trials, noise_level=_nl,
                                                proportional_error=args.prop_error, digital_bias=False,
                                                ideal=False, weight_bits=args.weight_bits,
                                                input_bits=args.input_bits, adc_bits=args.adc_bits,
                                                bias_rows=args.bias_rows)
        acc_log_dict[_nl] = "{:.2f} ± {:.2f}".format(_mean, _std)
        acc_list_dict[_nl] = _acc

    pkl_name = get_exp_name(proportional_error=args.prop_error, weight_bits=args.weight_bits, input_bits=args.input_bits,
                            adc_bits=args.adc_bits, bias_rows=args.bias_rows, noise_level_list=noise_level_list_)
    os.makedirs(os.path.join("logs/cross_sim_res", args.model_name), exist_ok=True)
    pkl_path = os.path.join("logs/cross_sim_res", args.model_name, "{}.pkl".format(pkl_name))
    with open(pkl_path, "wb") as fp:
        pickle.dump(acc_list_dict, fp)
        print("Model acc list saved to: {}".format(pkl_path))

    print("-------- Model name: {} --------".format(args.model_name))
    for _nl, _acc in acc_log_dict.items():
        print("Noise level: {}, Acc: {}".format(_nl, _acc))


if __name__ == "__main__":
    run_cross_sim_inference()
