import torch
import torch.nn as nn
import torchvision
import os
import argparse
import logging

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from pc_conv import PCConvNoisy, PCConv, PartialTiedPCConv
from pc_model import PCNet, PCNetWithMiddleConv, PCN_CLASSES, PC_CONV_CLASS
from inference_utils import load_and_prepare_model, replace_transpose_conv

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from cross_sim.dnn_inference_params import dnn_inference_params
from cross_sim.cross_bar_params import base_params_args

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
    parser.add_argument("--noisy_test", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True)
    parser.add_argument("--test_only", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=False)
    return parser.parse_args()


def cross_sim_inference(args, n=9, Nruns=10, noise_level=0.0, proportional_error=True, digital_bias=False,
                            ideal=False, weight_bits=8, input_bits=8, adc_bits=0, bias_rows=0):
    # args = parse_args()
    if args.test_only:
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
    with torch.no_grad():
        if args.test_only:
            logging.info("----- Running one forward pass for model: {} -----".format(args.model_name))
            net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                          pc_conv_layer=pc_conv, data_parallel=False,
                                          noise_to_bn=False, noise_to_linear=False,
                                          fuse_bn=False, conv_only=True, **noisy_params)
            net_.eval()

    # Get noise-free model
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
    input_ranges = torch.stack([torch.tensor([-2.64, 2.64])] + [torch.tensor([-1, 1])] * (n_layers - 1))

    ### Load ADC limits
    # Todo: Skipped for now

    ### Set the parameters
    for k in range(n_layers):
        params_args_k = params_args.copy()
        params_args_k['positiveInputsOnly'] = input_ranges[k][0] >= 0
        params_args_k['input_range'] = input_ranges[k]
        # params_args_k['adc_range'] = adc_ranges[k] # Todo: Skipped for now
        params_list[k] = dnn_inference_params(**params_args_k)

    #### Convert PyTorch layers to analog layers
    print("----- Start to convert from torch -----")
    analog_resnet = from_torch(net_, params_list, fuse_batchnorm=True, bias_rows=bias_rows)
    print("----- Successfully converted from torch -----")





