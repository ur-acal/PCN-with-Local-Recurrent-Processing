import torch
import torch.nn as nn
import torchvision
import os
import argparse
import logging

from pc_conv import PCConvNoisy, PCConv
from pc_model import PCNet, PCN_CLASSES, PC_CONV_CLASS
from inference_utils import load_and_prepare_model
from inference_utils import replace_transpose_conv, get_train_data
from ode_pc import ODEBlockPC
from torchdiffeq import odeint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCConvNoisy tests with custom noise parameters"
    )
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing the saved model checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Identifier or filename of the model to load")
    parser.add_argument("--w_type", type=str, default="fb_flip",
                        help="Type of weight perturbation")
    parser.add_argument("--noise_to_ff", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=True, help="Noise to the first feed-forward")
    parser.add_argument("--noise_to_bp", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=True, help="Noise to bypass")
    parser.add_argument("--tie_noise", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="Tie the noise of FB to that of FF")
    parser.add_argument("--tie_noise_bp", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="Tie the noise of Bypass to that of FF")
    parser.add_argument("--diff_noise", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="If True, set different noise for each FF/FB call in recurrence")
    parser.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys())+[None],
                   default=None)
    parser.add_argument("--conv_only", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False)
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level")
    return parser.parse_args()

def get_layer_input_shape(net: PCNet, layer_idx, sample_inp):
    assert layer_idx < net.num_layers
    for i in range(net.num_layers):
        sample_out = net.PcConvs[i](sample_inp)
        if i == layer_idx:
            break
        sample_inp = sample_out
    return sample_inp.size(), sample_out.size()

def get_response_from_layer(net: PCNet, layer_idx, sample_x, sample_y0=None):
    pc_layer = net.PcConvs[layer_idx]

    if sample_y0 is None:
        sample_y0 = pc_layer.relu(pc_layer.FFconv(sample_x))
    res = pc_layer.find_optimal_r(sample_x, sample_y0)
    return res

def get_response_from_ode_layer(net: PCNet, layer_idx, ode_params: dict, sample_x, sample_y0=None):
    pc_layer = net.PcConvs[layer_idx]
    noise_level = net.noise_level
    ode_layer = ODEBlockPC(pc_layer, noise_level=noise_level, **ode_params)

    if sample_y0 is None:
        sample_y0 = ode_layer.act_fn(ode_layer.FFconv(sample_x))

    def ode_func(t, y):
        return ode_layer.FFconv(ode_layer.act_fn(sample_x - ode_layer.FBconv(y)))

    integration_time = ode_layer.integration_time.type_as(sample_x)
    res = odeint(ode_func, sample_y0, integration_time, rtol=ode_layer.tol, atol=ode_layer.tol, method=ode_layer.method)
    res = res[-1]
    return res


def run_test():
    expand_weights = False

    args = parse_args()
    noisy_args = ["w_type", "noise_to_ff", "noise_to_bp", "tie_noise", "tie_noise_bp", "diff_noise"] # skip plotting for noisy exp
    logging.warning("Running test with parameters:")
    noisy_params = {}
    for name, val in vars(args).items():
        logging.warning(f"  {name}: {val}")
        if name in noisy_args:
            noisy_params[name] = val
    noisy_params["weight"] = None

    # Get pc_conv_layer to use
    pc_conv = PC_CONV_CLASS.get(args.pc_conv, PCConvNoisy)
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv.__name__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader = get_train_data()
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_best_ckpt.pth")

    with torch.no_grad():
        logging.info("----- Running one forward pass for model: {} -----".format(args.model_name))
        noisy_params["noise_level"] = args.noise_level
        noisy_params["weight"] = None
        net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                      pc_conv_layer=pc_conv, data_parallel=False,
                                      noise_to_bn=True, noise_to_linear=True,
                                      fuse_bn=False, conv_only=args.conv_only, **noisy_params)
        net_.eval()
        sample_batch = next(iter(train_dataloader))[0].to(device)[:64]
        _ = net_(sample_batch)


        ################################################################################################
        # Getting sample response starts from here
        ################################################################################################
        # Configurations
        layer_i = 3 # selected from list(range(net_.num_layers))

        # Get required shape; Only the second dimension can not be changed. (Batch, Channel, H, W)
        inp_shape, out_shape = get_layer_input_shape(net_, layer_i, sample_batch)
        print("Required sample_x shape for layer {}: {}".format(layer_i, inp_shape))
        print("Required sample_y0 shape for layer {}: {}".format(layer_i, out_shape))

        sample_x_ = torch.randn([1] + list(inp_shape)[1:], requires_grad=False, device=device)
        # Optional to pass in a sample_y0
        # sample_y0_ = torch.randn([1] + list(out_shape)[1:], requires_grad=False, device=device)
        sample_y0_ = None
        sample_res = get_response_from_layer(net_, layer_i, sample_x_, sample_y0_)
        print("Sample res from digital calculation: {}".format(sample_res))

        # ODE layer configurations
        cls, t_step = net_.PcConvs[0].cls, net_.PcConvs[0].lr
        t_end = cls * t_step
        ts_scale = 1
        delta_t = 0.0 # deviation from t_end, set within ~-/+0.2
        # Method selected from adaptive method: ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']
        # fixed method: ['euler', 'midpoint', 'heun2', 'heun3', 'rk4', 'explicit_adams', 'implicit_adams', 'fixed_adams']
        # t_step is meaning for fixed method
        ode_params_ = {"method": "dopri5", "t_end": t_end + delta_t, "tol": 1e-3, "t_step": t_step / ts_scale}
        sample_ode_res = get_response_from_ode_layer(net_, layer_i, ode_params_, sample_x_, sample_y0_)
        print("Sample res from ode calculation: {}".format(sample_ode_res))

if __name__ == "__main__":
    run_test()
