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
from inference_utils import load_and_prepare_model, expand_and_save_weights, plot_layer_pcn_loss, run_noise_experiment
from inference_utils import run_lr_cls_experiment, get_val_scale


def get_test_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=2)
    return test_loader

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCConvNoisy tests with custom noise parameters"
    )
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Directory containing the saved model checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Identifier or filename of the model to load")
    parser.add_argument("--weight", type=str, default=None,
                        help="The large dir that holds expanded weights")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="Where to save the PCN loss plot")
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
    parser.add_argument("--noise_to_bn", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="Noise to batch norm for PCN")
    parser.add_argument("--noise_to_linear", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="Noise to linear layer for PCN")
    parser.add_argument("--diff_noise", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=False, help="If True, set different noise for each FF/FB call in recurrence")
    parser.add_argument("--fuse_bn", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True, help="Fuse batch norm into conv")
    parser.add_argument("--val_scale_frac", type=float,
                        default=0.0, help="Fraction of train samples used to calculate the value scaler")
    parser.add_argument("--val_scale", type=float,
                        default=0.0, help="Input value scaler")
    parser.add_argument("--pc_conv", type=str, choices=list(PC_CONV_CLASS.keys())+[None],
                   default=None)
    parser.add_argument("--noisy_test", type=lambda v: v.lower() in ('yes', 'true', 't', '1'),
                        default=True)
    parser.add_argument("--test_only", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=False)
    return parser.parse_args()

def run_test():
    expand_weights = False
    loss_plot = False

    args = parse_args()
    if args.test_only:
        # set level in the very beginning before calling logging.warning, otherwise the line below will not work
        logging.basicConfig(level=logging.INFO)
    noisy_args = ["w_type", "noise_to_ff", "noise_to_bp", "tie_noise", "tie_noise_bp", "diff_noise"] # skip plotting for noisy exp
    logging.warning("Running test with parameters:")
    noisy_params = {}
    for name, val in vars(args).items():
        logging.warning(f"  {name}: {val}")
        if name in noisy_args:
            noisy_params[name] = val
    noisy_params["weight"] = os.path.join(args.weight, args.model_name) if expand_weights else None
    # noise_level and plot_path are passed in separately when calling the plot function or noise_exp function

    # Get pc_conv_layer to use
    pc_conv = PC_CONV_CLASS.get(args.pc_conv, PCConvNoisy)
    logging.warning("----- Using PC Conv layer: {} -----".format(pc_conv.__name__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = get_test_data()
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_best_ckpt.pth")

    # DO NOT USE print with logging, the outputs of the two will be out-of-order
    # IF have to use, run with python -u run_test.py
    with torch.no_grad():
        if args.test_only:
            logging.info("----- Running one forward pass for model: {} -----".format(args.model_name))
            noisy_params["noise_level"] = 0.4
            noisy_params["weight"] = None
            net_ = load_and_prepare_model(model_path=ckpt_path, device=device, model_struct=PCNet,
                                          pc_conv_layer=pc_conv, data_parallel=False,
                                          noise_to_bn=args.noise_to_bn, noise_to_linear=args.noise_to_linear,
                                          fuse_bn=args.fuse_bn, **noisy_params)
            net_.eval()
            test_batch = next(iter(test_dataloader))[0].to(device)[:512]
            logging.info("===== Before the input are scaled =====")
            _ = net_(test_batch)
            _, predicted_raw = torch.max(_, 1)

            # _max_scale = net_.get_max_hidden_val(test_batch)
            # _max_scale = 1
            # logging.info("===== After the input are scaled =====")
            # test_batch_scaled = test_batch / args.val_scale
            # _ = net_(test_batch_scaled)
            # _, predicted = torch.max(_, 1)
            # logging.info("Prediction: {}".format(predicted_raw))
            # logging.info("Prediction scaled: {} with scaler={}".format(predicted, args.val_scale))
            # logging.info("=====> Prediction acc after scaling: {}".format(
            #     torch.sum(predicted_raw == predicted) / len(predicted_raw)))
            # logging.info("Output shape: {}".format(_.shape))
            # logging.info(_)
            logging.info("===== Inspecting the range of the weights =====")
            for _name, _p in net_.named_parameters():
                print("Name: {}, max: {}, min: {}, median: {}, mean: {}".format(
                    _name, _p.max(), _p.min(), _p.median(), _p.mean()))
            for _name, _buf in net_.named_buffers():
                if "beta" in _name:
                    print("Name: {}, val: {}".format(_name, _buf))
            exit(0)

        # Get val_scale
        noisy_params_vs = deepcopy(noisy_params)
        noisy_params_vs.update({"noise_level": 0.0, "weight": None})
        if args.val_scale != 0.0:
            val_scale = args.val_scale
        else:
            val_scale = get_val_scale(model_path=ckpt_path, device=device, model_struct=PCNet, pc_conv_layer=pc_conv,
                                      data_parallel=False, noise_to_bn=args.noise_to_bn,
                                      noise_to_linear=args.noise_to_linear, fuse_bn=args.fuse_bn,
                                      val_scale_frac=args.val_scale_frac, **noisy_params_vs)

        # works with saved init_args
        if expand_weights:
            expand_and_save_weights(next(iter(test_dataloader))[0], model_path=ckpt_path, device=device,
                                    model_struct=PCNet, pc_conv_layer=pc_conv, data_parallel=False,
                                    noise_to_bn=args.noise_to_bn, noise_to_linear=args.noise_to_linear,
                                    weight_dir=args.weight, model_name=args.model_name)
        # plot_path specified inside
        if loss_plot:
            plot_layer_pcn_loss(next(iter(test_dataloader))[0], model_path=ckpt_path, device=device,
                                model_struct=PCNet, pc_conv_layer=pc_conv, data_parallel=False,
                                noise_to_bn=args.noise_to_bn, noise_to_linear=args.noise_to_linear,
                                loss_plot_dir=args.plot_path, model_name=args.model_name)

        noise_level_list_ = [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40]
        # noise_level_list_ = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
        #                      0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20,
        #                      .25, .30, .35, .40]
        # noise_level_list_ = [0, 0.1, .20, .30, .40]
        noisy_trials = 20
        if not args.noisy_test:
            scale_factor = [1, 2, 4, 6, 8, 10, 12, 16]
            # scale_factor = [0] # Not use PC
            _ = run_lr_cls_experiment(ckpt_path, test_dataloader, noise_level_list=noise_level_list_,
                                      model_struct=PCNet, pc_conv_layer=pc_conv, data_parallel=False,
                                      device=device, noisy_trials=noisy_trials, model_name=args.model_name,
                                      noise_to_bn=args.noise_to_bn, noise_to_linear=args.noise_to_linear,
                                      fuse_bn=args.fuse_bn, scale_factor=scale_factor, val_scale=val_scale,
                                      **noisy_params)
        else:
            # specify noise level inside, plot path omitted
            _ = run_noise_experiment(ckpt_path, test_dataloader, noise_level_list=noise_level_list_,
                                     model_struct=PCNet, pc_conv_layer=pc_conv, data_parallel=False,
                                     device=device, noisy_trials=noisy_trials, model_name=args.model_name,
                                     noise_to_bn=args.noise_to_bn, noise_to_linear=args.noise_to_linear,
                                     fuse_bn=args.fuse_bn, val_scale=val_scale, **noisy_params)

if __name__ == "__main__":
    run_test()
