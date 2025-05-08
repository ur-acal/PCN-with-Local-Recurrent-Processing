import torch
import torch.nn as nn
import torchvision
import os
import argparse

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from pc_conv import PCConvNoisy, PCConv
from pc_model import PCNet
from inference_utils import load_and_prepare_model, expand_and_save_weights, plot_layer_pcn_loss, run_noise_experiment


def get_test_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4096 * 2, shuffle=False, num_workers=2)
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
    return parser.parse_args()

def run_test():
    args = parse_args()
    noisy_args = ["w_type", "noise_to_ff", "noise_to_bp"] # skip plotting for noisy exp
    print("Running test with parameters:")
    noisy_params = {}
    for name, val in vars(args).items():
        print(f"  {name}: {val}")
        if name in noisy_args:
            noisy_params[name] = val
    noisy_params["weight"] = os.path.join(args.weight, args.model_name)
    # noise_level and plot_path are passed in separately when calling the plot function or noise_exp function

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = get_test_data()
    ckpt_path = os.path.join(args.model_dir, args.model_name, args.model_name + "_best_ckpt.pth")

    with torch.no_grad():
        # works with saved init_args
        expand_and_save_weights(next(iter(test_dataloader))[0], model_path=ckpt_path, device=device,
                                model_struct=PCNet, pc_conv_layer=PCConvNoisy, data_parallel=False,
                                weight_dir=args.weight, model_name=args.model_name)
        # plot_path specified inside
        plot_layer_pcn_loss(next(iter(test_dataloader))[0], model_path=ckpt_path, device=device,
                            model_struct=PCNet, pc_conv_layer=PCConvNoisy, data_parallel=False,
                            loss_plot_dir=args.plot_path, model_name=args.model_name)

        # specify noise level inside, plot path omitted
        noise_level_list_ = [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40]
        run_noise_experiment(ckpt_path, test_dataloader, noise_level_list=noise_level_list_,
                             model_struct=PCNet, pc_conv_layer=PCConvNoisy, data_parallel=False,
                             device=device, noisy_trials=5, **noisy_params)

if __name__ == "__main__":
    run_test()
