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
    parser.add_argument("--noise_level", type=float, default=None,
                        help="Noise level to apply")
    parser.add_argument("--weight", type=str, default=None,
                        help="Path to a weight override (.pth file), or leave None")
    parser.add_argument("--layer_idx", type=int, default=None,
                        help="Which layer index to inject noise into")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="Where to save any diagnostic plots")
    parser.add_argument("--w_type", type=str, default="fb_flip",
                        help="Type of weight perturbation")
    parser.add_argument("--noise_to_ff", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=True, help="Noise to feed-forward?")
    parser.add_argument("--noise_to_bp", type=lambda v: v.lower() in ('yes','true','t','1'),
                        default=True, help="Noise to bypass?")
    return parser.parse_args()

def run_test():
    args = parse_args()
    noisy_args = ["noise_level", "weight", "layer_idx", "plot_path", "w_type", "noise_to_ff", "noise_to_bp"]
    print("Running test with parameters:")
    noisy_params = {}
    for name, val in vars(args).items():
        print(f"  {name}: {val}")
        if name in noisy_args:
            noisy_params[name] = val

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = get_test_data()
    ckpt_path = os.path.join(args.model_dir, args.model_name)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = PCNet(pc_conv_layer=PCConv,
                   **ckpt["init_args"]["model_args"],
                   **ckpt["init_args"]["kwargs"])
    model.load_state_dict(ckpt["net"])

if __name__ == "__main__":
    run_test()

def run_test():
    noisy_params = {
        "noise_level": None, "weight": None, "layer_idx": None,
        "plot_path": None, "w_type": "fb_flip",
        "noise_to_ff": True, "noise_to_bp": True}
