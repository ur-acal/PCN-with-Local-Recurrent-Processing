import torch
import torch.nn as nn
import torchvision
import os
import inspect
import sys

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from cifar_test import PredNetBpD
from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy

import logging
log = logging.getLogger(__name__)
log.propagate = False

# attach a handler that only prints the message
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


def load_and_prepare_model(model_path, device, model_struct=PredNetBpD, pc_conv_layer=PCConvNoisy,
                           data_parallel=True, **kwargs):
    # Todo: When loading the model, filter out unused init_args that does not exist in pc_conv_layer, e.g. tie_frac...
    checkpoint_weight = torch.load(model_path, map_location=device)  # weights_only=False

    # check if model_struct is the old model defined or the new one
    sig = inspect.signature(model_struct.__init__)
    init_kwargs = dict(kwargs)
    if "pc_conv_layer" in sig.parameters and pc_conv_layer is not None:
        init_kwargs["pc_conv_layer"] = pc_conv_layer

    # check if model init kwargs are stored in checkpoint, instead of passing in as arguments
    if "init_args" in checkpoint_weight:
        # if yes, then kwargs should contain only the parameters related to noise
        init_kwargs = {
            **init_kwargs,
            **checkpoint_weight["init_args"]["model_args"],
            **checkpoint_weight["init_args"]["kwargs"]
        }

    net_ = model_struct(**init_kwargs)
    net_ = net_.to(device)
    if data_parallel:
        net_ = nn.DataParallel(net_)
        net_.load_state_dict(checkpoint_weight['net'])
        net_ = net_.module
    else:
        net_.load_state_dict(checkpoint_weight['net'])

    # Add noise
    if hasattr(net_, "add_noise"):
        net_.add_noise()
    log.warning("----- Model loaded -----")
    return net_


def expand_and_save_weights(sample_imgs, model_path, device="cpu", model_struct=PredNetBpD, pc_conv_layer=PCConvNoisy,
                            data_parallel=True, weight_dir="expanded_weights", model_suffix=".pt", model_name=None, **kwargs):
    log.warning("----- Start to expand and save weights -----")
    # Load model
    net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel, **kwargs)
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


def plot_layer_pcn_loss(sample_imgs, model_path, device="cpu", model_struct=PredNetBpD, pc_conv_layer=PCConvNoisy,
                        data_parallel=True, loss_plot_dir="loss_plot", model_suffix=".t7", model_name=None, **kwargs):
    log.warning("----- Start to plot layer PCN loss -----")
    noise_level = kwargs.get("noise_level", 0.0)
    if model_name is not None:
        loss_plot_dir = os.path.join(loss_plot_dir, model_name, "noise_level_{}".format(noise_level))
    else:
        loss_plot_dir = os.path.join(
            loss_plot_dir, model_path.split('/')[-1].split(model_suffix)[0], "noise_level_{}".format(noise_level))
    os.makedirs(loss_plot_dir, exist_ok=True)
    kwargs.update({"plot_path": loss_plot_dir})
    net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel, **kwargs)
    net_.eval()
    _ = net_(sample_imgs.to(device))
    log.warning("----- Loss is plotted -----")


def run_noise_experiment(model_path, test_loader, noise_level_list, device="cpu", model_struct=PredNetBpD,
                         pc_conv_layer=PCConvNoisy, data_parallel=True, noisy_trials=10, model_name=None, **kwargs):
    noise_acc = {}
    for noise_level in noise_level_list:
        trials = noisy_trials if noise_level > 0 else 1
        acc_list = []
        for t in range(trials):
            # reinitialize net with different noise during each trial
            params_ = deepcopy(kwargs)
            params_.update({"noise_level": noise_level})
            net_ = load_and_prepare_model(model_path, device, model_struct, pc_conv_layer, data_parallel, **params_)
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
