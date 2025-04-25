import torch
import torch.nn as nn
import torchvision
import os

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from cifar_test import PredNetBpD


def load_and_prepare_model(model_path, device, model_struct=PredNetBpD, **kwargs):
    checkpoint_weight = torch.load(model_path, map_location=device)  # weights_only=False
    net_ = model_struct(**kwargs)
    net_ = net_.to(device)
    net_ = nn.DataParallel(net_)
    net_.load_state_dict(checkpoint_weight['net'])
    net_ = net_.module
    return net_


def expand_and_save_weights(sample_imgs, model_path, device="cpu", model_struct=PredNetBpD,
                            weight_dir="expanded_weights", model_suffix=".t7", **kwargs):
    # Load model
    net_ = load_and_prepare_model(model_path, device, model_struct, **kwargs)
    # Save expanded weights
    weight_path = os.path.join(weight_dir, model_path.split('/')[-1].split(model_suffix)[0])
    os.makedirs(weight_path, exist_ok=True)
    # save_expanded_weights(prednet, sample_imgs.to(device), weight_path)
    net_.save_expanded_weights(sample_imgs.to(device), weight_path)


def plot_layer_pcn_loss(sample_imgs, model_path, device="cpu", model_struct=PredNetBpD,
                        loss_plot_dir="loss_plot", model_suffix=".t7", **kwargs):
    noise_level = kwargs.get("noise_level", 0.0)
    loss_plot_dir = os.path.join(
        loss_plot_dir, model_path.split('/')[-1].split(model_suffix)[0], "noise_level_{}".format(noise_level))
    os.makedirs(loss_plot_dir, exist_ok=True)
    kwargs.update({"plot_path": loss_plot_dir})
    net_ = load_and_prepare_model(model_path, device, model_struct, **kwargs)
    net_.eval()
    _ = net_(sample_imgs.to(device))


def run_noise_experiment(model_path, test_loader, noise_level_list, device="cpu", model_struct=PredNetBpD,
                         noisy_trials=10, **kwargs):
    noise_acc = {}
    for noise_level in noise_level_list:
        trials = noisy_trials if noise_level > 0 else 1
        acc_list = []
        for t in range(trials):
            # reinitialize net with different noise during each trial
            params_ = deepcopy(kwargs)
            params_.update({"noise_level": noise_level})
            net_ = load_and_prepare_model(model_path, device, model_struct, **params_)
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
            print(f'Test Accuracy at noise level {noise_level}: {accuracy:.2f}%')
        avg_acc = sum(acc_list) / len(acc_list)
        noise_acc[noise_level] = avg_acc
        print("Average test acc over {} trials is {}".format(trials, avg_acc))
    print("-------- Final Result --------")
    print(noise_acc)


if __name__ == '__main__':
    batch_size = 4096 * 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    num_samples = len(test_set)

    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # set different model initialization parameters here
    model_path_ = "checkpoint/PredNetBpD_5_30CLS_FalseNes_0.001WD_FalseTIED_4REP_best_ckpt.t7"
    weight_dir_ = "exp_weight_test_"
    loss_plot_dir_ = "loss_plt_test_"
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
