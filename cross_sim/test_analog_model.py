"""
Parameterizable inference simulation script for CIFAR-10 ResNets.
"""
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
__package__ = "cross_sim"

from copy import deepcopy

import torch
from torchvision import datasets, transforms
import numpy as np
import warnings, sys, time
import pickle, copy
import tqdm
from .build_resnet_cifar10 import ResNet_cifar10
warnings.filterwarnings('ignore')
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from .find_adc_range import find_adc_range
from .dnn_inference_params import dnn_inference_params
from .cross_bar_params import base_params_args


def test_analog_model(analog_model, Nruns, N, device, batch_size, data_loader=None):
    cifar10_dataloader = data_loader
    if cifar10_dataloader is None:
        #### Load and transform CIFAR-10 dataset
        normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225])
        dataset = datasets.CIFAR10(root='../../data',train=False, download=True,
            transform= transforms.Compose([transforms.ToTensor(), normalize]))
        dataset = torch.utils.data.Subset(dataset, np.arange(N))
        cifar10_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    #### Run inference and evaluate accuracy
    accuracies = np.zeros(Nruns)
    for m in range(Nruns):
        with torch.no_grad():
            T1 = time.time()
            y_pred, y, k = np.zeros(N), np.zeros(N), 0
            for inputs, labels in cifar10_dataloader:
                inputs = inputs.to(device)
                output = analog_model(inputs)
                output = output.to(device)
                y_pred_k = output.data.cpu().detach().numpy()
                if batch_size == 1:
                    y_pred[k] = y_pred_k.argmax()
                    y[k] = labels.cpu().detach().numpy()
                    k += 1
                else:
                    batch_size_k = y_pred_k.shape[0]
                    y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
                    y[k:(k+batch_size_k)] = labels.cpu().detach().numpy()
                    k += batch_size_k
                print("Image {:d}/{:d}, accuracy so far = {:.2f}%".format(
                    k, N, 100*np.sum(y[:k] == y_pred[:k])/k), end="\r")

            T2 = time.time()
            top1 = np.sum(y == y_pred)/len(y)
            accuracies[m] = top1
            print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
            print('Accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
            if m < (Nruns - 1):
                reinitialize(analog_model)

    if Nruns > 1:
        print("==========")
        print("Mean accuracy:  {:.2f}%".format(100*np.mean(accuracies)))
        print("Stdev accuracy: {:.2f}%".format(100*np.std(accuracies)))

    return 100*np.mean(accuracies), 100*np.std(accuracies), accuracies


def run_resnet(n=9, Nruns=10, noise_level=0.0, proportional_error=True, digital_bias=False, ideal=False,
               weight_bits=8, input_bits=8, adc_bits=0, bias_rows=0):
    useGPU = True # use GPU?
    N = 10000 # number of images
    batch_size = 256
    Nruns = Nruns

    depth = 6*n+2
    print("Model: ResNet-{:d}".format(depth))
    print("CIFAR-10: using "+("GPU" if useGPU else "CPU"))
    print("Number of images: {:d}".format(N))
    print("Number of runs: {:d}".format(Nruns))
    print("Batch size: {:d}".format(batch_size))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

    ##### Load Pytorch model
    resnet_model = ResNet_cifar10(n)
    resnet_model = resnet_model.to(device)
    resnet_model.load_state_dict(
        torch.load('./models/resnet{:d}_cifar10.pth'.format(depth),
        map_location=torch.device(device)))
    resnet_model.eval()
    n_layers = len(convertible_modules(resnet_model))

    ##### Set the simulation parameters

    # Create a list of CrossSimParameters objects
    params_list = [None] * n_layers

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
        'useGPU' : useGPU
    })

    ### Load input limits
    input_ranges = np.load("./calibrated_config/input_limits_ResNet{:d}.npy".format(depth))

    ### Load ADC limits
    adc_ranges = find_adc_range(params_args, n_layers, depth)

    ### Set the parameters
    for k in range(n_layers):
        params_args_k = params_args.copy()
        params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
        params_args_k['input_range'] = input_ranges[k]
        params_args_k['adc_range'] = adc_ranges[k]
        params_list[k] = dnn_inference_params(**params_args_k)

    #### Convert PyTorch layers to analog layers
    print("----- Start to convert from torch -----")
    analog_resnet = from_torch(resnet_model, params_list, fuse_batchnorm=True, bias_rows=bias_rows)
    print("----- Successfully converted from torch -----")

    # for _name, _param in analog_resnet.named_parameters():
    #     print("{}: {}".format(_name, _param))

    mean_acc, std_acc, acc_list = test_analog_model(analog_model=analog_resnet, Nruns=Nruns, N=N, device=device,
                                                    batch_size=batch_size)

    return mean_acc, std_acc, acc_list


def get_exp_name(proportional_error, weight_bits, input_bits, adc_bits, bias_rows, noise_level_list):
    if proportional_error:
        exp_name = "ProportionalMismatch_"
    else:
        exp_name = "AdditiveMismatch_"

    exp_name += "{}WeightBits_".format(weight_bits)
    exp_name += "{}InputBits_".format(input_bits)
    exp_name += "{}AdcBits_".format(adc_bits)
    exp_name += "{}BiasRows_".format(bias_rows)
    exp_name += "{}".format(",".join(str(_).replace(".", "p") for _ in noise_level_list))

    return exp_name


def run_exp_config(prop_error, n_weight_bits, n_input_bits, n_adc_bits, n_bias_rows):
    n_trials = 20
    noise_level_list_ = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                         0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20,
                         .25, .30, .35, .40]
    n_layer_list = [3, 5, 9]
    # noise_level_list_ = [0, 0.4]
    model_acc_log, model_acc_list = {}, {}
    for _n in n_layer_list:
        acc_log_dict, acc_list_dict = {}, {}
        for _nl in noise_level_list_:
            _n_trials = n_trials if _nl > 0.0 else 1
            _mean, _std, _acc = run_resnet(n=_n, Nruns=_n_trials, noise_level=_nl, proportional_error=prop_error,
                                           ideal=False, weight_bits=n_weight_bits, input_bits=n_input_bits,
                                           adc_bits=n_adc_bits, bias_rows=n_bias_rows)
            acc_log_dict[_nl] = "{:.2f} ± {:.2f}".format(_mean, _std)
            acc_list_dict[_nl] = _acc
        with open("models/acc_spec/resnet{}_{}_acc.pkl".format(6*_n+2, noise_level_list_), "wb") as fp:
            pickle.dump(acc_list_dict, fp)
        model_acc_log["resnet{}".format(6*_n+2)] = acc_log_dict
        model_acc_list["resnet{}".format(6*_n+2)] = acc_list_dict

    pkl_name = get_exp_name(proportional_error=prop_error, weight_bits=n_weight_bits, input_bits=n_input_bits,
                            adc_bits=n_adc_bits, bias_rows=n_bias_rows, noise_level_list=noise_level_list_)
    with open("models/acc_spec/{}.pkl".format(pkl_name), "wb") as fp:
        pickle.dump(model_acc_list, fp)
        print("All model acc list saved to: {}".format(fp.name))

    for _n in n_layer_list:
        cur_model_name = "resnet{}".format(6*_n+2)
        print("-------- Model name: {} --------".format(cur_model_name))
        for _nl, _acc in model_acc_log[cur_model_name].items():
            print("Noise level: {}, Acc: {}".format(_nl, _acc))

if __name__ == "__main__":
    ## Depth parameter for model selection
    # Follows definition in original ResNet paper (He et al, CVPR 2016)
    # n = 2 : ResNet-14 (175K weights)
    # n = 3 : ResNet-20 (272K weights)
    # n = 5 : ResNet-32 (467K weights)
    # n = 9 : ResNet-56 (856K weights)

    ## Digital bias control
    # bias digital or not seems to be controlled by bias_rows only. Bias row = 0 means digital bias.
    # After setting bias_row = 1:
    # 1. digital_bias argument seems to have no effect
    # 2. setting adc_bits to non-zero values (<8) will decrease the acc
    # Setting bias_row > 1 can increase acc

    test_only = True
    if test_only:
        run_resnet(n=9, Nruns=1, noise_level=0.3, proportional_error=True, ideal=False,
                   weight_bits=8, input_bits=8, adc_bits=0, bias_rows=1)
        exit(0)

    ###############################
    ## Configurations
    # prop_error_ = True
    n_weight_bits_ = 8
    n_input_bits_ = 8
    n_adc_bits_ = 0
    # n_bias_rows_ = 0
    ###############################

    for _pe in [True, False]:
        for nbr_ in [0, 1]:
            run_exp_config(prop_error=_pe, n_weight_bits=n_weight_bits_, n_input_bits=n_input_bits_,
                           n_adc_bits=n_adc_bits_, n_bias_rows=nbr_)
