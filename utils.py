'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.init as init


def load_res_vs_vin(dir_path=os.path.dirname(os.path.abspath(__file__)), R=50e3, R_max=180e3,
                    dtype=torch.float32, device="cpu"):
    data_dir = os.path.join(dir_path, "hardware_data", "res_vs_vin_{}k_{}k.csv".format(
        str(int(R/1e3)), str(int(R_max/1e3))))
    if not os.path.exists(data_dir):
        return None, None, None
    df = pd.read_csv(data_dir)
    v_grid = df[df.columns[0]].values
    R_codes = [float(_) for _ in list(df.columns)[1:]]
    R_table = df[list(df.columns)[1:]].values

    return (
        torch.tensor(v_grid, dtype=dtype, device=device),
        torch.tensor(R_codes, dtype=dtype, device=device),
        torch.tensor(R_table, dtype=dtype, device=device)
    )


def expand_weights_to_matrix(input_shape, weight_tensor, stride=1, padding=0, flip_weight=False):
    if flip_weight:
        weight_tensor = weight_tensor.flip([2, 3])

    C_in, H_in, W_in = input_shape
    C_out, _, K, _ = weight_tensor.shape

    # Compute output dimensions
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    # List to store sparse indices and values
    indices = []
    values = []

    for c_out in range(C_out):
        for h in range(H_out):
            for w in range(W_out):
                start_h = h * stride
                start_w = w * stride
                filter_idx = c_out * H_out * W_out + h * W_out + w
                for c_in in range(C_in):
                    for i in range(K):
                        for j in range(K):
                            # c_in * (H_in + 2 * padding) * (W_in + 2 * padding)
                            # + (start_h + i) * (W_in + 2 * padding) + (start_w + j)
                            input_idx = (c_in * (H_in + 2 * padding) + (start_h + i)) * (W_in + 2 * padding) + (start_w + j)
                            value = weight_tensor[c_out, c_in, i, j].item()
                            if value != 0:
                                indices.append([filter_idx, input_idx])
                                values.append(value)
    # Convert to sparse tensor
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float32)
    size = (C_out * H_out * W_out, C_in * (H_in + 2 * padding) * (W_in + 2 * padding))
    expanded_weights = torch.sparse_coo_tensor(indices, values, size=size)

    return expanded_weights


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 77

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


