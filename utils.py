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
import numpy as np
import torch.nn as nn
import torch.nn.init as init


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


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

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

class BRIMconfig:
    def __init__(
        self,
        R=31e3,
        C=49e-15,
        temperature_start=0.5, # 2.4
        temperature_end=0.5, #0.5
        t_step=2.2e-11,
        t_stop=10e-9,
        h2v=True,
        fixed_temperature=False
    ) -> None:
        self.R: float = R
        self.C: float = C
        if fixed_temperature:
            self.temperature_start: float = temperature_start
            self.temperature_end: float = temperature_start
        else:
            self.temperature_start: float = temperature_start
            self.temperature_end: float = temperature_end
        self.t_step: float = t_step
        self.t_stop: float = t_stop
        self.h2v: bool = h2v


def absv(spin):
    return torch.sign(spin)


def make_equivalent_ising(couplings: torch.tensor,
                          visible_bias: torch.tensor,
                          hidden_bias: torch.tensor,
                          visible_binary: torch.tensor,
                          hidden_binary: torch.tensor):
    
    J_new = couplings / 4
    a_new = couplings.sum(axis=1) / 4 + visible_bias / 2
    b_new = couplings.sum(axis=0) / 4 + hidden_bias / 2
    offset = (couplings / 4).sum() +\
        visible_bias.sum() / 2 + hidden_bias.sum() / 2
    visible_spins = 2 * visible_binary - 1
    hidden_spins = 2 * hidden_binary - 1
    
    return J_new, a_new, b_new, offset, visible_spins, hidden_spins


def format_check(W: torch.tensor, 
                 visible: torch.tensor, 
                 hidden: torch.tensor, 
                 visible_bias: torch.tensor, 
                 hidden_bias: torch.tensor):
    offset = 0
    convert_flag = 0
    # convert to spin if input are binary
    if (torch.all((visible == 0) | (visible == 1)) and torch.all((hidden == 0) | (hidden == 1))):
        W, visible_bias, hidden_bias, offset, visible, hidden =\
        make_equivalent_ising(W, visible_bias, hidden_bias, visible, hidden)
        convert_flag = 1
    # pass if input are spin
    elif (torch.all((visible == -1) | (visible == 1)) and torch.all((hidden == -1) | (hidden == 1))):
        convert_flag = 0
    # not support other input
    else:
        raise ValueError("visible and hidden are not binary or spin")
    
    return W, visible, hidden, visible_bias, hidden_bias, offset, convert_flag


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_logprob(W,  # (num_visible, num_hidden)
                      visible_bias,  # (1, num_visible)
                      hidden_bias,  # (1, num_hidden)
                      logZ,
                      data,  # (num_samples, num_visible)
                      offset):
    print(f'weight: {W.shape}, visible_bias: {visible_bias.shape}, hidden_bias: {hidden_bias.shape}')
    hidden = (np.sum(_sigmoid((hidden_bias + np.matmul(data, W))), axis=0) / len(data))[None, :]
    visible = (np.sum(data, axis=0) / len(data))[None, :]
    expectation_energy = -(visible @ W @ hidden.T + visible @ visible_bias.T + hidden_bias @ hidden.T)
    logprob = - expectation_energy - logZ
    return logprob

def normalize_and_quantize(spin, quantize):
    normalized_spin = (spin - spin.min()) / (spin.max() - spin.min()) * 2 - 1
    normalized_spin = ((normalized_spin + 1) / 2 * (quantize - 1)).round() / (quantize - 1) * 2 - 1
    return normalized_spin