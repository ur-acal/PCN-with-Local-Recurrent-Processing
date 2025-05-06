import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvDS(nn.Module):
    """
    Implementing forward pass of normal convolutional layer using gradient descent.
    """
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__()
        pass

    def forward(self, x):
        pass