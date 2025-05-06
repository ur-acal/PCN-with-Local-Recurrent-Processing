import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pc_conv import PCConv, PCConvNoisy

class PCNet(nn.Module):
    def __init__(self, inp_channels, out_channels, num_classes=10, **kwargs):
        super().__init__()
        self.ics = inp_channels # input channels
        self.ocs = out_channels # output channels
        self.max_pool = [False, True, False, True, False] # downsample flag
        self.num_layers = len(self.ics)

        # PC recurrent layers
        self.PCConv_layers = nn.ModuleList(
            [PCConv(inp_chan=self.ics[i], out_chan=self.ocs[i], **kwargs) for i in range(self.num_layers)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.num_layers)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BN_end = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.BNs[i](x)
            x = self.PCConv_layers[i](x, i)  # ReLU + Conv
            if self.max_pool[i]:
                x = self.max_pool2d(x)

        # classifier
        out = F.avg_pool2d(self.relu(self.BN_end(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
