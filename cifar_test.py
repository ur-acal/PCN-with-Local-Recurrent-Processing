'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PcConvBp_SGD(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, cls=0, bias=False, num_iterations=5):
        super().__init__()
        self.num_iterations = num_iterations
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = inchan
        self.C_out = outchan
        self.FFconv = nn.Conv2d(inchan, outchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, outchan, 1, 1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x, layer_idx):
        y = self.relu(self.FFconv(x))
        y, energies = self.find_optimal_r(x, y, layer_idx)
        y = y + self.bypass(x)
        return y

    def find_optimal_r(self, x, y, layer_idx):
        expanded_weights = torch.load(f'./expanded_weights/expanded_weights_{layer_idx}.pt')
        expanded_weights = expanded_weights.to(y.device)
        flattened_x = torch.flatten(x, start_dim=1).clone().detach()

        """
        Implement with SGD
        """
        # Initialize flattened_y as a tensor with requires_grad=True
        y = F.pad(y, (self.padding, self.padding, self.padding, self.padding))
        flattened_y = torch.flatten(y, start_dim=1).clone().detach().requires_grad_(True)
        energy_record = []
        optimizer = torch.optim.SGD([flattened_y], lr=0.01)
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            # energy = self.Energy_Function(flattened_x, expanded_weights, flattened_y)
            energy = torch.norm(flattened_x - flattened_y @ expanded_weights.T, p=2)
            energy.backward()
            optimizer.step()
            energy_record.append(energy.item())

        # Reshape the flattened_y to the original shape
        _, C_in, H_in, W_in = y.shape
        H_out = (H_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        optimal_y = flattened_y.view(-1, self.C_out, H_out, W_out)
        del flattened_y, flattened_x, energy, expanded_weights
        # Cut off the padding area
        optimal_y = optimal_y[:, :, self.padding:-self.padding, self.padding:-self.padding]
        optimal_y = optimal_y.to(y.device)
        
        return optimal_y.detach(), energy_record

    def Energy_Function(self, x, W, y):
        energy = torch.sqrt(x @ x.T -2* x @ W @ y.T + y @ (W.T @ W) @ y.T)
        return energy
    
    def expand_weights_to_matrix(self, input_shape, weight_tensor, stride=1, padding=0):
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
    
    
''' Architecture PredNetBpD '''
from prednet import PcConvBp
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False, solver=None, layer_number=None, num_iterations=None):
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            if solver is None:
                print('No solver in used, still using convolution in recurrent layer')
                assert layer_number is None, 'layer_number must be None if solver is None'
                self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
            elif solver == 'SGD':
                print(f'Solver {solver} is in use')
                assert layer_number is not None, 'layer_number must be provided if solver is not None'
                assert layer_number <= self.nlays, f'layer_number must be less than or equal to the number of layers: {self.nlays}'
                self.PcConvs = nn.ModuleList()
                for i in range(self.nlays):
                    # if i <= (layer_number-1):
                    if i == (layer_number-1):
                        self.PcConvs.append(PcConvBp_SGD(self.ics[i], self.ocs[i], cls=self.cls, num_iterations=num_iterations))
                    else:
                        self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i], cls=self.cls))
            else:
                print(f'Solver {solver} not supported')
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])

        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
      


if __name__ == '__main__':
    batchsize = 500
    test_ratio = 1
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    num_samples = len(testset)
    subset_size = int(test_ratio * num_samples)

    # Create a subset of the test set
    indices = list(range(num_samples))
    subset_indices = indices[:subset_size]
    test_subset = Subset(testset, subset_indices)

    # Create a DataLoader for the subset
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=batchsize, shuffle=True, num_workers=6)

    # Create an instance of the PredNetBpD class
    checkpoint_weight = torch.load('checkpoint/PredNetBpD_5CLS_FalseNes_0.001WD_FalseTIED_1REP_best_ckpt.t7')
    prednet = PredNetBpD(num_classes=10, cls=5, Tied=False, solver='SGD', layer_number=1, num_iterations=1)
    prednet = nn.DataParallel(prednet)
    prednet.load_state_dict(checkpoint_weight['net'])
    prednet = prednet.cuda()
    prednet.eval()
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        output_tensor = prednet(inputs)
        # Get the predicted class
        _, predicted = torch.max(output_tensor, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        print(f' Temporal Accuracy: {100 * correct / total:.2f}%')

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')