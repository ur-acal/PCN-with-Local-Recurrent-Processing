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
from scipy.optimize import dual_annealing
import numpy as np
from utils import expand_weights_to_matrix, save_expanded_weights
import matplotlib.pyplot as plt

class PcConvBp_DS(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, lr=1e-2, bias=False,
                 solver='SGD', num_iterations=5, train_weight=False, noise_level=None, weight_path=None,
                 layer_idx=None, plot_path=None):
        super().__init__()
        self.noise_level = noise_level
        self.solver = solver
        self.train_weight = train_weight
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = inchan
        self.C_out = outchan
        self.FFconv = nn.Conv2d(inchan, outchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, outchan, 1, 1))])
        self.relu = nn.ReLU(inplace=True)
        self.num_iterations = num_iterations
        self.lr = lr
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)
        self.noise_ff_matrix = torch.randn_like(self.FFconv.weight) * (0.0 if noise_level is None else noise_level)
        self.noise_fb_matrix = torch.randn_like(self.FBconv.weight) * (0.0 if noise_level is None else noise_level)

        self.plot_path = plot_path
        self.weight_path = weight_path
        self.layer_idx = layer_idx
        self._load_expanded_weights(noise_level)

    def forward(self, x, layer_idx, w_type="fb_flip"):
        # Todo: should we add noise to bypass and ff convolution?
        noise_ff = (self.noise_ff_matrix.to(device=self.FFconv.weight.device) + 1) * self.FFconv.weight
        y = self.relu(self.FFconv(x))
        # y = self.relu(torch.conv2d(x, noise_ff, padding=self.FFconv.padding))
        y = self.find_optimal_r(x, y, layer_idx, solver=self.solver, w_type=w_type)
        y = y + self.bypass(x)
        return y

    def find_optimal_r(self, x, y, layer_idx, solver, w_type=None):
        noise_ff = (self.noise_ff_matrix.to(device=self.FFconv.weight.device) + 1) * self.FFconv.weight
        noise_fb = self.FBconv.weight * (1 + self.noise_fb_matrix.to(device=self.FFconv.weight.device))

        if w_type is not None:
            assert w_type in {"fb", "fb_flip", "ff", "bp"}
        expanded_weights = self.expanded_weights.get(w_type, "fb_flip")
        flattened_x = torch.flatten(x, start_dim=1).clone().detach()
        if solver == 'SGD':
            """ Implement with SGD """
            y = F.pad(y, (self.padding, self.padding, self.padding, self.padding))
            # Initialize flattened_y as a tensor with requires_grad=True
            expanded_weights = expanded_weights.to(y.device)
            flattened_y = torch.flatten(y, start_dim=1).clone().detach().requires_grad_(True)
            flattened_y.retain_grad()
            optimizer_y = torch.optim.SGD([flattened_y], lr=self.lr)
            optimizer_w = torch.optim.SGD([expanded_weights], lr=self.lr) if self.train_weight else None
            energy_list = []
            for _ in range(self.num_iterations):
                optimizer_y.zero_grad()
                energy = self.Energy_Function(flattened_x, expanded_weights, flattened_y)
                # energy = torch.norm(( (flattened_x - flattened_y @ expanded_weights.T) @ expanded_weights) , p=2)
                # energy = (flattened_x - flattened_y @ expanded_weights.T).T.mm(flattened_x - flattened_y @ expanded_weights.T)
                
                energy.backward()
                optimizer_y.step()
                energy_list.append(energy.item())
                
            if self.train_weight:
                for _ in range(5):
                    optimizer_w.zero_grad()
                    energy = torch.norm(flattened_x - flattened_y @ expanded_weights.T, p=2)
                    energy.backward()
                    optimizer_w.step()
                    torch.save(expanded_weights, f'./expanded_weights_train/expanded_weights_{layer_idx}.pt')

            if self.plot_path is not None:
                plot_save_path = os.path.join(self.plot_path,
                                         'pcn_loss_layer_{}_{}_lr_{}.pdf'.format(layer_idx, w_type, self.lr))
                self.plot_and_save(energy_list, plot_save_path, plot_title=None)
            
        elif solver == 'SA':
            flattened_x_np = flattened_x.cpu().numpy()
            expanded_weights_np = expanded_weights.to_dense().numpy()
            flattened_y_np = torch.flatten(y, start_dim=1).cpu().detach()
            flattened_y_np = flattened_y_np.numpy()

            def e_f(y, x, W):
                energy = np.linalg.norm(x - y @ W.T, ord=2)
                return energy.item()

            # Define bounds for each element in flattened_y_np
            bounds = [(-2.5, 2.5) for _ in range(flattened_y_np.size)]

            result = dual_annealing(e_f, bounds, x0=np.squeeze(flattened_y_np), args=(flattened_x_np, expanded_weights_np), maxiter=self.num_iterations, maxfun=5)
            flattened_y = torch.tensor(result.x, dtype=torch.float32)
            
        elif solver == 'LD':
            def LD(r0, r1, lr=self.lr, sd0=0.5, sd1=0.1):
                # Q is  W.T @ W
                # c is  -2 * r0 @ W
                with torch.no_grad():
                    mom = 0.99
                    x = r0.clone().detach()
                    y = r1.clone().detach()
                    prev_y = y.clone().detach()
                    sd = torch.linspace(sd0, sd1, self.num_iterations)
                    for i in range(self.num_iterations):
                        # Perform sparse matrix multiplication instead of forming Q explicitly
                        error = self.relu(x - torch.conv_transpose2d(y, noise_fb, padding=self.FBconv.padding))
                        y += lr * torch.conv2d(error, noise_ff, padding=self.FFconv.padding)# + np.sqrt(2 * lr) * sd[i] * torch.randn_like(y)
                    return y
            optimal_y = LD(x, y)
            return optimal_y.detach()
        else:
            raise ValueError(f'Solver {solver} not supported')
        
        # Reshape the flattened_y to the original shape
        _, C_in, H_in, W_in = y.shape
        H_out = (H_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        optimal_y = flattened_y.view(-1, self.C_out, H_out, W_out)
        del flattened_y, flattened_x, expanded_weights
        # Cut off the padding area
        optimal_y = optimal_y[:, :, self.padding:-self.padding, self.padding:-self.padding]
        optimal_y = optimal_y.to(y.device)
        
        return optimal_y.detach()

    def _load_expanded_weights(self, noise_level):
        fb_weight = torch.load(os.path.join(self.weight_path, 'expanded_weights_layer_fb_{}.pt'.format(self.layer_idx + 1)))
        fb_weight_flip = torch.load(
            os.path.join(self.weight_path, 'expanded_weights_layer_fb_{}_flip.pt'.format(self.layer_idx + 1)))
        ff_weight = torch.load(
            os.path.join(self.weight_path, 'expanded_weights_layer_ff_{}.pt'.format(self.layer_idx + 1)))
        bp_weight = torch.load(
            os.path.join(self.weight_path, 'expanded_weights_layer_bp_{}.pt'.format(self.layer_idx + 1)))
        self.expanded_weights = {"fb": fb_weight, "fb_flip": fb_weight_flip, "ff": ff_weight, "bp": bp_weight}
        if noise_level is not None:
            for _w_type, _weight in self.expanded_weights.items():
                noise_ = torch.randn_like(_weight) * noise_level
                self.expanded_weights[_w_type] = _weight * (1 + noise_)

    @staticmethod
    def Energy_Function(x, W, y):
        energy = torch.sqrt(x @ x.T -2* x @ W @ y.T + (y @ W.T) @ (W @ y.T))
        return energy

    def plot_and_save(self, energy_list, plot_save_path, plot_title=None):
        plt.rcParams['font.family'] = 'Times New Roman'

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(range(len(energy_list)), [_eng ** 2 for _eng in energy_list])

        # Labeling
        ax.set_xlabel('Iteration', fontname='Times New Roman')
        ax.set_ylabel('PCN loss', fontname='Times New Roman')
        if plot_title:
            ax.set_title(plot_title, fontname='Times New Roman')

        # Improve layout and save
        plt.tight_layout()
        fig.savefig(plot_save_path, format='pdf')
        plt.close(fig)

    
''' Architecture PredNetBpD '''
from prednet import PcConvBp
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, lr=1e-4,
                 solver=None, layer_number=None, num_iterations=None, train_weight=False,
                 noise_level=None, weight_path=None, plot_path=None, pcn_weight_type="fb_flip"):
        super().__init__()
        self.ics = [ 3, 32, 64,  64, 128] # input chanels
        self.ocs = [32, 64, 64, 128, 128] # output chanels
        self.maxpool = [False, True, False, True, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)
        self.pcn_weight_type = pcn_weight_type

        # construct PC layers
        self.solver = solver
        if solver is None:
            print('No solver in used, still using convolution in recurrent layer')
            assert layer_number is None, 'layer_number must be None if solver is None'
            self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls, lr=0.01) for i in range(self.nlays)])
        elif solver in ['SGD', 'SA', 'LD']:
            print(f'Solver {solver} is in use')
            assert layer_number is not None, 'layer_number must be provided if solver is not None'
            assert set(layer_number).issubset(range(self.nlays)), f'layer_numbers must be less than or equal to the number of layers: {self.nlays}'
            self.PcConvs = nn.ModuleList()
            for i in range(self.nlays):
                # if i <= (layer_number-1):
                if i in layer_number:
                    self.PcConvs.append(PcConvBp_DS(self.ics[i], self.ocs[i], lr=lr,
                                                    solver=solver, num_iterations=num_iterations, train_weight=train_weight,
                                                    noise_level=noise_level, weight_path=weight_path, layer_idx=i,
                                                    plot_path=plot_path))
                else:
                    self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i], cls=self.cls, lr=1e-2))
        else:
            print(f'Solver {solver} not supported')
        if noise_level is not None:
            print(f'Adding noise to the solver {solver} with noise level {noise_level}')
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            if self.solver in ['SGD', 'SA', 'LD']:
                x = self.PcConvs[i](x, i, self.pcn_weight_type)  # ReLU + Conv
            else:
                x = self.PcConvs[i](x, i)
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    batchsize =512
    test_ratio = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
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
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=batchsize, shuffle=False, num_workers=2)

    # Load model
    model_path = 'checkpoint/PredNetBpD_5_30CLS_FalseNes_0.001WD_FalseTIED_4REP_best_ckpt.t7'
    # checkpoint_weight = torch.load('checkpoint/PredNetBpD_5_0CLS_FalseNes_0.001WD_FalseTIED_1REP_last_ckpt_no_recurr.t7', map_location=device)
    checkpoint_weight = torch.load(model_path, map_location=device)
    prednet = PredNetBpD(num_classes=10, cls=30, lr=1e-2,
                         noise_level=0,
                         solver='LD', layer_number=[0, 1, 2, 3, 4], num_iterations=3000, train_weight=False)
    prednet = prednet.to(device)
    prednet = nn.DataParallel(prednet)
    prednet.load_state_dict(checkpoint_weight['net'])

    # Save expanded weights (optional)
    save_weight = True
    weight_path = os.path.join("expanded_weights", model_path.split('/')[-1].split('.t7')[0])
    plot_loss_path = os.path.join("loss_plot", model_path.split('/')[-1].split('.t7')[0])
    if save_weight:
        sample_imgs, _ = next(iter(testloader))
        save_expanded_weights(prednet, sample_imgs.to(device), weight_path)
        exit(0)

    # noise experiments
    for noise_level in [0, 0.05, 0.1, 0.15, .20, .25, .30, .35, .40]:
        trials = 3 if noise_level > 0 else 1
        for t in range(trials):
            prednet = PredNetBpD(num_classes=10, cls=30, lr=1e-4,
                                 noise_level=noise_level, solver='SGD',
                                 layer_number=[0, 1, 2, 3, 4], num_iterations=3000,
                                 train_weight=False,
                                 weight_path=weight_path, plot_path=plot_loss_path)
            prednet = prednet.to(device)
            prednet = nn.DataParallel(prednet)
            prednet.load_state_dict(checkpoint_weight['net'])
            prednet.eval()
            total = 0
            correct = 0
            
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader), disable=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    output_tensor = prednet(inputs)
                
                # Get the predicted class
                _, predicted = torch.max(output_tensor, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                # print(f' Temporary Accuracy: {100 * correct / total:.2f}%')

            # Calculate the accuracy
            accuracy = 100 * correct / total
            print(f'Test Accuracy at noise level {noise_level}: {accuracy:.2f}%')
