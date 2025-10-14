import logging
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.parametrize as P
import torchvision
import torchvision.transforms as transforms
import argparse
import tqdm
import subprocess
import json

from pc_model import PCNet
from data_utils import ToPackedRGGB, RawImgDataset, load_and_register_buffer, get_parametrized_weight_mods
from scangen.data import NoiseCIFARDataset, MyNoiseCIFARDataset

class TrainerCiFar(object):
    def __init__(self, model, model_name, save_path,
                 batch_size=512, optim_type="Adam", weight_decay=1e-3,
                 loss_fn=nn.CrossEntropyLoss(),
                 learning_rate=0.01, num_epochs=300, warmup_epoch=1,
                 lr_reduce_on="80,122,150,225,262", test_bs=512, max_norm=None, aug=False, T0=None,
                 eval_every=1, img_type="rgb"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('----- Using {} device -----'.format(self.device))

        model = model.to(self.device)
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.optimizer = self._get_optimizer(optim_type, lr=learning_rate, weight_decay=weight_decay)
        # Reuse the LR schedule epoch as before
        # Todo: Change the scheduler to some more flexible one
        if warmup_epoch > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=0.01, total_iters=100)
        if T0 is not None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                            T_0=T0, T_mult=2)
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                            milestones=list(map(int, lr_reduce_on.split(","))))
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epoch = warmup_epoch
        self.test_batch_size = test_bs
        self.max_norm = max_norm
        self.aug = aug # use the augmentation in convMixer or not
        self.eval_every = eval_every

        self._prepare_cifar(img_type)

    def train(self):
        train_loss_list, val_acc_list = [], []
        best_acc, val_acc, best_epoch = 0.0, 0.0, 0
        best_model_path = None
        for epoch in range(self.num_epochs):
            print("Training epoch {} / {}".format(epoch, self.num_epochs))
            train_loss = self.train_one_epoch(epoch)
            if (epoch + 1) % self.eval_every == 0:
                train_acc, _, _ = self.evaluate(self.train_dataloader)
                val_acc, _, _ = self.evaluate(self.val_dataloader)
                train_loss_list.append(train_loss)
                val_acc_list.append(val_acc)
                print("Validation acc: {}, Train acc: {}".format(val_acc, train_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    best_model_path = self._save_model_ckpt(val_acc, epoch + 1, "_best_ckpt.pth")
            self.scheduler.step()
        _ = self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")
        print("----- Train finished, Model Name: {} -----".format(self.model_name))
        print("----- Total number of parameters: {} M -----".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        print("----- Best acc: {}, Best epoch: {} -----".format(best_acc, best_epoch))
        print("----- Model path: {} -----".format(best_model_path))
        print("--------------------------------------------------------------------------")
        return train_loss_list, val_acc_list

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, n_samples = 0.0, 0
        progress_bar = tqdm.tqdm(enumerate(self.train_dataloader),
                            total=len(self.train_dataloader), desc="Training")
        for _i, _data in progress_bar:
            inputs, labels = _data
            n_samples += inputs.size(0)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            # Update running loss and compute average loss
            running_loss += loss.item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            # Update the tqdm progress bar with current iteration and loss
            progress_bar.set_postfix({
                "Iter": f"{_i + 1}/{len(self.train_dataloader)}",
                "Loss": f"{avg_loss:.4f}",
                "LR": self.optimizer.param_groups[0]["lr"]
            })

            if epoch < self.warmup_epoch:
                self.warmup_scheduler.step()

        if epoch < self.warmup_epoch - 1:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=0.1,
                                                                total_iters=10)
        running_loss /= n_samples
        return running_loss

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        running_loss = 0.0
        pred_list, label_list = [], []
        with torch.no_grad():
            self.model.eval()
            for data in dataloader:
                inputs, labels = data

                # move the data to GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate outputs by running inputs through the network
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pred_list.append(predicted)
                label_list.append(labels)

        accuracy = correct / total
        return accuracy, torch.cat(label_list), torch.cat(pred_list)

    def _save_then_load(self, save_to):
        model_class = self.model.__class__
        tmp_sd = {
            'net': self.model.state_dict(),
            'init_args': self.model.init_args,
        }
        tmp_sp = os.path.join(str(save_to), "__tmp_model.pth")
        torch.save(tmp_sd, tmp_sp)
        tmp_sd = torch.load(tmp_sp, weights_only=False)
        decoupled_model = model_class(
            **{**tmp_sd['init_args']['model_args'], **tmp_sd['init_args']['kwargs']}).to(self.device)
        p_dict = get_parametrized_weight_mods(self.model)
        _ = load_and_register_buffer(decoupled_model, tmp_sd['net'], self.device, p_dict)
        return decoupled_model

    def _save_model_ckpt(self, acc, epoch, suffix=""):
        save_to = os.path.join(self.save_path, self.model_name)
        os.makedirs(save_to, exist_ok=True)
        save_pth_path = os.path.join(str(save_to), self.model_name + suffix)

        # Need to save then the load the model to totally decouple the parameterization
        flat_model = self._save_then_load(save_to)
        parametrize_flag = False
        for _mod in flat_model.modules():
            if P.is_parametrized(_mod):
                parametrize_flag = True
                P.remove_parametrizations(_mod, "weight", leave_parametrized=True) # Keep the parametrized res
        if parametrize_flag:
            logging.warning("Model Includes parametrized module, saving the non-parametrized model with param baked in.")
            flat_state = {
                'net': flat_model.state_dict(),
                'init_args': self.model.init_args,
                'net_type': self.model.__class__.__name__,
                'acc': acc,
                'epoch': epoch,
            }
            # save the flat model with the same name as before
            torch.save(flat_state, save_pth_path)
            # modify the model name with "full_param" to save the model with full parametrization
            save_pth_path = os.path.join(str(save_to), self.model_name + "_full_param" + suffix)

        state = {
            'net': self.model.state_dict(),
            'init_args': self.model.init_args,
            'net_type': self.model.__class__.__name__,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, save_pth_path)
        return save_pth_path

    def _get_optimizer(self, optim_type, lr, weight_decay):
        if optim_type == "SGD":
            return optim.SGD(self.model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay, nesterov=False)
        elif optim_type == "Adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer: {}".format(optim_type))

    def _prepare_cifar(self, img_type):
        """
        Todo: Actually the validation dataset should be split from the train_set.
        After the split, we can change the scheduler into other types depending on the validation result.
        """
        if img_type in {"rgb", "rggb"}:
            if self.aug:
                if img_type == "rgb":
                    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandAugment(num_ops=1, magnitude=8),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                        transforms.RandomErasing(p=0.25),
                    ])
                else:
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        ToPackedRGGB(return_orig=False),
                        transforms.RandomResizedCrop(16, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.2),
                        transforms.RandomErasing(p=0.1),
                    ])
            else:
                if img_type == "rgb":
                    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
                else:
                    # Todo: Normalize rggb data?
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        ToPackedRGGB(return_orig=False),
                        transforms.RandomCrop(16, padding=2),
                        transforms.RandomHorizontalFlip(),
                    ])
            if img_type == "rgb":
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    ToPackedRGGB(return_orig=False), ])
            self.train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
            self.val_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        elif img_type == "scanGFI":
            conf_file = "./{}config.json".format(self.model_name)
            subprocess.run("uv run scangen create-config {}".format(conf_file), shell=True)
            with open("./{}".format(conf_file)) as fp:
                scangen_config = json.load(fp)
            self.train_set = MyNoiseCIFARDataset(
                root=os.path.join(os.path.abspath(__file__).rpartition("/")[0].rpartition("/")[0],
                                  "cifar-10-data", img_type),
                input_name="cifar10",
                train=True,
                noise_config=scangen_config["noise"],
                device=self.device,
                transform=transforms.Compose([
                    transforms.RandomCrop(16, padding=2),
                    transforms.RandomHorizontalFlip(),
                ])
            )
            self.val_set = MyNoiseCIFARDataset(
                root=os.path.join(os.path.abspath(__file__).rpartition("/")[0].rpartition("/")[0],
                                  "cifar-10-data", img_type),
                input_name="cifar10",
                train=False,
                noise_config=scangen_config["noise"],
                device=self.device,
            )
            subprocess.run("rm ./{}".format(conf_file), shell=True)
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(16, padding=2),
                transforms.RandomHorizontalFlip(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.train_set = RawImgDataset(root=os.path.join("../cifar-10-data", img_type), train=True, transform=transform_train)
            self.val_set = RawImgDataset(root=os.path.join("../cifar-10-data", img_type), train=False, transform=transform_test)

        # Get dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=2)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False,
                                                          num_workers=2)

