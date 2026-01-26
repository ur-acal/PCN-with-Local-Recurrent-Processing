import logging
import os
import tempfile
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import tqdm
import subprocess
import json

from pc_model import PCNet
from data_utils import ToPackedRGGB, RawImgDataset, load_and_register_buffer, get_parametrized_weight_mods, get_quant_model
from scangen.data import NoiseCIFARDataset, MyNoiseCIFARDataset

class TrainerCiFar(object):
    def __init__(self, model, model_name, save_path,
                 batch_size=512, optim_type="Adam", weight_decay=1e-3,
                 loss_fn=nn.CrossEntropyLoss(), quant_params=None, q_calib_bs=256,
                 learning_rate=0.01, num_epochs=300, warmup_epoch=1,
                 lr_reduce_on="80,122,150,225,262", test_bs=512, max_norm=None, aug=False, T0=None,
                 eval_every=1, img_type="rgb", noise_level=None, task="cifar10", noise_type=None,
                 distill_type=None, teacher=None, distill_T=1, distill_w="1|0|0"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.warning('----- Using {} device -----'.format(self.device))

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

        # noise inject training
        self.noisy_model = None
        if noise_level is not None:
            self.noisy_model = WrappedNoisyModel(model=self.model, noise_level=noise_level, noise_type=noise_type)

        self._prepare_cifar(img_type, task)
        if distill_type is not None and teacher is not None:
            self.loss_fn = self._get_distill_cls(distill_type, distill_T)
            self.teacher = teacher.to(self.device)
            self.teacher.eval()
            self.distill_weights = list(map(lambda _x: float(_x), distill_w.split("|")))

        # QAT for PPCN
        self.qat_params = quant_params
        if self.qat_params is not None:
            self._calib_model(q_calib_bs)

    def _calib_model(self, q_calib_bs):
        if q_calib_bs <= self.batch_size:
            calib_batch = next(iter(self.train_dataloader))[0][:q_calib_bs].to(self.device)
            _ = self.model(calib_batch)
        else:
            n_batches = q_calib_bs // self.batch_size
            rem_samples = q_calib_bs % self.batch_size
            calib_batch = []
            for _i, _batch in enumerate(self.train_dataloader):
                _inp, _ = _batch
                _inp = _inp.to(self.device)
                if _i == n_batches:
                    if rem_samples > 0:
                        calib_batch.append(_inp[:rem_samples])
                    break
                calib_batch.append(_inp)
            calib_batch = torch.cat(calib_batch, dim=0)
            _ = self.model(calib_batch)
        logging.warning("Calibration done with calib batch: {}".format(calib_batch.shape))

    @staticmethod
    def max_param_change(before, after):
        # before = {n: copy.deepcopy(p.detach().clone()) for n, p in m.named_parameters()}
        # after = {n: p.detach() for n, p in m.named_parameters()}
        mx = 0.0
        arg = None
        for n in before:
            d = (before[n] - after[n]).abs().max().item()
            if d > mx:
                mx = d
                arg = n
        print("max |delta w| = {}".format(mx))

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

                if val_acc <= 0.15 and epoch + 1 >= 20:
                    print("Train failed, stopped at epoch: {}".format(epoch + 1))
                    break
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
            if self.noisy_model and isinstance(self.noisy_model, nn.Module):
                outputs = self.noisy_model(inputs)
            else:
                outputs = self.model(inputs)

            # Loss calculation
            if not isinstance(self.loss_fn, nn.ModuleList):
                # Normal training process
                loss = self.loss_fn(outputs, labels)
            else:
                # Distillation
                loss = self._calc_distill_loss(inputs, outputs, labels)
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

                if not isinstance(self.loss_fn, nn.ModuleList):
                    loss = self.loss_fn(outputs, labels)
                else:
                    # Distillation
                    loss_fn_cls = self.loss_fn[0]
                    loss = loss_fn_cls(outputs, labels)
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
        if self.qat_params is not None:
            _ = get_quant_model(decoupled_model, device=self.device, **self.qat_params)
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

    def _prepare_cifar(self, img_type, task):
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
            with tempfile.TemporaryDirectory() as tmpdir:
                conf_file = os.path.join(tmpdir, "config.json")
                subprocess.run("uv run scangen create-config --dataset {} {}".format(task, conf_file), shell=True)
                with open("{}".format(conf_file)) as fp:
                    scangen_config = json.load(fp)
                self.train_set = MyNoiseCIFARDataset(
                    root=os.path.join(os.path.abspath(__file__).rpartition("/")[0].rpartition("/")[0],
                                      "cifar-10-data", img_type),
                    input_name=task,
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
                    input_name=task,
                    train=False,
                    noise_config=scangen_config["noise"],
                    device=self.device,
                )
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

    def _get_distill_cls(self, distill_type, distill_T):
        self.distill_type = distill_type
        if distill_type == "VanillaKD":
            return nn.ModuleList([self.loss_fn, VanillaKD(distill_T)])
        elif distill_type == "CRD":
            return nn.ModuleList([self.loss_fn, VanillaKD(distill_T), CRD()])

    def _calc_distill_loss(self, inputs, outputs, labels):
        gamma, alpha, beta = self.distill_weights
        criterion_cls = self.loss_fn[0]
        loss_cls = criterion_cls(outputs, labels)

        criterion_kl = self.loss_fn[1]
        with torch.no_grad():
            out_teacher = self.teacher(inputs)
        loss_kl = criterion_kl(y_t=out_teacher.detach(), y_s=outputs)

        # Loss from different kinds of distillation methods
        if self.distill_type == "VanillaKD":
            return gamma * loss_cls + alpha * loss_kl
        elif self.distill_type == "CRD":
            # Todo: Implement CRD
            criterion_crd = self.loss_fn[2]
            return gamma * loss_cls
        else:
            return loss_cls


class WrappedNoisyModel(nn.Module):
    """
    Todo: Used this module only when doing noise-inject training independent not with QAT.
    """
    def __init__(self, model: nn.Module, noise_level, noise_type="mul"):
        super().__init__()
        self.model = model
        self.noise_level = noise_level
        assert noise_type.lower() in {"mul", "add"}
        self.noise_type = noise_type.lower()
        self._gen_noisy_p = self._apply_noise_mul if self.noise_type == "mul" else self._apply_noise_add
        # Keeps a list of params free from noise
        self.noise_free_params = {"s_w_Param"}

    def _check_noise_free(self, p_name):
        for _nf_p in self.noise_free_params:
            if _nf_p in p_name:
                return True
        return False

    def gen_noisy_params(self):
        noisy_params = {}
        for _name, _param in self.model.named_parameters():
            if self._check_noise_free(_name):
                noisy_params[_name] = _param
            else:
                noisy_params[_name] = self._gen_noisy_p(_param) # type: ignore[misc]
        return noisy_params

    def _apply_noise_mul(self, p: nn.Parameter):
        noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * self.noise_level
        # Todo: In this case, the noise is also applied to the gradient. Should we use
        #   return p + p.detach() * noise_ ?
        return p.mul(1 + noise_)

    def _apply_noise_add(self, p: nn.Parameter):
        p_max = p.detach().abs().max()
        noise_ = torch.randn_like(p, device=p.device, requires_grad=False) * self.noise_level * p_max
        return p.add(noise_)

    def forward(self, x):
        if not self.model.training:
            return self.model(x)
        noisy_params = self.gen_noisy_params()
        return torch.func.functional_call(self.model, noisy_params, (x,))


class VanillaKD(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, y_t, y_s):
        log_p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T ** 2)


class CRD(nn.Module):
    pass


KD_CLASSES = {
    "VanillaKD": VanillaKD,
    "CRD": CRD,
}