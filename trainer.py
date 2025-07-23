import logging
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.parametrize as P
import torch.quantization as quantization
import torchvision
import torchvision.transforms as transforms
import argparse
import tqdm

from pc_model import PCNet

class TrainerCiFar(object):
    def __init__(self, model, model_name, save_path,
                 batch_size=512, optim_type="Adam", weight_decay=1e-3,
                 loss_fn=nn.CrossEntropyLoss(),
                 learning_rate=0.01, num_epochs=300, warmup_epoch=1,
                 lr_reduce_on="80,122,150,225,262", test_bs=512, max_norm=None,
                 qat=False, qat_backend="fbgemm", qat_start_epoch=0):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('----- Using {} device -----'.format(self.device))

        model = model.to(self.device)
        self.model = model
        self.model_name = model_name
        self.save_path = save_path

        # Store optimizer parameters for QAT
        self._optim_type = optim_type
        self._lr = learning_rate
        self._weight_decay = weight_decay

        self.optimizer = self._get_optimizer(optim_type, lr=learning_rate, weight_decay=weight_decay)
        # Reuse the LR schedule epoch as before
        # Todo: Change the scheduler to some more flexible one
        if warmup_epoch > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=0.01, total_iters=100)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                        milestones=list(map(int, lr_reduce_on.split(","))))
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epoch = warmup_epoch
        self.test_batch_size = test_bs
        self.max_norm = max_norm
        self.lr_reduce_on = lr_reduce_on

        # Quantization Aware Training parameters
        self.qat = qat
        self.qat_backend = qat_backend
        self.qat_start_epoch = qat_start_epoch
        self.qat_prepared = False

        if self.qat:
            print('----- Quantization Aware Training enabled -----')
            print('----- QAT Backend: {} -----'.format(self.qat_backend))
            print('----- QAT Start Epoch: {} -----'.format(self.qat_start_epoch))

        self._prepare_cifar()

    def train(self):
        train_loss_list, val_acc_list = [], []
        best_acc, val_acc, best_epoch = 0.0, 0.0, 0
        best_model_path = None
        for epoch in range(self.num_epochs):
            print("Training epoch {} / {}".format(epoch, self.num_epochs))

            # Enable QAT at specified epoch
            if self.qat and epoch == self.qat_start_epoch and not self.qat_prepared:
                print(f"\n🔄 ENABLING QAT AT EPOCH {epoch}")
                self._prepare_qat()
                # Recreate scheduler for new optimizer
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optimizer,
                    milestones=list(map(int, self.lr_reduce_on.split(",")))
                )
                # Adjust scheduler state to current epoch
                for _ in range(epoch):
                    self.scheduler.step()
                print(f"✓ QAT enabled and scheduler adjusted for epoch {epoch}\n")

            train_loss = self.train_one_epoch(epoch)
            train_acc, _, _ = self.evaluate(self.train_dataloader)
            val_acc, _, _ = self.evaluate(self.val_dataloader)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)

            # QAT status indicator
            qat_status = ""
            if self.qat:
                if self.qat_prepared:
                    qat_status = " [QAT-ACTIVE]"
                elif epoch >= self.qat_start_epoch:
                    qat_status = " [QAT-READY]"
                else:
                    qat_status = f" [QAT-PENDING@{self.qat_start_epoch}]"

            print("Validation acc: {}, Train acc: {}{}".format(val_acc, train_acc, qat_status))
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                best_model_path = self._save_model_ckpt(val_acc, epoch + 1, "_best_ckpt.pth")
            self.scheduler.step()
        _ = self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")

        # Convert QAT model to quantized model for inference
        if self.qat and self.qat_prepared:
            print("\n" + "="*60)
            print("CONVERTING QAT MODEL TO QUANTIZED MODEL")
            print("="*60)

            # Get model size before quantization
            qat_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

            self.model.eval()
            quantized_model = quantization.convert(self.model, inplace=False)

            # Test quantized model
            try:
                dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
                with torch.no_grad():
                    qat_output = self.model(dummy_input)
                    quant_output = quantized_model(dummy_input.cpu())  # Quantized models typically run on CPU
                print("✓ Quantized model forward pass test successful")
                print(f"QAT output shape: {qat_output.shape}, Quantized output shape: {quant_output.shape}")
            except Exception as e:
                print(f"✗ Quantized model test failed: {e}")

            # Count quantized modules
            quant_modules = []
            for name, module in quantized_model.named_modules():
                if 'quantized' in str(type(module)).lower():
                    quant_modules.append(name)

            print(f"✓ Found {len(quant_modules)} quantized modules")
            print(f"QAT model size: {qat_size / 1024:.2f} KB")

            quantized_model_path = self._save_quantized_model(quantized_model, val_acc, self.num_epochs)

            # Check saved file size
            if os.path.exists(quantized_model_path):
                file_size = os.path.getsize(quantized_model_path)
                print(f"Quantized model file size: {file_size / 1024:.2f} KB")
                compression_ratio = qat_size / file_size if file_size > 0 else 0
                print(f"Compression ratio: {compression_ratio:.2f}x")

            print(f"✓ Quantized model saved: {quantized_model_path}")
            print("="*60)

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
        tmp_sd = torch.load(tmp_sp)
        decoupled_model = model_class(
            **{**tmp_sd['init_args']['model_args'], **tmp_sd['init_args']['kwargs']}).to(self.device)
        decoupled_model.load_state_dict(tmp_sd['net'])
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

    def _save_quantized_model(self, quantized_model, acc, epoch, suffix="_quantized.pth"):
        """Save quantized model"""
        save_to = self.save_path
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        save_pth_path = os.path.join(str(save_to), self.model_name + suffix)

        # Note: Quantized models may not have init_args, so we save what we can
        state = {
            'net': quantized_model.state_dict(),
            'net_type': quantized_model.__class__.__name__,
            'acc': acc,
            'epoch': epoch,
            'quantized': True,
            'backend': self.qat_backend,
        }
        torch.save(state, save_pth_path)
        return save_pth_path

    def _prepare_qat(self):
        """Prepare model for Quantization Aware Training"""
        if not self.qat or self.qat_prepared:
            return

        print('----- Preparing model for QAT -----')

        # Count original parameters
        original_params = sum(p.numel() for p in self.model.parameters())
        print(f'Original model parameters: {original_params:,}')

        # Set quantization config for INT8
        self.model.qconfig = quantization.get_default_qat_qconfig(self.qat_backend)
        print(f'QConfig: {self.model.qconfig}')

        # Prepare model for QAT
        self.model = quantization.prepare_qat(self.model, inplace=False)

        # Count QAT modules
        qat_modules = []
        fake_quant_modules = []
        observer_modules = []

        for name, module in self.model.named_modules():
            module_type = str(type(module))
            if 'fake_quant' in module_type.lower():
                fake_quant_modules.append(name)
            elif 'observer' in module_type.lower():
                observer_modules.append(name)
            elif any(qat_term in module_type.lower() for qat_term in ['qat', 'quantiz']):
                qat_modules.append(name)

        print(f'✓ Found {len(fake_quant_modules)} FakeQuantize modules')
        print(f'✓ Found {len(observer_modules)} Observer modules')
        print(f'✓ Found {len(qat_modules)} other QAT modules')

        # Move model back to device after QAT preparation
        self.model = self.model.to(self.device)

        # Update optimizer to work with QAT model
        if hasattr(self, 'optimizer'):
            # Create new optimizer for QAT model parameters
            if hasattr(self, '_optim_type') and hasattr(self, '_lr') and hasattr(self, '_weight_decay'):
                self.optimizer = self._get_optimizer(self._optim_type, self._lr, self._weight_decay)
                print(f'✓ Optimizer updated for QAT model')

        # Test forward pass
        try:
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            print('✓ QAT model forward pass test successful')
        except Exception as e:
            print(f'✗ QAT model forward pass test failed: {e}')
            raise

        self.qat_prepared = True
        print('----- Model prepared for QAT -----')

    def _get_optimizer(self, optim_type, lr, weight_decay):
        if optim_type == "SGD":
            return optim.SGD(self.model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay, nesterov=False)
        elif optim_type == "Adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer: {}".format(optim_type))

    def _prepare_cifar(self):
        """
        Todo: Actually the validation dataset should be split from the train_set.
        After the split, we can change the scheduler into other types depending on the validation result.
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
        self.train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
