import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import tqdm

from pc_model import PCNet

class TrainerCiFar(object):
    def __init__(self, model, model_name, save_path,
                 batch_size=128, optim_type=torch.optim.Adam, weight_decay=1e-3,
                 loss_fn=nn.CrossEntropyLoss(),
                 learning_rate=0.01, num_epochs=300):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.optimizer = optim_type(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Reuse the LR schedule epoch as before
        # Todo: Change the scheduler to some more flexible one
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[150, 225, 262])
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self._prepare_cifar()

    def train(self):
        train_loss_list, val_acc_list = [], []
        best_acc, val_acc = 0.0, 0.0
        for epoch in range(self.num_epochs):
            print("Training epoch {} / {}".format(epoch, self.num_epochs))
            train_loss = self.train_one_epoch()
            val_acc, _, _ = self.evaluate()
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            print("Validation acc: {}".format(val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_model_ckpt(val_acc, epoch + 1, "_best_ckpt.pth")
        self._save_model_ckpt(val_acc, self.num_epochs, "_last_ckpt.pth")
        return train_loss_list, val_acc_list

    def train_one_epoch(self):
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
            self.optimizer.step()
            self.scheduler.step()

            # Update running loss and compute average loss
            running_loss += loss.item() * inputs.size(0)
            avg_loss = running_loss / n_samples

            # Update the tqdm progress bar with current iteration and loss
            progress_bar.set_postfix({
                "Iter": f"{_i + 1}/{len(self.train_dataloader)}",
                "Loss": f"{avg_loss:.4f}"
            })

        running_loss /= n_samples
        return running_loss

    def evaluate(self):
        correct = 0
        total = 0
        running_loss = 0.0
        pred_list, label_list = [], []
        with torch.no_grad():
            self.model.eval()
            for data in self.val_dataloader:
                inputs, labels = data

                # move the data to GPU
                inputs, labels = inputs.to(self.device).permute(1, 0, 2), labels.to(self.device)

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

    def _save_model_ckpt(self, acc, epoch, suffix=""):
        save_to = os.path.join(self.save_path, self.model_name)
        os.makedirs(save_to, exist_ok=True)
        save_pth_path = os.path.join(str(save_to), self.model_name + suffix)
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, save_pth_path)

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
        self.val_dataloader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size * 4, shuffle=False, num_workers=2)
