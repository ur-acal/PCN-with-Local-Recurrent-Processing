"""Simplified MNIST trainer; reuse CIFAR train-step, evaluation and serialization."""
import json
from pathlib import Path

import torch

from trainer_timm import TrainerCiFarTimmStyle
from mnist_train_eval.mnist_data import mnist_loader, MEAN, STD


class MNISTTrainer(TrainerCiFarTimmStyle):
    @staticmethod
    def normalize_dataset_name(name):
        if name != 'mnist':
            raise ValueError('MNISTTrainer only supports mnist')
        return name

    def __init__(self, model, args, name):
        self.mnist_args = args
        super().__init__(
            model=model, model_name=name, save_path=str(args.output_dir),
            dataset_name='mnist', img_type='gray', in_chans=1,
            batch_size=args.batch_size, test_bs=args.test_batch_size,
            num_workers=args.num_workers, num_epochs=args.epochs,
            learning_rate=args.lr, weight_decay=args.weight_decay,
            warmup_epoch=0, distill_method='none', noise_level=None,
            timm_aug=False, mixup_alpha=0., cutmix_alpha=0.,
            label_smoothing=0., hflip=0., color_jitter=0., re_prob=0.,
            auto_augment=None, final_eval_only=True,
            health_check_epochs=args.health_check_epochs,
            health_check_batches=args.health_check_batches, health_check_seed=args.seed,
            timm_input_size=(1, 28, 28), timm_mean=MEAN, timm_std=STD,
            is_timm_model=True, save_flattened_and_full_param=True)

    def _prepare_cifar(self, img_type, dataset_name):
        # Override the data boundary, not the inherited compute/training paths.
        a = self.mnist_args
        common = dict(data_dir=a.data_dir, num_workers=a.num_workers,
                      seed=a.seed, download=a.download)
        self.train_dataloader = mnist_loader(
            **common, train=True, batch_size=a.batch_size,
            limit_samples=a.limit_train_samples)
        self.val_dataloader = mnist_loader(
            **common, train=False, batch_size=a.test_batch_size,
            limit_samples=a.limit_test_samples)
        self.train_set, self.val_set = self.train_dataloader.dataset, self.val_dataloader.dataset

    def _get_optimizer(self, optim_type, lr, weight_decay):
        a = self.mnist_args
        if a.optimizer == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=lr,
                rho=a.rho, eps=a.eps, weight_decay=weight_decay)
        if a.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr,
                eps=a.eps, weight_decay=weight_decay)
        return torch.optim.SGD(self.model.parameters(), lr=lr,
            momentum=a.momentum, weight_decay=weight_decay)

    def _build_timm_scheduler(self):
        a = self.mnist_args
        if a.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=a.step_size, gamma=a.gamma)
        elif a.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=a.epochs, eta_min=a.min_lr)
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)

    def _save_model_ckpt(self, acc, epoch, suffix=''):
        path = super()._save_model_ckpt(acc, epoch, suffix)
        # Serialize reconstruction/config alongside both ordinary and QAT files.
        paths = {Path(path), Path(self.save_path) / self.model_name / (self.model_name + suffix)}
        for dest in paths:
            checkpoint = torch.load(dest, map_location='cpu', weights_only=False)
            checkpoint['mnist_config'] = vars(self.mnist_args).copy()
            torch.save(checkpoint, dest)
        return path

    def train(self):
        # No best/latest selection, no intermediate test access. Full-parameter
        # last is the QAT continuation artifact; ordinary last has baked weights.
        self._maybe_profile_ode_rhs_checkpointing()
        history = []
        for epoch in range(self.num_epochs):
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Training epoch {epoch + 1}/{self.num_epochs}; LR={lr:g}', flush=True)
            loss = self.train_one_epoch(epoch)
            row = dict(epoch=epoch + 1, lr=lr, loss=loss)
            if epoch + 1 in self.health_check_epochs:
                row['training_health_accuracy'], _ = self._evaluate_training_health()
                print(f'Training health: {row["training_health_accuracy"]:.4f}', flush=True)
            self.scheduler.step()  # Exactly one decay AFTER each training epoch.
            self._save_model_ckpt(None, epoch + 1, '_last_ckpt.pth')
            history.append(row)
        accuracy, _, _, _ = self.evaluate(self.val_dataloader)
        path = self._save_model_ckpt(accuracy, self.num_epochs, '_last_ckpt.pth')
        result = dict(final_accuracy=accuracy, checkpoint=path, history=history,
                      parameters=sum(p.numel() for p in self.model.parameters()),
                      config=vars(self.mnist_args))
        dest = Path(self.save_path) / self.model_name / 'mnist_training_result.json'
        dest.write_text(json.dumps(result, indent=2, default=str) + '\n')
        print(f'Final test accuracy: {accuracy:.4%}\nCheckpoint: {path}', flush=True)
        return result
