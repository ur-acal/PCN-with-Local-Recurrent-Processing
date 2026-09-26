"""Benchmark SRRL teacher/student forward order on the real recovery run."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_ode_cifar as entry
from trainer_timm import TrainerCiFarTimmStyle
from training_recovery import restore_latest


parser = argparse.ArgumentParser()
parser.add_argument("--order", choices=("student_first", "teacher_first"), required=True)
parser.add_argument("--iterations", type=int, default=4)
parser.add_argument("--save-state", type=Path)
args = parser.parse_args()

run_root = ROOT / (
    "saved_ckpt_runs/tc_rgb_cifar100_state1_pcn_resnet_depth_study_"
    "ft_study_all_on_timm_aug"
)
paths = list(run_root.glob("*/*_latest_ckpt.pth"))
if len(paths) != 1:
    raise RuntimeError(f"Expected one recovery checkpoint, found: {paths}")
checkpoint_path = paths[0]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = dict(checkpoint["training_recovery"]["config"])
config.update(
    model_name=checkpoint_path.parent.name,
    save_path=str(run_root),
    output_save_path=str(run_root),
    ckpt="latest",
    mem_frac=1.0,
)
del checkpoint

entry.get_args = lambda: argparse.Namespace(**config)
entry.evaluate_teacher = lambda *unused_args, **unused_kwargs: None


def benchmark(self):
    recovered = restore_latest(self)
    if recovered is None:
        raise RuntimeError("The requested checkpoint is not a recovery checkpoint")

    self.model.train()
    self.teacher_model.eval()
    self._teacher_extractor.eval()
    self._feature_kd_loss.train()
    iterator = iter(self.train_dataloader)
    records = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for iteration in range(args.iterations):
        inputs, teacher_inputs, labels = self._unpack_train_batch(next(iterator))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if teacher_inputs is not None:
            teacher_inputs = teacher_inputs.to(self.device)

        labels_for_ce = labels
        if self._paired_mixup_active:
            inputs, teacher_inputs, labels_for_ce = self._paired_timm_mixup_cutmix(
                inputs, teacher_inputs, labels)
        elif self.mixup_fn is not None:
            if teacher_inputs is not None:
                raise RuntimeError("Unexpected teacher inputs for ordinary mixup")
            inputs, labels_for_ce = self.mixup_fn(inputs, labels)

        self.optimizer.zero_grad()
        teacher_input = teacher_inputs if teacher_inputs is not None else inputs
        if args.order == "teacher_first":
            teacher_logits, teacher_features = self.teacher_forward_for_distillation(teacher_input)
            outputs, student_features = self._student_forward_feature_kd(inputs)
        else:
            outputs, student_features = self._student_forward_feature_kd(inputs)
            teacher_logits, teacher_features = self.teacher_forward_for_distillation(teacher_input)

        ce_loss = self.train_loss_fn(outputs, labels_for_ce)
        kd_loss = None
        if self._kd_enabled:
            kd_loss = self._kd_loss(outputs, teacher_logits)
            base_loss = ((1.0 - self.distill_alpha) * ce_loss
                         + self.distill_alpha * kd_loss)
        else:
            base_loss = ce_loss
        feature_loss, feature_logs = self._compute_feature_kd_loss(
            student_features, teacher_features, teacher_logits)
        loss = base_loss + self.feature_kd_beta * feature_loss
        loss.backward()
        self.optimizer.step()
        records.append(dict(
            loss=float(loss.detach()), ce=float(ce_loss.detach()),
            feature=float(feature_loss.detach()),
            stat=float(feature_logs["Stat"]), pred=float(feature_logs["Pred"]),
            output_sum=float(outputs.detach().double().sum()),
        ))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    parameter_sum = sum(float(value.detach().double().sum())
                        for value in self.model.parameters())
    report = dict(
        order=args.order, iterations=args.iterations,
        seconds=elapsed, seconds_per_iteration=elapsed / args.iterations,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        parameter_sum=parameter_sum, records=records,
    )
    if args.save_state is not None:
        torch.save(dict(
            model={key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            feature_kd={key: value.detach().cpu()
                        for key, value in self._feature_kd_loss.state_dict().items()},
        ), args.save_state)
    print("ORDER_BENCHMARK " + json.dumps(report), flush=True)


TrainerCiFarTimmStyle.train = benchmark
entry.main()
