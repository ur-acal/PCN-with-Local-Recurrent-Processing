"""ReviewKD auxiliary modules adapted from the official CIFAR implementation.

Reference:
https://github.com/JIA-Lab-research/ReviewKD/blob/master/CIFAR-100/model/reviewkd.py
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .teacher_feat_extract import TeacherFeatureExtractor


def _as_4d_tensor(output, name):
    if isinstance(output, (tuple, list)):
        tensors = [value for value in output if torch.is_tensor(value)]
        if len(tensors) != 1:
            raise RuntimeError(f"{name} did not produce exactly one tensor.")
        output = tensors[0]
    if not torch.is_tensor(output) or output.dim() != 4:
        shape = tuple(output.shape) if torch.is_tensor(output) else type(output)
        raise RuntimeError(f"{name} must produce a 4D feature map, got {shape}.")
    return output


class ReviewKDTeacherFeatureExtractor(TeacherFeatureExtractor):
    """Return the deepest spatial teacher stages and the global pooled feature.

    ReviewKD's CIFAR implementation uses three spatial stage outputs followed by
    the global-average-pooled classifier input for its default four matches. An
    EfficientNet has more spatial reductions than a CIFAR ResNet, so the deepest
    ``num_stages - 1`` distinct spatial resolutions are selected. The final
    spatial hook is placed before the head activation when that module is
    exposed, matching the original final-preactivation / activated-pool split.
    """

    def __init__(self, teacher: nn.Module, num_stages: int = 4) -> None:
        super().__init__(teacher)
        self.num_stages = int(num_stages)
        if self.num_stages < 1:
            raise ValueError("ReviewKD num_stages must be at least 1.")

        self._spatial_hook_specs, self._pool_hook_spec = self._build_hook_specs()

    @staticmethod
    def _last_batch_norm(module):
        batch_norm = None
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                batch_norm = child
        return batch_norm

    def _build_hook_specs(self):
        # Each spatial spec is (module, clone_output). Cloning is required when
        # an in-place activation consumes a hooked pre-activation tensor.
        if self.kind == "hankyul2_effnetv2":
            spatial = [(block, False) for block in self.teacher.blocks.children()]
            head_feature = self.teacher.head[0]
            head_pre_activation = self._last_batch_norm(head_feature)
            if head_pre_activation is None:
                spatial.append((head_feature, False))
            else:
                spatial.append((head_pre_activation, True))
            return spatial, (self.teacher.head[1], False)

        if self.kind == "efficientnet_pytorch":
            spatial = [(block, False) for block in self.teacher._blocks]
            head_pre_activation = getattr(self.teacher, "_bn1", None)
            if head_pre_activation is None:
                head_pre_activation = self.teacher._conv_head
            spatial.append((head_pre_activation, True))
            return spatial, (self.teacher._avg_pooling, False)

        if self.kind == "torchvision_efficientnet":
            feature_modules = list(self.teacher.features.children())
            spatial = [(module, False) for module in feature_modules[:-1]]
            final_feature = feature_modules[-1]
            head_pre_activation = self._last_batch_norm(final_feature)
            if head_pre_activation is None:
                spatial.append((final_feature, False))
            else:
                spatial.append((head_pre_activation, True))
            return spatial, (self.teacher.avgpool, False)

        raise RuntimeError(f"Unknown teacher kind: {self.kind}")

    @staticmethod
    def _deepest_distinct_spatial_features(features, count):
        # Keep the last feature produced at each resolution. This makes the
        # final head feature replace the last backbone feature at the same H/W.
        distinct = []
        for feature in features:
            feature = _as_4d_tensor(feature, "ReviewKD teacher spatial hook")
            if feature.shape[-2:] == (1, 1):
                continue
            if distinct and distinct[-1].shape[-2:] == feature.shape[-2:]:
                distinct[-1] = feature
            else:
                distinct.append(feature)

        if len(distinct) < count:
            shapes = [tuple(feature.shape) for feature in distinct]
            raise RuntimeError(
                f"ReviewKD requested {count} spatial teacher stages, but only "
                f"{len(distinct)} distinct stages were captured: {shapes}."
            )
        return distinct[-count:] if count else []

    @torch.no_grad()
    def forward_with_features(self, x: torch.Tensor):
        spatial_outputs = []
        pooled_outputs = []
        handles = []

        def make_spatial_hook(clone_output):
            def hook(_module, _inputs, output):
                output = _as_4d_tensor(output, "ReviewKD teacher spatial hook")
                spatial_outputs.append(output.clone() if clone_output else output)
            return hook

        def pool_hook(_module, _inputs, output):
            pooled_outputs.append(
                _as_4d_tensor(output, "ReviewKD teacher pooling hook")
            )

        for module, clone_output in self._spatial_hook_specs:
            handles.append(module.register_forward_hook(make_spatial_hook(clone_output)))
        handles.append(self._pool_hook_spec[0].register_forward_hook(pool_hook))

        try:
            logits = self.teacher(x)
        finally:
            for handle in handles:
                handle.remove()

        if not pooled_outputs:
            raise RuntimeError("ReviewKD failed to capture the teacher pooled feature.")

        spatial_count = self.num_stages - 1
        features = self._deepest_distinct_spatial_features(
            spatial_outputs,
            spatial_count,
        )
        pooled = pooled_outputs[-1]
        if pooled.shape[-2:] != (1, 1):
            raise RuntimeError(
                "ReviewKD expected a 1x1 teacher pooled feature, got "
                f"{tuple(pooled.shape)}."
            )
        features.append(pooled)
        return logits, features


class ABF(nn.Module):
    """Attention-based fusion module from the official ReviewKD code."""

    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, residual=None, out_size=None):
        n, _, h, w = x.shape
        x = self.conv1(x)

        if self.att_conv is not None:
            if residual is None:
                raise RuntimeError("ReviewKD ABF fusion requires a residual feature.")
            residual = F.interpolate(residual, (h, w), mode="nearest")
            attention = self.att_conv(torch.cat([x, residual], dim=1))
            x = (
                x * attention[:, 0].view(n, 1, h, w)
                + residual * attention[:, 1].view(n, 1, h, w)
            )

        if out_size is not None and x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, tuple(out_size), mode="nearest")

        output = self.conv2(x)
        return output, x


def hierarchical_context_loss(student_features, teacher_features):
    """The original ReviewKD HCL: full map plus 4x4, 2x2, and 1x1."""
    if len(student_features) != len(teacher_features):
        raise RuntimeError(
            "ReviewKD HCL feature count mismatch: "
            f"student={len(student_features)}, teacher={len(teacher_features)}."
        )

    stage_losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise RuntimeError(
                "ReviewKD HCL feature shape mismatch: "
                f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}."
            )

        height = student.shape[-2]
        loss = F.mse_loss(student, teacher, reduction="mean")
        weight = 1.0
        total_weight = 1.0

        for level in (4, 2, 1):
            if level >= height:
                continue
            pooled_student = F.adaptive_avg_pool2d(student, (level, level))
            pooled_teacher = F.adaptive_avg_pool2d(teacher, (level, level))
            weight /= 2.0
            loss = loss + F.mse_loss(
                pooled_student,
                pooled_teacher,
                reduction="mean",
            ) * weight
            total_weight += weight

        stage_losses.append(loss / total_weight)

    if not stage_losses:
        raise RuntimeError("ReviewKD HCL received no feature stages.")
    return torch.stack(stage_losses).sum(), stage_losses


class ReviewKDLoss(nn.Module):
    """Deep-to-shallow ABF review path followed by the original HCL."""

    def __init__(self, student_features: Sequence[torch.Tensor],
                 teacher_features: Sequence[torch.Tensor]):
        super().__init__()
        if len(student_features) != len(teacher_features):
            raise ValueError(
                "ReviewKD feature count mismatch during initialization: "
                f"student={len(student_features)}, teacher={len(teacher_features)}."
            )
        if not student_features:
            raise ValueError("ReviewKD requires at least one feature stage.")

        self.num_stages = len(student_features)
        student_channels = [int(feature.shape[1]) for feature in student_features]
        teacher_channels = [int(feature.shape[1]) for feature in teacher_features]
        self.teacher_sizes = [tuple(feature.shape[-2:]) for feature in teacher_features]
        mid_channel = min(512, student_channels[-1])

        abfs = []
        for index, (student_channel, teacher_channel) in enumerate(
                zip(student_channels, teacher_channels)):
            abfs.append(
                ABF(
                    student_channel,
                    mid_channel,
                    teacher_channel,
                    fuse=index < self.num_stages - 1,
                )
            )
        self.abfs = nn.ModuleList(abfs[::-1])

    def trainable_parameters(self):
        return self.parameters()

    def forward(self, student_features, teacher_features, return_dict=False):
        if len(student_features) != self.num_stages:
            raise RuntimeError(
                f"ReviewKD expected {self.num_stages} student stages, got "
                f"{len(student_features)}."
            )
        if len(teacher_features) != self.num_stages:
            raise RuntimeError(
                f"ReviewKD expected {self.num_stages} teacher stages, got "
                f"{len(teacher_features)}."
            )

        reversed_student = list(student_features)[::-1]
        reversed_sizes = self.teacher_sizes[::-1]
        reviewed = []

        output, residual = self.abfs[0](
            reversed_student[0],
            out_size=reversed_sizes[0],
        )
        reviewed.append(output)

        for feature, abf, out_size in zip(
                reversed_student[1:], self.abfs[1:], reversed_sizes[1:]):
            output, residual = abf(
                feature,
                residual=residual,
                out_size=out_size,
            )
            reviewed.insert(0, output)

        loss, stage_losses = hierarchical_context_loss(reviewed, teacher_features)
        if return_dict:
            return loss, {
                "reviewkd_hcl_loss": loss,
                "reviewkd_stage_losses": stage_losses,
            }
        return loss
