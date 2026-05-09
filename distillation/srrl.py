"""SRRL helpers.

SRRL = Softmax Regression Representation Learning.

This implementation follows the practical code structure from KD_SRRL:

    feat_s = connector(feat_s)
    loss_stat = statm_loss(feat_s, feat_t)
    pred_sc = teacher_classifier(feat_s)
    loss_pred = mse(pred_sc, pred_t)

The connector is training-only. The student model architecture is unchanged
for inference.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .teacher_feat_extract import _require_4d, _align_spatial_by_pooling

class SRRLConnector(nn.Module):
    """
    Official-style SRRL connector:
        1x1 conv -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.connector = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, feat_student: torch.Tensor) -> torch.Tensor:
        feat_student = _require_4d(feat_student, "feat_student")
        return self.connector(feat_student)


class SRRLStatLoss(nn.Module):
    """
    Same idea as KD_SRRL statm_loss.

    It compares channel-wise spatial means:
        mean_gap = || mean_hw(F_s) - mean_hw(F_t) ||^2
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feat_student: torch.Tensor, feat_teacher: torch.Tensor) -> torch.Tensor:
        feat_student = _require_4d(feat_student, "feat_student")
        feat_teacher = _require_4d(feat_teacher, "feat_teacher")

        feat_student = feat_student.view(feat_student.size(0), feat_student.size(1), -1)
        feat_teacher = feat_teacher.view(feat_teacher.size(0), feat_teacher.size(1), -1)

        mean_student = feat_student.mean(dim=2)
        mean_teacher = feat_teacher.mean(dim=2)

        mean_gap = (mean_student - mean_teacher).pow(2).mean(dim=1)
        return mean_gap.mean()


class SRRLLoss(nn.Module):
    """
    SRRL training-only loss.

    Inputs:
        feat_student: raw student 4D feature
        feat_teacher: teacher 4D feature
        teacher_logits: teacher logits for same input
        teacher_head: frozen teacher classifier head callable

    Computes:
        transformed_student = connector(feat_student)
        stat_loss = stat_loss(transformed_student, feat_teacher)
        pred_student_teacher_head = teacher_head(transformed_student)
        pred_loss = mse(pred_student_teacher_head, teacher_logits)

    Total:
        stat_weight * stat_loss + pred_weight * pred_loss
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stat_weight: float = 1.0,
        pred_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.connector = SRRLConnector(in_channels, out_channels)
        self.stat_loss_fn = SRRLStatLoss()

        self.stat_weight = float(stat_weight)
        self.pred_weight = float(pred_weight)

    def trainable_parameters(self):
        return self.connector.parameters()

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_head,
        return_dict: bool = False,
    ):
        feat_student = _require_4d(feat_student, "feat_student")
        feat_teacher = _require_4d(feat_teacher, "feat_teacher")

        feat_student, feat_teacher = _align_spatial_by_pooling(
            feat_student,
            feat_teacher,
        )

        trans_feat_student = self.connector(feat_student)
        feat_teacher = feat_teacher.detach()
        teacher_logits = teacher_logits.detach()

        stat_loss = self.stat_loss_fn(trans_feat_student, feat_teacher)

        # Important: do not wrap this in no_grad.
        # Teacher head params are frozen, but gradients must flow through
        # the head operations back to trans_feat_student and connector/student.
        pred_student_in_teacher = teacher_head(trans_feat_student)
        pred_loss = F.mse_loss(pred_student_in_teacher, teacher_logits)

        loss = self.stat_weight * stat_loss + self.pred_weight * pred_loss

        if not return_dict:
            return loss

        return loss, {
            "srrl_stat_loss": stat_loss.detach(),
            "srrl_pred_loss": pred_loss.detach(),
            "srrl_logits": pred_student_in_teacher,
        }
