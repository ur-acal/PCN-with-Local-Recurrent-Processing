"""Official-style SimKD utilities.

This version follows the implementation idea of SimKD:

    student feature map F_s
      -> transfer module T_phi(F_s)
      -> match teacher feature map F_t by MSE
      -> feed T_phi(F_s) into frozen teacher classifier head

No memory bank is used, so this is compatible with Mixup/CutMix.
"""

from __future__ import annotations

from typing import Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F

from .teacher_feat_extract import _require_4d, _align_spatial_by_pooling


class SimKD(nn.Module):
    """
    Official-style SimKD transfer module.

    Input:
        feat_student: [B, C_s, H_s, W_s]
        feat_teacher: [B, C_t, H_t, W_t]

    Output:
        trans_feat_student: [B, C_t, H, W]
        trans_feat_teacher: [B, C_t, H, W]
        pred_feat_student: teacher-head logits from trans_feat_student
    """

    def __init__(
        self,
        s_n: int,
        t_n: int,
        factor: int = 2,
    ) -> None:
        super().__init__()

        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}.")

        hidden = max(t_n // factor, 1)

        self.transfer = nn.Sequential(
            nn.Conv2d(s_n, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, t_n, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        teacher_head: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_dict: bool = False,
    ):
        feat_student = _require_4d(feat_student, "feat_student")
        feat_teacher = _require_4d(feat_teacher, "feat_teacher")

        feat_student, feat_teacher = _align_spatial_by_pooling(
            feat_student,
            feat_teacher,
        )

        trans_feat_student = self.transfer(feat_student)
        trans_feat_teacher = feat_teacher.detach()

        loss = F.mse_loss(trans_feat_student, trans_feat_teacher)

        pred_feat_student = None
        if teacher_head is not None:
            # teacher_head parameters are frozen, but gradients must flow
            # through its operations back to trans_feat_student.
            pred_feat_student = teacher_head(trans_feat_student)

        if not return_dict:
            return trans_feat_student, trans_feat_teacher, pred_feat_student

        return loss, {
            "trans_feat_student": trans_feat_student,
            "trans_feat_teacher": trans_feat_teacher,
            "simkd_logits": pred_feat_student,
            "simkd_mse_loss": loss.detach(),
        }

    def trainable_parameters(self):
        return self.transfer.parameters()
