"""Masked Generative Distillation (MGD).

This is adapted for your trainer structure:

    student feature F_s
        -> optional 1x1 align to teacher channels
        -> random mask
        -> 3x3 conv + ReLU + 3x3 conv generator
        -> match teacher feature F_t

The auxiliary MGD modules are training-only. The student inference
architecture is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .teacher_feat_extract import _require_4d, _align_spatial_by_pooling


class MGDLoss(nn.Module):
    """
    Masked Generative Distillation loss.

    Args:
        student_channels: channels of the raw student feature map.
        teacher_channels: channels of the teacher feature map.
        alpha_mgd: overall MGD loss weight, matching the official code's alpha_mgd.
        lambda_mgd: masking ratio. Larger means more features are masked.
        mask_mode:
            "channel": mask shape [N, C, 1, 1], matching the released MGD cls code.
            "spatial": mask shape [N, 1, H, W], closer to pixel/spatial masking.
            "pixel":   mask shape [N, C, H, W], most fine-grained.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        alpha_mgd: float = 7e-5,
        lambda_mgd: float = 0.5,
        mask_mode: str = "channel",
    ) -> None:
        super().__init__()

        self.student_channels = int(student_channels)
        self.teacher_channels = int(teacher_channels)
        self.alpha_mgd = float(alpha_mgd)
        self.lambda_mgd = float(lambda_mgd)
        self.mask_mode = mask_mode.lower()

        if self.mask_mode not in {"channel", "spatial", "pixel"}:
            raise ValueError(
                f"Unsupported mask_mode={mask_mode}. "
                "Use 'channel', 'spatial', or 'pixel'."
            )

        if self.student_channels != self.teacher_channels:
            self.align = nn.Conv2d(
                self.student_channels,
                self.teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(
                self.teacher_channels,
                self.teacher_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.teacher_channels,
                self.teacher_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def trainable_parameters(self):
        return self.parameters()

    def _make_mask(self, feat: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feat.shape

        if self.mask_mode == "channel":
            shape = (n, c, 1, 1)
        elif self.mask_mode == "spatial":
            shape = (n, 1, h, w)
        else:
            shape = (n, c, h, w)

        mask = torch.rand(shape, device=feat.device, dtype=feat.dtype)
        mask = torch.where(
            mask < self.lambda_mgd,
            torch.zeros((), device=feat.device, dtype=feat.dtype),
            torch.ones((), device=feat.device, dtype=feat.dtype),
        )
        return mask

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        return_dict: bool = False,
    ):
        feat_student = _require_4d(feat_student, "feat_student")
        feat_teacher = _require_4d(feat_teacher, "feat_teacher")

        # Your PCNet and EfficientNet teacher can have different final H/W.
        feat_student, feat_teacher = _align_spatial_by_pooling(
            feat_student,
            feat_teacher,
        )

        if self.align is not None:
            feat_student = self.align(feat_student)

        feat_teacher = feat_teacher.detach()

        if feat_student.shape != feat_teacher.shape:
            raise RuntimeError(
                "MGD feature shape mismatch after alignment: "
                f"student={tuple(feat_student.shape)}, "
                f"teacher={tuple(feat_teacher.shape)}."
            )

        mask = self._make_mask(feat_student)
        masked_feat_student = feat_student * mask
        generated_feat = self.generation(masked_feat_student)

        # Match official code style: reduction='sum' / N.
        n = feat_teacher.size(0)
        raw_loss = F.mse_loss(
            generated_feat,
            feat_teacher,
            reduction="sum",
        ) / n

        loss = self.alpha_mgd * raw_loss

        if not return_dict:
            return loss

        return loss, {
            "mgd_raw_loss": raw_loss.detach(),
            "mgd_loss": loss.detach(),
        }
