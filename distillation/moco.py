import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MoCoInfoNCELoss(nn.Module):
    """InfoNCE loss where the positive logit is in column 0."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MoCoContrast(nn.Module):
    """
    MoCo-style symmetric queue contrast.

    Student branch:
        query      = feat_student
        positive   = feat_teacher
        negatives  = teacher queue

    Teacher branch:
        query      = feat_teacher
        positive   = feat_student
        negatives  = student queue
    """

    def __init__(
        self,
        feature_dim: int,
        queue_size: int,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if queue_size <= 0:
            raise ValueError("queue_size must be positive for MoCoContrast.")

        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.temperature = temperature

        # Original MoCo-style practice: initialize queue with random normalized vectors.
        queue_student = F.normalize(torch.randn(queue_size, feature_dim), dim=1)
        queue_teacher = F.normalize(torch.randn(queue_size, feature_dim), dim=1)

        self.register_buffer("queue_student", queue_student)
        self.register_buffer("queue_teacher", queue_teacher)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
    ) -> None:
        feat_student = F.normalize(feat_student.detach(), dim=1)
        feat_teacher = F.normalize(feat_teacher.detach(), dim=1)

        batch_size = feat_student.size(0)
        qsize = self.queue_size

        if batch_size >= qsize:
            self.queue_student.copy_(feat_student[-qsize:])
            self.queue_teacher.copy_(feat_teacher[-qsize:])
            self.queue_ptr[0] = 0
            return

        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size

        if end <= qsize:
            self.queue_student[ptr:end] = feat_student
            self.queue_teacher[ptr:end] = feat_teacher
        else:
            first = qsize - ptr
            second = batch_size - first

            self.queue_student[ptr:] = feat_student[:first]
            self.queue_student[:second] = feat_student[first:]

            self.queue_teacher[ptr:] = feat_teacher[:first]
            self.queue_teacher[:second] = feat_teacher[first:]

        self.queue_ptr[0] = (ptr + batch_size) % qsize

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_student = F.normalize(feat_student, dim=1)
        feat_teacher = F.normalize(feat_teacher, dim=1)

        # Take immutable snapshots for this forward pass
        queue_teacher = self.queue_teacher.detach().clone()
        queue_student = self.queue_student.detach().clone()

        # Positive logits: [B, 1]
        l_pos_s = torch.sum(feat_student * feat_teacher, dim=1, keepdim=True)
        l_pos_t = torch.sum(feat_teacher * feat_student, dim=1, keepdim=True)

        # Negative logits from queue snapshots: [B, Q]
        l_neg_s = torch.matmul(feat_student, queue_teacher.t())
        l_neg_t = torch.matmul(feat_teacher, queue_student.t())

        # Positive is column 0
        out_s = torch.cat([l_pos_s, l_neg_s], dim=1) / self.temperature
        out_t = torch.cat([l_pos_t, l_neg_t], dim=1) / self.temperature

        # Now update the real queues
        self._dequeue_and_enqueue(feat_student, feat_teacher)

        return out_s, out_t
