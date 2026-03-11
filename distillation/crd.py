"""Contrastive Representation Distillation loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .memory import ContrastMemory
from .moco import MoCoContrast, MoCoInfoNCELoss

EPS = 1e-7


@dataclass
class CRDOptions:
    """Hyper-parameters for CRD loss."""

    contrast_method: str = "memory"
    feat_dim: int = 128
    nce_k: int = 16384
    nce_t: float = 0.07
    nce_m: float = 0.5
    n_data: int = 0
    s_dim: Optional[int] = None
    t_dim: Optional[int] = None


class CRDLoss(nn.Module):
    """CRD loss with symmetric contrastive terms."""

    def __init__(self, options: CRDOptions) -> None:
        super().__init__()
        if options.s_dim is None or options.t_dim is None:
            raise ValueError("s_dim and t_dim must be provided for CRDLoss.")
        if options.n_data <= 0:
            raise ValueError("n_data must be positive for CRDLoss.")

        self.embed_s = Embed(options.s_dim, options.feat_dim)
        self.embed_t = Embed(options.t_dim, options.feat_dim)

        self.contrast_method = getattr(options, "contrast_method", "memory")

        if self.contrast_method == "memory":
            self.contrast = ContrastMemory(
                options.feat_dim,
                options.n_data,
                options.nce_k,
                options.nce_t,
                options.nce_m,
            )
            self.criterion_s = ContrastLoss(options.n_data)
            self.criterion_t = ContrastLoss(options.n_data)

        elif self.contrast_method == "moco":
            queue_size = getattr(options, "moco_queue_size", None)
            if queue_size is None:
                queue_size = options.nce_k

            self.contrast = MoCoContrast(
                feature_dim=options.feat_dim,
                queue_size=queue_size,
                temperature=options.nce_t,
            )
            self.criterion_s = MoCoInfoNCELoss()
            self.criterion_t = MoCoInfoNCELoss()

        else:
            raise ValueError(
                f"Unsupported contrast_method: {self.contrast_method}. "
                "Expected 'memory' or 'moco'."
            )

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        indices: torch.Tensor,
        contrast_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embed_s = self.embed_s(feat_student)
        embed_t = self.embed_t(feat_teacher)

        if self.contrast_method == "memory":
            out_s, out_t = self.contrast(embed_s, embed_t, indices, contrast_idx)
        else:  # moco
            out_s, out_t = self.contrast(embed_s, embed_t)

        loss_s = self.criterion_s(out_s)
        loss_t = self.criterion_t(out_t)
        return loss_s + loss_t


class ContrastLoss(nn.Module):
    """Noise-contrastive loss as defined in the CRD paper."""

    def __init__(self, n_data: int) -> None:
        super().__init__()
        self.n_data = n_data

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        num_neg = logits.size(1) - 1
        pn = 1 / float(self.n_data)

        pos = logits.select(1, 0)
        log_d1 = torch.div(pos, pos.add(num_neg * pn + EPS)).log_()

        neg = logits.narrow(1, 1, num_neg)
        log_d0 = torch.div(neg.clone().fill_(num_neg * pn), neg.add(num_neg * pn + EPS)).log_()

        loss = -(log_d1.sum(0) + log_d0.view(-1, 1).sum(0)) / batch_size
        return loss


class Embed(nn.Module):
    """Linear projection followed by L2 normalization."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.l2norm(x)


class Normalize(nn.Module):
    """L2 normalization layer."""

    def __init__(self, power: int = 2) -> None:
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm)
