"""Memory bank utilities for CRD."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class AliasMethod:
    """Efficient multinomial sampling via alias method."""

    def __init__(self, probs: torch.Tensor) -> None:
        if probs.sum() > 1:
            probs = probs / probs.sum()
        k = len(probs)
        self.prob = torch.zeros(k, dtype=torch.float)
        self.alias = torch.zeros(k, dtype=torch.long)

        smaller: list[int] = []
        larger: list[int] = []
        for idx, prob in enumerate(probs):
            self.prob[idx] = k * prob
            if self.prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for remaining in smaller + larger:
            self.prob[remaining] = 1.0

    def to(self, device: torch.device) -> AliasMethod:
        self.prob = self.prob.to(device=device)
        self.alias = self.alias.to(device=device)
        return self

    def draw(self, n: int) -> torch.Tensor:
        device = self.prob.device
        k = self.alias.numel()
        kk = torch.randint(0, k, (n,), device=device)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)

        bernoulli = torch.bernoulli(prob)
        oq = kk * bernoulli.long()
        oj = alias * (1 - bernoulli.long())
        return oq + oj


class ContrastMemory(nn.Module):
    """Memory buffer that supplies additional negatives for CRD."""

    def __init__(
        self,
        feature_dim: int,
        n_data: int,
        nce_k: int,
        temperature: float = 0.07,
        momentum: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_data = n_data
        self.nce_k = nce_k

        unigrams = torch.ones(n_data)
        self.multinomial = AliasMethod(unigrams)

        stdv = 1.0 / math.sqrt(feature_dim / 3)
        self.register_buffer(
            "params",
            torch.tensor([float(nce_k), temperature, -1.0, -1.0, momentum], dtype=torch.float),
        )
        self.register_buffer(
            "memory_student",
            torch.rand(n_data, feature_dim).mul_(2 * stdv).add_(-stdv),
        )
        self.register_buffer(
            "memory_teacher",
            torch.rand(n_data, feature_dim).mul_(2 * stdv).add_(-stdv),
        )
        self._sync_alias_method_device()

    def _sync_alias_method_device(self) -> None:
        device = self.params.device
        self.multinomial.to(device)

    def _maybe_sync(self) -> None:
        if self.multinomial.prob.device != self.params.device:
            self._sync_alias_method_device()

    def forward(
        self,
        feat_student: torch.Tensor,
        feat_teacher: torch.Tensor,
        indices: torch.Tensor,
        sampled_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_sync()
        device = self.params.device

        batch_size = feat_student.size(0)
        feature_dim = feat_student.size(1)
        k = int(self.params[0].item())
        temperature = self.params[1].item()
        z_student = self.params[2].item()
        z_teacher = self.params[3].item()
        momentum = self.params[4].item()

        if sampled_idx is None:
            sampled_idx = self.multinomial.draw(batch_size * (k + 1)).view(batch_size, -1)
            sampled_idx = sampled_idx.to(device)
            sampled_idx[:, 0].copy_(indices)
        else:
            sampled_idx = sampled_idx.to(device)

        weight_teacher = self.memory_teacher.index_select(0, sampled_idx.view(-1)).view(batch_size, k + 1, feature_dim)
        out_student = torch.bmm(weight_teacher, feat_student.view(batch_size, feature_dim, 1))
        out_student = torch.exp(out_student.div(temperature))

        weight_student = self.memory_student.index_select(0, sampled_idx.view(-1)).view(batch_size, k + 1, feature_dim)
        out_teacher = torch.bmm(weight_student, feat_teacher.view(batch_size, feature_dim, 1))
        out_teacher = torch.exp(out_teacher.div(temperature))

        if z_student < 0:
            self.params[2] = out_student.mean() * self.n_data
            z_student = self.params[2].clone().detach().item()
        if z_teacher < 0:
            self.params[3] = out_teacher.mean() * self.n_data
            z_teacher = self.params[3].clone().detach().item()

        out_student = out_student.div(z_student).contiguous()
        out_teacher = out_teacher.div(z_teacher).contiguous()

        with torch.no_grad():
            pos_student = self.memory_student.index_select(0, indices.view(-1))
            pos_student.mul_(momentum)
            pos_student.add_(feat_student.mul(1 - momentum))
            norm_student = pos_student.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_student = pos_student.div(norm_student)
            self.memory_student.index_copy_(0, indices, updated_student)

            pos_teacher = self.memory_teacher.index_select(0, indices.view(-1))
            pos_teacher.mul_(momentum)
            pos_teacher.add_(feat_teacher.mul(1 - momentum))
            norm_teacher = pos_teacher.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_teacher = pos_teacher.div(norm_teacher)
            self.memory_teacher.index_copy_(0, indices, updated_teacher)

        return out_student, out_teacher
