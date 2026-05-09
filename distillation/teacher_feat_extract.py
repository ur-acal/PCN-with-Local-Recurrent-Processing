import logging
import math
from typing import Optional


import torch
from torch import nn
import torch.nn.functional as F


def _require_4d(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a tensor, got {type(x)}.")
    if x.dim() != 4:
        raise ValueError(
            f"{name} must be a 4D feature map [B,C,H,W], got shape {tuple(x.shape)}."
        )
    return x


def _align_spatial_by_pooling(
    feat_student: torch.Tensor,
    feat_teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align spatial sizes by adaptive average pooling the larger map
    to the smaller H/W.

    This lets PCNet and EfficientNet have different feature resolutions.
    """
    feat_student = _require_4d(feat_student, "feat_student")
    feat_teacher = _require_4d(feat_teacher, "feat_teacher")

    hs, ws = feat_student.shape[-2:]
    ht, wt = feat_teacher.shape[-2:]

    if (hs, ws) == (ht, wt):
        return feat_student, feat_teacher

    target_size = (min(hs, ht), min(ws, wt))

    if feat_student.shape[-2:] != target_size:
        feat_student = F.adaptive_avg_pool2d(feat_student, target_size)

    if feat_teacher.shape[-2:] != target_size:
        feat_teacher = F.adaptive_avg_pool2d(feat_teacher, target_size)

    return feat_student, feat_teacher


class EfficientNetPyTorchHead(nn.Module):
    """
    Head for efficientnet_pytorch.EfficientNet.

    Expected structure:
        feat = teacher.extract_features(x)
        logits = _avg_pooling -> flatten -> _dropout -> _fc
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.avg_pool = teacher._avg_pooling
        self.dropout = teacher._dropout
        self.fc = teacher._fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Hankyul2EffNetV2Head(nn.Module):
    """
    Head for hankyul2/EfficientNetV2-pytorch.

    We define teacher feature after head[0], so the head is head[1:].
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        children = list(teacher.head.children())
        if len(children) < 2:
            raise RuntimeError("Unexpected hankyul2 EfficientNetV2 head structure.")
        self.head_tail = nn.Sequential(*children[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_tail(x)


class TorchvisionEfficientNetHead(nn.Module):
    """
    Head for torchvision EfficientNet / EfficientNetV2.
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.avgpool = teacher.avgpool
        self.classifier = teacher.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class TeacherFeatureExtractor(nn.Module):
    """
    Extract final 4D teacher feature and provide frozen teacher classifier head.

    Supported:
      1. efficientnet_pytorch EfficientNet
      2. hankyul2/EfficientNetV2-pytorch
      3. torchvision EfficientNet/EfficientNetV2
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher

        if hasattr(teacher, "extract_features") and hasattr(teacher, "_fc"):
            logging.warning("Teacher is efficientnet_pytorch.")
            self.kind = "efficientnet_pytorch"
            self.head = EfficientNetPyTorchHead(teacher)

        elif hasattr(teacher, "stem") and hasattr(teacher, "blocks") and hasattr(teacher, "head"):
            logging.warning("Teacher is hankyul2.")
            self.kind = "hankyul2_effnetv2"
            self.head = Hankyul2EffNetV2Head(teacher)

        elif hasattr(teacher, "features") and hasattr(teacher, "avgpool") and hasattr(teacher, "classifier"):
            logging.warning("Teacher is torchvision_efficientnet.")
            self.kind = "torchvision_efficientnet"
            self.head = TorchvisionEfficientNetHead(teacher)

        else:
            raise ValueError(
                "Unsupported teacher for SRRL feature extraction. "
                "Expected efficientnet_pytorch EfficientNet, hankyul2 EfficientNetV2, "
                "or torchvision EfficientNet/EfficientNetV2."
            )

        self.teacher.eval()
        self.head.eval()

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        for p in self.head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "efficientnet_pytorch":
            return self.teacher.extract_features(x)

        if self.kind == "hankyul2_effnetv2":
            x = self.teacher.stem(x)
            x = self.teacher.blocks(x)
            x = self.teacher.head[0](x)
            return x

        if self.kind == "torchvision_efficientnet":
            return self.teacher.features(x)

        raise RuntimeError(f"Unknown teacher kind: {self.kind}")

    def classify_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Frozen teacher classifier head.

        No no_grad here, because SRRL needs gradients to flow through
        this head back into the transformed student feature.
        """
        self.head.eval()
        return self.head(feat)

    @torch.no_grad()
    def forward_logits_from_feature(self, feat: torch.Tensor) -> torch.Tensor:
        self.head.eval()
        return self.head(feat)

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)
