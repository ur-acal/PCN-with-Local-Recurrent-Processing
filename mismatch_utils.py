import torch


ADDITIVE_SCALE_MODES = ("max_abs", "rms")


def additive_mismatch_scale(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if tensor.numel() == 0:
        raise ValueError("Cannot compute an additive mismatch scale for an empty tensor.")
    if mode == "max_abs":
        return tensor.abs().max()
    if mode == "rms":
        return tensor.square().mean().sqrt()
    raise ValueError(f"Unsupported additive scale mode: {mode}")
