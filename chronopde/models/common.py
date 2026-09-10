"""Shared conditioning and parameter-count helpers for model baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def conditioned_input(state: Tensor, delta_t: Tensor, parameters: Tensor) -> Tensor:
    if state.ndim != 4 or state.shape[1] != 2:
        raise ValueError("state must have shape [B,2,H,W]")
    batch, _, height, width = state.shape
    if delta_t.shape != (batch,) or parameters.shape != (batch, 3):
        raise ValueError("delta_t and parameters must have shapes [B] and [B,3]")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite")
    if not bool(torch.isfinite(delta_t).all()) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("delta_t and parameters must be finite")
    condition = torch.cat((delta_t[:, None], parameters), dim=1)
    condition_map = condition[:, :, None, None].expand(-1, -1, height, width)
    x = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=state.device, dtype=state.dtype)
    y = torch.linspace(
        -1 + 1 / height, 1 - 1 / height, height, device=state.device, dtype=state.dtype
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx, yy)).unsqueeze(0).expand(batch, -1, -1, -1)
    return torch.cat((state, condition_map.to(state), coordinates), dim=1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
