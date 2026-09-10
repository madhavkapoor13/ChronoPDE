"""Metrics used by baseline training and ID evaluation."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def nrmse(prediction: Tensor, target: Tensor, dimensions: tuple[int, ...]) -> Tensor:
    numerator = torch.mean(torch.square(prediction - target), dim=dimensions).sqrt()
    denominator = torch.mean(torch.square(target), dim=dimensions).sqrt().clamp_min(1e-12)
    return numerator / denominator


def rollout_nrmse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("rollout tensors must have matching [B,T,C,H,W] shapes")
    return nrmse(prediction, target, (2, 3, 4))


def relative_l2_per_trajectory(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("trajectory tensors must have matching shapes")
    error = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    scale = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return cast(Tensor, error / scale)
