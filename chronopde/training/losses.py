"""Losses shared by continuous-time spectral models."""

from __future__ import annotations

import torch
from torch import Tensor

from chronopde.contracts import VelocityLossBreakdown
from chronopde.numerics import dct2


def velocity_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    spectral_weight: float = 0.05,
    modes_y: int = 12,
    modes_x: int = 12,
    epsilon: float = 1e-8,
) -> VelocityLossBreakdown:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("velocity tensors must have the same [B,C,H,W] shape")
    if spectral_weight < 0 or epsilon <= 0:
        raise ValueError("spectral_weight and epsilon must be valid")
    if modes_y > prediction.shape[-2] or modes_x > prediction.shape[-1]:
        raise ValueError("spectral modes do not fit the velocity grid")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("velocity tensors must be finite")
    physical_mse = torch.mean(torch.square(prediction - target))
    predicted_coefficients = dct2(prediction)[..., :modes_y, :modes_x]
    target_coefficients = dct2(target)[..., :modes_y, :modes_x]
    numerator = torch.sum(torch.square(predicted_coefficients - target_coefficients), dim=(1, 2, 3))
    denominator = torch.sum(torch.square(target_coefficients), dim=(1, 2, 3)).clamp_min(epsilon)
    spectral_relative_error = torch.mean(numerator / denominator)
    total = physical_mse + spectral_weight * spectral_relative_error
    return VelocityLossBreakdown(total, physical_mse, spectral_relative_error)
