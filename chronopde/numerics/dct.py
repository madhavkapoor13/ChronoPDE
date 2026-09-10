"""Differentiable orthonormal DCT-II and inverse DCT-III transforms."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor


def dct_1d(x: Tensor, dim: int = -1) -> Tensor:
    """Apply an orthonormal DCT-II along ``dim`` using one complex FFT."""

    if not x.is_floating_point():
        raise TypeError("DCT input must be a floating-point tensor")
    dim = dim % x.ndim
    moved = x.movedim(dim, -1)
    n = moved.shape[-1]
    if n < 1:
        raise ValueError("DCT dimension cannot be empty")
    reordered = torch.cat((moved[..., ::2], moved[..., 1::2].flip(-1)), dim=-1)
    spectrum = torch.fft.fft(reordered, dim=-1)
    angles = -torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2 * n))
    values = spectrum.real * torch.cos(angles) - spectrum.imag * torch.sin(angles)
    scale = torch.full((n,), math.sqrt(2.0 / n), device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(1.0 / n)
    return cast(Tensor, (values * scale).movedim(-1, dim))


def idct_1d(x: Tensor, dim: int = -1) -> Tensor:
    """Apply the orthonormal DCT-III, inverse of :func:`dct_1d`."""

    if not x.is_floating_point():
        raise TypeError("IDCT input must be a floating-point tensor")
    dim = dim % x.ndim
    moved = x.movedim(dim, -1)
    n = moved.shape[-1]
    if n < 1:
        raise ValueError("IDCT dimension cannot be empty")
    unscaled = moved.clone()
    unscaled[..., 0] *= math.sqrt(n)
    if n > 1:
        unscaled[..., 1:] *= math.sqrt(n / 2.0)
    imaginary = torch.cat((torch.zeros_like(unscaled[..., :1]), -unscaled.flip(-1)[..., :-1]), -1)
    angles = torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2 * n))
    real = unscaled * torch.cos(angles) - imaginary * torch.sin(angles)
    imag = unscaled * torch.sin(angles) + imaginary * torch.cos(angles)
    reordered = torch.fft.ifft(torch.complex(real, imag), dim=-1).real
    restored = torch.empty_like(reordered)
    restored[..., ::2] = reordered[..., : n - n // 2]
    restored[..., 1::2] = reordered.flip(-1)[..., : n // 2]
    return restored.movedim(-1, dim)


def dct2(x: Tensor) -> Tensor:
    """Apply an orthonormal 2D DCT-II over the last two dimensions."""

    if x.ndim < 2:
        raise ValueError("dct2 requires at least two dimensions")
    return dct_1d(dct_1d(x, -1), -2)


def idct2(x: Tensor) -> Tensor:
    """Apply the inverse orthonormal 2D DCT over the last two dimensions."""

    if x.ndim < 2:
        raise ValueError("idct2 requires at least two dimensions")
    return idct_1d(idct_1d(x, -1), -2)
