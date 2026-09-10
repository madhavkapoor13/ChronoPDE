"""From-scratch periodic FFT-FNO autoregressive baseline."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from chronopde.models.common import conditioned_input


class FourierSpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1 / channels
        shape = (channels, channels, modes_y, modes_x)
        self.upper_weight = nn.Parameter(scale * torch.randn(*shape, dtype=torch.complex64))
        self.lower_weight = nn.Parameter(scale * torch.randn(*shape, dtype=torch.complex64))

    @staticmethod
    def _multiply(values: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("biyx,ioyx->boyx", values, weights)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("spectral input must have shape [B,C,H,W]")
        _, _, height, width = x.shape
        if 2 * self.modes_y > height or self.modes_x > width // 2 + 1:
            raise ValueError("retained Fourier modes do not fit the spatial grid")
        spectrum = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(
            x.shape[0],
            self.channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=spectrum.dtype,
        )
        output[:, :, : self.modes_y, : self.modes_x] = self._multiply(
            spectrum[:, :, : self.modes_y, : self.modes_x], self.upper_weight
        )
        output[:, :, -self.modes_y :, : self.modes_x] = self._multiply(
            spectrum[:, :, -self.modes_y :, : self.modes_x], self.lower_weight
        )
        return cast(Tensor, torch.fft.irfft2(output, s=(height, width), norm="ortho"))


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int, activation: bool) -> None:
        super().__init__()
        self.spectral = FourierSpectralConv2d(width, modes_y, modes_x)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        result = self.spectral(x) + self.pointwise(x)
        return functional.gelu(result) if self.activation else result


class FNOAutoregressive(nn.Module):
    def __init__(self, width: int = 29, modes_y: int = 12, modes_x: int = 12) -> None:
        super().__init__()
        self.lift = nn.Conv2d(8, width, 1)
        self.blocks = nn.ModuleList(
            FourierBlock(width, modes_y, modes_x, activation=index < 3) for index in range(4)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, 2, 1)
        )

    def forward(self, state: Tensor, delta_t: Tensor, parameters: Tensor) -> Tensor:
        x = self.lift(conditioned_input(state, delta_t, parameters))
        for block in self.blocks:
            x = block(x)
        return cast(Tensor, self.projection(x))
