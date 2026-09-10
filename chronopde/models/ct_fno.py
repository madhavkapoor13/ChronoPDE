"""FiLM-conditioned continuous-time FFT neural operator."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from chronopde.models.fno import FourierSpectralConv2d


class FiLMConditioner(nn.Module):
    def __init__(self, width: int, blocks: int, hidden_width: int = 128) -> None:
        super().__init__()
        self.width = width
        self.blocks = blocks
        self.network = nn.Sequential(
            nn.Linear(4, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, blocks * 2 * width),
        )

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != 4:
            raise ValueError("FiLM condition must have shape [B,4]")
        values = self.network(condition).reshape(-1, self.blocks, 2, self.width)
        return values[:, :, 0], values[:, :, 1]


class ContinuousFourierBlock(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int, activation: bool) -> None:
        super().__init__()
        self.spectral = FourierSpectralConv2d(width, modes_y, modes_x)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activation = activation

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        if scale.shape != x.shape[:2] or bias.shape != x.shape[:2]:
            raise ValueError("FiLM scale and bias must have shape [B,C]")
        result = self.spectral(x) + self.pointwise(x)
        result = result * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return functional.gelu(result) if self.activation else result


class FFTContinuousVectorField(nn.Module):
    def __init__(
        self,
        width: int = 29,
        modes_y: int = 12,
        modes_x: int = 12,
        blocks: int = 4,
        film_hidden_width: int = 128,
        time_range: tuple[float, float] = (0.0, 50.0),
    ) -> None:
        super().__init__()
        if blocks < 1 or time_range[1] <= time_range[0]:
            raise ValueError("blocks and time range must be valid")
        self.time_start, self.time_end = time_range
        self.lift = nn.Conv2d(2, width, 1)
        self.conditioner = FiLMConditioner(width, blocks, film_hidden_width)
        self.blocks = nn.ModuleList(
            ContinuousFourierBlock(width, modes_y, modes_x, activation=index < blocks - 1)
            for index in range(blocks)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, 2, 1)
        )

    def forward(self, state: Tensor, time: Tensor, parameters: Tensor) -> Tensor:
        if state.ndim != 4 or state.shape[1] != 2:
            raise ValueError("state must have shape [B,2,H,W]")
        if time.shape != (state.shape[0],) or parameters.shape != (state.shape[0], 3):
            raise ValueError("time and parameters must have shapes [B] and [B,3]")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        if not bool(torch.isfinite(time).all()) or not bool(torch.isfinite(parameters).all()):
            raise ValueError("time and parameters must be finite")
        normalized_time = 2 * (time - self.time_start) / (self.time_end - self.time_start) - 1
        condition = torch.cat((normalized_time[:, None], parameters), dim=1).to(state)
        scales, biases = self.conditioner(condition)
        x = self.lift(state)
        for index, block in enumerate(self.blocks):
            x = block(x, scales[:, index], biases[:, index])
        return cast(Tensor, self.projection(x))
