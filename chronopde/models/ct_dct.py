"""Boundary-aware continuous-time neural operator using cosine modes."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from chronopde.models.ct_fno import FiLMConditioner
from chronopde.numerics import dct2, idct2


class CosineSpectralConv2d(nn.Module):
    """Real-valued spectral convolution in an orthonormal DCT basis."""

    def __init__(self, channels: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        if channels < 1 or modes_y < 1 or modes_x < 1:
            raise ValueError("channels and retained cosine modes must be positive")
        self.channels = channels
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes_y, modes_x)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("spectral input must have shape [B,C,H,W]")
        _, _, height, width = x.shape
        if self.modes_y > height or self.modes_x > width:
            raise ValueError("retained cosine modes do not fit the spatial grid")
        spectrum = dct2(x)
        output = torch.zeros(
            x.shape[0],
            self.channels,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
        )
        output[:, :, : self.modes_y, : self.modes_x] = torch.einsum(
            "biyx,ioyx->boyx",
            spectrum[:, :, : self.modes_y, : self.modes_x],
            self.weight,
        )
        return idct2(output)


class ContinuousCosineBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes_y: int,
        modes_x: int,
        activation: bool,
        residual_skip: bool = False,
    ) -> None:
        super().__init__()
        self.spectral = CosineSpectralConv2d(width, modes_y, modes_x)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activation = activation
        self.residual_skip = residual_skip

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        if scale.shape != x.shape[:2] or bias.shape != x.shape[:2]:
            raise ValueError("FiLM scale and bias must have shape [B,C]")
        result = self.spectral(x) + self.pointwise(x)
        if self.residual_skip:
            result = result + x
        result = result * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return functional.gelu(result) if self.activation else result


class DCTContinuousVectorField(nn.Module):
    """FiLM-conditioned velocity field aligned with Neumann boundaries."""

    def __init__(
        self,
        width: int = 57,
        modes_y: int = 12,
        modes_x: int = 12,
        blocks: int = 4,
        film_hidden_width: int = 128,
        time_range: tuple[float, float] = (0.0, 50.0),
        residual_skip: bool = False,
    ) -> None:
        super().__init__()
        if blocks < 1 or time_range[1] <= time_range[0]:
            raise ValueError("blocks and time range must be valid")
        self.time_start, self.time_end = time_range
        self.lift = nn.Conv2d(2, width, 1)
        self.conditioner = FiLMConditioner(width, blocks, film_hidden_width)
        self.blocks = nn.ModuleList(
            ContinuousCosineBlock(
                width,
                modes_y,
                modes_x,
                activation=index < blocks - 1,
                residual_skip=residual_skip,
            )
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
