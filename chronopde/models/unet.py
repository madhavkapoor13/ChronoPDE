"""Reflection-padded residual U-Net autoregressive baseline."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from chronopde.models.common import conditioned_input


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if output_channels % 8:
            raise ValueError("U-Net channels must be divisible by eight")
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, output_channels, 3),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_channels, output_channels, 3),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.layers(x))


class UNetAutoregressive(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("U-Net requires four channel levels")
        c0, c1, c2, c3 = channels
        self.encoder0 = ConvBlock(8, c0)
        self.encoder1 = ConvBlock(c0, c1)
        self.encoder2 = ConvBlock(c1, c2)
        self.bottleneck = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decoder2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decoder1 = ConvBlock(c1 + c1, c1)
        self.up0 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.decoder0 = ConvBlock(c0 + c0, c0)
        self.projection = nn.Conv2d(c0, 2, 1)

    def forward(self, state: Tensor, delta_t: Tensor, parameters: Tensor) -> Tensor:
        x = conditioned_input(state, delta_t, parameters)
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(self.pool(e1))
        latent = self.bottleneck(self.pool(e2))
        d2 = self.decoder2(torch.cat((self.up2(latent), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.up1(d2), e1), dim=1))
        d0 = self.decoder0(torch.cat((self.up0(d1), e0), dim=1))
        return cast(Tensor, self.projection(d0))
