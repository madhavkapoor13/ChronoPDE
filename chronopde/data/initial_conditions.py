"""Deterministic DCT-band-limited initial conditions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.fft import idctn

from chronopde.config import InitialConditionConfig
from chronopde.contracts import FloatArray, GridSpec

InitialConditionRegime = Literal["train", "ood"]


def _band_limited_field(
    grid: GridSpec,
    rng: np.random.Generator,
    mode_band: tuple[int, int],
    standard_deviation: float,
) -> FloatArray:
    low, high = mode_band
    if high >= min(grid.height, grid.width):
        raise ValueError("initial-condition modes must fit inside the grid")
    coefficients = np.zeros((grid.height, grid.width), dtype=np.float64)
    coefficients[low : high + 1, low : high + 1] = rng.standard_normal(
        (high - low + 1, high - low + 1)
    )
    coefficients[0, 0] = 0.0
    field = idctn(coefficients, type=2, norm="ortho")
    field -= np.mean(field)
    observed_std = float(np.std(field))
    if not np.isfinite(observed_std) or observed_std <= 0:
        raise RuntimeError("initial-condition coefficients produced a degenerate field")
    field *= standard_deviation / observed_std
    return np.asarray(field, dtype=np.float32)


def generate_initial_condition(
    grid: GridSpec,
    seed: int,
    regime: InitialConditionRegime,
    config: InitialConditionConfig | None = None,
) -> FloatArray:
    """Generate a deterministic state ``[u, v]`` with independent RNG streams."""

    grid.validate()
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if regime not in ("train", "ood"):
        raise ValueError("regime must be 'train' or 'ood'")
    ic_config = config or InitialConditionConfig()
    if regime == "train":
        mode_band = ic_config.train_modes
        standard_deviation = ic_config.train_standard_deviation
    else:
        mode_band = ic_config.ood_modes
        standard_deviation = ic_config.ood_standard_deviation

    child_sequences = np.random.SeedSequence(seed).spawn(2)
    fields = [
        _band_limited_field(
            grid,
            np.random.default_rng(child_sequence),
            mode_band,
            standard_deviation,
        )
        for child_sequence in child_sequences
    ]
    return np.stack(fields, axis=0).astype(np.float32, copy=False)
