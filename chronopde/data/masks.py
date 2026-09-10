"""Deterministic temporal observation masks."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from numpy.typing import NDArray

MaskRegime = Literal["full", "irregular_50", "irregular_25"]


def observation_mask(
    time_count: int, trajectory_id: str, regime: MaskRegime, base_seed: int
) -> NDArray[np.bool_]:
    if time_count < 3:
        raise ValueError("time_count must be at least three")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    mask = np.zeros(time_count, dtype=np.bool_)
    mask[[0, -1]] = True
    if regime == "full":
        mask[:] = True
        return mask
    retained = {"irregular_50": 50, "irregular_25": 25}.get(regime)
    if retained is None:
        raise ValueError(f"unsupported observation regime: {regime}")
    retained = min(retained, time_count - 2)
    payload = f"{base_seed}|{trajectory_id}|{regime}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    mask[rng.choice(np.arange(1, time_count - 1), retained, replace=False)] = True
    return mask
