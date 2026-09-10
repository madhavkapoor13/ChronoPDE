from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dctn

from chronopde.config import load_config
from chronopde.data.initial_conditions import generate_initial_condition
from chronopde.numerics.laplacian import build_grid

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("regime", "expected_std", "band"),
    [("train", 0.5, (1, 8)), ("ood", 0.75, (9, 16))],
)
def test_initial_condition_statistics_and_spectral_support(
    regime: str, expected_std: float, band: tuple[int, int]
) -> None:
    config = load_config(ROOT / "configs/project.yaml")
    grid = build_grid(config.pde)
    state = generate_initial_condition(
        grid,
        1729,
        regime,
        config.data.initial_conditions,  # type: ignore[arg-type]
    )
    assert state.shape == (2, 64, 64)
    assert state.dtype == np.float32
    np.testing.assert_allclose(np.mean(state, axis=(1, 2)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(state, axis=(1, 2)), expected_std, atol=1e-5)

    low, high = band
    for channel in state:
        coefficients = dctn(channel, type=2, norm="ortho")
        mask = np.zeros_like(coefficients, dtype=bool)
        mask[low : high + 1, low : high + 1] = True
        total_energy = float(np.sum(coefficients**2))
        band_energy = float(np.sum(coefficients[mask] ** 2))
        assert band_energy / total_energy > 0.999


def test_initial_conditions_are_deterministic_and_seeded() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    grid = build_grid(config.pde)
    first = generate_initial_condition(grid, 10, "train", config.data.initial_conditions)
    repeated = generate_initial_condition(grid, 10, "train", config.data.initial_conditions)
    different = generate_initial_condition(grid, 11, "train", config.data.initial_conditions)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert not np.array_equal(first[0], first[1])


def test_invalid_seed_and_regime_are_rejected() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    grid = build_grid(config.pde)
    with pytest.raises(ValueError, match="non-negative"):
        generate_initial_condition(grid, -1, "train", config.data.initial_conditions)
    with pytest.raises(ValueError, match="regime"):
        generate_initial_condition(
            grid,
            0,
            "unknown",
            config.data.initial_conditions,  # type: ignore[arg-type]
        )
