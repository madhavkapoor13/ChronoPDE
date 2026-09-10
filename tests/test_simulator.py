import json
from pathlib import Path

import numpy as np
import pytest

from chronopde.config import PDEConfig, load_config
from chronopde.contracts import PhysicalParameters
from chronopde.data.simulator import reaction_diffusion_rhs, simulate_trajectory
from chronopde.numerics.laplacian import build_grid, build_neumann_laplacian

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/pdebench_reference.npz"
METADATA = ROOT / "tests/fixtures/pdebench_reference.json"


def small_pde() -> PDEConfig:
    base = load_config(ROOT / "configs/project.yaml").pde
    return base.model_copy(update={"height": 8, "width": 8, "t_end": 1.0, "stored_times": 5})


def test_uniform_state_rhs_matches_reaction_terms() -> None:
    pde = small_pde()
    grid = build_grid(pde)
    laplacian = build_neumann_laplacian(grid)
    params = PhysicalParameters(0.001, 0.005, 0.005)
    u_value, v_value = 0.25, -0.1
    state = np.concatenate(
        (np.full(64, u_value, dtype=np.float64), np.full(64, v_value, dtype=np.float64))
    )
    rhs = reaction_diffusion_rhs(0.0, state, params, laplacian)
    np.testing.assert_allclose(rhs[:64], u_value - u_value**3 - params.k - v_value)
    np.testing.assert_allclose(rhs[64:], u_value - v_value)


def test_simulator_shape_dtype_metadata_and_determinism() -> None:
    pde = small_pde()
    rng = np.random.default_rng(7)
    initial = rng.standard_normal((2, 8, 8)).astype(np.float32)
    params = PhysicalParameters(pde.du, pde.dv, pde.k)
    first = simulate_trajectory(initial, params, pde)
    second = simulate_trajectory(initial, params, pde)
    assert first.diagnostics.success
    assert first.states.shape == (5, 2, 8, 8)
    assert first.states.dtype == np.float32
    assert first.times.dtype == np.float64
    assert first.diagnostics.nfev > 0
    assert first.diagnostics.runtime_seconds >= 0
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.times, np.linspace(0, 1, 5))


def test_simulator_matches_pdebench_reference_fixture() -> None:
    assert FIXTURE.is_file(), "generate the committed PDEBench fixture first"
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        initial = np.asarray(fixture["initial_state"], dtype=np.float32)
        expected = np.asarray(fixture["states"], dtype=np.float32)
        params = PhysicalParameters(*np.asarray(fixture["params"]).tolist())
    pde = small_pde().model_copy(
        update={
            "solver": small_pde().solver.model_copy(
                update={
                    "method": metadata["solver"],
                    "rtol": metadata["rtol"],
                    "atol": metadata["atol"],
                }
            )
        }
    )
    result = simulate_trajectory(initial, params, pde)
    relative_error = np.linalg.norm(result.states - expected) / np.linalg.norm(expected)
    assert relative_error < 1e-6


def test_invalid_inputs_and_divergence_are_reported() -> None:
    pde = small_pde()
    params = PhysicalParameters(pde.du, pde.dv, pde.k)
    with pytest.raises(ValueError, match="shape"):
        simulate_trajectory(np.zeros((2, 7, 8), dtype=np.float32), params, pde)
    invalid = np.zeros((2, 8, 8), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_trajectory(invalid, params, pde)
    initial = np.full((2, 8, 8), 0.5, dtype=np.float32)
    divergent = simulate_trajectory(initial, params, pde, divergence_threshold=0.1)
    assert not divergent.diagnostics.success
    assert "divergence threshold" in divergent.diagnostics.message
    with pytest.raises(ValueError, match="positive"):
        reaction_diffusion_rhs(
            0.0,
            np.zeros(128),
            PhysicalParameters(-0.1, 0.005, 0.005),
            build_neumann_laplacian(build_grid(pde)),
        )
