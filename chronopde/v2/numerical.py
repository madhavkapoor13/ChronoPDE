"""Independent numerical oracles for ChronoPDE V2 Phase 2."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn, idctn

from chronopde.config import PDEConfig
from chronopde.contracts import GridSpec, PhysicalParameters
from chronopde.data.simulator import reaction_diffusion_rhs
from chronopde.numerics.laplacian import (
    apply_neumann_stencil,
    build_grid,
    build_neumann_laplacian,
)
from chronopde.v2.protocol import NumericalTolerances

Float64 = NDArray[np.float64]
EPSILON = float(np.finfo(np.float64).eps)


def _norm(value: NDArray[np.float64]) -> float:
    return float(np.asarray(np.linalg.norm(value)).item())


def relative_l2(actual: NDArray[np.floating[Any]], expected: NDArray[np.floating[Any]]) -> float:
    """Return a finite relative L2 error with a scale-aware denominator."""

    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    denominator: float = max(_norm(expected64.ravel()), EPSILON)
    return _norm((actual64 - expected64).ravel()) / denominator


def neumann_eigenvalues(grid: GridSpec) -> Float64:
    """Analytical eigenvalues of the cell-centred Neumann FV Laplacian."""

    ky = np.arange(grid.height, dtype=np.float64)
    kx = np.arange(grid.width, dtype=np.float64)
    eig_y = -4.0 * np.sin(np.pi * ky / (2.0 * grid.height)) ** 2 / grid.dy**2
    eig_x = -4.0 * np.sin(np.pi * kx / (2.0 * grid.width)) ** 2 / grid.dx**2
    return np.asarray(eig_y[:, None] + eig_x[None, :], dtype=np.float64)


def apply_dct_neumann_laplacian(field: Float64, grid: GridSpec) -> Float64:
    """Apply the same Laplacian through its orthonormal DCT-II eigensystem."""

    if field.shape != (grid.height, grid.width):
        raise ValueError("field shape must match the grid")
    coefficients = dctn(np.asarray(field, dtype=np.float64), type=2, norm="ortho")
    return np.asarray(
        idctn(coefficients * neumann_eigenvalues(grid), type=2, norm="ortho"),
        dtype=np.float64,
    )


def rhs_from_stencil(state: Float64, params: PhysicalParameters, grid: GridSpec) -> Float64:
    """Evaluate the PDE using the independent reflected-cell stencil."""

    if state.shape != (2, grid.height, grid.width):
        raise ValueError("state must have shape [2,H,W]")
    params.validate()
    u, v = np.asarray(state, dtype=np.float64)
    lap_u = apply_neumann_stencil(u, grid)
    lap_v = apply_neumann_stencil(v, grid)
    return np.asarray(
        np.stack(
            (
                params.du * lap_u + u - u**3 - params.k - v,
                params.dv * lap_v + u - v,
            )
        ),
        dtype=np.float64,
    )


def rhs_from_dct(state: Float64, params: PhysicalParameters, grid: GridSpec) -> Float64:
    """Evaluate the PDE using the analytical DCT Laplacian oracle."""

    if state.shape != (2, grid.height, grid.width):
        raise ValueError("state must have shape [2,H,W]")
    params.validate()
    u, v = np.asarray(state, dtype=np.float64)
    lap_u = apply_dct_neumann_laplacian(u, grid)
    lap_v = apply_dct_neumann_laplacian(v, grid)
    return np.asarray(
        np.stack(
            (
                params.du * lap_u + u - u**3 - params.k - v,
                params.dv * lap_v + u - v,
            )
        ),
        dtype=np.float64,
    )


def exact_discrete_rhs(state: Float64, params: PhysicalParameters, pde: PDEConfig) -> Float64:
    """Evaluate the production sparse-matrix RHS and return ``[2,H,W]``."""

    state64 = np.asarray(state, dtype=np.float64)
    if state64.shape != (2, pde.height, pde.width):
        raise ValueError("state must have shape [2,H,W]")
    grid = build_grid(pde)
    laplacian = build_neumann_laplacian(grid)
    flat = np.concatenate((state64[0].ravel(), state64[1].ravel()))
    result = reaction_diffusion_rhs(0.0, flat, params, laplacian)
    return np.asarray(
        np.stack(
            (
                result[: pde.height * pde.width].reshape(pde.height, pde.width),
                result[pde.height * pde.width :].reshape(pde.height, pde.width),
            )
        ),
        dtype=np.float64,
    )


def normalized_physical_rhs(rhs: Float64, state_std: Float64) -> Float64:
    """Normalize a velocity expressed per physical unit of time."""

    rhs64 = np.asarray(rhs, dtype=np.float64)
    scale = np.asarray(state_std, dtype=np.float64)
    if rhs64.ndim < 3 or rhs64.shape[-3] != 2 or scale.shape != (2,):
        raise ValueError("rhs must end in [2,H,W] and state_std must have shape [2]")
    if not np.all(np.isfinite(rhs64)) or not np.all(np.isfinite(scale)):
        raise ValueError("rhs and state_std must be finite")
    if np.any(scale <= 0):
        raise ValueError("state_std must be positive")
    shape = (1,) * (rhs64.ndim - 3) + (2, 1, 1)
    return np.asarray(rhs64 / scale.reshape(shape), dtype=np.float64)


def run_numerical_audit(pde: PDEConfig, tolerances: NumericalTolerances) -> dict[str, Any]:
    """Run deterministic structural, DCT, RHS, precision, and units checks."""

    grid = build_grid(pde)
    matrix = build_neumann_laplacian(grid)
    rng = np.random.default_rng(20260922)
    first = rng.standard_normal((pde.height, pde.width))
    second = rng.standard_normal((pde.height, pde.width))
    flat_first = first.ravel()
    flat_second = second.ravel()
    scale: float = max(_norm(np.asarray(matrix.data, dtype=np.float64)), EPSILON)
    symmetry_error = _norm(np.asarray((matrix - matrix.T).data, dtype=np.float64)) / scale
    constant_error = _norm(
        np.asarray(matrix @ np.ones(matrix.shape[0]), dtype=np.float64)
    ) / scale
    mass_denominator: float = max(
        _norm(np.asarray(matrix @ flat_first, dtype=np.float64)), EPSILON
    )
    mass_error = abs(float(np.sum(matrix @ flat_first))) / mass_denominator
    green_left = float(np.dot(flat_first, np.asarray(matrix @ flat_second)))
    green_right = float(np.dot(np.asarray(matrix @ flat_first), flat_second))
    green_denominator: float = max(abs(green_left), abs(green_right), EPSILON)
    green_error = abs(green_left - green_right) / green_denominator
    maximum_eigenvalue = float(np.max(neumann_eigenvalues(grid)))
    stencil_error = relative_l2(
        (matrix @ flat_first).reshape(pde.height, pde.width),
        apply_neumann_stencil(first, grid),
    )
    dct_error = relative_l2(
        (matrix @ flat_first).reshape(pde.height, pde.width),
        apply_dct_neumann_laplacian(first, grid),
    )

    state = rng.standard_normal((2, pde.height, pde.width)) * 0.4
    params = PhysicalParameters(pde.du, pde.dv, pde.k)
    sparse_rhs = exact_discrete_rhs(state, params, pde)
    stencil_rhs = rhs_from_stencil(state, params, grid)
    dct_rhs = rhs_from_dct(state, params, grid)
    stencil_rhs_error = relative_l2(stencil_rhs, sparse_rhs)
    dct_rhs_error = relative_l2(dct_rhs, sparse_rhs)
    stored_state = state.astype(np.float32)
    rhs_from_stored = exact_discrete_rhs(stored_state.astype(np.float64), params, pde)
    float32_error = relative_l2(rhs_from_stored, sparse_rhs)
    state_std = np.asarray([0.75, 1.25], dtype=np.float64)
    normalized = normalized_physical_rhs(sparse_rhs, state_std)
    restored = normalized * state_std[:, None, None]
    normalization_error = relative_l2(restored, sparse_rhs)

    structural = {
        "symmetry_relative_error": symmetry_error,
        "constant_nullspace_relative_error": constant_error,
        "diffusion_mass_relative_error": mass_error,
        "green_identity_relative_error": green_error,
        "maximum_analytical_eigenvalue": maximum_eigenvalue,
        "sparse_vs_reflected_stencil_relative_error": stencil_error,
        "sparse_vs_dct_laplacian_relative_error": dct_error,
    }
    rhs = {
        "sparse_vs_stencil_relative_error": stencil_rhs_error,
        "sparse_vs_dct_relative_error": dct_rhs_error,
        "float32_state_rhs_relative_error": float32_error,
        "normalization_roundtrip_relative_error": normalization_error,
        "target_time_units": "per_physical_time",
    }
    structural_values = [
        symmetry_error,
        constant_error,
        mass_error,
        green_error,
        max(maximum_eigenvalue, 0.0),
        stencil_error,
        dct_error,
        normalization_error,
    ]
    passed = bool(
        max(structural_values) <= tolerances.structural_relative
        and max(stencil_rhs_error, dct_rhs_error) <= tolerances.rhs_relative
        and float32_error <= tolerances.float32_relative
    )
    return {
        "passed": passed,
        "structural": structural,
        "rhs_and_units": rhs,
        "tolerances": tolerances.model_dump(mode="json"),
        "basis_statement": (
            "The DCT-II basis diagonalizes the frozen cell-centred Neumann operator. "
            "The complete neural network is basis-aligned, not boundary-enforcing: "
            "pointwise paths, nonlinearities, FiLM, lifting, and projection are not "
            "explicit physical-wall constraints."
        ),
    }
