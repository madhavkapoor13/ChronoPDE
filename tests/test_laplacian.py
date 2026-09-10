from pathlib import Path

import numpy as np

from chronopde.config import PDEConfig, load_config
from chronopde.numerics.laplacian import (
    apply_neumann_stencil,
    build_grid,
    build_neumann_laplacian,
)

ROOT = Path(__file__).resolve().parents[1]


def small_pde(height: int = 4, width: int = 5) -> PDEConfig:
    base = load_config(ROOT / "configs/project.yaml").pde
    return base.model_copy(update={"height": height, "width": width})


def test_laplacian_shape_symmetry_and_null_space() -> None:
    grid = build_grid(small_pde())
    matrix = build_neumann_laplacian(grid)
    assert matrix.shape == (20, 20)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=0, rtol=0)
    residual = matrix @ np.ones(20)
    assert float(np.max(np.abs(residual))) < 1e-12


def test_laplacian_is_negative_semidefinite() -> None:
    grid = build_grid(small_pde(3, 4))
    eigenvalues = np.linalg.eigvalsh(build_neumann_laplacian(grid).toarray())
    assert float(np.max(eigenvalues)) < 1e-10


def test_matrix_matches_explicit_boundary_stencil() -> None:
    grid = build_grid(small_pde())
    rng = np.random.default_rng(42)
    field = rng.standard_normal((grid.height, grid.width))
    expected = apply_neumann_stencil(field, grid)
    actual = (build_neumann_laplacian(grid) @ field.ravel()).reshape(field.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_horizontal_rows_do_not_wrap() -> None:
    grid = build_grid(small_pde())
    matrix = build_neumann_laplacian(grid)
    assert matrix[grid.width - 1, grid.width] == 0
    assert matrix[grid.width, grid.width - 1] == 0

