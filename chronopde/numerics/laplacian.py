"""Cell-centred finite-volume Laplacian with homogeneous Neumann boundaries."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from chronopde.config import PDEConfig
from chronopde.contracts import Float64Array, GridSpec


def build_grid(config: PDEConfig) -> GridSpec:
    """Build the frozen cell-centred spatial grid."""

    dx = (config.x_max - config.x_min) / config.width
    dy = (config.y_max - config.y_min) / config.height
    x = np.linspace(config.x_min + dx / 2, config.x_max - dx / 2, config.width)
    y = np.linspace(config.y_min + dy / 2, config.y_max - dy / 2, config.height)
    grid = GridSpec(
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        dx=float(dx),
        dy=float(dy),
        height=config.height,
        width=config.width,
    )
    grid.validate()
    return grid


def build_neumann_laplacian(grid: GridSpec) -> csr_matrix:
    """Return a symmetric CSR Laplacian for a row-major cell-centred grid."""

    grid.validate()
    height, width = grid.height, grid.width
    size = height * width
    inv_dx2 = 1.0 / grid.dx**2
    inv_dy2 = 1.0 / grid.dy**2

    main = np.full(size, -2.0 * inv_dx2 - 2.0 * inv_dy2, dtype=np.float64)
    for row in range(height):
        offset = row * width
        main[offset] += inv_dx2
        main[offset + width - 1] += inv_dx2
    main[:width] += inv_dy2
    main[-width:] += inv_dy2

    horizontal = np.full(size - 1, inv_dx2, dtype=np.float64)
    row_boundaries = np.arange(width - 1, size - 1, width)
    horizontal[row_boundaries] = 0.0
    vertical = np.full(size - width, inv_dy2, dtype=np.float64)

    matrix = diags(
        diagonals=(vertical, horizontal, main, horizontal, vertical),
        offsets=(-width, -1, 0, 1, width),
        shape=(size, size),
        format="csr",
        dtype=np.float64,
    )
    return csr_matrix(matrix)


def apply_neumann_stencil(field: Float64Array, grid: GridSpec) -> Float64Array:
    """Explicit reflected-boundary stencil used as an independent test oracle."""

    if field.shape != (grid.height, grid.width):
        raise ValueError("field shape must match the grid")
    padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    horizontal = (padded[1:-1, :-2] - 2 * field + padded[1:-1, 2:]) / grid.dx**2
    vertical = (padded[:-2, 1:-1] - 2 * field + padded[2:, 1:-1]) / grid.dy**2
    return np.asarray(horizontal + vertical, dtype=np.float64)
