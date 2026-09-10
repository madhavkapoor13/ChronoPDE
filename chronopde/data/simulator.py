"""Reference reaction-diffusion simulator."""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from chronopde.config import PDEConfig
from chronopde.contracts import (
    Float64Array,
    FloatArray,
    PhysicalParameters,
    SimulationDiagnostics,
    SimulationResult,
)
from chronopde.numerics.laplacian import build_grid, build_neumann_laplacian


def reaction_diffusion_rhs(
    time_value: float,
    flat_state: Float64Array,
    params: PhysicalParameters,
    laplacian: csr_matrix,
) -> Float64Array:
    """Evaluate the coupled FitzHugh-Nagumo semi-discrete dynamics."""

    del time_value
    params.validate()
    state = np.asarray(flat_state, dtype=np.float64)
    if state.ndim != 1 or state.size % 2:
        raise ValueError("flat_state must be a one-dimensional concatenation of u and v")
    spatial_size = state.size // 2
    if laplacian.shape != (spatial_size, spatial_size):
        raise ValueError("laplacian shape is incompatible with flat_state")
    if not np.all(np.isfinite(state)):
        raise ValueError("flat_state must contain only finite values")
    u = state[:spatial_size]
    v = state[spatial_size:]
    diffusion_u = np.asarray(laplacian.dot(u), dtype=np.float64)
    diffusion_v = np.asarray(laplacian.dot(v), dtype=np.float64)
    du_dt = params.du * diffusion_u + u - u**3 - params.k - v
    dv_dt = params.dv * diffusion_v + u - v
    return np.concatenate((du_dt, dv_dt)).astype(np.float64, copy=False)


def simulate_trajectory(
    initial_state: FloatArray,
    params: PhysicalParameters,
    pde: PDEConfig,
    *,
    divergence_threshold: float = 10.0,
) -> SimulationResult:
    """Integrate one trajectory and return states plus complete diagnostics."""

    params.validate()
    if divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive")
    expected_shape = (2, pde.height, pde.width)
    initial = np.asarray(initial_state)
    if initial.shape != expected_shape:
        raise ValueError(f"initial_state must have shape {expected_shape}")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain only finite values")

    grid = build_grid(pde)
    laplacian = build_neumann_laplacian(grid)
    times = np.linspace(pde.t_start, pde.t_end, pde.stored_times, dtype=np.float64)
    flat_initial = np.concatenate(
        (initial[0].ravel(), initial[1].ravel())
    ).astype(np.float64, copy=False)

    started = time.perf_counter()
    solution = solve_ivp(
        reaction_diffusion_rhs,
        (pde.t_start, pde.t_end),
        flat_initial,
        args=(params, laplacian),
        method=pde.solver.method,
        t_eval=times,
        rtol=pde.solver.rtol,
        atol=pde.solver.atol,
    )
    runtime = time.perf_counter() - started

    complete = solution.y.shape == (2 * pde.height * pde.width, pde.stored_times)
    if complete:
        u = solution.y[: pde.height * pde.width].T.reshape(
            pde.stored_times, pde.height, pde.width
        )
        v = solution.y[pde.height * pde.width :].T.reshape(
            pde.stored_times, pde.height, pde.width
        )
        states64 = np.stack((u, v), axis=1)
    else:
        states64 = np.empty((0, 2, pde.height, pde.width), dtype=np.float64)

    finite = bool(states64.size and np.all(np.isfinite(states64)))
    max_abs_state = float(np.max(np.abs(states64))) if finite else float("inf")
    below_threshold = max_abs_state <= divergence_threshold
    success = bool(solution.success and complete and finite and below_threshold)

    message_parts = [str(solution.message)]
    if not complete:
        message_parts.append("solver did not return every requested time")
    if not finite:
        message_parts.append("trajectory contains non-finite values")
    if finite and not below_threshold:
        message_parts.append(
            f"trajectory exceeded divergence threshold {divergence_threshold:g}"
        )

    diagnostics = SimulationDiagnostics(
        success=success,
        message="; ".join(message_parts),
        nfev=int(solution.nfev),
        runtime_seconds=float(runtime),
        max_abs_state=max_abs_state,
    )
    return SimulationResult(
        states=np.asarray(states64, dtype=np.float32),
        times=np.asarray(solution.t, dtype=np.float64),
        params=params,
        diagnostics=diagnostics,
    )
