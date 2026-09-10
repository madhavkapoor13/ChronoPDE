"""Differentiable fixed-step Euler, Heun, and RK4 integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from chronopde.contracts import IntegrationResult

VectorField = Callable[[Tensor, Tensor, Tensor | None], Tensor]


def euler_step(
    vector_field: VectorField,
    state: Tensor,
    time: Tensor,
    dt: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    return state + dt * vector_field(state, time, condition)


def heun_step(
    vector_field: VectorField,
    state: Tensor,
    time: Tensor,
    dt: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    k1 = vector_field(state, time, condition)
    k2 = vector_field(state + dt * k1, time + dt, condition)
    return state + 0.5 * dt * (k1 + k2)


def rk4_step(
    vector_field: VectorField,
    state: Tensor,
    time: Tensor,
    dt: Tensor,
    condition: Tensor | None = None,
) -> Tensor:
    k1 = vector_field(state, time, condition)
    k2 = vector_field(state + 0.5 * dt * k1, time + 0.5 * dt, condition)
    k3 = vector_field(state + 0.5 * dt * k2, time + 0.5 * dt, condition)
    k4 = vector_field(state + dt * k3, time + dt, condition)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_fixed_step(
    vector_field: VectorField,
    initial_state: Tensor,
    query_times: Tensor,
    steps_per_interval: int,
    method: Literal["euler", "heun", "rk4"],
    condition: Tensor | None = None,
) -> IntegrationResult:
    if initial_state.ndim < 1 or not initial_state.is_floating_point():
        raise ValueError("initial_state must be a batched floating-point tensor")
    if not bool(torch.isfinite(initial_state).all()):
        raise ValueError("initial state must be finite")
    if query_times.ndim != 1 or len(query_times) < 1:
        raise ValueError("query_times must be a non-empty one-dimensional tensor")
    query_times = query_times.to(device=initial_state.device, dtype=initial_state.dtype)
    if not bool(torch.isfinite(query_times).all()):
        raise ValueError("query times must be finite")
    if len(query_times) > 1 and not bool((torch.diff(query_times) > 0).all()):
        raise ValueError("query times must be strictly increasing")
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be positive")
    methods = {"euler": (euler_step, 1), "heun": (heun_step, 2), "rk4": (rk4_step, 4)}
    if method not in methods:
        raise ValueError(f"unsupported integration method: {method}")
    step, evaluations = methods[method]
    state = initial_state
    states = [state]
    for interval_index in range(len(query_times) - 1):
        dt = (query_times[interval_index + 1] - query_times[interval_index]) / steps_per_interval
        for substep in range(steps_per_interval):
            scalar_time = query_times[interval_index] + substep * dt
            time = scalar_time.expand(initial_state.shape[0])
            state = step(vector_field, state, time, dt, condition)
        states.append(state)
    nfev = (len(query_times) - 1) * steps_per_interval * evaluations
    return IntegrationResult(states=torch.stack(states, dim=1), times=query_times, nfev=nfev)
