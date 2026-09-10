"""Shared autoregressive prediction and rollout functions."""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor

from chronopde.contracts import IntegrationResult
from chronopde.numerics import integrate_fixed_step


class AutoregressiveModel(Protocol):
    def __call__(self, state: Tensor, delta_t: Tensor, parameters: Tensor) -> Tensor: ...


class ContinuousModel(Protocol):
    def __call__(self, state: Tensor, time: Tensor, parameters: Tensor) -> Tensor: ...


def predict_next_state(
    model: AutoregressiveModel,
    state: Tensor,
    delta_t: Tensor,
    parameters: Tensor,
) -> Tensor:
    residual = model(state, delta_t, parameters)
    if residual.shape != state.shape:
        raise ValueError("model residual must have the same shape as the input state")
    return state + residual


def autoregressive_rollout(
    model: AutoregressiveModel,
    initial_state: Tensor,
    query_times: Tensor,
    parameters: Tensor,
    *,
    reference_dt: float = 0.5,
) -> Tensor:
    if initial_state.ndim != 4 or parameters.shape != (initial_state.shape[0], 3):
        raise ValueError("initial_state and parameters have incompatible shapes")
    if query_times.ndim == 1:
        query_times = query_times.unsqueeze(0).expand(initial_state.shape[0], -1)
    if query_times.ndim != 2 or query_times.shape[0] != initial_state.shape[0]:
        raise ValueError("query_times must have shape [T] or [B,T]")
    if reference_dt <= 0 or not bool((torch.diff(query_times, dim=1) > 0).all()):
        raise ValueError("query times must increase and reference_dt must be positive")
    state = initial_state
    states = [state]
    for index in range(query_times.shape[1] - 1):
        delta_t = (query_times[:, index + 1] - query_times[:, index]) / reference_dt
        state = predict_next_state(model, state, delta_t.to(state), parameters)
        states.append(state)
    return torch.stack(states, dim=1)


def persistence_rollout(initial_state: Tensor, time_count: int) -> Tensor:
    if time_count < 1:
        raise ValueError("time_count must be positive")
    return initial_state[:, None].expand(-1, time_count, *initial_state.shape[1:])


def continuous_rollout(
    model: ContinuousModel,
    initial_state: Tensor,
    query_times: Tensor,
    parameters: Tensor,
    *,
    steps_per_interval: int = 2,
    method: Literal["euler", "heun", "rk4"] = "rk4",
) -> IntegrationResult:
    if query_times.ndim == 2:
        if not bool(torch.allclose(query_times, query_times[:1].expand_as(query_times))):
            raise ValueError("continuous rollout requires a shared query-time grid")
        query_times = query_times[0]
    if query_times.ndim != 1:
        raise ValueError("query_times must have shape [T] or a shared [B,T]")

    def vector_field(state: Tensor, time: Tensor, condition: Tensor | None) -> Tensor:
        if condition is None:
            raise ValueError("continuous rollout requires parameters")
        return model(state, time, condition)

    return integrate_fixed_step(
        vector_field,
        initial_state,
        query_times,
        steps_per_interval,
        method,
        parameters,
    )
