"""Shared autoregressive prediction and rollout functions."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class AutoregressiveModel(Protocol):
    def __call__(self, state: Tensor, delta_t: Tensor, parameters: Tensor) -> Tensor: ...


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
