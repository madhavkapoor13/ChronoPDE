"""Batched CFO-style quintic paths for irregularly sampled trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QuinticSpline:
    coefficients: Tensor
    start_times: Tensor
    end_times: Tensor


def _batched_times(states: Tensor, times: Tensor) -> Tensor:
    if states.ndim < 3:
        raise ValueError("states must have shape [B,T,...]")
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(states.shape[0], -1)
    if times.shape != states.shape[:2]:
        raise ValueError("times must have shape [T] or [B,T]")
    return times.to(device=states.device, dtype=states.dtype)


def _validate(states: Tensor, times: Tensor) -> Tensor:
    if not states.is_floating_point() or not times.is_floating_point():
        raise TypeError("states and times must be floating-point tensors")
    if states.shape[1] < 3:
        raise ValueError("at least three spline knots are required")
    times = _batched_times(states, times)
    if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(times).all()):
        raise ValueError("spline inputs must be finite")
    if not bool((torch.diff(times, dim=1) > 0).all()):
        raise ValueError("times must be strictly increasing")
    return times


def _expand(values: Tensor, state_ndim: int) -> Tensor:
    return values.reshape(values.shape + (1,) * (state_ndim - 2))


def estimate_knot_derivatives(states: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate first and second physical-time derivatives at every knot."""

    times = _validate(states, times)
    h = _expand(torch.diff(times, dim=1), states.ndim)
    h0, h1 = h[:, :-1], h[:, 1:]
    first_centre = (
        -h1 / (h0 * (h0 + h1)) * states[:, :-2]
        + (h1 - h0) / (h0 * h1) * states[:, 1:-1]
        + h0 / (h1 * (h0 + h1)) * states[:, 2:]
    )
    left0, left1 = h[:, 0], h[:, 1]
    first_left = (
        -(2 * left0 + left1) / (left0 * (left0 + left1)) * states[:, 0]
        + (left0 + left1) / (left0 * left1) * states[:, 1]
        - left0 / (left1 * (left0 + left1)) * states[:, 2]
    )
    right0, right1 = h[:, -2], h[:, -1]
    first_right = (
        right1 / (right0 * (right0 + right1)) * states[:, -3]
        - (right0 + right1) / (right0 * right1) * states[:, -2]
        + (2 * right1 + right0) / (right1 * (right0 + right1)) * states[:, -1]
    )
    first = torch.cat((first_left[:, None], first_centre, first_right[:, None]), dim=1)

    second_centre = (
        2
        * ((states[:, 2:] - states[:, 1:-1]) / h1 - (states[:, 1:-1] - states[:, :-2]) / h0)
        / (h0 + h1)
    )
    second_left = 2 * (
        states[:, 0] / (left0 * (left0 + left1))
        - states[:, 1] / (left0 * left1)
        + states[:, 2] / (left1 * (left0 + left1))
    )
    second_right = 2 * (
        states[:, -3] / (right0 * (right0 + right1))
        - states[:, -2] / (right0 * right1)
        + states[:, -1] / (right1 * (right0 + right1))
    )
    second = torch.cat((second_left[:, None], second_centre, second_right[:, None]), dim=1)
    return first, second


def build_quintic_spline(states: Tensor, times: Tensor) -> QuinticSpline:
    times = _validate(states, times)
    first, second = estimate_knot_derivatives(states, times)
    dt = _expand(torch.diff(times, dim=1), states.ndim)
    a0 = states[:, :-1]
    a1 = first[:, :-1] * dt
    a2 = 0.5 * second[:, :-1] * dt.square()
    s1 = states[:, 1:] - a0 - a1 - a2
    s2 = first[:, 1:] * dt - a1 - 2 * a2
    s3 = second[:, 1:] * dt.square() - 2 * a2
    a3 = 10 * s1 - 4 * s2 + 0.5 * s3
    a4 = -15 * s1 + 7 * s2 - s3
    a5 = 6 * s1 - 3 * s2 + 0.5 * s3
    coefficients = torch.stack((a0, a1, a2, a3, a4, a5), dim=2)
    return QuinticSpline(coefficients, times[:, :-1], times[:, 1:])


def _evaluate(spline: QuinticSpline, query_times: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if query_times.ndim == 1:
        query_times = query_times.unsqueeze(0).expand(spline.start_times.shape[0], -1)
    if query_times.ndim != 2 or query_times.shape[0] != spline.start_times.shape[0]:
        raise ValueError("query_times must have shape [Q] or [B,Q]")
    query_times = query_times.to(spline.start_times)
    if not bool(torch.isfinite(query_times).all()):
        raise ValueError("query times must be finite")
    outputs: list[Tensor] = []
    derivatives: list[Tensor] = []
    taus: list[Tensor] = []
    for batch in range(query_times.shape[0]):
        query = query_times[batch]
        if bool((query < spline.start_times[batch, 0]).any()) or bool(
            (query > spline.end_times[batch, -1]).any()
        ):
            raise ValueError("query times must lie inside the spline domain")
        indices = torch.searchsorted(spline.start_times[batch], query, right=True) - 1
        indices = indices.clamp(0, spline.start_times.shape[1] - 1)
        start = spline.start_times[batch, indices]
        dt = spline.end_times[batch, indices] - start
        tau = (query - start) / dt
        coeff = spline.coefficients[batch, indices]
        shape = tau.shape + (1,) * (coeff.ndim - 2)
        z = tau.reshape(shape)
        value = (
            (((coeff[:, 5] * z + coeff[:, 4]) * z + coeff[:, 3]) * z + coeff[:, 2]) * z
            + coeff[:, 1]
        ) * z + coeff[:, 0]
        derivative = (
            coeff[:, 1]
            + 2 * coeff[:, 2] * z
            + 3 * coeff[:, 3] * z.square()
            + 4 * coeff[:, 4] * z.pow(3)
            + 5 * coeff[:, 5] * z.pow(4)
        ) / dt.reshape(shape)
        outputs.append(value)
        derivatives.append(derivative)
        taus.append(tau)
    return torch.stack(outputs), torch.stack(derivatives), torch.stack(taus)


def evaluate_quintic_spline(spline: QuinticSpline, query_times: Tensor) -> tuple[Tensor, Tensor]:
    values, derivatives, _ = _evaluate(spline, query_times)
    return values, derivatives


def sample_conditional_path(
    spline: QuinticSpline,
    query_times: Tensor,
    noise: Tensor,
    gamma: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    values, derivatives, tau = _evaluate(spline, query_times)
    if noise.shape != values.shape:
        raise ValueError("noise must have the same shape as evaluated path states")
    interval = spline.end_times - spline.start_times
    dt_values: list[Tensor] = []
    query = (
        query_times if query_times.ndim == 2 else query_times.unsqueeze(0).expand(tau.shape[0], -1)
    )
    for batch in range(tau.shape[0]):
        indices = (
            torch.searchsorted(
                spline.start_times[batch], query[batch].to(spline.start_times), right=True
            )
            - 1
        )
        dt_values.append(interval[batch, indices.clamp(0, interval.shape[1] - 1)])
    dt = torch.stack(dt_values).reshape(tau.shape + (1,) * (values.ndim - 2))
    z = tau.reshape(tau.shape + (1,) * (values.ndim - 2))
    perturbation = gamma * z.pow(3) * (1 - z).pow(3)
    perturbation_derivative = gamma * 3 / dt * z.square() * (1 - z).square() * (1 - 2 * z)
    return values + perturbation * noise, derivatives + perturbation_derivative * noise
