import pytest
import torch

from chronopde.numerics.splines import (
    build_quintic_spline,
    estimate_knot_derivatives,
    evaluate_quintic_spline,
    sample_conditional_path,
)


def _quadratic_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.tensor([0.0, 0.2, 0.7, 1.5, 2.0], dtype=torch.float64)
    values = (times.square() + 2 * times + 1).reshape(1, -1, 1)
    return values, times


def test_derivatives_are_exact_for_irregular_quadratic() -> None:
    states, times = _quadratic_fixture()
    first, second = estimate_knot_derivatives(states, times)
    torch.testing.assert_close(first[0, :, 0], 2 * times + 2, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(second[0, :, 0], torch.full_like(times, 2), rtol=1e-12, atol=1e-12)


def test_quintic_interpolates_knots_and_analytic_derivative() -> None:
    states, times = _quadratic_fixture()
    spline = build_quintic_spline(states, times)
    values, derivatives = evaluate_quintic_spline(spline, times)
    torch.testing.assert_close(values, states, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(derivatives[0, :, 0], 2 * times + 2, rtol=1e-11, atol=1e-11)
    query = torch.tensor([0.31, 1.11], dtype=torch.float64)
    epsilon = 1e-6
    upper, _ = evaluate_quintic_spline(spline, query + epsilon)
    lower, _ = evaluate_quintic_spline(spline, query - epsilon)
    _, analytic = evaluate_quintic_spline(spline, query)
    torch.testing.assert_close((upper - lower) / (2 * epsilon), analytic, rtol=1e-7, atol=1e-7)


def test_conditional_perturbation_vanishes_at_endpoints() -> None:
    states, times = _quadratic_fixture()
    spline = build_quintic_spline(states, times)
    baseline, baseline_derivative = evaluate_quintic_spline(spline, times)
    noise = torch.ones_like(baseline)
    path, target = sample_conditional_path(spline, times, noise)
    torch.testing.assert_close(path, baseline)
    torch.testing.assert_close(target, baseline_derivative)


def test_spline_validation() -> None:
    states = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="increasing"):
        build_quintic_spline(states, torch.tensor([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="three"):
        build_quintic_spline(torch.zeros(1, 2, 2), torch.tensor([0.0, 1.0]))
