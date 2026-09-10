import math

import pytest
import torch

from chronopde.numerics.integrators import integrate_fixed_step


def exponential_field(
    state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor | None
) -> torch.Tensor:
    del time, condition
    return state


@pytest.mark.parametrize(
    ("method", "minimum_ratio", "evaluations"),
    [("euler", 1.8, 1), ("heun", 3.5, 2), ("rk4", 12.0, 4)],
)
def test_integrator_convergence_and_nfe(
    method: str, minimum_ratio: float, evaluations: int
) -> None:
    initial = torch.ones(2, 1, dtype=torch.float64)
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    coarse = integrate_fixed_step(exponential_field, initial, times, 4, method)  # type: ignore[arg-type]
    fine = integrate_fixed_step(exponential_field, initial, times, 8, method)  # type: ignore[arg-type]
    exact = math.e
    coarse_error = abs(coarse.states[0, -1, 0].item() - exact)
    fine_error = abs(fine.states[0, -1, 0].item() - exact)
    assert coarse_error / fine_error > minimum_ratio
    assert coarse.nfev == 4 * evaluations
    assert fine.states.shape == (2, 2, 1)


def test_integrator_supports_irregular_queries_and_gradients() -> None:
    initial = torch.ones(1, 2, requires_grad=True)
    times = torch.tensor([0.0, 0.2, 1.0])
    result = integrate_fixed_step(exponential_field, initial, times, 2, "rk4")
    result.states[:, -1].sum().backward()
    assert initial.grad is not None
    assert result.nfev == 16
    torch.testing.assert_close(result.times, times)


def test_integrator_validation() -> None:
    initial = torch.ones(1, 1)
    with pytest.raises(ValueError, match="increasing"):
        integrate_fixed_step(exponential_field, initial, torch.tensor([0.0, 0.0]), 1, "euler")
    with pytest.raises(ValueError, match="positive"):
        integrate_fixed_step(exponential_field, initial, torch.tensor([0.0, 1.0]), 0, "euler")
