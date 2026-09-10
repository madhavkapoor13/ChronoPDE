import pytest
import torch

from chronopde.evaluation.metrics import relative_l2_per_trajectory, rollout_nrmse
from chronopde.evaluation.rollout import (
    autoregressive_rollout,
    persistence_rollout,
    predict_next_state,
)


class ConstantResidual:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(
        self, state: torch.Tensor, delta_t: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        del parameters
        return self.value * delta_t[:, None, None, None].expand_as(state)


def test_zero_residual_rollout_equals_persistence() -> None:
    initial = torch.randn(2, 2, 8, 8)
    times = torch.tensor([0.0, 0.5, 1.0])
    prediction = autoregressive_rollout(ConstantResidual(0), initial, times, torch.zeros(2, 3))
    torch.testing.assert_close(prediction, persistence_rollout(initial, 3))


def test_constant_residual_accumulates_with_irregular_time() -> None:
    initial = torch.zeros(1, 2, 4, 4)
    times = torch.tensor([0.0, 0.5, 1.5])
    prediction = autoregressive_rollout(ConstantResidual(2), initial, times, torch.zeros(1, 3))
    assert prediction[0, -1, 0, 0, 0].item() == pytest.approx(6.0)
    assert (
        predict_next_state(ConstantResidual(2), initial, torch.ones(1), torch.zeros(1, 3))[
            0, 0, 0, 0
        ].item()
        == 2.0
    )


def test_rollout_metrics() -> None:
    target = torch.ones(2, 3, 2, 4, 4)
    prediction = target * 2
    torch.testing.assert_close(rollout_nrmse(prediction, target), torch.ones(2, 3))
    torch.testing.assert_close(relative_l2_per_trajectory(prediction, target), torch.ones(2))
