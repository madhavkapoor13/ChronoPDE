import pytest
import torch
from torch import nn

from chronopde.evaluation import continuous_rollout


class ConstantField(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        del time, parameters
        return torch.ones_like(state)


def test_continuous_rollout_uses_physical_time_and_exact_nfe() -> None:
    initial = torch.zeros(2, 2, 4, 4, requires_grad=True)
    times = torch.tensor([[0.0, 0.5, 2.0], [0.0, 0.5, 2.0]])
    result = continuous_rollout(
        ConstantField(), initial, times, torch.zeros(2, 3), steps_per_interval=2
    )
    states = result.states
    assert isinstance(states, torch.Tensor)
    torch.testing.assert_close(states[:, -1], torch.full_like(initial, 2.0))
    assert result.nfev == 16
    states.sum().backward()
    assert initial.grad is not None


def test_continuous_rollout_rejects_per_sample_time_grids() -> None:
    times = torch.tensor([[0.0, 1.0], [0.0, 1.1]])
    with pytest.raises(ValueError, match="shared"):
        continuous_rollout(ConstantField(), torch.zeros(2, 2, 4, 4), times, torch.zeros(2, 3))
