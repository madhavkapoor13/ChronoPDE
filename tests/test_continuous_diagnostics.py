from pathlib import Path

import torch
from torch import Tensor, nn

from chronopde.config import load_config
from chronopde.diagnostics.continuous_gate import _boundary_normal_mse, _velocity_metrics

ROOT = Path(__file__).resolve().parents[1]


class ZeroVelocity(nn.Module):
    def forward(self, state: Tensor, time: Tensor, parameters: Tensor) -> Tensor:
        del time, parameters
        return torch.zeros_like(state)


def test_diagnostic_velocity_metrics_are_per_channel_and_physical() -> None:
    target = torch.stack((torch.ones(2, 4, 4), 2 * torch.ones(2, 4, 4)), dim=1)
    batch = {
        "state": torch.zeros_like(target),
        "target_velocity": target,
        "time": torch.zeros(2),
        "parameters": torch.zeros(2, 3),
    }
    metrics = _velocity_metrics(
        ZeroVelocity(),
        [batch],
        torch.device("cpu"),
        spectral_weight=0.05,
        modes_y=4,
        modes_x=4,
        state_std=torch.tensor([2.0, 3.0]),
    )
    assert metrics["velocity_nrmse"] == 1.0
    assert metrics["velocity_nrmse_u"] == 1.0
    assert metrics["velocity_nrmse_v"] == 1.0
    assert metrics["physical_velocity_mse"] == 20.0
    assert metrics["spectral_relative_error"] == 1.0


def test_boundary_normal_mse_is_zero_for_constant_fields() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    states = torch.ones(2, 5, 2, config.pde.height, config.pde.width)
    assert _boundary_normal_mse(states, config) == 0.0
