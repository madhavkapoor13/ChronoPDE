import pytest
import torch

from chronopde.training.losses import velocity_loss


def test_velocity_loss_zero_and_gradients() -> None:
    prediction = torch.randn(2, 2, 16, 16, requires_grad=True)
    exact = velocity_loss(prediction, prediction, modes_y=4, modes_x=4)
    torch.testing.assert_close(exact.total, torch.tensor(0.0))
    target = torch.zeros_like(prediction)
    result = velocity_loss(prediction, target, modes_y=4, modes_x=4)
    assert torch.isfinite(result.total)
    assert result.total > result.physical_mse
    result.total.backward()
    assert prediction.grad is not None


def test_velocity_loss_validation() -> None:
    with pytest.raises(ValueError, match="same"):
        velocity_loss(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 7, 8))
    with pytest.raises(ValueError, match="do not fit"):
        velocity_loss(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), modes_y=9)
