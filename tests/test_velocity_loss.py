import pytest
import torch

from chronopde.training.losses import full_field_relative_velocity_loss, velocity_loss


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


def test_full_field_relative_velocity_loss_formula_gradients_and_scale() -> None:
    prediction = torch.tensor([[[[2.0]]], [[[3.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0]]], [[[1.0]]]])
    loss = full_field_relative_velocity_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    scaled = full_field_relative_velocity_loss(7 * prediction, 7 * target)
    torch.testing.assert_close(scaled, loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_full_field_relative_velocity_loss_clamp_and_validation() -> None:
    prediction = torch.ones(1, 2, 2, 2)
    target = torch.zeros_like(prediction)
    loss = full_field_relative_velocity_loss(prediction, target, epsilon=2.0)
    torch.testing.assert_close(loss, torch.tensor(4.0))
    with pytest.raises(ValueError, match="positive"):
        full_field_relative_velocity_loss(prediction, target, epsilon=0.0)
    with pytest.raises(ValueError, match="same"):
        full_field_relative_velocity_loss(prediction, target[..., :1])
    invalid = prediction.clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        full_field_relative_velocity_loss(invalid, target)


def test_existing_velocity_loss_default_is_unchanged() -> None:
    generator = torch.Generator().manual_seed(3)
    prediction = torch.randn(2, 2, 8, 8, generator=generator)
    target = torch.randn(2, 2, 8, 8, generator=generator)
    implicit = velocity_loss(prediction, target, modes_y=4, modes_x=4)
    explicit = velocity_loss(
        prediction,
        target,
        spectral_weight=0.05,
        modes_y=4,
        modes_x=4,
        epsilon=1e-8,
    )
    torch.testing.assert_close(implicit.total, explicit.total)
