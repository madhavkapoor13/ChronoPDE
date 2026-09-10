import pytest
import torch

from chronopde.models import (
    FNOAutoregressive,
    FourierSpectralConv2d,
    UNetAutoregressive,
    trainable_parameter_count,
)


@pytest.mark.parametrize("model", [UNetAutoregressive(), FNOAutoregressive()])
def test_ar_model_shape_finite_and_gradients(model: torch.nn.Module) -> None:
    state = torch.randn(2, 2, 32, 32, requires_grad=True)
    result = model(state, torch.ones(2), torch.zeros(2, 3))
    assert result.shape == state.shape
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert state.grad is not None


def test_real_parameter_counts_are_matched() -> None:
    unet_count = trainable_parameter_count(UNetAutoregressive())
    fno_count = trainable_parameter_count(FNOAutoregressive())
    assert unet_count == 1_929_890
    assert fno_count == 1_943_263
    assert abs(unet_count - fno_count) / unet_count < 0.10


def test_spectral_layer_validates_modes() -> None:
    layer = FourierSpectralConv2d(4, modes_y=5, modes_x=5)
    with pytest.raises(ValueError, match="do not fit"):
        layer(torch.randn(1, 4, 8, 8))


def test_conditioning_rejects_nonfinite_values() -> None:
    model = FNOAutoregressive(width=4, modes_y=2, modes_x=2)
    state = torch.zeros(2, 2, 8, 8)
    parameters = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="finite"):
        model(state, torch.tensor([float("nan"), 1.0]), parameters)


def test_deterministic_initialization_and_forward() -> None:
    torch.manual_seed(10)
    first = FNOAutoregressive(width=8, modes_y=3, modes_x=3)
    torch.manual_seed(10)
    second = FNOAutoregressive(width=8, modes_y=3, modes_x=3)
    state = torch.randn(1, 2, 16, 16)
    torch.testing.assert_close(
        first(state, torch.ones(1), torch.zeros(1, 3)),
        second(state, torch.ones(1), torch.zeros(1, 3)),
    )
