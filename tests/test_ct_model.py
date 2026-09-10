import pytest
import torch

from chronopde.models import (
    FFTContinuousVectorField,
    FiLMConditioner,
    FNOAutoregressive,
    UNetAutoregressive,
    trainable_parameter_count,
)


def test_ct_fno_shape_gradients_and_conditioning() -> None:
    model = FFTContinuousVectorField(width=8, modes_y=3, modes_x=3, film_hidden_width=16)
    state = torch.randn(2, 2, 16, 16, requires_grad=True)
    parameters = torch.zeros(2, 3)
    result = model(state, torch.tensor([0.0, 25.0]), parameters)
    assert result.shape == state.shape
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert state.grad is not None
    same_state = torch.zeros(2, 2, 16, 16)
    conditioned = model(same_state, torch.tensor([0.0, 50.0]), parameters)
    assert not torch.allclose(conditioned[0], conditioned[1])


def test_ct_fno_parameter_count_matches_week4_models() -> None:
    ct_count = trainable_parameter_count(FFTContinuousVectorField())
    comparisons = (
        trainable_parameter_count(FNOAutoregressive()),
        trainable_parameter_count(UNetAutoregressive()),
    )
    assert ct_count == 1_973_657
    assert all(abs(ct_count - count) / count < 0.10 for count in comparisons)


def test_film_shapes_and_input_validation() -> None:
    conditioner = FiLMConditioner(width=5, blocks=3, hidden_width=8)
    scales, biases = conditioner(torch.randn(4, 4))
    assert scales.shape == biases.shape == (4, 3, 5)
    assert not torch.equal(scales[:, 0], scales[:, 1])
    with pytest.raises(ValueError, match="shape"):
        conditioner(torch.randn(4, 3))
    model = FFTContinuousVectorField(width=4, modes_y=2, modes_x=2)
    with pytest.raises(ValueError, match="finite"):
        model(torch.zeros(1, 2, 8, 8), torch.tensor([float("nan")]), torch.zeros(1, 3))
