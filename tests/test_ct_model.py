from pathlib import Path

import pytest
import torch

from chronopde.config import load_config
from chronopde.models import (
    CosineSpectralConv2d,
    DCTContinuousVectorField,
    FFTContinuousVectorField,
    FiLMConditioner,
    FNOAutoregressive,
    UNetAutoregressive,
    trainable_parameter_count,
)
from chronopde.training.continuous import _completed_budget_status, build_continuous_model


def test_cosine_spectral_convolution_shape_modes_and_gradients() -> None:
    layer = CosineSpectralConv2d(channels=4, modes_y=3, modes_x=2)
    values = torch.randn(2, 4, 9, 8, requires_grad=True)
    result = layer(values)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert values.grad is not None
    assert layer.weight.grad is not None
    with pytest.raises(ValueError, match="do not fit"):
        CosineSpectralConv2d(4, 10, 2)(values)


def test_chronopde_shape_gradients_and_conditioning() -> None:
    model = DCTContinuousVectorField(
        width=8, modes_y=3, modes_x=3, film_hidden_width=16
    )
    state = torch.randn(2, 2, 16, 16, requires_grad=True)
    parameters = torch.zeros(2, 3)
    result = model(state, torch.tensor([0.0, 25.0]), parameters)
    assert result.shape == state.shape
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert state.grad is not None
    conditioned = model(torch.zeros_like(state), torch.tensor([0.0, 50.0]), parameters)
    assert not torch.allclose(conditioned[0], conditioned[1])


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


def test_chronopde_parameter_count_matches_controlled_baselines() -> None:
    chronopde_count = trainable_parameter_count(DCTContinuousVectorField())
    comparisons = (
        trainable_parameter_count(FFTContinuousVectorField()),
        trainable_parameter_count(FNOAutoregressive()),
        trainable_parameter_count(UNetAutoregressive()),
    )
    assert chronopde_count == 1_951_125
    assert all(abs(chronopde_count - count) / count < 0.10 for count in comparisons)


def test_continuous_builder_selects_distinct_model_families() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/project.yaml")
    assert isinstance(build_continuous_model(config, "chronopde"), DCTContinuousVectorField)
    assert isinstance(build_continuous_model(config, "fno_ct"), FFTContinuousVectorField)


def test_smoke_budget_completion_cannot_bypass_strict_gate() -> None:
    passed, message = _completed_budget_status(
        smoke_overfit=True,
        best_metric=0.1,
        persistence_metric=1.0,
        history=[{"epoch": 149.0}],
        minimum_epochs=25,
        last_stable=True,
    )
    assert passed is False
    assert "not reached" in message


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
