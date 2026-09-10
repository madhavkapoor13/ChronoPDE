import numpy as np
import pytest

import chronopde
from chronopde.contracts import TrajectorySample


def valid_sample() -> TrajectorySample:
    return TrajectorySample(
        state=np.zeros((5, 2, 8, 8), dtype=np.float32),
        times=np.linspace(0, 1, 5, dtype=np.float32),
        params=np.array([0.001, 0.005, 0.005], dtype=np.float32),
        observation_mask=np.array([True, False, True, False, True]),
        trajectory_id="train-0000",
        ic_seed=11,
    )


def test_valid_sample_contract() -> None:
    valid_sample().validate()


def test_week2_public_api_is_exported() -> None:
    assert callable(chronopde.build_neumann_laplacian)
    assert callable(chronopde.generate_initial_condition)
    assert callable(chronopde.simulate_trajectory)
    assert chronopde.PhysicalParameters(0.001, 0.005, 0.005).du == 0.001


def test_week3_public_api_is_exported() -> None:
    for name in (
        "build_dataset_manifest",
        "observation_mask",
        "dct2",
        "idct2",
        "build_quintic_spline",
        "sample_conditional_path",
        "integrate_fixed_step",
    ):
        assert callable(getattr(chronopde, name))


def test_week4_public_models_are_exported() -> None:
    assert callable(chronopde.UNetAutoregressive)
    assert callable(chronopde.FNOAutoregressive)


def test_week5_public_api_is_exported() -> None:
    assert callable(chronopde.FFTContinuousVectorField)
    assert chronopde.VelocitySample is not None
    assert chronopde.VelocityLossBreakdown is not None


def test_endpoint_retention_is_required() -> None:
    sample = valid_sample()
    bad = TrajectorySample(
        state=sample.state,
        times=sample.times,
        params=sample.params,
        observation_mask=np.array([False, True, True, True, True]),
        trajectory_id=sample.trajectory_id,
        ic_seed=sample.ic_seed,
    )
    with pytest.raises(ValueError, match="retain both endpoints"):
        bad.validate()


def test_channel_order_is_enforced() -> None:
    sample = valid_sample()
    bad = TrajectorySample(
        state=np.zeros((5, 8, 8, 2), dtype=np.float32),
        times=sample.times,
        params=sample.params,
        observation_mask=sample.observation_mask,
        trajectory_id=sample.trajectory_id,
        ic_seed=sample.ic_seed,
    )
    with pytest.raises(ValueError, match=r"\[T, 2, H, W\]"):
        bad.validate()
