import numpy as np
import pytest

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

