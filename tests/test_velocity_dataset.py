from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from chronopde.data.datasets import HDF5VelocityDataset, collate_velocity_samples
from chronopde.diagnostics.balanced_batch import resolve_balanced_sample_indices
from chronopde.numerics import build_quintic_spline, evaluate_quintic_spline


@pytest.fixture
def velocity_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "velocity.h5"
    times = np.asarray([0.0, 0.2, 0.7, 1.4, 2.0], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        normalization = handle.create_group("normalization")
        normalization.create_dataset("state_mean", data=np.asarray([1.0, -1.0]))
        normalization.create_dataset("state_std", data=np.asarray([2.0, 4.0]))
        normalization.create_dataset("parameter_mean", data=np.asarray([1.0, 2.0, 3.0]))
        normalization.create_dataset("parameter_std", data=np.asarray([0.5, 1.0, 2.0]))
        for split, count in (("train", 2), ("validation", 1), ("id", 1)):
            group = handle.create_group(f"splits/{split}")
            states = np.empty((count, 5, 2, 4, 4), dtype=np.float32)
            for trajectory in range(count):
                values = times**2 + 2 * times + trajectory
                states[trajectory, :, 0] = values[:, None, None]
                states[trajectory, :, 1] = (2 * values)[:, None, None]
            group.create_dataset("states", data=states)
            group.create_dataset("times", data=np.repeat(times[None], count, axis=0))
            group.create_dataset("parameters", data=np.repeat([[1.5, 3.0, 5.0]], count, axis=0))
            group.create_dataset(
                "trajectory_ids",
                data=np.asarray([f"{split}-{index:04d}" for index in range(count)], dtype="S"),
            )
            masks = group.create_group("masks")
            masks.create_dataset("full", data=np.ones((count, 5), dtype=np.bool_))
            masks.create_dataset(
                "irregular_50",
                data=np.repeat([[True, False, True, False, True]], count, axis=0),
            )
            masks.create_dataset(
                "irregular_25",
                data=np.repeat([[True, True, False, False, True]], count, axis=0),
            )
    return path


def test_velocity_dataset_counts_determinism_and_epoch_variation(
    velocity_hdf5: Path,
) -> None:
    dataset = HDF5VelocityDataset(velocity_hdf5, "train", "full", seed=11, gamma=0.0)
    assert len(dataset) == 8
    first = dataset[2]
    repeated = dataset[2]
    torch.testing.assert_close(first.state, repeated.state)
    assert first.time.item() == repeated.time.item()
    dataset.set_epoch(1)
    changed = dataset[2]
    assert first.time.item() != changed.time.item()
    with pytest.raises(ValueError, match="train or validation"):
        HDF5VelocityDataset(velocity_hdf5, "id", "full", seed=0)
    dataset.close()


def test_local_velocity_sample_matches_full_spline(velocity_hdf5: Path) -> None:
    dataset = HDF5VelocityDataset(
        velocity_hdf5,
        "train",
        "full",
        seed=3,
        gamma=0.0,
        trajectory_ids=("train-0000",),
    )
    sample = dataset[1]
    with h5py.File(velocity_hdf5, "r") as handle:
        states = torch.from_numpy(np.asarray(handle["splits/train/states"][0]))
        states = dataset.normalization.normalize_state(states)
        times = torch.from_numpy(np.asarray(handle["splits/train/times"][0], dtype=np.float32))
    spline = build_quintic_spline(states[None], times)
    value, derivative = evaluate_quintic_spline(spline, sample.time[None])
    torch.testing.assert_close(sample.state, value[0, 0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sample.target_velocity, derivative[0, 0], rtol=1e-5, atol=1e-6)
    batch = collate_velocity_samples([sample, sample])
    assert batch["state"].shape == (2, 2, 4, 4)
    assert batch["time"].shape == (2,)
    dataset.close()


def test_velocity_perturbation_is_deterministic(velocity_hdf5: Path) -> None:
    perturbed = HDF5VelocityDataset(velocity_hdf5, "validation", "full", 9, gamma=1e-5)
    baseline = HDF5VelocityDataset(velocity_hdf5, "validation", "full", 9, gamma=0.0)
    first = perturbed[0]
    repeated = perturbed[0]
    plain = baseline[0]
    torch.testing.assert_close(first.state, repeated.state)
    assert not torch.equal(first.state, plain.state)
    assert torch.isfinite(first.target_velocity).all()
    perturbed.close()
    baseline.close()


def test_fixed_velocity_samples_do_not_change_between_epochs(velocity_hdf5: Path) -> None:
    fixed = HDF5VelocityDataset(
        velocity_hdf5,
        "train",
        "full",
        seed=9,
        gamma=0.0,
        resample_each_epoch=False,
    )
    resampled = HDF5VelocityDataset(velocity_hdf5, "train", "full", seed=9, gamma=0.0)
    fixed_before = fixed[0]
    resampled_before = resampled[0]
    fixed.set_epoch(3)
    resampled.set_epoch(3)
    torch.testing.assert_close(fixed[0].state, fixed_before.state)
    assert fixed[0].time.item() == fixed_before.time.item()
    assert resampled[0].time.item() != resampled_before.time.item()
    fixed.close()
    resampled.close()


def test_balanced_sample_indices_are_resolved_by_identity(velocity_hdf5: Path) -> None:
    dataset = HDF5VelocityDataset(
        velocity_hdf5,
        "train",
        "full",
        seed=0,
        gamma=0.0,
        resample_each_epoch=False,
    )
    indices = resolve_balanced_sample_indices(
        dataset,
        trajectory_ids=("train-0000", "train-0001"),
        intervals=(0, 3),
    )
    assert indices == (0, 3, 4, 7)
    assert [dataset.sample_identity(index) for index in indices] == [
        ("train-0000", 0),
        ("train-0000", 3),
        ("train-0001", 0),
        ("train-0001", 3),
    ]
    dataset.close()
