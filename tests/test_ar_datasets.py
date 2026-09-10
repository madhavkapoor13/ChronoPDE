from pathlib import Path

import h5py
import numpy as np
import torch

from chronopde.data.datasets import HDF5AutoregressiveDataset, HDF5RolloutDataset


def make_dataset(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        normalization = handle.create_group("normalization")
        normalization.create_dataset("state_mean", data=[1.0, -1.0])
        normalization.create_dataset("state_std", data=[2.0, 4.0])
        normalization.create_dataset("parameter_mean", data=[1.0, 2.0, 3.0])
        normalization.create_dataset("parameter_std", data=[0.5, 1.0, 2.0])
        splits = handle.create_group("splits")
        for split in ("train", "validation", "id"):
            group = splits.create_group(split)
            states = np.zeros((2, 5, 2, 16, 16), dtype=np.float32)
            for trajectory in range(2):
                for time_index in range(5):
                    states[trajectory, time_index, 0] = 1 + 2 * time_index
                    states[trajectory, time_index, 1] = -1 + 4 * time_index
            group.create_dataset("states", data=states)
            group.create_dataset("times", data=np.tile(np.arange(5) * 0.5, (2, 1)))
            group.create_dataset("parameters", data=np.tile([1.5, 3.0, 5.0], (2, 1)))
            string_type = h5py.string_dtype("utf-8")
            group.create_dataset(
                "trajectory_ids",
                data=np.asarray([f"{split}-0000", f"{split}-0001"], dtype=object),
                dtype=string_type,
            )
            masks = group.create_group("masks")
            masks.create_dataset("full", data=np.ones((2, 5), dtype=np.bool_))
            masks.create_dataset(
                "irregular_50",
                data=np.asarray([[1, 0, 1, 0, 1], [1, 1, 0, 0, 1]], dtype=np.bool_),
            )
            masks.create_dataset(
                "irregular_25",
                data=np.asarray([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], dtype=np.bool_),
            )
    return path


def test_pair_dataset_uses_adjacent_retained_times_and_normalization(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "tiny.h5")
    dataset = HDF5AutoregressiveDataset(path, "train", "irreg50", ("train-0000",))
    assert len(dataset) == 2
    first = dataset[0]
    assert (first.start_index, first.end_index) == (0, 2)
    assert first.delta_t.item() == 2.0
    torch.testing.assert_close(first.current_state, torch.zeros_like(first.current_state))
    torch.testing.assert_close(first.residual_target, torch.full_like(first.residual_target, 2.0))
    torch.testing.assert_close(first.parameters, torch.ones(3))
    state = torch.randn(2, 16, 16)
    torch.testing.assert_close(
        dataset.normalization.denormalize_state(dataset.normalization.normalize_state(state)), state
    )
    dataset.close()


def test_rollout_dataset_is_lazy_and_trajectory_isolated(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "tiny.h5")
    dataset = HDF5RolloutDataset(path, "validation", "full", ("validation-0001",))
    assert dataset._handle is None
    sample = dataset[0]
    assert sample.trajectory_id == "validation-0001"
    assert sample.states.shape == (5, 2, 16, 16)
    assert dataset._handle is not None
    dataset.close()
    assert dataset._handle is None
