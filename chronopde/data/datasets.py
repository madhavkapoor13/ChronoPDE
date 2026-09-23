"""Lazy, worker-safe HDF5 datasets for autoregressive learning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from chronopde.config import RegimeName
from chronopde.contracts import AutoregressivePair, RolloutSample, VelocitySample
from chronopde.numerics.splines import estimate_knot_derivatives

MASK_NAMES: dict[RegimeName, str] = {
    "full": "full",
    "irreg50": "irregular_50",
    "irreg25": "irregular_25",
}


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: Tensor
    state_std: Tensor
    parameter_mean: Tensor
    parameter_std: Tensor

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean[:, None, None]) / self.state_std[:, None, None]

    def denormalize_state(self, state: Tensor) -> Tensor:
        return state * self.state_std[:, None, None] + self.state_mean[:, None, None]

    def normalize_parameters(self, parameters: Tensor) -> Tensor:
        return (parameters - self.parameter_mean) / self.parameter_std


def load_normalization(path: str | Path) -> NormalizationStats:
    with h5py.File(path, "r") as handle:
        group = handle["normalization"]
        return NormalizationStats(
            state_mean=torch.from_numpy(np.asarray(group["state_mean"], dtype=np.float32)),
            state_std=torch.from_numpy(np.asarray(group["state_std"], dtype=np.float32)),
            parameter_mean=torch.from_numpy(np.asarray(group["parameter_mean"], dtype=np.float32)),
            parameter_std=torch.from_numpy(np.asarray(group["parameter_std"], dtype=np.float32)),
        )


class _LazyHDF5Dataset(Dataset[Any]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        self._handle: h5py.File | None = None
        self.normalization = load_normalization(self.path)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()


class HDF5AutoregressiveDataset(_LazyHDF5Dataset):
    """Pairs formed only between adjacent retained observations."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        regime: RegimeName,
        trajectory_ids: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(path)
        if split not in {"train", "validation", "id"}:
            raise ValueError("autoregressive split must be train, validation, or id")
        self.split = split
        self.regime = regime
        self.reference_dt = 0.5
        with h5py.File(self.path, "r") as handle:
            group = handle[f"splits/{split}"]
            ids = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in group["trajectory_ids"][:]
            ]
            selected = set(trajectory_ids) if trajectory_ids is not None else None
            if selected is not None and not selected.issubset(ids):
                missing = selected.difference(ids)
                raise ValueError(f"unknown trajectory IDs: {sorted(missing)}")
            masks = np.asarray(group[f"masks/{MASK_NAMES[regime]}"], dtype=np.bool_)
            self._pairs: list[tuple[int, int, int, str]] = []
            for trajectory_index, trajectory_id in enumerate(ids):
                if selected is not None and trajectory_id not in selected:
                    continue
                retained = np.flatnonzero(masks[trajectory_index])
                self._pairs.extend(
                    (trajectory_index, int(start), int(end), trajectory_id)
                    for start, end in pairwise(retained)
                )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> AutoregressivePair:
        trajectory, start, end, trajectory_id = self._pairs[index]
        group = self._file()[f"splits/{self.split}"]
        current = torch.from_numpy(np.asarray(group["states"][trajectory, start]))
        following = torch.from_numpy(np.asarray(group["states"][trajectory, end]))
        times = np.asarray(group["times"][trajectory, [start, end]], dtype=np.float32)
        parameters = torch.from_numpy(np.asarray(group["parameters"][trajectory], dtype=np.float32))
        current_normalized = self.normalization.normalize_state(current)
        following_normalized = self.normalization.normalize_state(following)
        return AutoregressivePair(
            current_state=current_normalized,
            residual_target=following_normalized - current_normalized,
            delta_t=torch.tensor((times[1] - times[0]) / self.reference_dt),
            parameters=self.normalization.normalize_parameters(parameters),
            trajectory_id=trajectory_id,
            start_index=start,
            end_index=end,
        )


class HDF5RolloutDataset(_LazyHDF5Dataset):
    def __init__(
        self,
        path: str | Path,
        split: str,
        regime: RegimeName,
        trajectory_ids: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(path)
        if split not in {"train", "validation", "id"}:
            raise ValueError("rollout split must be train, validation, or id")
        self.split = split
        self.regime = regime
        self.reference_dt = 0.5
        with h5py.File(self.path, "r") as handle:
            ids = handle[f"splits/{split}/trajectory_ids"][:]
            decoded = [item.decode() if isinstance(item, bytes) else str(item) for item in ids]
            selected = set(trajectory_ids) if trajectory_ids is not None else None
            self._indices = [
                index
                for index, trajectory_id in enumerate(decoded)
                if selected is None or trajectory_id in selected
            ]
            self._ids = [decoded[index] for index in self._indices]
            if selected is not None and selected != set(self._ids):
                raise ValueError(
                    f"unknown trajectory IDs: {sorted(selected.difference(self._ids))}"
                )

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> RolloutSample:
        group = self._file()[f"splits/{self.split}"]
        trajectory_index = self._indices[index]
        mask = np.asarray(
            group[f"masks/{MASK_NAMES[self.regime]}"][trajectory_index], dtype=np.bool_
        )
        states = torch.from_numpy(np.asarray(group["states"][trajectory_index, mask]))
        times = torch.from_numpy(
            np.asarray(group["times"][trajectory_index, mask], dtype=np.float32)
        )
        parameters = torch.from_numpy(
            np.asarray(group["parameters"][trajectory_index], dtype=np.float32)
        )
        return RolloutSample(
            states=self.normalization.normalize_state(states),
            times=times,
            parameters=self.normalization.normalize_parameters(parameters),
            trajectory_id=self._ids[index],
        )


class HDF5VelocityDataset(_LazyHDF5Dataset):
    """Deterministic spline-path samples, one for every retained interval."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        regime: RegimeName,
        seed: int,
        gamma: float = 1e-5,
        trajectory_ids: tuple[str, ...] | None = None,
        *,
        resample_each_epoch: bool = True,
    ) -> None:
        super().__init__(path)
        if split not in {"train", "validation"}:
            raise ValueError("velocity split must be train or validation")
        if seed < 0 or gamma < 0:
            raise ValueError("seed and gamma must be non-negative")
        self.split = split
        self.regime = regime
        self.seed = seed
        self.gamma = gamma
        self.epoch = 0
        self.resample_each_epoch = resample_each_epoch
        with h5py.File(self.path, "r") as handle:
            group = handle[f"splits/{split}"]
            ids = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in group["trajectory_ids"][:]
            ]
            selected = set(trajectory_ids) if trajectory_ids is not None else None
            if selected is not None and not selected.issubset(ids):
                raise ValueError(f"unknown trajectory IDs: {sorted(selected.difference(ids))}")
            masks = np.asarray(group[f"masks/{MASK_NAMES[regime]}"], dtype=np.bool_)
            self._retained: dict[int, NDArray[np.int64]] = {}
            self._samples: list[tuple[int, int, str]] = []
            for trajectory_index, trajectory_id in enumerate(ids):
                if selected is not None and trajectory_id not in selected:
                    continue
                retained = np.flatnonzero(masks[trajectory_index])
                if len(retained) < 3:
                    raise ValueError("velocity splines require at least three retained states")
                self._retained[trajectory_index] = retained
                self._samples.extend(
                    (trajectory_index, interval, trajectory_id)
                    for interval in range(len(retained) - 1)
                )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._samples)

    def sample_identity(self, index: int) -> tuple[str, int]:
        """Return the trajectory and interval without loading spline tensors."""
        _, interval, trajectory_id = self._samples[index]
        return trajectory_id, interval

    def _generator(self, trajectory_id: str, interval: int) -> torch.Generator:
        sample_epoch = self.epoch if self.resample_each_epoch else 0
        payload = f"{self.seed}:{sample_epoch}:{trajectory_id}:{interval}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return torch.Generator().manual_seed(sample_seed)

    def _knot_data(
        self, group: h5py.Group, trajectory: int, retained: NDArray[np.int64], knot: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = len(retained)
        if knot == 0:
            positions = np.asarray((0, 1, 2))
            selected_position = 0
        elif knot == count - 1:
            positions = np.asarray((count - 3, count - 2, count - 1))
            selected_position = 2
        else:
            positions = np.asarray((knot - 1, knot, knot + 1))
            selected_position = 1
        stored_indices = retained[positions]
        states = torch.from_numpy(np.asarray(group["states"][trajectory, stored_indices]))
        states = self.normalization.normalize_state(states)
        times = torch.from_numpy(
            np.asarray(group["times"][trajectory, stored_indices], dtype=np.float32)
        )
        first, second = estimate_knot_derivatives(states[None], times)
        return states[selected_position], first[0, selected_position], second[0, selected_position]

    def __getitem__(self, index: int) -> VelocitySample:
        trajectory, interval, trajectory_id = self._samples[index]
        retained = self._retained[trajectory]
        group = self._file()[f"splits/{self.split}"]
        left, left_first, left_second = self._knot_data(group, trajectory, retained, interval)
        right, right_first, right_second = self._knot_data(
            group, trajectory, retained, interval + 1
        )
        stored = retained[[interval, interval + 1]]
        times = torch.from_numpy(np.asarray(group["times"][trajectory, stored], dtype=np.float32))
        dt = times[1] - times[0]
        generator = self._generator(trajectory_id, interval)
        tau = torch.rand((), generator=generator)
        a0 = left
        a1 = left_first * dt
        a2 = 0.5 * left_second * dt.square()
        s1 = right - a0 - a1 - a2
        s2 = right_first * dt - a1 - 2 * a2
        s3 = right_second * dt.square() - 2 * a2
        a3 = 10 * s1 - 4 * s2 + 0.5 * s3
        a4 = -15 * s1 + 7 * s2 - s3
        a5 = 6 * s1 - 3 * s2 + 0.5 * s3
        state = ((((a5 * tau + a4) * tau + a3) * tau + a2) * tau + a1) * tau + a0
        velocity = (
            a1 + 2 * a2 * tau + 3 * a3 * tau.square() + 4 * a4 * tau.pow(3) + 5 * a5 * tau.pow(4)
        ) / dt
        noise = torch.randn(state.shape, generator=generator, dtype=state.dtype)
        envelope = self.gamma * tau.pow(3) * (1 - tau).pow(3)
        envelope_derivative = (
            self.gamma * 3 / dt * tau.square() * (1 - tau).square() * (1 - 2 * tau)
        )
        parameters = torch.from_numpy(np.asarray(group["parameters"][trajectory], dtype=np.float32))
        return VelocitySample(
            state=state + envelope * noise,
            target_velocity=velocity + envelope_derivative * noise,
            time=times[0] + tau * dt,
            parameters=self.normalization.normalize_parameters(parameters),
            trajectory_id=trajectory_id,
            interval_index=interval,
        )


def collate_autoregressive_pairs(samples: list[AutoregressivePair]) -> dict[str, Any]:
    return {
        "current_state": torch.stack([sample.current_state for sample in samples]),
        "residual_target": torch.stack([sample.residual_target for sample in samples]),
        "delta_t": torch.stack([sample.delta_t for sample in samples]),
        "parameters": torch.stack([sample.parameters for sample in samples]),
        "trajectory_id": [sample.trajectory_id for sample in samples],
        "start_index": torch.tensor([sample.start_index for sample in samples]),
        "end_index": torch.tensor([sample.end_index for sample in samples]),
    }


def collate_rollouts(samples: list[RolloutSample]) -> dict[str, Any]:
    lengths = {len(sample.times) for sample in samples}
    if len(lengths) != 1:
        raise ValueError("rollouts in one batch must contain the same number of retained times")
    return {
        "states": torch.stack([sample.states for sample in samples]),
        "times": torch.stack([sample.times for sample in samples]),
        "parameters": torch.stack([sample.parameters for sample in samples]),
        "trajectory_id": [sample.trajectory_id for sample in samples],
    }


def collate_velocity_samples(samples: list[VelocitySample]) -> dict[str, Any]:
    return {
        "state": torch.stack([sample.state for sample in samples]),
        "target_velocity": torch.stack([sample.target_velocity for sample in samples]),
        "time": torch.stack([sample.time for sample in samples]),
        "parameters": torch.stack([sample.parameters for sample in samples]),
        "trajectory_id": [sample.trajectory_id for sample in samples],
        "interval_index": torch.tensor([sample.interval_index for sample in samples]),
    }
