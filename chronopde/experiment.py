"""Experiment naming and dry-run planning utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from chronopde.config import ModelName, ProjectConfig, RegimeName, SplitName

ExperimentKind = Literal[
    "id_rollout",
    "sparse_50",
    "sparse_25",
    "hidden_time",
    "ood_params",
    "ood_ic",
    "basis_ablation",
    "integrator_sweep",
    "mode_sweep",
]


@dataclass(frozen=True)
class DryRunPlan:
    command: str
    experiment_id: str
    artifact_directory: str
    trajectory_count: int
    state_shape: tuple[int, int, int, int, int]
    estimated_state_gib: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def experiment_id(model: ModelName, regime: RegimeName, split: SplitName, seed: int) -> str:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return f"{model}-{regime}-{split}-s{seed}"


def state_shape(config: ProjectConfig) -> tuple[int, int, int, int, int]:
    return (
        config.data.trajectory_count,
        config.pde.stored_times,
        2,
        config.pde.height,
        config.pde.width,
    )


def estimated_state_gib(config: ProjectConfig) -> float:
    element_count = 1
    for dimension in state_shape(config):
        element_count *= dimension
    return element_count * 4 / 1024**3


def build_dry_run_plan(
    config: ProjectConfig,
    *,
    command: str,
    model: ModelName = "chronopde",
    regime: RegimeName = "full",
    split: SplitName = "train",
    seed: int = 0,
) -> DryRunPlan:
    run_id = experiment_id(model, regime, split, seed)
    artifact_directory = str(Path(config.project.artifact_root) / run_id)
    return DryRunPlan(
        command=command,
        experiment_id=run_id,
        artifact_directory=artifact_directory,
        trajectory_count=config.data.trajectory_count,
        state_shape=state_shape(config),
        estimated_state_gib=round(estimated_state_gib(config), 3),
        notes=(
            "Dry-run only: no data, checkpoints, or artifacts were written.",
            "Stored-state estimate excludes HDF5 metadata and compression.",
        ),
    )

