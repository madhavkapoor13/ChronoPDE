"""Validated configuration models for every ChronoPDE command."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["chronopde", "fno_ct", "fno_ar", "unet_ar"]
RegimeName = Literal["full", "irreg50", "irreg25"]
SplitName = Literal["train", "validation", "id", "oodparam", "oodic", "hidden_time"]
IntegratorName = Literal["euler", "heun", "rk4"]


class StrictModel(BaseModel):
    """Base model that rejects silent misspellings in YAML files."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSection(StrictModel):
    name: str = "chronopde"
    artifact_root: Path = Path("artifacts/runs")


class SolverConfig(StrictModel):
    method: Literal["RK45", "DOP853", "BDF"] = "DOP853"
    rtol: float = Field(default=1e-7, gt=0)
    atol: float = Field(default=1e-9, gt=0)


class PDEConfig(StrictModel):
    du: float = Field(default=1e-3, gt=0)
    dv: float = Field(default=5e-3, gt=0)
    k: float = Field(default=5e-3, gt=0)
    t_start: float = 0.0
    t_end: float = 50.0
    stored_times: int = Field(default=101, ge=3)
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    height: int = Field(default=64, ge=8)
    width: int = Field(default=64, ge=8)
    boundary: Literal["neumann"] = "neumann"
    solver: SolverConfig = SolverConfig()

    @model_validator(mode="after")
    def validate_intervals(self) -> PDEConfig:
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be greater than t_start")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("spatial maxima must be greater than spatial minima")
        return self


class ParameterRanges(StrictModel):
    du: tuple[float, float]
    dv: tuple[float, float]
    k: tuple[float, float]

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterRanges:
        for name in ("du", "dv", "k"):
            low, high = getattr(self, name)
            if low <= 0 or high <= low:
                raise ValueError(f"{name} range must be positive and increasing")
        return self


class OODParameterRanges(StrictModel):
    du: tuple[tuple[float, float], tuple[float, float]]
    dv: tuple[tuple[float, float], tuple[float, float]]
    k: tuple[tuple[float, float], tuple[float, float]]


class InitialConditionConfig(StrictModel):
    train_modes: tuple[int, int] = (1, 8)
    train_standard_deviation: float = Field(default=0.5, gt=0)
    ood_modes: tuple[int, int] = (9, 16)
    ood_amplitude_multiplier: float = Field(default=1.5, gt=1)

    @model_validator(mode="after")
    def validate_modes(self) -> InitialConditionConfig:
        train_low, train_high = self.train_modes
        ood_low, ood_high = self.ood_modes
        if train_low < 1 or train_high < train_low:
            raise ValueError("train_modes must be positive and increasing")
        if ood_low <= train_high or ood_high < ood_low:
            raise ValueError("ood_modes must be increasing and disjoint above train_modes")
        return self

    @property
    def ood_standard_deviation(self) -> float:
        return self.train_standard_deviation * self.ood_amplitude_multiplier


class PilotConfig(StrictModel):
    trajectories: int = Field(default=24, ge=1)
    divergence_threshold: float = Field(default=10.0, gt=0)
    maximum_failure_fraction: float = Field(default=0.10, ge=0, lt=1)
    maximum_adjustment_rounds: int = Field(default=2, ge=0)
    workers: int = Field(default=4, ge=1)
    output_directory: Path = Path("artifacts/pilot/week2")


class DataConfig(StrictModel):
    output_path: Path = Path("data/chronopde.h5")
    sample_path: Path = Path("data/sample.h5")
    train_trajectories: int = Field(default=320, ge=1)
    validation_trajectories: int = Field(default=60, ge=1)
    id_test_trajectories: int = Field(default=100, ge=1)
    ood_parameter_trajectories: int = Field(default=120, ge=1)
    ood_ic_trajectories: int = Field(default=120, ge=1)
    train_ranges: ParameterRanges
    ood_ranges: OODParameterRanges
    retention_fractions: tuple[float, ...] = (1.0, 0.5, 0.25)
    base_seed: int = Field(default=1729, ge=0)
    initial_conditions: InitialConditionConfig = InitialConditionConfig()
    pilot: PilotConfig = PilotConfig()

    @model_validator(mode="after")
    def validate_data_protocol(self) -> DataConfig:
        if set(self.retention_fractions) != {1.0, 0.5, 0.25}:
            raise ValueError("retention_fractions must contain exactly 1.0, 0.5, and 0.25")
        for name in ("du", "dv", "k"):
            train_low, train_high = getattr(self.train_ranges, name)
            low_band, high_band = getattr(self.ood_ranges, name)
            for low, high in (low_band, high_band):
                if low <= 0 or high <= low:
                    raise ValueError(f"{name} OOD bands must be positive and increasing")
            if low_band[1] >= train_low or high_band[0] <= train_high:
                raise ValueError(f"{name} OOD bands must not overlap the training range")
        if self.pilot.trajectories != 24:
            raise ValueError("the frozen Week 2 pilot must contain exactly 24 trajectories")
        return self

    @property
    def trajectory_count(self) -> int:
        return sum(
            (
                self.train_trajectories,
                self.validation_trajectories,
                self.id_test_trajectories,
                self.ood_parameter_trajectories,
                self.ood_ic_trajectories,
            )
        )


class ModelConfig(StrictModel):
    name: ModelName = "chronopde"
    state_channels: int = Field(default=2, ge=1)
    width: int = Field(default=32, ge=4)
    spectral_modes_x: int = Field(default=12, ge=1)
    spectral_modes_y: int = Field(default=12, ge=1)
    blocks: int = Field(default=4, ge=1)
    conditioning_features: int = Field(default=4, ge=1)
    activation: Literal["gelu"] = "gelu"
    spline: Literal["quintic", "linear"] = "quintic"
    perturbation_gamma: float = Field(default=1e-5, ge=0)


class TrainingConfig(StrictModel):
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_epochs: int = Field(default=5, ge=0)
    max_epochs: int = Field(default=150, ge=1)
    patience: int = Field(default=20, ge=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    spectral_loss_weight: float = Field(default=0.05, ge=0)
    precision: Literal["float32", "amp"] = "float32"
    seed: int = Field(default=0, ge=0)


class EvaluationConfig(StrictModel):
    solver: IntegratorName = "rk4"
    steps_per_interval: int = Field(default=2, ge=1)
    correlation_threshold: float = Field(default=0.9, gt=0, le=1)
    divergence_threshold: float = Field(default=10.0, gt=0)
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    timing_warmups: int = Field(default=10, ge=0)
    timing_repetitions: int = Field(default=100, ge=1)


class ProjectConfig(StrictModel):
    project: ProjectSection = ProjectSection()
    pde: PDEConfig
    data: DataConfig
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def validate_cross_section_contracts(self) -> ProjectConfig:
        if self.model is not None:
            if self.model.spectral_modes_x > self.pde.width:
                raise ValueError("spectral_modes_x cannot exceed grid width")
            if self.model.spectral_modes_y > self.pde.height:
                raise ValueError("spectral_modes_y cannot exceed grid height")
        highest_ic_mode = self.data.initial_conditions.ood_modes[1]
        if highest_ic_mode >= min(self.pde.height, self.pde.width):
            raise ValueError("initial-condition modes must fit inside the configured grid")
        return self


def load_config(path: str | Path) -> ProjectConfig:
    """Load and strictly validate a ChronoPDE YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ProjectConfig.model_validate(raw)
