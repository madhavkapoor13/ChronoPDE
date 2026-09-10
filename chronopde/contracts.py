"""Shape and call contracts shared by future simulator and model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

FloatArray: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class PhysicalParameters:
    du: float
    dv: float
    k: float

    def validate(self) -> None:
        if not np.all(np.isfinite((self.du, self.dv, self.k))):
            raise ValueError("physical parameters must be finite")
        if self.du <= 0 or self.dv <= 0 or self.k <= 0:
            raise ValueError("physical parameters must be positive")

    def as_array(self) -> Float64Array:
        return np.asarray((self.du, self.dv, self.k), dtype=np.float64)


@dataclass(frozen=True)
class GridSpec:
    x: Float64Array
    y: Float64Array
    dx: float
    dy: float
    height: int
    width: int

    def validate(self) -> None:
        if self.height < 2 or self.width < 2:
            raise ValueError("grid dimensions must be at least 2")
        if self.x.shape != (self.width,) or self.y.shape != (self.height,):
            raise ValueError("coordinate shapes must match grid width and height")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("grid spacing must be positive")


@dataclass(frozen=True)
class SimulationDiagnostics:
    success: bool
    message: str
    nfev: int
    runtime_seconds: float
    max_abs_state: float


@dataclass(frozen=True)
class SimulationResult:
    states: FloatArray
    times: Float64Array
    params: PhysicalParameters
    diagnostics: SimulationDiagnostics


@dataclass(frozen=True)
class TrajectoryManifestEntry:
    """One immutable row in the full-dataset generation manifest."""

    trajectory_id: str
    split: Literal["train", "validation", "id", "oodparam", "oodic"]
    index: int
    params: PhysicalParameters
    ic_seed: int
    ic_regime: Literal["train", "ood"]
    parameter_band: str


@dataclass(frozen=True)
class GenerationDiagnostics:
    trajectory_id: str
    success: bool
    runtime_seconds: float
    nfev: int
    max_abs_state: float
    message: str


@dataclass(frozen=True)
class AutoregressivePair:
    current_state: Tensor
    residual_target: Tensor
    delta_t: Tensor
    parameters: Tensor
    trajectory_id: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class RolloutSample:
    states: Tensor
    times: Tensor
    parameters: Tensor
    trajectory_id: str


@dataclass(frozen=True)
class VelocitySample:
    state: Tensor
    target_velocity: Tensor
    time: Tensor
    parameters: Tensor
    trajectory_id: str
    interval_index: int


@dataclass(frozen=True)
class VelocityLossBreakdown:
    total: Tensor
    physical_mse: Tensor
    spectral_relative_error: Tensor


@dataclass(frozen=True)
class TrajectorySample:
    """One trajectory using channels-first state layout ``[T, 2, H, W]``."""

    state: FloatArray
    times: FloatArray
    params: FloatArray
    observation_mask: BoolArray
    trajectory_id: str
    ic_seed: int

    def validate(self) -> None:
        if self.state.ndim != 4 or self.state.shape[1] != 2:
            raise ValueError("state must have shape [T, 2, H, W]")
        time_count = self.state.shape[0]
        if self.times.shape != (time_count,):
            raise ValueError("times must have shape [T]")
        if self.params.shape != (3,):
            raise ValueError("params must have shape [3] ordered as [Du, Dv, k]")
        if self.observation_mask.shape != (time_count,):
            raise ValueError("observation_mask must have shape [T]")
        if not bool(self.observation_mask[0]) or not bool(self.observation_mask[-1]):
            raise ValueError("observation masks must retain both endpoints")


class ContinuousVectorField(Protocol):
    """Future CT models map ``(x, t, p)`` to ``dx/dt`` with x-shaped output."""

    def __call__(self, state: object, time: object, params: object) -> object: ...


class AutoregressiveStep(Protocol):
    """Future AR models map ``(x, dt, p)`` to a residual state update."""

    def __call__(self, state: object, delta_t: object, params: object) -> object: ...


@dataclass(frozen=True)
class IntegrationResult:
    states: object
    times: object
    nfev: int

    @property
    def query_times(self) -> object:
        """Backward-compatible alias for the Week 1 contract name."""

        return self.times

    @property
    def function_evaluations(self) -> int:
        """Backward-compatible alias for the Week 1 contract name."""

        return self.nfev


class FixedStepIntegrator(Protocol):
    def __call__(
        self,
        vector_field: ContinuousVectorField,
        initial_state: object,
        start_time: float,
        end_time: float,
        steps: int,
        params: object,
    ) -> IntegrationResult: ...
