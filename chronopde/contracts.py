"""Shape and call contracts shared by future simulator and model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]


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
    query_times: object
    function_evaluations: int


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

