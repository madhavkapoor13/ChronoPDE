"""Strict protocol models for the second ChronoPDE V2 phase."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chronopde.config import ParameterRanges, PDEConfig, SolverConfig
from chronopde.v2.common import canonical_json_bytes


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExactTargetContract(_StrictModel):
    kind: Literal["exact_discrete_rhs"]
    time_units: Literal["physical"]
    normalized_formula: Literal["physical_rhs / training_state_std"]
    spline_targets: Literal[False]
    perturbations: Literal[False]


class SeedOffsets(_StrictModel):
    pilot: int
    training: int
    validation: int
    confirmatory: int

    @model_validator(mode="after")
    def unique_positive_offsets(self) -> SeedOffsets:
        values = (self.pilot, self.training, self.validation, self.confirmatory)
        if len(set(values)) != 4 or min(values) <= 0:
            raise ValueError("seed offsets must be unique and positive")
        return self


class V2InitialCondition(_StrictModel):
    maximum_mode: int = Field(ge=1)
    spectral_decay: float = Field(gt=0)
    standard_deviation: float = Field(gt=0)


class V2DataContract(_StrictModel):
    master_seed: Literal[20260922]
    pilot_trajectories: Literal[24]
    training_trajectories: Literal[512]
    validation_trajectories: Literal[128]
    confirmatory_trajectories: Literal[256]
    seed_offsets: SeedOffsets
    parameter_ranges: ParameterRanges
    initial_condition: V2InitialCondition
    tight_solver_indices: tuple[int, ...]
    tight_solver: SolverConfig

    @model_validator(mode="after")
    def validate_tight_subset(self) -> V2DataContract:
        if len(self.tight_solver_indices) != 6 or len(set(self.tight_solver_indices)) != 6:
            raise ValueError("exactly six unique tight-solver indices are required")
        if min(self.tight_solver_indices) < 0 or max(self.tight_solver_indices) >= 24:
            raise ValueError("tight-solver indices must identify pilot trajectories")
        if self.initial_condition.maximum_mode >= 64:
            raise ValueError("pilot modes must fit inside the frozen grid")
        return self


class ComparatorContract(_StrictModel):
    width: int = Field(ge=4)
    blocks: int = Field(ge=1)
    film_hidden_width: int = Field(ge=4)
    fft_modes_y: int = Field(ge=1)
    fft_modes_x: int = Field(ge=1)
    expected_dct_modes_y: int = Field(ge=1)
    expected_dct_modes_x: int = Field(ge=1)
    total_parameter_tolerance: float = Field(gt=0)
    spectral_dof_tolerance: float = Field(gt=0)
    physical_cutoff_tolerance: float = Field(gt=0)
    search_width_min: int = Field(ge=4)
    search_width_max: int = Field(ge=4)
    search_modes_min: int = Field(ge=1)
    search_modes_max: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_search(self) -> ComparatorContract:
        if self.search_width_max < self.search_width_min:
            raise ValueError("invalid comparator width search interval")
        if self.search_modes_max < self.search_modes_min:
            raise ValueError("invalid comparator mode search interval")
        return self


class NumericalTolerances(_StrictModel):
    structural_relative: float = Field(gt=0)
    rhs_relative: float = Field(gt=0)
    float32_relative: float = Field(gt=0)
    solver_state_p95_relative_l2: float = Field(gt=0)
    solver_final_relative_l2: float = Field(gt=0)
    divergence_threshold: float = Field(gt=0)


class MetricsContract(_StrictModel):
    primary: Literal["held_out_rollout_relative_l2_over_physical_time"]
    secondary: tuple[str, ...]
    boundary_attribution: tuple[str, ...]
    forbidden_wording: Literal["physical_wall_flux"]


class OutputContract(_StrictModel):
    artifact_root: Path
    report_root: Path

    @model_validator(mode="after")
    def isolated_roots(self) -> OutputContract:
        if self.artifact_root != Path("artifacts/chronopde_v2/runs"):
            raise ValueError("V2 artifacts must use the isolated V2 root")
        if not self.report_root.is_relative_to(Path("reports/chronopde_v2")):
            raise ValueError("V2 reports must use the isolated V2 report root")
        return self


class RestrictionContract(_StrictModel):
    gpu_training: Literal[False]
    full_dataset_generation: Literal[False]
    sparse_time_claims: Literal[False]
    ood_claims: Literal[False]
    legacy_id_confirmatory_reuse: Literal[False]
    confirmatory_access_requires_frozen_model: Literal[True]


class Phase2Protocol(_StrictModel):
    schema_version: Literal[1]
    study_id: Literal["chronopde_v2"]
    phase: Literal[2]
    protocol_version: Literal[1]
    parent_phase_commit: Literal["58b0bd84ffab353c45f6df30a222edbc57c512c9"]
    parent_descriptor: Literal["configs/chronopde_v2/study.yaml"]
    primary_question: str
    target: ExactTargetContract
    pde: PDEConfig
    data: V2DataContract
    comparison: ComparatorContract
    tolerances: NumericalTolerances
    metrics: MetricsContract
    outputs: OutputContract
    restrictions: RestrictionContract

    @model_validator(mode="after")
    def validate_frozen_grid(self) -> Phase2Protocol:
        if (self.pde.height, self.pde.width) != (64, 64):
            raise ValueError("Phase 2 uses the frozen 64x64 grid")
        if self.data.initial_condition.maximum_mode >= min(self.pde.height, self.pde.width):
            raise ValueError("initial-condition modes must fit inside the grid")
        if self.data.tight_solver.rtol >= self.pde.solver.rtol:
            raise ValueError("tight solver rtol must be stricter than the standard solver")
        if self.data.tight_solver.atol >= self.pde.solver.atol:
            raise ValueError("tight solver atol must be stricter than the standard solver")
        if self.outputs.report_root != Path("reports/chronopde_v2/phase2"):
            raise ValueError("Phase 2 reports must use the Phase 2 report root")
        return self

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_phase2_protocol(path: Path) -> Phase2Protocol:
    """Load a strict, immutable Phase 2 protocol."""

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Phase 2 protocol root must be a mapping")
    return Phase2Protocol.model_validate(raw)


class Phase3SplitContract(_StrictModel):
    training: Literal[512]
    validation: Literal[128]


class Phase3Restrictions(_StrictModel):
    generate_confirmatory: Literal[False]
    gpu_training: Literal[False]
    model_training: Literal[False]
    use_pilot_normalization: Literal[False]
    use_legacy_data: Literal[False]


class Phase3Protocol(_StrictModel):
    schema_version: Literal[1]
    study_id: Literal["chronopde_v2"]
    phase: Literal[3]
    protocol_version: Literal[1]
    parent_phase_commit: Literal["a18f651"]
    phase2_descriptor: Literal["configs/chronopde_v2/phase2.yaml"]
    phase2_protocol_sha256: Literal[
        "017a1857d6d34f778455f23648728696db7120a2b9504354f00fd97a2776780a"
    ]
    frozen_future_manifest_sha256: Literal[
        "273f7ba6b7a0645fc3f5df236aa1886cca67a4ce7897270a5119e5ad35cea7c5"
    ]
    target: Literal["exact_discrete_rhs"]
    time_units: Literal["physical"]
    development_splits: Phase3SplitContract
    sealed_confirmatory_trajectories: Literal[256]
    workers: int = Field(ge=1, le=16)
    rhs_spot_checks: int = Field(ge=1)
    outputs: OutputContract
    restrictions: Phase3Restrictions

    @model_validator(mode="after")
    def validate_phase3_outputs(self) -> Phase3Protocol:
        if self.outputs.report_root != Path("reports/chronopde_v2/phase3"):
            raise ValueError("Phase 3 reports must use the Phase 3 report root")
        if self.rhs_spot_checks > self.development_splits.training:
            raise ValueError("RHS spot checks cannot exceed the training split")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def load_phase3_protocol(path: Path) -> Phase3Protocol:
    """Load the strict Phase 3 development-data protocol."""

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Phase 3 protocol root must be a mapping")
    return Phase3Protocol.model_validate(raw)


class Phase4Models(_StrictModel):
    width: Literal[29]
    blocks: Literal[4]
    film_hidden_width: Literal[128]
    fft_modes: tuple[Literal[12], Literal[12]]
    dct_modes: tuple[Literal[24], Literal[24]]
    residual_skip: Literal[False]


class Phase4Training(_StrictModel):
    seed: Literal[0]
    maximum_steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    minimum_learning_rate: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    evaluation_interval: int = Field(ge=1)
    checkpoint_interval: int = Field(ge=1)
    validation_velocity_samples: int = Field(ge=1)
    validation_rollout_trajectories: int = Field(ge=1)
    rollout_steps_per_interval: int = Field(ge=1)
    objective: Literal["mean_per_sample_full_field_relative_rhs_error"]

    @model_validator(mode="after")
    def valid_schedule(self) -> Phase4Training:
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum learning rate cannot exceed the initial rate")
        if self.warmup_steps >= self.maximum_steps:
            raise ValueError("warmup must finish before the training budget")
        if self.evaluation_interval > self.maximum_steps:
            raise ValueError("evaluation interval exceeds the training budget")
        return self


class Phase4Gate(_StrictModel):
    minimum_validation_loss_reduction: float = Field(gt=1)
    maximum_velocity_nrmse: float = Field(gt=0)
    require_rollout_better_than_persistence: Literal[True]
    maximum_divergence_fraction: float = Field(ge=0, le=0)


class Phase4Restrictions(_StrictModel):
    development_only: Literal[True]
    generate_confirmatory: Literal[False]
    inspect_confirmatory: Literal[False]
    ood_evaluation: Literal[False]
    superiority_claim: Literal[False]
    model_selection_split: Literal["validation"]


class Phase4Protocol(_StrictModel):
    schema_version: Literal[1]
    study_id: Literal["chronopde_v2"]
    phase: Literal[4]
    protocol_version: Literal[1]
    parent_phase_commit: Literal["53fc3e9"]
    phase2_descriptor: Literal["configs/chronopde_v2/phase2.yaml"]
    phase3_descriptor: Literal["configs/chronopde_v2/phase3.yaml"]
    phase3_protocol_sha256: Literal[
        "f8e716c73eb09341f015b4783c6d0ee7971daaf215c3c59ee98b2dd3aaf5bb34"
    ]
    development_dataset_sha256: Literal[
        "4eb0482bdadb5fe836076131ecb01588409bdb7c3aa0914727a69107deaeded6"
    ]
    normalization_sha256: Literal[
        "c9b334d077ef1075dc3d7e0c0da230954a6b4abfae4b0ce02ecfa69505581c1d"
    ]
    target: Literal["exact_discrete_rhs"]
    time_units: Literal["physical"]
    models: Phase4Models
    training: Phase4Training
    gate: Phase4Gate
    checkpoint_selection: Literal[
        "minimum_validation_rollout_relative_l2_then_velocity_nrmse"
    ]
    outputs: OutputContract
    restrictions: Phase4Restrictions

    @model_validator(mode="after")
    def validate_phase4(self) -> Phase4Protocol:
        if self.outputs.report_root != Path("reports/chronopde_v2/phase4"):
            raise ValueError("Phase 4 reports must use the Phase 4 report root")
        if self.training.validation_rollout_trajectories > 128:
            raise ValueError("rollout subset exceeds the frozen validation split")
        if self.training.validation_velocity_samples > 128 * 101:
            raise ValueError("velocity subset exceeds the frozen validation split")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def load_phase4_protocol(path: Path) -> Phase4Protocol:
    """Load the strict, development-only Phase 4 protocol."""

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Phase 4 protocol root must be a mapping")
    return Phase4Protocol.model_validate(raw)
