"""Immutable run identities and validated lifecycle transitions for ChronoPDE V2."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from chronopde.v2.common import atomic_write_json, read_json_object

SCHEMA_VERSION = 1
STUDY_ID = "chronopde_v2"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TOKEN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ExecutionStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ScientificOutcome(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


TERMINAL_STATUSES = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.FAILED,
    ExecutionStatus.INTERRUPTED,
}
ALLOWED_TRANSITIONS = {
    ExecutionStatus.PLANNED: {ExecutionStatus.RUNNING, ExecutionStatus.FAILED},
    ExecutionStatus.RUNNING: TERMINAL_STATUSES,
    ExecutionStatus.COMPLETED: set(),
    ExecutionStatus.FAILED: set(),
    ExecutionStatus.INTERRUPTED: set(),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_hash(value: str | None, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _validate_token(value: str, field: str) -> None:
    if not TOKEN_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must contain only lowercase identifier characters")


def make_run_id(
    *, phase: int, pde: str, task: str, model: str, regime: str, seed: int, config_hash: str
) -> str:
    if phase < 1 or seed < 0:
        raise ValueError("phase must be positive and seed must be non-negative")
    _validate_hash(config_hash, "config_hash")
    for field, value in (("pde", pde), ("task", task), ("model", model), ("regime", regime)):
        _validate_token(value, field)
    return f"{STUDY_ID}-p{phase}-{pde}-{task}-{model}-{regime}-s{seed}-{config_hash[:8]}"


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    study_id: str
    phase: int
    run_id: str
    code_commit: str
    config_hash: str
    dataset_hash: str | None
    normalization_hash: str | None
    model: str
    task: str
    regime: str
    seed: int
    artifact_directory: str
    created_at: str
    updated_at: str
    execution_status: ExecutionStatus
    scientific_outcome: ScientificOutcome
    checkpoint_selection: str
    failure_reason: str | None
    parent_run: str | None
    finalized: bool

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.study_id != STUDY_ID:
            raise ValueError("unsupported V2 run manifest identity")
        _validate_hash(self.config_hash, "config_hash")
        _validate_hash(self.dataset_hash, "dataset_hash", nullable=True)
        _validate_hash(self.normalization_hash, "normalization_hash", nullable=True)
        if self.execution_status != ExecutionStatus.PLANNED and (
            self.dataset_hash is None or self.normalization_hash is None
        ):
            raise ValueError(
                "running or terminal runs require dataset and normalization identities"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", self.code_commit):
            raise ValueError("code_commit must be a full lowercase Git commit")
        expected = make_run_id(
            phase=self.phase,
            pde=self.run_id.split("-")[2],
            task=self.task,
            model=self.model,
            regime=self.regime,
            seed=self.seed,
            config_hash=self.config_hash,
        )
        if self.run_id != expected:
            raise ValueError("run_id does not match the manifest identity")
        if Path(self.artifact_directory).parts[:3] != ("artifacts", STUDY_ID, "runs"):
            raise ValueError("V2 artifacts must use the isolated V2 run root")
        if not self.checkpoint_selection.strip():
            raise ValueError("checkpoint_selection must be explicit")
        if self.finalized != (self.execution_status in TERMINAL_STATUSES):
            raise ValueError("terminal execution states and finalized must agree")
        if self.execution_status != ExecutionStatus.FAILED and self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for failed execution")
        if self.execution_status == ExecutionStatus.FAILED and not self.failure_reason:
            raise ValueError("failed execution requires a failure_reason")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["execution_status"] = self.execution_status.value
        payload["scientific_outcome"] = self.scientific_outcome.value
        return cast(dict[str, object], payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        manifest = cls(
            schema_version=int(payload["schema_version"]),
            study_id=str(payload["study_id"]),
            phase=int(payload["phase"]),
            run_id=str(payload["run_id"]),
            code_commit=str(payload["code_commit"]),
            config_hash=str(payload["config_hash"]),
            dataset_hash=cast(str | None, payload["dataset_hash"]),
            normalization_hash=cast(str | None, payload["normalization_hash"]),
            model=str(payload["model"]),
            task=str(payload["task"]),
            regime=str(payload["regime"]),
            seed=int(payload["seed"]),
            artifact_directory=str(payload["artifact_directory"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            execution_status=ExecutionStatus(payload["execution_status"]),
            scientific_outcome=ScientificOutcome(payload["scientific_outcome"]),
            checkpoint_selection=str(payload["checkpoint_selection"]),
            failure_reason=cast(str | None, payload["failure_reason"]),
            parent_run=cast(str | None, payload["parent_run"]),
            finalized=bool(payload["finalized"]),
        )
        manifest.validate()
        return manifest


def create_run_manifest(
    path: Path,
    *,
    phase: int,
    pde: str,
    task: str,
    model: str,
    regime: str,
    seed: int,
    config_hash: str,
    code_commit: str,
    checkpoint_selection: str,
    dataset_hash: str | None = None,
    normalization_hash: str | None = None,
    parent_run: str | None = None,
    timestamp: str | None = None,
) -> RunManifest:
    run_id = make_run_id(
        phase=phase,
        pde=pde,
        task=task,
        model=model,
        regime=regime,
        seed=seed,
        config_hash=config_hash,
    )
    now = timestamp or utc_now()
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        study_id=STUDY_ID,
        phase=phase,
        run_id=run_id,
        code_commit=code_commit,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        normalization_hash=normalization_hash,
        model=model,
        task=task,
        regime=regime,
        seed=seed,
        artifact_directory=f"artifacts/{STUDY_ID}/runs/{run_id}",
        created_at=now,
        updated_at=now,
        execution_status=ExecutionStatus.PLANNED,
        scientific_outcome=ScientificOutcome.NOT_EVALUATED,
        checkpoint_selection=checkpoint_selection,
        failure_reason=None,
        parent_run=parent_run,
        finalized=False,
    )
    manifest.validate()
    if path.exists():
        existing = RunManifest.from_dict(read_json_object(path))
        identity_fields = (
            "schema_version",
            "study_id",
            "phase",
            "run_id",
            "code_commit",
            "config_hash",
            "dataset_hash",
            "normalization_hash",
            "model",
            "task",
            "regime",
            "seed",
            "artifact_directory",
            "checkpoint_selection",
            "parent_run",
        )
        if any(getattr(existing, field) != getattr(manifest, field) for field in identity_fields):
            raise FileExistsError(f"run manifest already exists with another identity: {path}")
        return existing
    atomic_write_json(path, manifest.to_dict())
    return manifest


def transition_run_manifest(
    path: Path,
    status: ExecutionStatus,
    *,
    outcome: ScientificOutcome | None = None,
    failure_reason: str | None = None,
    timestamp: str | None = None,
) -> RunManifest:
    current = RunManifest.from_dict(read_json_object(path))
    if current.finalized or status not in ALLOWED_TRANSITIONS[current.execution_status]:
        raise ValueError(f"invalid execution transition: {current.execution_status} -> {status}")
    updated = replace(
        current,
        execution_status=status,
        scientific_outcome=outcome or current.scientific_outcome,
        failure_reason=failure_reason,
        updated_at=timestamp or utc_now(),
        finalized=status in TERMINAL_STATUSES,
    )
    updated.validate()
    atomic_write_json(path, updated.to_dict())
    return updated
