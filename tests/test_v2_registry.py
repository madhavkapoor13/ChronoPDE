from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronopde.v2.registry import (
    ExecutionStatus,
    ScientificOutcome,
    create_run_manifest,
    make_run_id,
    transition_run_manifest,
)

CONFIG_HASH = "a" * 64
COMMIT = "b" * 40
DATASET_HASH = "c" * 64
NORMALIZATION_HASH = "d" * 64


def _create(path: Path, **overrides: object):
    values = {
        "phase": 2,
        "pde": "reaction_diffusion",
        "task": "boundary",
        "model": "dct_ar",
        "regime": "full",
        "seed": 0,
        "config_hash": CONFIG_HASH,
        "code_commit": COMMIT,
        "checkpoint_selection": "minimum_validation_rollout",
        "dataset_hash": DATASET_HASH,
        "normalization_hash": NORMALIZATION_HASH,
        "timestamp": "2026-09-20T00:00:00+00:00",
    }
    values.update(overrides)
    return create_run_manifest(path, **values)  # type: ignore[arg-type]


def test_run_id_contract() -> None:
    assert make_run_id(
        phase=2,
        pde="reaction_diffusion",
        task="boundary",
        model="dct_ar",
        regime="full",
        seed=3,
        config_hash=CONFIG_HASH,
    ) == "chronopde_v2-p2-reaction_diffusion-boundary-dct_ar-full-s3-aaaaaaaa"


def test_manifest_transitions_and_completion_does_not_pass(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    _create(path)
    transition_run_manifest(
        path, ExecutionStatus.RUNNING, timestamp="2026-09-20T00:01:00+00:00"
    )
    completed = transition_run_manifest(
        path, ExecutionStatus.COMPLETED, timestamp="2026-09-20T00:02:00+00:00"
    )
    assert completed.scientific_outcome == ScientificOutcome.NOT_EVALUATED
    assert completed.finalized is True
    with pytest.raises(ValueError, match="invalid execution transition"):
        transition_run_manifest(path, ExecutionStatus.RUNNING)


def test_duplicate_run_identity_is_idempotent_but_conflict_fails(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    original = _create(path)
    repeated = _create(path, timestamp="2026-09-21T00:00:00+00:00")
    assert repeated == original
    payload = json.loads(path.read_text())
    payload["model"] = "other_model"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileExistsError)):
        _create(path)


def test_failed_run_requires_reason_and_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    _create(path)
    transition_run_manifest(path, ExecutionStatus.RUNNING)
    with pytest.raises(ValueError, match="failure_reason"):
        transition_run_manifest(path, ExecutionStatus.FAILED)
    failed = transition_run_manifest(
        path, ExecutionStatus.FAILED, failure_reason="controlled test failure"
    )
    assert failed.finalized
    with pytest.raises(ValueError):
        transition_run_manifest(path, ExecutionStatus.RUNNING)


def test_run_cannot_start_before_data_identities_exist(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    _create(path, dataset_hash=None, normalization_hash=None)
    with pytest.raises(ValueError, match="require dataset and normalization"):
        transition_run_manifest(path, ExecutionStatus.RUNNING)
