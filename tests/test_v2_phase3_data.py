from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from chronopde.v2.development_data import (
    generate_development_dataset,
    validate_development_dataset,
)
from chronopde.v2.pilot import development_rows, future_manifest, manifest_hash
from chronopde.v2.protocol import load_phase2_protocol, load_phase3_protocol

ROOT = Path(__file__).resolve().parents[1]


def _protocols():
    phase2 = load_phase2_protocol(ROOT / "configs/chronopde_v2/phase2.yaml")
    phase3 = load_phase3_protocol(ROOT / "configs/chronopde_v2/phase3.yaml")
    return phase2, phase3


def _small_protocols():
    phase2, phase3 = _protocols()
    pde = phase2.pde.model_copy(
        update={"height": 8, "width": 8, "t_end": 0.1, "stored_times": 3}
    )
    initial = phase2.data.initial_condition.model_copy(update={"maximum_mode": 3})
    phase2_data = phase2.data.model_copy(
        update={
            "training_trajectories": 4,
            "validation_trajectories": 2,
            "confirmatory_trajectories": 2,
            "initial_condition": initial,
        }
    )
    phase2 = phase2.model_copy(update={"pde": pde, "data": phase2_data})
    rows = future_manifest(phase2)
    splits = phase3.development_splits.model_copy(update={"training": 4, "validation": 2})
    phase3 = phase3.model_copy(
        update={
            "phase2_protocol_sha256": phase2.digest,
            "frozen_future_manifest_sha256": manifest_hash(rows),
            "development_splits": splits,
            "sealed_confirmatory_trajectories": 2,
            "workers": 2,
            "rhs_spot_checks": 3,
        }
    )
    return phase2, phase3


def test_phase3_protocol_freezes_development_only_scope() -> None:
    phase2, phase3 = _protocols()
    assert phase3.phase2_protocol_sha256 == phase2.digest
    assert phase3.development_splits.training == 512
    assert phase3.development_splits.validation == 128
    assert phase3.sealed_confirmatory_trajectories == 256
    assert phase3.restrictions.generate_confirmatory is False
    assert phase3.restrictions.model_training is False
    assert phase3.restrictions.use_pilot_normalization is False


def test_small_development_generation_is_resumable_exact_and_sealed(tmp_path: Path) -> None:
    phase2, phase3 = _small_protocols()
    first, qa = generate_development_dataset(phase2, phase3, tmp_path)
    second, second_qa = generate_development_dataset(phase2, phase3, tmp_path)
    assert first == second
    assert len(qa) == 6
    assert all(row["cached"] for row in second_qa)
    assert first["successful_trajectories"] == 6
    assert first["training_trajectories"] == 4
    assert first["validation_trajectories"] == 2
    assert first["confirmatory_trajectories_generated"] == 0
    assert first["maximum_rhs_spot_check_relative_l2"] <= 1e-6
    dataset = tmp_path / "chronopde_v2_development.h5"
    rows = development_rows(future_manifest(phase2))
    validated = validate_development_dataset(dataset, rows, phase2, phase3)
    assert validated["logical_content_sha256"] == first["logical_content_sha256"]
    with h5py.File(dataset) as handle:
        assert set(handle["splits"]) == {"training", "validation"}
        assert handle["normalization"].attrs["source_split"] == "training"
        assert np.all(np.asarray(handle["normalization/state_std"]) > 0)
    with h5py.File(dataset, "r+") as handle:
        handle["normalization/state_mean"][0] += 0.1
    with pytest.raises(ValueError, match="training-only normalization mismatch"):
        validate_development_dataset(dataset, rows, phase2, phase3)


def test_corrupt_trajectory_cache_is_rejected_before_reuse(tmp_path: Path) -> None:
    phase2, phase3 = _small_protocols()
    generate_development_dataset(phase2, phase3, tmp_path)
    cache = next((tmp_path / "trajectory_cache/training").glob("*.npz"))
    cache.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid trajectory cache"):
        generate_development_dataset(phase2, phase3, tmp_path)
