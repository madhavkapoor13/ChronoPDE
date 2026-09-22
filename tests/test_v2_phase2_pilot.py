from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.fft import dctn

from chronopde.contracts import PhysicalParameters
from chronopde.v2.numerical import exact_discrete_rhs
from chronopde.v2.pilot import (
    confirmatory_rows,
    development_rows,
    future_manifest,
    generate_cpu_pilot,
    generate_v2_initial_condition,
    manifest_summary,
    pilot_manifest,
    validate_pilot_file,
)
from chronopde.v2.protocol import load_phase2_protocol

ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return load_phase2_protocol(ROOT / "configs/chronopde_v2/phase2.yaml")


def _small_protocol():
    protocol = _protocol()
    pde = protocol.pde.model_copy(
        update={
            "height": 8,
            "width": 8,
            "t_end": 0.1,
            "stored_times": 3,
        }
    )
    initial = protocol.data.initial_condition.model_copy(update={"maximum_mode": 3})
    data = protocol.data.model_copy(update={"initial_condition": initial})
    return protocol.model_copy(update={"pde": pde, "data": data})


def test_manifest_namespaces_are_disjoint_and_deterministic() -> None:
    protocol = _protocol()
    first = manifest_summary(protocol)
    second = manifest_summary(protocol)
    assert first == second
    assert first["future_split_counts"] == {
        "training": 512,
        "validation": 128,
        "confirmatory": 256,
    }
    pilot_ids = {row["trajectory_id"] for row in pilot_manifest(protocol)}
    future_ids = {row["trajectory_id"] for row in future_manifest(protocol)}
    assert pilot_ids.isdisjoint(future_ids)


def test_confirmatory_rows_are_sealed_by_default() -> None:
    rows = future_manifest(_protocol())
    assert {row["split"] for row in development_rows(rows)} == {"training", "validation"}
    with pytest.raises(PermissionError, match="frozen model"):
        confirmatory_rows(rows)
    assert len(confirmatory_rows(rows, frozen_model=True)) == 256


def test_v2_initial_conditions_are_deterministic_smooth_and_scaled() -> None:
    protocol = _protocol()
    first = generate_v2_initial_condition(protocol, 123)
    second = generate_v2_initial_condition(protocol, 123)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 64, 64)
    np.testing.assert_allclose(np.mean(first, axis=(1, 2)), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.std(first, axis=(1, 2)), 0.5, rtol=1e-6)
    spectrum = dctn(first.astype(np.float64), type=2, norm="ortho", axes=(-2, -1))
    assert np.max(np.abs(spectrum[:, 17:, :])) <= 1e-5
    assert np.max(np.abs(spectrum[:, :, 17:])) <= 1e-5


def test_small_cpu_pilot_regenerates_identically_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    protocol = _small_protocol()
    dataset = tmp_path / "pilot.h5"
    summary_path = tmp_path / "summary.json"
    first = generate_cpu_pilot(protocol, dataset, summary_path)
    second = generate_cpu_pilot(protocol, dataset, summary_path)
    assert first == second
    assert first["passed"] is True
    assert first["trajectory_count"] == 24
    assert first["logical_content_sha256"] == second["logical_content_sha256"]
    validate_pilot_file(dataset, protocol)
    regenerated = generate_cpu_pilot(
        protocol, tmp_path / "regenerated.h5", tmp_path / "regenerated.json"
    )
    assert regenerated["logical_content_sha256"] == first["logical_content_sha256"]
    assert regenerated["file_sha256"] == first["file_sha256"]
    with h5py.File(dataset, "r") as handle:
        state = np.asarray(handle["states"][0, 0], dtype=np.float64)
        stored_rhs = np.asarray(handle["rhs"][0, 0], dtype=np.float64)
        parameters = np.asarray(handle["parameters"][0], dtype=np.float64)
    params = PhysicalParameters(*parameters.tolist())
    expected_rhs = exact_discrete_rhs(state, params, protocol.pde)
    np.testing.assert_allclose(stored_rhs, expected_rhs, rtol=1e-6, atol=1e-6)
    with h5py.File(dataset, "r+") as handle:
        handle.attrs["logical_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="logical content checksum"):
        validate_pilot_file(dataset, protocol)
