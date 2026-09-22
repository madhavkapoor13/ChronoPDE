from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from chronopde.models import trainable_parameter_count
from chronopde.v2.phase4 import phase4_decision, validate_phase4_contract
from chronopde.v2.phase4_training import (
    DevelopmentExactRHSDataset,
    ReplacementSampler,
    build_phase4_model,
    fixed_validation_indices,
    phase4_gate,
    train_phase4_model,
)
from chronopde.v2.protocol import load_phase4_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chronopde_v2/phase4.yaml"


def test_phase4_contract_is_matched_and_development_only() -> None:
    protocol = load_phase4_protocol(CONFIG)
    report = validate_phase4_contract(ROOT, protocol)
    assert report["passed"] is True
    assert report["parameter_counts"] == {"fft": 1_973_657, "dct": 1_973_657}
    assert report["confirmatory_access_allowed"] is False
    assert protocol.restrictions.superiority_claim is False
    assert protocol.training.objective == "mean_per_sample_full_field_relative_rhs_error"
    assert trainable_parameter_count(build_phase4_model(protocol, "fft")) == 1_973_657
    assert trainable_parameter_count(build_phase4_model(protocol, "dct")) == 1_973_657


def test_phase4_fixed_validation_samples_are_deterministic() -> None:
    protocol = load_phase4_protocol(CONFIG)
    first = fixed_validation_indices(128 * 101, 128, protocol)
    second = fixed_validation_indices(128 * 101, 128, protocol)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(np.unique(first[0])) == protocol.training.validation_velocity_samples
    assert len(np.unique(first[1])) == protocol.training.validation_rollout_trajectories


def test_replacement_sampler_resumes_exactly() -> None:
    sampler = ReplacementSampler(100, 7)
    _ = sampler.next(11)
    state = sampler.state_dict()
    expected = sampler.next(25)
    resumed = ReplacementSampler(100, 999, state)
    assert np.array_equal(resumed.next(25), expected)
    with pytest.raises(ValueError, match="sample count"):
        ReplacementSampler(99, 7, state)


def _gate_rows(
    *, complete: bool, velocity: float, rollout: float, persistence: float
) -> list[dict[str, object]]:
    return [
        {
            "optimizer_steps": 0,
            "validation_relative_loss": 2.0,
            "velocity_nrmse": 1.0,
            "rollout_relative_l2": 2.0,
            "persistence_relative_l2": persistence,
            "divergence_fraction": 0.0,
        },
        {
            "optimizer_steps": 10_000 if complete else 9_000,
            "validation_relative_loss": 0.01,
            "velocity_nrmse": velocity,
            "rollout_relative_l2": rollout,
            "persistence_relative_l2": persistence,
            "divergence_fraction": 0.0,
        },
    ]


def test_phase4_gate_requires_full_budget_and_all_metrics() -> None:
    protocol = load_phase4_protocol(CONFIG)
    assert phase4_gate(
        _gate_rows(complete=True, velocity=0.10, rollout=0.2, persistence=0.4), protocol
    )["passed"]
    assert not phase4_gate(
        _gate_rows(complete=False, velocity=0.10, rollout=0.2, persistence=0.4), protocol
    )["passed"]
    assert not phase4_gate(
        _gate_rows(complete=True, velocity=0.20, rollout=0.2, persistence=0.4), protocol
    )["passed"]
    assert not phase4_gate(
        _gate_rows(complete=True, velocity=0.10, rollout=0.5, persistence=0.4), protocol
    )["passed"]


@pytest.mark.parametrize(
    ("fft", "dct", "decision"),
    [
        (True, True, "phase4_passed_phase5_multiseed_validation_allowed"),
        (True, False, "single_backbone_pass_controlled_investigation_required"),
        (False, True, "single_backbone_pass_controlled_investigation_required"),
        (False, False, "phase4_failed_pause_continuous_time_study"),
    ],
)
def test_phase4_routes_all_gate_combinations(fft: bool, dct: bool, decision: str) -> None:
    results = {name: {"gate": {"passed": value}} for name, value in (("fft", fft), ("dct", dct))}
    report = phase4_decision(results)
    assert report["decision"] == decision
    assert report["superiority_claim_authorized"] is False


def _write_minimal_dataset(
    path: Path, *, include_confirmatory: bool, populate: bool = False
) -> None:
    summary = json.loads((ROOT / "reports/chronopde_v2/phase3/dataset_summary.json").read_text())
    normalization = summary["normalization"]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "chronopde_v2_development_1"
        handle.attrs["logical_content_sha256"] = summary["logical_content_sha256"]
        handle.attrs["confirmatory_status"] = "sealed_not_generated"
        splits = handle.create_group("splits")
        for split in ("training", "validation"):
            group = splits.create_group(split)
            if populate:
                mean = np.asarray(normalization["state_mean"], dtype=np.float32)
                states = np.broadcast_to(
                    mean[None, None, :, None, None], (1, 2, 2, 64, 64)
                ).copy()
                states[:, 1] += 0.01
                rhs = np.full_like(states, 0.01)
                group.create_dataset("states", data=states)
                group.create_dataset("rhs", data=rhs)
                group.create_dataset("times", data=np.asarray([[0.0, 0.5]]))
                group.create_dataset(
                    "parameters",
                    data=np.asarray([normalization["parameter_mean"]], dtype=np.float64),
                )
        if include_confirmatory:
            splits.create_group("confirmatory")
        group = handle.create_group("normalization")
        for name, values in normalization.items():
            group.create_dataset(name, data=np.asarray(values, dtype=np.float64))


def test_dataset_reader_rejects_confirmatory_states(tmp_path: Path) -> None:
    protocol = load_phase4_protocol(CONFIG)
    path = tmp_path / "development.h5"
    _write_minimal_dataset(path, include_confirmatory=True)
    with pytest.raises(ValueError, match="confirmatory"):
        DevelopmentExactRHSDataset(path, protocol)


def test_phase4_notebook_code_cells_compile() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/chronopde_v2_phase4_kaggle.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "phase4_notebook", "exec")


def test_phase4_one_step_cpu_smoke_creates_verified_recovery(tmp_path: Path) -> None:
    protocol = load_phase4_protocol(CONFIG)
    training = protocol.training.model_copy(
        update={
            "maximum_steps": 1,
            "batch_size": 1,
            "warmup_steps": 0,
            "evaluation_interval": 1,
            "checkpoint_interval": 1,
            "validation_velocity_samples": 1,
            "validation_rollout_trajectories": 1,
        }
    )
    smoke_protocol = protocol.model_copy(update={"training": training})
    data_path = tmp_path / "development.h5"
    _write_minimal_dataset(data_path, include_confirmatory=False, populate=True)
    result = train_phase4_model(
        tmp_path,
        smoke_protocol,
        data_path,
        "fft",
        torch.device("cpu"),
        code_commit="0" * 40,
    )
    run_root = tmp_path / "artifacts/chronopde_v2/runs" / result["run_id"]
    assert result["optimizer_steps"] == 1
    assert (run_root / "last.pt").is_file()
    assert (run_root / "last.pt.sha256").is_file()
    assert (run_root / "recovery.zip").is_file()
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["execution_status"] == "completed"
