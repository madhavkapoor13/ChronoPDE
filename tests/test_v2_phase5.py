from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from chronopde.models import trainable_parameter_count
from chronopde.v2.phase5 import (
    build_phase5_model,
    phase5_decision,
    train_phase5_run,
    validate_phase5_contract,
)
from chronopde.v2.protocol import load_phase4_protocol, load_phase5_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chronopde_v2/phase5.yaml"


def test_phase5_contract_is_fresh_multiseed_and_development_only() -> None:
    protocol = load_phase5_protocol(CONFIG)
    result = validate_phase5_contract(ROOT, protocol)
    assert result["passed"] is True
    assert result["parameter_counts"] == {"fft": 1_973_657, "dct": 1_973_657}
    assert result["seeds"] == [0, 1, 2, 3, 4]
    assert result["rollout_steps_per_interval"] == 8
    assert result["confirmatory_access_allowed"] is False
    assert protocol.restrictions.fresh_runs is True
    assert protocol.restrictions.reuse_phase4_checkpoints is False
    assert protocol.restrictions.superiority_claim is False
    for model in ("fft", "dct"):
        assert trainable_parameter_count(build_phase5_model(protocol, model)) == 1_973_657


def _summary(
    *, velocity: float, rollout: float | None, persistence: float, divergence: float
) -> dict[str, object]:
    return {
        "checkpoint_selection": {"velocity_nrmse": velocity},
        "rollout": {
            "rollout_relative_l2": rollout,
            "rollout_relative_l2_all_finite": rollout,
            "persistence_relative_l2_all": persistence,
            "divergence_fraction": divergence,
        },
    }


def test_phase5_gate_freezes_dct_only_after_four_of_five_paired_wins() -> None:
    protocol = load_phase5_protocol(CONFIG)
    summaries = {}
    for seed in protocol.training.seeds:
        summaries[("fft", seed)] = _summary(
            velocity=0.20, rollout=1.10, persistence=1.0, divergence=0.5
        )
        summaries[("dct", seed)] = _summary(
            velocity=0.10, rollout=0.80, persistence=1.0, divergence=0.0
        )
    decision = phase5_decision(summaries, protocol)
    assert decision["passed"] is True
    assert decision["dct_rollout_wins"] == 5
    assert decision["dct_velocity_wins"] == 5
    assert decision["frozen_candidate_model"] == "dct"
    assert decision["phase6_confirmatory_generation_allowed"] is True
    assert decision["superiority_claim_authorized"] is False


def test_phase5_incomplete_budget_never_authorizes_phase6() -> None:
    protocol = load_phase5_protocol(CONFIG)
    decision = phase5_decision(
        {("dct", 0): _summary(velocity=0.1, rollout=0.8, persistence=1.0, divergence=0.0)},
        protocol,
    )
    assert decision["complete"] is False
    assert decision["phase6_confirmatory_generation_allowed"] is False


def test_phase5_dct_instability_blocks_phase6() -> None:
    protocol = load_phase5_protocol(CONFIG)
    summaries = {}
    for seed in protocol.training.seeds:
        summaries[("fft", seed)] = _summary(
            velocity=0.20, rollout=1.10, persistence=1.0, divergence=0.5
        )
        summaries[("dct", seed)] = _summary(
            velocity=0.10,
            rollout=0.80,
            persistence=1.0,
            divergence=0.1 if seed == 4 else 0.0,
        )
    decision = phase5_decision(summaries, protocol)
    assert decision["passed"] is False
    assert decision["decision"] == "phase5_failed_no_confirmatory_generation"
    assert decision["phase6_confirmatory_generation_allowed"] is False


def test_phase5_all_divergent_dct_rollouts_fail_without_crashing() -> None:
    protocol = load_phase5_protocol(CONFIG)
    summaries = {}
    for seed in protocol.training.seeds:
        summaries[("fft", seed)] = _summary(
            velocity=0.20, rollout=None, persistence=1.0, divergence=1.0
        )
        summaries[("dct", seed)] = _summary(
            velocity=0.10, rollout=None, persistence=1.0, divergence=1.0
        )
    decision = phase5_decision(summaries, protocol)
    assert decision["passed"] is False
    assert decision["phase6_confirmatory_generation_allowed"] is False


def test_phase5_notebook_code_cells_compile() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/chronopde_v2_phase5_kaggle.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "phase5_notebook", "exec")


def _write_minimal_dataset(path: Path) -> None:
    summary = json.loads(
        (ROOT / "reports/chronopde_v2/phase3/dataset_summary.json").read_text()
    )
    normalization = summary["normalization"]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "chronopde_v2_development_1"
        handle.attrs["logical_content_sha256"] = summary["logical_content_sha256"]
        handle.attrs["confirmatory_status"] = "sealed_not_generated"
        splits = handle.create_group("splits")
        for split in ("training", "validation"):
            group = splits.create_group(split)
            mean = np.asarray(normalization["state_mean"], dtype=np.float32)
            states = np.broadcast_to(
                mean[None, None, :, None, None], (1, 2, 2, 64, 64)
            ).copy()
            states[:, 1] += 0.01
            group.create_dataset("states", data=states)
            group.create_dataset("rhs", data=np.full_like(states, 0.01))
            group.create_dataset("times", data=np.asarray([[0.0, 0.5]]))
            group.create_dataset(
                "parameters",
                data=np.asarray([normalization["parameter_mean"]], dtype=np.float64),
            )
            group.create_dataset("trajectory_ids", data=np.asarray([b"test-0000"]))
        group = handle.create_group("normalization")
        for name, values in normalization.items():
            group.create_dataset(name, data=np.asarray(values, dtype=np.float64))


def test_phase5_one_step_cpu_smoke_is_fresh_and_recoverable(tmp_path: Path) -> None:
    protocol = load_phase5_protocol(CONFIG)
    training = protocol.training.model_copy(
        update={
            "maximum_steps": 1,
            "batch_size": 1,
            "warmup_steps": 0,
            "evaluation_interval": 1,
            "checkpoint_interval": 1,
            "validation_velocity_samples": 1,
            "validation_rollout_trajectories": 1,
            "rollout_steps_per_interval": 1,
        }
    )
    smoke_protocol = protocol.model_copy(update={"training": training})
    phase4 = load_phase4_protocol(ROOT / protocol.phase4_descriptor)
    phase4_training = phase4.training.model_copy(
        update={
            "validation_velocity_samples": 1,
            "validation_rollout_trajectories": 1,
        }
    )
    smoke_phase4 = phase4.model_copy(update={"training": phase4_training})
    data_path = tmp_path / "development.h5"
    _write_minimal_dataset(data_path)
    result = train_phase5_run(
        tmp_path,
        smoke_protocol,
        smoke_phase4,
        data_path,
        "fft",
        0,
        torch.device("cpu"),
        code_commit="0" * 40,
    )
    run_root = tmp_path / "artifacts/chronopde_v2/runs" / result["run_id"]
    assert result["optimizer_steps"] == 1
    assert result["confirmatory_accessed"] is False
    assert (run_root / "best.pt").is_file()
    assert (run_root / "last.pt.sha256").is_file()
    assert (run_root / "recovery.zip").is_file()
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["execution_status"] == "completed"
