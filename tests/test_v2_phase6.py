from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from chronopde.v2.common import canonical_json_bytes
from chronopde.v2.phase6 import (
    MODEL_NAMES,
    _generate_one,
    phase6_decision,
    validate_phase6_contract,
)
from chronopde.v2.pilot import confirmatory_rows, future_manifest, manifest_hash
from chronopde.v2.protocol import load_phase2_protocol, load_phase6_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chronopde_v2/phase6.yaml"


def _summary(
    model: str,
    seed: int,
    *,
    rollout: float | None,
    velocity: float | None,
    persistence: float,
    divergence: float,
    boundary: float = 0.2,
    derivative: float = 0.2,
) -> dict[str, object]:
    return {
        "model": model,
        "seed": seed,
        "rollout_relative_l2_all_finite": rollout,
        "persistence_relative_l2_all": persistence,
        "divergence_fraction": divergence,
        "velocity": {"velocity_nrmse": velocity},
        "attribution_medians": {
            "boundary_strip_relative_l2": boundary,
            "first_interior_normal_derivative_relative_l2": derivative,
        },
    }


def _paired_summaries(*, dct_wins: int = 5) -> dict[tuple[str, int], dict[str, object]]:
    summaries = {}
    for seed in range(5):
        summaries[("fft", seed)] = _summary(
            "fft",
            seed,
            rollout=1.0,
            velocity=0.4,
            persistence=1.2,
            divergence=0.1,
            boundary=0.4,
            derivative=0.4,
        )
        winning = seed < dct_wins
        summaries[("dct", seed)] = _summary(
            "dct",
            seed,
            rollout=0.8 if winning else 1.1,
            velocity=0.2 if winning else 0.5,
            persistence=1.2,
            divergence=0.0,
            boundary=0.2 if winning else 0.5,
            derivative=0.2 if winning else 0.5,
        )
    return summaries


def test_phase6_protocol_freezes_inputs_and_prohibits_training() -> None:
    protocol = load_phase6_protocol(CONFIG)
    assert protocol.checkpoints.source_archive_sha256 == (
        "cf3eede191de4f9e1dd43b36f1d9f1000d314d3a0e01bc90f71a80d4ebee3c54"
    )
    assert protocol.phase5_protocol_sha256 == (
        "73d7ad0540351fc206574e1cc4d6f47685f7fe392b7c918a867e73d09b02776f"
    )
    assert protocol.confirmatory_manifest_sha256 == (
        "dbb177527a5e740df474d0bd26d39be42904fdc649bd1e81bd5f6ed98a1b85a1"
    )
    assert len(protocol.checkpoints.fft) == len(protocol.checkpoints.dct) == 5
    assert protocol.restrictions.training is False
    assert protocol.restrictions.optimizer_updates is False
    assert protocol.restrictions.checkpoint_writes is False
    assert protocol.restrictions.repeat_with_changed_settings is False


def test_phase6_confirmatory_identities_are_the_frozen_phase2_holdout() -> None:
    protocol = load_phase6_protocol(CONFIG)
    phase2 = load_phase2_protocol(ROOT / protocol.phase2_descriptor)
    rows = confirmatory_rows(future_manifest(phase2), frozen_model=True)
    assert len(rows) == 256
    assert manifest_hash(rows) == protocol.confirmatory_manifest_sha256
    assert not any(str(row["trajectory_id"]).startswith(("train", "validation")) for row in rows)


def test_phase6_strict_gate_passes_only_five_of_five_directional_wins() -> None:
    protocol = load_phase6_protocol(CONFIG)
    decision = phase6_decision(_paired_summaries(), protocol)  # type: ignore[arg-type]
    assert decision["passed"] is True
    assert decision["rollout_wins"] == 5
    assert decision["velocity_wins"] == 5
    assert decision["median_relative_improvement"] == pytest.approx(0.2)
    assert decision["one_sided_exact_sign_test_p"] == 0.03125
    assert decision["boundary_wording_authorized"] is True
    assert decision["superiority_claim_authorized"] is True


def test_phase6_four_of_five_wins_is_a_confirmatory_failure() -> None:
    protocol = load_phase6_protocol(CONFIG)
    decision = phase6_decision(_paired_summaries(dct_wins=4), protocol)  # type: ignore[arg-type]
    assert decision["passed"] is False
    assert decision["decision"] == "phase6_confirmatory_claim_failed_no_retuning"
    assert decision["superiority_claim_authorized"] is False


def test_phase6_nonfinite_result_is_unavailable_and_fails_without_crashing() -> None:
    protocol = load_phase6_protocol(CONFIG)
    summaries = _paired_summaries()
    summaries[("dct", 3)] = _summary(
        "dct", 3, rollout=None, velocity=None, persistence=1.2, divergence=1.0
    )
    decision = phase6_decision(summaries, protocol)  # type: ignore[arg-type]
    assert decision["passed"] is False
    assert decision["median_relative_improvement"] is None
    assert decision["paired_results"][3]["relative_improvement"] is None


def test_phase6_incomplete_execution_allows_only_identical_rerun() -> None:
    protocol = load_phase6_protocol(CONFIG)
    summaries = _paired_summaries()
    summaries.pop(("fft", 4))
    decision = phase6_decision(summaries, protocol)  # type: ignore[arg-type]
    assert decision["complete"] is False
    assert decision["decision"] == "phase6_incomplete_identical_rerun_only"
    assert decision["superiority_claim_authorized"] is False


def test_phase6_source_constructs_no_optimizer_and_writes_no_checkpoint() -> None:
    import chronopde.v2.phase6 as phase6

    source = inspect.getsource(phase6)
    assert "torch.optim" not in source
    assert "save_v2_checkpoint" not in source


def test_phase6_notebook_code_cells_compile() -> None:
    for name in (
        "chronopde_v2_phase6_generate_kaggle.ipynb",
        "chronopde_v2_phase6_evaluate_kaggle.ipynb",
    ):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), name, "exec")


def test_phase6_archive_hash_conflict_stops_before_extraction(tmp_path: Path) -> None:
    protocol = load_phase6_protocol(CONFIG)
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the frozen archive")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_phase6_contract(ROOT, protocol, archive)


def test_phase6_corrupt_resumable_cache_is_rejected(tmp_path: Path) -> None:
    protocol = load_phase6_protocol(CONFIG)
    phase2 = load_phase2_protocol(ROOT / protocol.phase2_descriptor)
    row = confirmatory_rows(future_manifest(phase2), frozen_model=True)[0]
    cache = tmp_path / f"{row['trajectory_id']}.npz"
    np.savez_compressed(
        cache,
        trajectory_id=row["trajectory_id"],
        protocol_hash=protocol.digest,
        row_json=canonical_json_bytes(row).decode("utf-8"),
        states=np.zeros((1,), dtype=np.float32),
        rhs=np.zeros((1,), dtype=np.float32),
        times=np.zeros((1,), dtype=np.float64),
        nfev=1,
        maximum=0.0,
        logical_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="corrupt Phase 6 cache payload"):
        _generate_one(row, tmp_path, phase2, protocol)


def test_phase6_models_are_exactly_the_two_frozen_backbones() -> None:
    assert MODEL_NAMES == ("fft", "dct")


def test_committed_phase6_evidence_records_the_frozen_passing_gate() -> None:
    protocol = load_phase6_protocol(CONFIG)
    report_root = ROOT / "reports/chronopde_v2/phase6"
    decision = json.loads((report_root / "decision_report.json").read_text())
    snapshot = json.loads((report_root / "protocol_snapshot.json").read_text())
    assert decision["protocol_sha256"] == protocol.digest
    assert decision["complete"] is decision["passed"] is True
    assert decision["superiority_claim_authorized"] is True
    assert decision["rollout_wins"] == decision["velocity_wins"] == 5
    assert decision["median_relative_improvement"] == pytest.approx(0.21135876577066035)
    assert decision["hierarchical_bootstrap"]["available_resamples"] == 20_000
    assert snapshot == protocol.model_dump(mode="json")
