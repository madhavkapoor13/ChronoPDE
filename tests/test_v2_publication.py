from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronopde.v2.publication import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_RESULTS_SHA256,
    load_release_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_release_evidence_matches_frozen_phase6_numbers() -> None:
    evidence = load_release_evidence(ROOT)
    assert evidence.median_relative_improvement == pytest.approx(0.21135876577066035)
    assert evidence.bootstrap_ci_low == pytest.approx(0.16353042010042135)
    assert evidence.bootstrap_ci_high == pytest.approx(0.2772115352816319)
    assert evidence.sign_test_p == pytest.approx(0.03125)
    assert evidence.dataset_sha256 == EXPECTED_DATASET_SHA256
    assert evidence.results_sha256 == EXPECTED_RESULTS_SHA256
    assert evidence.fft_divergent_rollouts == 255
    assert evidence.dct_divergent_rollouts == 0


def test_release_manifest_uses_explicit_availability_states() -> None:
    manifest = json.loads(
        (ROOT / "reports/chronopde_v2/release/artifact_manifest.json").read_text()
    )
    assert manifest["schema_version"] == 1
    assert manifest["study_id"] == "chronopde_v2"
    assert {item["availability"] for item in manifest["artifacts"]} <= {
        "verified_local",
        "verified_local_non_authoritative",
        "kaggle_private_only",
    }
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_public_headline_is_derived_from_frozen_evidence() -> None:
    evidence = load_release_evidence(ROOT)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "21.14%" in readme
    assert "16.35%" in readme and "27.72%" in readme
    assert "1,280" in readme
    assert evidence.permitted_claim in (
        ROOT / "reports/chronopde_v2/phase6/decision_report.json"
    ).read_text(encoding="utf-8")


def test_demo_routes_v2_and_historical_v1_evidence() -> None:
    source = (ROOT / "demo/app.py").read_text(encoding="utf-8")
    assert "load_release_evidence" in source
    assert "Historical V1" in source
    historical = json.loads((ROOT / "reports/final/final_summary.json").read_text())
    assert historical["decision"] == "stop_and_document_model_or_conditioning_limitation"


def test_report_builder_consumes_only_committed_release_evidence() -> None:
    source = (ROOT / "scripts/build_v2_confirmatory_report.py").read_text(encoding="utf-8")
    compile(source, "build_v2_confirmatory_report.py", "exec")
    assert "load_release_evidence(ROOT)" in source
    assert "chronopde_v2_confirmatory_report.pdf" in source


def test_release_verifier_reports_resume_headline() -> None:
    from scripts.verify_v2_release import verified_summary

    summary = verified_summary(ROOT)
    assert summary["passed"] is True
    assert summary["confirmatory_trajectories"] == 256
    assert summary["checkpoint_pairs"] == 5
    assert summary["rollout_wins"] == 5
    assert summary["velocity_wins"] == 5
    assert summary["median_paired_rollout_improvement"] == pytest.approx(
        0.21135876577066035
    )
    assert summary["dct_divergent_rollouts"] == 0
    assert summary["total_dct_rollouts"] == 1280
