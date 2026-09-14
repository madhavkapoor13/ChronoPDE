from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chronopde.analysis.week6_recovery import (
    BOOTSTRAP_RESAMPLES,
    EXPECTED_ROUTE,
    _paired_bootstrap,
    analyze_week6_failure,
    verify_archives,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports/diagnostics/week6/evidence_manifest.json"


def test_paired_bootstrap_is_deterministic() -> None:
    differences = np.asarray([-0.1, -0.2, 0.05, -0.04])
    assert _paired_bootstrap(differences) == _paired_bootstrap(differences)
    assert BOOTSTRAP_RESAMPLES == 10_000


def test_archive_verification_rejects_checksum_mismatch(tmp_path: Path) -> None:
    (tmp_path / "evidence.zip").write_bytes(b"not the declared archive")
    manifest = {
        "archives": [
            {
                "id": "evidence",
                "filename": "evidence.zip",
                "git_commit": "1" * 40,
                "sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archives(manifest, tmp_path)


def test_archive_verification_rejects_path_traversal(tmp_path: Path) -> None:
    manifest = {
        "archives": [
            {
                "id": "evidence",
                "filename": "../evidence.zip",
                "git_commit": "1" * 40,
                "sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(ValueError, match="escapes declared root"):
        verify_archives(manifest, tmp_path)


def test_archive_verification_rejects_invalid_commit(tmp_path: Path) -> None:
    manifest = {
        "archives": [
            {
                "id": "evidence",
                "filename": "evidence.zip",
                "git_commit": "not-a-commit",
                "sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(ValueError, match="invalid Git commit"):
        verify_archives(manifest, tmp_path)


def test_committed_evidence_excludes_large_or_nested_artifacts() -> None:
    evidence = ROOT / "reports/diagnostics/week6/evidence"
    forbidden_suffixes = {".h5", ".hdf5", ".pt", ".pth", ".zip"}
    assert not [path for path in evidence.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    assert not [path for path in evidence.rglob(".git")]


def test_frozen_failure_analysis_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = analyze_week6_failure(ROOT, MANIFEST, first)
    repeated = analyze_week6_failure(ROOT, MANIFEST, second)

    assert report == repeated
    assert report["decision"] == EXPECTED_ROUTE
    assert len(report["archive_hashes_declared"]) == 7
    assert report["gate"]["passed"] is False
    assert report["dct_wins"] == 16
    assert report["models"]["fno_ct"]["best_eligible_step"] == 4100
    assert report["models"]["chronopde"]["best_eligible_step"] == 5000
    assert report["models"]["fno_ct"]["historical_gate_median_nrmse"] == pytest.approx(
        0.018945075571537018
    )
    assert report["models"]["chronopde"]["historical_gate_median_nrmse"] == pytest.approx(
        0.014205897226929665
    )
    assert report["models"]["fno_ct"]["conventional_sample_median_nrmse"] == pytest.approx(
        0.019495482556521893
    )
    assert report["models"]["chronopde"]["conventional_sample_median_nrmse"] == pytest.approx(
        0.015137499198317528
    )
    assert report["bootstrap"]["confidence_interval_95"][1] < 0
    assert (first / "failure_analysis.json").read_bytes() == (
        second / "failure_analysis.json"
    ).read_bytes()
    assert (first / "paired_sample_comparison.csv").read_bytes() == (
        second / "paired_sample_comparison.csv"
    ).read_bytes()

    loaded = json.loads((first / "failure_analysis.json").read_text())
    assert loaded["target_audit"]["passed"] is True
    for name in (
        "loss_alignment_convergence.png",
        "paired_model_comparison.png",
        "conditioning_analysis.png",
        "objective_alignment.png",
    ):
        assert (first / name).stat().st_size > 10_000
