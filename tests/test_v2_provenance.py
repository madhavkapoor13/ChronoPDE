from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronopde.v2.phase1 import main as phase1_main
from chronopde.v2.phase1 import phase1_report
from chronopde.v2.provenance import discover_archives, validate_claim_ledger

ROOT = Path(__file__).resolve().parents[1]


def _declaration(digest: str, filename: str = "original.zip") -> dict[str, str]:
    return {
        "id": "example",
        "filename": filename,
        "git_commit": "1" * 40,
        "sha256": digest,
    }


def test_phase1_reproduces_frozen_headline() -> None:
    report = phase1_report(ROOT)
    frozen = report["frozen_diagnostic"]
    assert frozen["sample_count"] == 16
    assert frozen["dct_wins"] == 16
    assert frozen["models"]["fno_ct"]["best_eligible_step"] == 4100
    assert frozen["models"]["chronopde"]["best_eligible_step"] == 5000
    assert frozen["models"]["fno_ct"]["historical_gate_median_nrmse"] == pytest.approx(
        0.018945075571537018
    )
    assert frozen["models"]["chronopde"]["historical_gate_median_nrmse"] == pytest.approx(
        0.014205897226929665
    )
    assert frozen["models"]["fno_ct"]["passed"] is False
    assert frozen["models"]["chronopde"]["passed"] is False


def test_archive_discovery_uses_hash_and_reports_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "renamed.zip"
    second = tmp_path / "nested" / "copy.zip"
    first.write_bytes(b"evidence")
    second.parent.mkdir()
    second.write_bytes(b"evidence")
    import hashlib

    digest = hashlib.sha256(b"evidence").hexdigest()
    record = discover_archives([_declaration(digest)], tmp_path)[0]
    assert record["status"] == "verified_locally"
    assert record["duplicate_matches"] == 1
    assert len(record["matching_paths"]) == 2


def test_archive_discovery_distinguishes_missing_and_mismatch(tmp_path: Path) -> None:
    expected = "0" * 64
    (tmp_path / "original.zip").write_bytes(b"wrong")
    assert discover_archives([_declaration(expected)], tmp_path)[0]["status"] == "hash_mismatch"
    assert (
        discover_archives([_declaration(expected, "absent.zip")], tmp_path)[0]["status"]
        == "missing"
    )


def test_archive_declaration_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe archive"):
        discover_archives([_declaration("0" * 64, "../escape.zip")], tmp_path)


def test_claim_ledger_is_complete_and_explicit() -> None:
    path = ROOT / "reports/chronopde_v2/phase1/claim_ledger.yaml"
    claims = validate_claim_ledger(ROOT, path)
    by_id = {claim["id"]: claim for claim in claims}
    assert by_id["week5_four_trajectory_gate"]["status"] == "unsupported"
    assert by_id["general_dct_superiority"]["status"] == "unsupported"
    assert by_id["sparse_time_performance"]["status"] == "not_run"
    assert by_id["ood_performance"]["status"] == "not_run"


def test_committed_inventory_contains_no_large_artifacts() -> None:
    inventory = json.loads(
        (ROOT / "reports/chronopde_v2/phase1/evidence_inventory.json").read_text()
    )
    assert len(inventory["archives"]) == 7
    assert all(item["status"] == "declared_only" for item in inventory["archives"])
    forbidden = {".h5", ".hdf5", ".pt", ".pth", ".zip"}
    assert not [
        item for item in inventory["lightweight_evidence"] if Path(item["path"]).suffix in forbidden
    ]


def test_phase1_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert phase1_main(["phase1", "--format", "json"]) == 0
    capsys.readouterr()
    assert phase1_main(["phase1", "--archive-root", str(tmp_path), "--format", "json"]) == 2
    capsys.readouterr()
    (tmp_path / "results.zip").write_bytes(b"not the declared archive")
    assert phase1_main(["phase1", "--archive-root", str(tmp_path), "--format", "json"]) == 3
