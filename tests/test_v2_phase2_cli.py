from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronopde.v2 import cli
from chronopde.v2.phase2 import _decision, verify_phase2_package

ROOT = Path(__file__).resolve().parents[1]


def test_phase2_decision_routes_are_explicit() -> None:
    passed = {"passed": True}
    failed = {"passed": False}
    assert _decision(failed, passed, passed)[0] == "stop_and_repair_numerical_implementation"
    assert _decision(passed, failed, passed)[0] == "stop_no_fair_comparator"
    assert _decision(passed, passed, failed)[0] == "stop_and_revise_simulator_or_data_protocol"
    assert _decision(passed, passed, passed) == (
        "phase2_complete_phase3_data_generation_allowed",
        True,
    )


def test_cli_dispatches_phase2_without_cuda_or_training(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {
        "decision": "phase2_complete_phase3_data_generation_allowed",
        "numerical_audit": {"passed": True},
        "comparator": {"passed": True},
        "pilot": {"passed": True},
    }
    monkeypatch.setattr("chronopde.v2.phase2.phase2_report", lambda *_: report)
    assert cli.main(["phase2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"].startswith("phase2_complete")
    source = (ROOT / "chronopde/v2/phase2.py").read_text(encoding="utf-8").lower()
    assert "cuda" not in source
    assert "chronopde.training" not in source


def test_phase2_recovery_package_is_verified_and_path_safe(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    from chronopde.v2.phase2 import _package_evidence

    package = tmp_path / "recovery.zip"
    created = _package_evidence(package, [evidence], tmp_path)
    verified = verify_phase2_package(package)
    assert created == verified
    assert verified["partial"] is False
