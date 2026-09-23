from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronopde.v2 import cli
from chronopde.v2.phase2 import _package_evidence, verify_evidence_package


def test_cli_dispatches_phase3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {
        "decision": "phase3_complete_phase4_feasibility_training_allowed",
        "dataset": {
            "successful_trajectories": 640,
            "logical_content_sha256": "a" * 64,
        },
    }
    monkeypatch.setattr("chronopde.v2.phase3.phase3_report", lambda *_: report)
    assert cli.main(["phase3", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "phase3_complete_phase4_feasibility_training_allowed"


def test_phase3_module_has_no_training_or_cuda_dependency() -> None:
    import chronopde.v2.phase3 as phase3

    source = Path(phase3.__file__).read_text(encoding="utf-8").lower()
    assert "chronopde.training" not in source
    assert "torch.cuda" not in source


def test_phase3_recovery_package_has_phase3_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "summary.json"
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    package = tmp_path / "phase3.zip"
    _package_evidence(package, [evidence], tmp_path, phase=3)
    manifest = verify_evidence_package(package, expected_phase=3)
    assert manifest["phase"] == 3
    with pytest.raises(ValueError, match="Phase 2 package identity"):
        verify_evidence_package(package, expected_phase=2)
