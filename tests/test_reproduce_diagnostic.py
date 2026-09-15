from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_diagnostic.py"


def _module() -> ModuleType:
    spec = spec_from_file_location("reproduce_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproducer_emits_exact_headline_result(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    assert module.main(["--model", "both", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"]["fno_ct"]["best_step"] == 4100
    assert payload["models"]["chronopde_dct"]["best_step"] == 5000
    assert payload["models"]["fno_ct"]["gate_nrmse"] == pytest.approx(0.0189450756)
    assert payload["models"]["chronopde_dct"]["gate_nrmse"] == pytest.approx(0.0142058972)
    assert payload["models"]["fno_ct"]["result"] == "FAIL"
    assert payload["models"]["chronopde_dct"]["result"] == "FAIL"
    assert payload["dct_wins"] == 16


def test_reproducer_text_is_concise(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    assert module.main(["--model", "chronopde_dct"]) == 0
    output = capsys.readouterr().out
    assert "Best step: 5000" in output
    assert "Median velocity nRMSE: 0.01421" in output
    assert "Gate: 0.01000" in output
    assert "Result: FAIL" in output


def test_reproducer_rejects_unavailable_seed() -> None:
    module = _module()
    with pytest.raises(SystemExit, match="only the frozen seed-0"):
        module.main(["--seed", "1"])
