import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/week6_gate_diagnostics_kaggle.ipynb"
MECHANICS_NOTEBOOK = ROOT / "notebooks/week6_mechanics_diagnostics_kaggle.ipynb"


def test_diagnostic_notebook_is_clean_syntactic_and_packages_failures() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = []
    for cell in payload["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    combined = "\n".join(code)
    assert "80bafb5f09a2d924535f1530ed104bda8d62a714" in combined
    assert "subprocess.run(command, check=False)" in combined
    assert "chronopde_week6_gate_diagnostics" in combined
    assert "ood" not in combined.lower()


def test_mechanics_notebook_is_clean_syntactic_and_packages_failures() -> None:
    payload = json.loads(MECHANICS_NOTEBOOK.read_text(encoding="utf-8"))
    code = []
    for cell in payload["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    combined = "\n".join(code)
    assert "68a511867ed4f02bbbefe7743ff42799d0fd8a66" in combined
    assert "scripts/diagnose_mechanics.py" in combined
    assert "subprocess.run(command, check=False)" in combined
    assert "chronopde_week6_mechanics_diagnostics" in combined
    assert "ignore_patterns('*.pt')" in combined
