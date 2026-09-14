import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/week6_gate_diagnostics_kaggle.ipynb"
MECHANICS_NOTEBOOK = ROOT / "notebooks/week6_mechanics_diagnostics_kaggle.ipynb"
AUDIT_NOTEBOOK = ROOT / "notebooks/week6_velocity_audit_kaggle.ipynb"
BALANCED_NOTEBOOK = ROOT / "notebooks/week6_balanced_batch_kaggle.ipynb"


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


def test_velocity_audit_notebook_is_cpu_only_and_packages_failures() -> None:
    payload = json.loads(AUDIT_NOTEBOOK.read_text(encoding="utf-8"))
    code = []
    for cell in payload["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    combined = "\n".join(code)
    assert "c7bda5977afe3ad24b204ddc94dda34d3abe87f1" in combined
    assert "scripts/audit_velocity.py" in combined
    assert "subprocess.run(command, cwd=REPOSITORY, check=False)" in combined
    assert "chronopde_week6_velocity_audit" in combined
    assert "'cpu'" in combined
    assert "'cuda'" not in combined
    assert "make_archive" in combined


def test_balanced_batch_notebook_is_pinned_bounded_and_packages_failures() -> None:
    payload = json.loads(BALANCED_NOTEBOOK.read_text(encoding="utf-8"))
    code = []
    for cell in payload["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)
    combined = "\n".join(code)
    assert "30381c36472d2fc98afdde977084745f5edd3504" in combined
    assert "scripts/diagnose_balanced_batch.py" in combined
    assert "subprocess.run(command, cwd=REPOSITORY, check=False)" in combined
    assert "'5000'" in combined
    assert "'100'" in combined
    assert "CUDA_VISIBLE_DEVICES" in combined
    assert "chronopde_week6_balanced_batch" in combined
    assert "make_archive" in combined
    assert "ood" not in combined.lower()
