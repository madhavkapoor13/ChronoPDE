from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_qualitative_comparison.py"


def _constant(name: str) -> object:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"constant not found: {name}")


def test_qualitative_protocol_is_frozen() -> None:
    assert _constant("REPRESENTATIVE_ID") == "id-0042"
    assert _constant("TIME_INDICES") == (20, 60, 100)
    assert _constant("EXPECTED_DATA_SHA256") == (
        "907aa0d79e604e68ce2d4f5cccfd93ffc64eb68c473caf3edae4be438472caec"
    )


def test_qualitative_script_discloses_exploratory_status() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "exploratory_context_only" in source
    assert "unequal training histories" in source
    assert "selection_rule" in source
