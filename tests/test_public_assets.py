from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_svg_is_valid_and_complete() -> None:
    path = ROOT / "figures/architecture_overview.svg"
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    text = " ".join(element.text or "" for element in root.iter())
    for label in ("FiLM", "DCT spectral", "RK4", "CT-FFT control", "Neumann"):
        assert label in text


def test_public_repository_has_no_large_training_artifacts() -> None:
    forbidden = {".h5", ".hdf5", ".pt", ".pth", ".zip"}
    result = run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_candidates = [ROOT / line for line in result.stdout.splitlines() if line]
    assert not [path for path in tracked_candidates if path.suffix.lower() in forbidden]


def test_public_documentation_links_exist() -> None:
    expected = (
        ROOT / "docs/PORTFOLIO.md",
        ROOT / "docs/REPRODUCTION.md",
        ROOT / "figures/architecture_overview.svg",
        ROOT / "figures/qualitative_rollout.png",
        ROOT / "figures/qualitative_rollout.json",
        ROOT / "notebooks/week6_loss_alignment_kaggle.ipynb",
        ROOT / "output/pdf/chronopde_negative_result_report.pdf",
    )
    assert all(path.is_file() for path in expected)
