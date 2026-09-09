import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_generate_dry_run_from_another_directory(tmp_path: Path) -> None:
    result = run_script(
        str(ROOT / "scripts/generate_data.py"),
        "--config",
        "configs/data.yaml",
        "--dry-run",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "generate_data"
    assert payload["trajectory_count"] == 720


def test_train_dry_run() -> None:
    result = run_script(
        "scripts/train.py",
        "--model",
        "chronopde",
        "--regime",
        "irreg25",
        "--seed",
        "0",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["experiment_id"] == "chronopde-irreg25-train-s0"


def test_evaluate_dry_run() -> None:
    result = run_script(
        "scripts/evaluate.py",
        "--experiment",
        "ood_params",
        "--checkpoint",
        "placeholder.pt",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["experiment_id"] == "chronopde-full-oodparam-s0"


def test_non_dry_run_is_guarded() -> None:
    result = run_script("scripts/generate_data.py")
    assert result.returncode != 0
    assert "scheduled for Week 2" in result.stderr
