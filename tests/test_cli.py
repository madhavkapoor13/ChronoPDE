import json
import subprocess
import sys
from pathlib import Path

import pytest

from chronopde.cli import generate_main
from chronopde.data.pilot import PilotReport

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


def test_train_help_exposes_week4_options() -> None:
    result = run_script("scripts/train.py", "--help")
    assert result.returncode == 0
    assert "--smoke-overfit" in result.stdout
    assert "--resume" in result.stdout
    assert "--data-path" in result.stdout
    assert "--learning-rate" in result.stdout
    assert "--steps-per-interval" in result.stdout


def test_continuous_diagnostic_dry_run() -> None:
    result = run_script("scripts/diagnose_continuous.py", "--device", "cpu", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "diagnose_continuous_gate"
    assert payload["trajectory_ids"] == [
        "train-0000",
        "train-0001",
        "train-0002",
        "train-0003",
    ]
    assert payload["fixed_samples"] is True
    assert payload["perturbation_gamma"] == 0.0
    assert payload["single_batch_steps"] == 2_000
    assert payload["four_trajectory_steps"] == 10_000


def test_chronopde_training_is_operational(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    from chronopde.training.continuous import ContinuousTrainingReport

    report = ContinuousTrainingReport(
        passed=True,
        model="chronopde",
        experiment_id="chronopde-full-train-s0",
        epochs=1,
        optimizer_steps=1,
        initial_loss=1.0,
        final_loss=0.1,
        velocity_nrmse=0.1,
        best_validation_nrmse=0.1,
        persistence_validation_nrmse=1.0,
        parameter_count=1,
        peak_device_memory_bytes=0,
        artifact_directory="artifacts/runs/chronopde-full-train-s0",
        message="ok",
    )
    def fake_train(*_args: object, **kwargs: object) -> ContinuousTrainingReport:
        captured.update(kwargs)
        return report

    monkeypatch.setattr("chronopde.training.train_continuous_time", fake_train)
    from chronopde.cli import train_main

    assert train_main(["--model", "chronopde", "--max-epochs", "1"]) == 0
    assert captured["model_name"] == "chronopde"


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


def test_generation_requires_explicit_mode() -> None:
    result = run_script("scripts/generate_data.py")
    assert result.returncode != 0
    assert "choose one of" in result.stderr


def test_generate_help_exposes_pilot_mode() -> None:
    result = run_script("scripts/generate_data.py", "--help")
    assert result.returncode == 0
    assert "--pilot" in result.stdout
    assert "--manifest-only" in result.stdout
    assert "--full" in result.stdout


def test_failed_pilot_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    report = PilotReport(
        passed=False,
        successful=21,
        failed=3,
        total=24,
        allowed_failures=2,
        config_hash="test",
        output_directory="raw",
        published_directory="published",
    )
    monkeypatch.setattr("chronopde.data.pilot.run_pilot", lambda *_args, **_kwargs: report)
    assert generate_main(["--config", "configs/project.yaml", "--pilot"]) == 2
