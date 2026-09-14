import json
from pathlib import Path

from chronopde.diagnostics.continuous_gate import TrainingDiagnosticReport
from chronopde.diagnostics.mechanics import _score_run


def _training_report(directory: Path) -> TrainingDiagnosticReport:
    return TrainingDiagnosticReport(
        passed=False,
        model="chronopde",
        kind="single_batch",
        optimizer_steps=2_000,
        initial_loss=1.0,
        final_loss=0.01,
        loss_reduction=100.0,
        velocity_nrmse=0.02,
        velocity_nrmse_u=0.02,
        velocity_nrmse_v=0.02,
        physical_velocity_mse=0.0,
        spectral_relative_error=0.0,
        rollout_nrmse=float("nan"),
        final_rollout_nrmse=float("nan"),
        boundary_normal_mse=float("nan"),
        stable=True,
        parameter_count=1,
        artifact_directory=str(directory),
    )


def test_mechanics_gate_uses_best_eligible_step_not_last_step(tmp_path: Path) -> None:
    rows = [
        {"optimizer_steps": 0, "loss": 1.0, "velocity_nrmse": 1.0},
        {"optimizer_steps": 100, "loss": 0.0009, "velocity_nrmse": 0.009},
        {"optimizer_steps": 200, "loss": 0.01, "velocity_nrmse": 0.02},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = _score_run("test", "chronopde", _training_report(tmp_path))
    assert result.passed is True
    assert result.best_step == 100
    assert result.best_velocity_nrmse == 0.009
    assert result.loss_reduction_at_best_velocity > 1_000
