"""Low-cost optimizer and loss diagnostics after the fixed-batch gate fails."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from chronopde.config import ProjectConfig
from chronopde.diagnostics.continuous_gate import (
    TrainingDiagnosticReport,
    _write_json,
    train_fixed_diagnostic,
)
from chronopde.reproducibility import environment_metadata
from chronopde.training.trainer import resolve_device


@dataclass(frozen=True)
class MechanicsRunReport:
    passed: bool
    protocol: str
    model: str
    best_step: int
    best_velocity_nrmse: float
    loss_reduction_at_best_velocity: float
    training: TrainingDiagnosticReport


@dataclass(frozen=True)
class MechanicsSuiteReport:
    passed: bool
    route: str
    selected_protocol: str | None
    runs: dict[str, dict[str, MechanicsRunReport]]
    output_directory: str


def _score_run(
    protocol: str,
    model_name: str,
    report: TrainingDiagnosticReport,
) -> MechanicsRunReport:
    metrics_path = Path(report.artifact_directory) / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    initial_loss = float(rows[0]["loss"])
    eligible = [
        row
        for row in rows[1:]
        if initial_loss / max(float(row["loss"]), 1e-30) >= 1_000
    ]
    candidates = eligible or rows[1:]
    best = min(candidates, key=lambda row: float(row["velocity_nrmse"]))
    reduction = initial_loss / max(float(best["loss"]), 1e-30)
    velocity_nrmse = float(best["velocity_nrmse"])
    return MechanicsRunReport(
        passed=reduction >= 1_000 and velocity_nrmse <= 0.01,
        protocol=protocol,
        model=model_name,
        best_step=int(best["optimizer_steps"]),
        best_velocity_nrmse=velocity_nrmse,
        loss_reduction_at_best_velocity=reduction,
        training=report,
    )


def run_single_batch_mechanics_suite(
    config: ProjectConfig,
    root: Path,
    data_path: Path,
    *,
    device_name: str = "auto",
    max_steps: int = 5_000,
    evaluation_interval: int = 100,
) -> MechanicsSuiteReport:
    """Find one shared fixed-batch protocol that both continuous models pass."""

    if config.training is None:
        raise ValueError("training configuration is required")
    if max_steps < 1 or evaluation_interval < 1:
        raise ValueError("mechanics step counts and intervals must be positive")
    device = resolve_device(device_name)
    output = root / "artifacts/diagnostics/week6/mechanics"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "environment.json", environment_metadata(root, 0))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    protocols = (
        ("lr3e-4_spectral", 3e-4, config.training.spectral_loss_weight),
        ("lr1e-4_spectral", 1e-4, config.training.spectral_loss_weight),
        ("lr3e-4_physical_only", 3e-4, 0.0),
    )
    runs: dict[str, dict[str, MechanicsRunReport]] = {}
    selected: str | None = None
    for protocol, learning_rate, spectral_weight in protocols:
        print(json.dumps({"mechanics_protocol": protocol, "status": "started"}))
        protocol_runs: dict[str, MechanicsRunReport] = {}
        for model_name in ("fno_ct", "chronopde"):
            training = train_fixed_diagnostic(
                config,
                root,
                data_path,
                device,
                model_name,
                "single_batch",
                max_steps,
                evaluation_interval,
                evaluation_interval,
                learning_rate=learning_rate,
                spectral_weight=spectral_weight,
                artifact_label=f"mechanics/{protocol}-{model_name}-s0",
            )
            scored = _score_run(protocol, model_name, training)
            protocol_runs[model_name] = scored
            print(json.dumps(asdict(scored), sort_keys=True))
        runs[protocol] = protocol_runs
        if all(result.passed for result in protocol_runs.values()):
            selected = protocol
            break
    passed = selected is not None
    route = (
        f"proceed_fixed_four_with_{selected}"
        if selected is not None
        else "repair_metric_denominator_or_velocity_target"
    )
    report = MechanicsSuiteReport(
        passed=passed,
        route=route,
        selected_protocol=selected,
        runs=runs,
        output_directory=str(output),
    )
    _write_json(output / "suite_summary.json", asdict(report))
    return report
