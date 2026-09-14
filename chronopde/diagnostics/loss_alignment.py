"""Loss-to-gate alignment diagnostic for the Week 6 continuous-time models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from chronopde.config import ProjectConfig
from chronopde.data.datasets import HDF5VelocityDataset
from chronopde.diagnostics.balanced_batch import (
    BALANCED_INTERVALS,
    resolve_balanced_sample_indices,
)
from chronopde.diagnostics.continuous_gate import SMOKE_IDS, _write_json, train_fixed_diagnostic
from chronopde.diagnostics.mechanics import MechanicsRunReport, _score_run
from chronopde.reproducibility import environment_metadata
from chronopde.training.trainer import resolve_device

REQUIRED_ALIGNMENT_STEPS = 5_000


@dataclass(frozen=True)
class LossAlignmentReport:
    passed: bool
    budget_complete: bool
    route: str
    sample_indices: list[int]
    runs: dict[str, MechanicsRunReport]
    output_directory: str


def loss_alignment_route(
    *, fft_passed: bool, dct_passed: bool, budget_complete: bool
) -> str:
    if not budget_complete:
        return "incomplete_budget_no_decision"
    if fft_passed and dct_passed:
        return "proceed_aligned_fixed_four_trajectory_comparison"
    if fft_passed:
        return "test_predeclared_dct_variants_with_aligned_loss"
    if dct_passed:
        return "stop_and_investigate_control_anomaly"
    return "stop_and_document_model_or_conditioning_limitation"


def run_loss_alignment_suite(
    config: ProjectConfig,
    root: Path,
    data_path: Path,
    *,
    device_name: str = "auto",
    max_steps: int = REQUIRED_ALIGNMENT_STEPS,
    evaluation_interval: int = 100,
) -> LossAlignmentReport:
    """Run one fixed full-field-relative objective for both balanced controls."""
    if config.training is None:
        raise ValueError("training configuration is required")
    if max_steps < 1 or max_steps > REQUIRED_ALIGNMENT_STEPS or evaluation_interval < 1:
        raise ValueError("loss-alignment steps must be in [1, 5000] and interval positive")
    dataset = HDF5VelocityDataset(
        data_path,
        "train",
        "full",
        seed=0,
        gamma=0.0,
        trajectory_ids=SMOKE_IDS,
        resample_each_epoch=False,
    )
    try:
        sample_indices = resolve_balanced_sample_indices(dataset)
        identities = [dataset.sample_identity(index) for index in sample_indices]
    finally:
        dataset.close()
    output = root / "artifacts/diagnostics/week6/loss_alignment"
    if output.exists():
        raise FileExistsError(f"loss-alignment output already exists: {output}")
    output.mkdir(parents=True)
    _write_json(output / "environment.json", environment_metadata(root, 0))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    _write_json(
        output / "protocol.json",
        {
            "batch_size": 16,
            "evaluation_interval": evaluation_interval,
            "fixed_samples": True,
            "interval_indices_per_trajectory": list(BALANCED_INTERVALS),
            "learning_rate": 3e-4,
            "max_steps": max_steps,
            "median_velocity_nrmse_threshold": 0.01,
            "minimum_loss_reduction": 1_000,
            "objective": "full_field_relative",
            "perturbation_gamma": 0.0,
            "relative_loss_epsilon": 1e-8,
            "sample_indices": list(sample_indices),
            "sample_identities": [
                {"trajectory_id": trajectory_id, "interval_index": interval}
                for trajectory_id, interval in identities
            ],
            "trajectory_ids": list(SMOKE_IDS),
            "weight_decay": 0.0,
        },
    )
    device = resolve_device(device_name)
    runs: dict[str, MechanicsRunReport] = {}
    for model_name in ("fno_ct", "chronopde"):
        print(json.dumps({"loss_alignment_model": model_name, "status": "started"}))
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
            learning_rate=3e-4,
            artifact_label=f"loss_alignment/full-relative-{model_name}-s0",
            sample_indices=sample_indices,
            objective="full_field_relative",
        )
        runs[model_name] = _score_run("full_field_relative", model_name, training)
        print(json.dumps(asdict(runs[model_name]), sort_keys=True))
    budget_complete = max_steps == REQUIRED_ALIGNMENT_STEPS
    fft_passed = budget_complete and runs["fno_ct"].passed
    dct_passed = budget_complete and runs["chronopde"].passed
    route = loss_alignment_route(
        fft_passed=fft_passed,
        dct_passed=dct_passed,
        budget_complete=budget_complete,
    )
    report = LossAlignmentReport(
        passed=fft_passed and dct_passed,
        budget_complete=budget_complete,
        route=route,
        sample_indices=list(sample_indices),
        runs=runs,
        output_directory=str(output),
    )
    _write_json(output / "suite_summary.json", asdict(report))
    return report
