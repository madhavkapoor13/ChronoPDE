"""Balanced fixed-batch follow-up for the Week 6 continuous-time gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from chronopde.config import ProjectConfig
from chronopde.data.datasets import HDF5VelocityDataset
from chronopde.diagnostics.continuous_gate import SMOKE_IDS, _write_json, train_fixed_diagnostic
from chronopde.diagnostics.mechanics import MechanicsRunReport, _score_run
from chronopde.reproducibility import environment_metadata
from chronopde.training.trainer import resolve_device

BALANCED_INTERVALS = (0, 33, 66, 99)


@dataclass(frozen=True)
class BalancedBatchReport:
    passed: bool
    route: str
    sample_indices: list[int]
    trajectory_ids: list[str]
    interval_indices: list[int]
    runs: dict[str, MechanicsRunReport]
    output_directory: str


def resolve_balanced_sample_indices(
    dataset: HDF5VelocityDataset,
    trajectory_ids: tuple[str, ...] = SMOKE_IDS,
    intervals: tuple[int, ...] = BALANCED_INTERVALS,
) -> tuple[int, ...]:
    """Resolve a trajectory-major dataset into a declared balanced batch."""
    requested = [
        (trajectory_id, interval)
        for trajectory_id in trajectory_ids
        for interval in intervals
    ]
    available = {dataset.sample_identity(index): index for index in range(len(dataset))}
    missing = [identity for identity in requested if identity not in available]
    if missing:
        raise ValueError(f"balanced sample identities are unavailable: {missing}")
    return tuple(available[identity] for identity in requested)


def run_balanced_batch_suite(
    config: ProjectConfig,
    root: Path,
    data_path: Path,
    *,
    device_name: str = "auto",
    max_steps: int = 5_000,
    evaluation_interval: int = 100,
) -> BalancedBatchReport:
    """Run the same representative 16-sample memorization protocol for both models."""
    if config.training is None:
        raise ValueError("training configuration is required")
    if max_steps < 1 or evaluation_interval < 1:
        raise ValueError("balanced diagnostic step counts must be positive")
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
    output = root / "artifacts/diagnostics/week6/balanced_batch"
    output.mkdir(parents=True, exist_ok=True)
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
            "perturbation_gamma": 0.0,
            "sample_indices": list(sample_indices),
            "sample_identities": [
                {"trajectory_id": trajectory_id, "interval_index": interval}
                for trajectory_id, interval in identities
            ],
            "spectral_weight": config.training.spectral_loss_weight,
            "trajectory_ids": list(SMOKE_IDS),
            "weight_decay": 0.0,
        },
    )
    device = resolve_device(device_name)
    runs: dict[str, MechanicsRunReport] = {}
    for model_name in ("fno_ct", "chronopde"):
        print(json.dumps({"balanced_batch_model": model_name, "status": "started"}))
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
            spectral_weight=config.training.spectral_loss_weight,
            artifact_label=f"balanced_batch/lr3e-4_spectral-{model_name}-s0",
            sample_indices=sample_indices,
        )
        runs[model_name] = _score_run("balanced_lr3e-4_spectral", model_name, training)
        print(json.dumps(asdict(runs[model_name]), sort_keys=True))
    fft_passed = runs["fno_ct"].passed
    dct_passed = runs["chronopde"].passed
    if fft_passed and dct_passed:
        route = "proceed_fixed_four_trajectory_comparison"
    elif fft_passed:
        route = "test_predeclared_dct_variants"
    else:
        route = "stop_and_redesign_continuous_time_gate"
    report = BalancedBatchReport(
        passed=fft_passed and dct_passed,
        route=route,
        sample_indices=list(sample_indices),
        trajectory_ids=list(SMOKE_IDS),
        interval_indices=list(BALANCED_INTERVALS),
        runs=runs,
        output_directory=str(output),
    )
    _write_json(output / "suite_summary.json", asdict(report))
    return report
