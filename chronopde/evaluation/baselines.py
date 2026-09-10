"""ID-rollout evaluation and versionable reports for AR baselines."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronopde.config import ProjectConfig, RegimeName
from chronopde.data.datasets import HDF5RolloutDataset, NormalizationStats, collate_rollouts
from chronopde.evaluation.metrics import relative_l2_per_trajectory, rollout_nrmse
from chronopde.evaluation.rollout import autoregressive_rollout, persistence_rollout
from chronopde.training.trainer import (
    AutoregressiveModelName,
    build_autoregressive_model,
    load_checkpoint,
    resolve_device,
)


@dataclass(frozen=True)
class BaselineEvaluationReport:
    model: str
    trajectories: int
    median_relative_l2: float
    median_final_nrmse: float
    persistence_final_nrmse: float
    divergence_fraction: float
    latency_seconds_per_trajectory: float
    output_directory: str


def _stats_to(stats: NormalizationStats, device: torch.device) -> NormalizationStats:
    return NormalizationStats(
        state_mean=stats.state_mean.to(device),
        state_std=stats.state_std.to(device),
        parameter_mean=stats.parameter_mean.to(device),
        parameter_std=stats.parameter_std.to(device),
    )


def _denormalize(states: Tensor, stats: NormalizationStats) -> Tensor:
    return (
        states * stats.state_std[None, None, :, None, None]
        + stats.state_mean[None, None, :, None, None]
    )


def evaluate_autoregressive_baseline(
    config: ProjectConfig,
    root: Path,
    model_name: AutoregressiveModelName,
    checkpoint: Path,
    *,
    regime: RegimeName = "full",
    data_path: Path | None = None,
    device_name: str = "auto",
) -> BaselineEvaluationReport:
    device = resolve_device(device_name)
    dataset = HDF5RolloutDataset(data_path or root / config.data.output_path, "id", regime)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_rollouts)
    model = build_autoregressive_model(model_name, config).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    stats = _stats_to(dataset.normalization, device)
    rows: list[dict[str, Any]] = []
    time_errors: list[np.ndarray] = []
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    persistence_final: list[float] = []
    elapsed = 0.0
    with torch.no_grad():
        for batch in loader:
            target = cast(Tensor, batch["states"]).to(device)
            times = cast(Tensor, batch["times"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            started = time.perf_counter()
            prediction = autoregressive_rollout(model, target[:, 0], times, parameters)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            prediction_physical = _denormalize(prediction, stats)
            target_physical = _denormalize(target, stats)
            errors = rollout_nrmse(prediction_physical, target_physical)
            relative = relative_l2_per_trajectory(prediction_physical, target_physical)
            persistence = persistence_rollout(target_physical[:, 0], target.shape[1])
            persistence_errors = rollout_nrmse(persistence, target_physical)
            time_errors.extend(errors.cpu().numpy())
            persistence_final.extend(persistence_errors[:, -1].cpu().tolist())
            identifiers = cast(list[str], batch["trajectory_id"])
            for index, trajectory_id in enumerate(identifiers):
                maximum = float(torch.max(torch.abs(prediction_physical[index])).item())
                rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "relative_l2": float(relative[index].item()),
                        "final_nrmse": float(errors[index, -1].item()),
                        "mean_nrmse": float(torch.mean(errors[index]).item()),
                        "max_abs_prediction": maximum,
                        "diverged": not math_isfinite(maximum) or maximum > 10,
                    }
                )
                if len(examples) < 3:
                    examples.append(
                        (
                            trajectory_id,
                            prediction_physical[index].cpu().numpy(),
                            target_physical[index].cpu().numpy(),
                        )
                    )
    output = root / "reports/baselines/week4" / model_name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    error_array = np.stack(time_errors)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(np.arange(error_array.shape[1]), np.median(error_array, axis=0), label=model_name)
    axis.set(xlabel="stored time index", ylabel="median nRMSE", title="ID rollout error")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output / "nrmse_over_time.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(
        len(examples), 3, figsize=(9, 3 * len(examples)), constrained_layout=True
    )
    for row_index, (trajectory_id, prediction_array, target_array) in enumerate(examples):
        axes[row_index, 0].imshow(target_array[-1, 0], cmap="coolwarm")
        axes[row_index, 1].imshow(prediction_array[-1, 0], cmap="coolwarm")
        axes[row_index, 2].imshow(
            np.abs(prediction_array[-1, 0] - target_array[-1, 0]), cmap="magma"
        )
        for column, title in enumerate(("target", "prediction", "absolute error")):
            axes[row_index, column].set_title(f"{trajectory_id}: {title}")
            axes[row_index, column].set_axis_off()
    figure.savefig(output / "field_panels.png", dpi=160)
    plt.close(figure)
    report = BaselineEvaluationReport(
        model=model_name,
        trajectories=len(rows),
        median_relative_l2=float(np.median([row["relative_l2"] for row in rows])),
        median_final_nrmse=float(np.median([row["final_nrmse"] for row in rows])),
        persistence_final_nrmse=float(np.median(persistence_final)),
        divergence_fraction=float(np.mean([row["diverged"] for row in rows])),
        latency_seconds_per_trajectory=elapsed / len(rows),
        output_directory=str(output),
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset.close()
    return report


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))
