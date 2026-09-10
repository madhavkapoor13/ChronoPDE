"""Frozen ID evaluation for the continuous-time FFT baseline."""

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
from chronopde.evaluation.rollout import continuous_rollout, persistence_rollout
from chronopde.models import trainable_parameter_count
from chronopde.training.continuous import build_continuous_model
from chronopde.training.trainer import load_checkpoint, resolve_device


@dataclass(frozen=True)
class ContinuousEvaluationReport:
    model: str
    trajectories: int
    median_relative_l2: float
    median_final_nrmse: float
    persistence_final_nrmse: float
    divergence_fraction: float
    latency_seconds_per_trajectory: float
    nfev_per_trajectory: int
    parameter_count: int
    peak_device_memory_bytes: int
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


def _write_time_errors(path: Path, errors: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["trajectory_index", *[f"time_{index}" for index in range(errors.shape[1])]]
        )
        writer.writerows([[index, *row] for index, row in enumerate(errors)])


def evaluate_continuous_baseline(
    config: ProjectConfig,
    root: Path,
    checkpoint: Path,
    *,
    regime: RegimeName = "full",
    data_path: Path | None = None,
    device_name: str = "auto",
    steps_per_interval: int | None = None,
) -> ContinuousEvaluationReport:
    if config.evaluation is None:
        raise ValueError("evaluation configuration is required")
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = HDF5RolloutDataset(data_path or root / config.data.output_path, "id", regime)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_rollouts)
    model = build_continuous_model(config).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    stats = _stats_to(dataset.normalization, device)
    integration_steps = (
        steps_per_interval
        if steps_per_interval is not None
        else config.evaluation.steps_per_interval
    )
    if integration_steps < 1:
        raise ValueError("steps_per_interval must be positive")
    rows: list[dict[str, Any]] = []
    time_errors: list[np.ndarray] = []
    persistence_final: list[float] = []
    elapsed = 0.0
    nfev = 0
    with torch.no_grad():
        for batch in loader:
            target = cast(Tensor, batch["states"]).to(device)
            times = cast(Tensor, batch["times"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            integration = continuous_rollout(
                model,
                target[:, 0],
                times,
                parameters,
                steps_per_interval=integration_steps,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            nfev = integration.nfev
            prediction = cast(Tensor, integration.states)
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
                        "diverged": not np.isfinite(maximum) or maximum > 10,
                    }
                )
    output = root / "reports/baselines/week5/fno_ct"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    error_array = np.stack(time_errors)
    _write_time_errors(output / "nrmse_by_time.csv", error_array)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(np.median(error_array, axis=0), label="fno_ct")
    persistence_curve: list[np.ndarray] = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        physical = _denormalize(sample.states[None].to(device), stats)
        persistent = persistence_rollout(physical[:, 0], physical.shape[1])
        persistence_curve.append(rollout_nrmse(persistent, physical)[0].cpu().numpy())
    axis.plot(np.median(np.stack(persistence_curve), axis=0), label="persistence")
    for baseline in ("unet_ar", "fno_ar"):
        source = root / f"reports/baselines/week4/{baseline}/nrmse_by_time.csv"
        if source.is_file():
            values = np.loadtxt(source, delimiter=",", skiprows=1)[:, 1:]
            axis.plot(np.median(values, axis=0), label=baseline)
    axis.set(xlabel="stored time index", ylabel="median nRMSE", title="Frozen ID rollout")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output / "model_comparison.png", dpi=160)
    plt.close(figure)
    order = np.argsort([row["mean_nrmse"] for row in rows])
    selected = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    figure, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for plot_row, dataset_index in enumerate(selected):
        batch = collate_rollouts([dataset[dataset_index]])
        target = cast(Tensor, batch["states"]).to(device)
        times = cast(Tensor, batch["times"]).to(device)
        parameters = cast(Tensor, batch["parameters"]).to(device)
        with torch.no_grad():
            prediction = cast(
                Tensor,
                continuous_rollout(
                    model,
                    target[:, 0],
                    times,
                    parameters,
                    steps_per_interval=integration_steps,
                ).states,
            )
        physical_prediction = _denormalize(prediction, stats)[0, -1, 0].cpu().numpy()
        physical_target = _denormalize(target, stats)[0, -1, 0].cpu().numpy()
        panels = (
            physical_target,
            physical_prediction,
            np.abs(physical_prediction - physical_target),
        )
        for column, (panel, title) in enumerate(
            zip(panels, ("target", "prediction", "absolute error"), strict=True)
        ):
            axes[plot_row, column].imshow(panel, cmap="magma" if column == 2 else "coolwarm")
            axes[plot_row, column].set_title(f"{rows[dataset_index]['trajectory_id']}: {title}")
            axes[plot_row, column].set_axis_off()
    figure.savefig(output / "field_panels.png", dpi=160)
    plt.close(figure)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    report = ContinuousEvaluationReport(
        model="fno_ct",
        trajectories=len(rows),
        median_relative_l2=float(np.median([row["relative_l2"] for row in rows])),
        median_final_nrmse=float(np.median([row["final_nrmse"] for row in rows])),
        persistence_final_nrmse=float(np.median(persistence_final)),
        divergence_fraction=float(np.mean([row["diverged"] for row in rows])),
        latency_seconds_per_trajectory=elapsed / len(rows),
        nfev_per_trajectory=nfev,
        parameter_count=trainable_parameter_count(model),
        peak_device_memory_bytes=peak_memory,
        output_directory=str(output),
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset.close()
    return report
