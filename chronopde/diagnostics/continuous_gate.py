"""Controlled diagnostics for the Week 6 continuous-time overfit gate."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronopde.config import ProjectConfig
from chronopde.contracts import PhysicalParameters
from chronopde.data.datasets import (
    HDF5RolloutDataset,
    HDF5VelocityDataset,
    collate_rollouts,
    collate_velocity_samples,
)
from chronopde.data.simulator import reaction_diffusion_rhs
from chronopde.evaluation.metrics import nrmse, rollout_nrmse
from chronopde.evaluation.rollout import continuous_rollout
from chronopde.models import (
    DCTContinuousVectorField,
    FFTContinuousVectorField,
    trainable_parameter_count,
)
from chronopde.numerics import build_grid, build_neumann_laplacian, dct2
from chronopde.reproducibility import environment_metadata, seed_everything
from chronopde.training.losses import velocity_loss
from chronopde.training.trainer import resolve_device, save_checkpoint

DiagnosticModelName = Literal[
    "chronopde",
    "fno_ct",
    "chronopde_residual",
    "chronopde_modes16",
]
SMOKE_IDS = tuple(f"train-{index:04d}" for index in range(4))


@dataclass(frozen=True)
class TargetAuditReport:
    passed: bool
    samples: int
    median_spline_pde_nrmse: float
    target_rms_u: float
    target_rms_v: float
    retained_energy: dict[str, float]
    time_min: float
    time_max: float
    maximum_time_roundtrip_error: float
    output_directory: str


@dataclass(frozen=True)
class TrainingDiagnosticReport:
    passed: bool
    model: str
    kind: str
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    loss_reduction: float
    velocity_nrmse: float
    velocity_nrmse_u: float
    velocity_nrmse_v: float
    physical_velocity_mse: float
    spectral_relative_error: float
    rollout_nrmse: float
    final_rollout_nrmse: float
    boundary_normal_mse: float
    stable: bool
    parameter_count: int
    artifact_directory: str


@dataclass(frozen=True)
class ContinuousGateSuiteReport:
    passed: bool
    route: str
    target_audit: TargetAuditReport
    single_batch: dict[str, TrainingDiagnosticReport]
    four_trajectory: dict[str, TrainingDiagnosticReport]
    variants: dict[str, TrainingDiagnosticReport]
    output_directory: str


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_for_diagnostic(config: ProjectConfig, name: DiagnosticModelName) -> nn.Module:
    if config.model is None:
        raise ValueError("model configuration is required")
    blocks = config.model.blocks
    film_hidden_width = config.model.film_hidden_width
    time_range = (config.pde.t_start, config.pde.t_end)
    if name == "fno_ct":
        return FFTContinuousVectorField(
            width=config.model.fno_width,
            modes_y=12,
            modes_x=12,
            blocks=blocks,
            film_hidden_width=film_hidden_width,
            time_range=time_range,
        )
    if name == "chronopde":
        return DCTContinuousVectorField(
            width=config.model.width,
            modes_y=12,
            modes_x=12,
            blocks=blocks,
            film_hidden_width=film_hidden_width,
            time_range=time_range,
        )
    if name == "chronopde_residual":
        return DCTContinuousVectorField(
            width=config.model.width,
            modes_y=12,
            modes_x=12,
            residual_skip=True,
            blocks=blocks,
            film_hidden_width=film_hidden_width,
            time_range=time_range,
        )
    if name == "chronopde_modes16":
        return DCTContinuousVectorField(
            width=43,
            modes_y=16,
            modes_x=16,
            blocks=blocks,
            film_hidden_width=film_hidden_width,
            time_range=time_range,
        )
    raise ValueError(f"unsupported diagnostic model: {name}")


def _velocity_metrics(
    model: nn.Module,
    batches: list[dict[str, Any]],
    device: torch.device,
    spectral_weight: float,
    modes_y: int,
    modes_x: int,
    state_std: Tensor,
) -> dict[str, float]:
    totals: list[float] = []
    physical: list[float] = []
    spectral: list[float] = []
    errors: list[Tensor] = []
    channel_errors: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            state = cast(Tensor, batch["state"]).to(device)
            target = cast(Tensor, batch["target_velocity"]).to(device)
            time = cast(Tensor, batch["time"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            prediction = model(state, time, parameters)
            breakdown = velocity_loss(
                prediction,
                target,
                spectral_weight=spectral_weight,
                modes_y=modes_y,
                modes_x=modes_x,
            )
            totals.append(float(breakdown.total.item()))
            scale = state_std.to(device)[None, :, None, None]
            physical.append(float(torch.mean(((prediction - target) * scale).square()).item()))
            spectral.append(float(breakdown.spectral_relative_error.item()))
            errors.append(nrmse(prediction, target, (1, 2, 3)))
            channel_errors.append(nrmse(prediction, target, (2, 3)))
    channels = torch.cat(channel_errors)
    return {
        "loss": float(np.mean(totals)),
        "physical_velocity_mse": float(np.mean(physical)),
        "spectral_relative_error": float(np.mean(spectral)),
        "velocity_nrmse": float(torch.median(torch.cat(errors)).item()),
        "velocity_nrmse_u": float(torch.median(channels[:, 0]).item()),
        "velocity_nrmse_v": float(torch.median(channels[:, 1]).item()),
    }


def _boundary_normal_mse(states: Tensor, config: ProjectConfig) -> float:
    dx = (config.pde.x_max - config.pde.x_min) / config.pde.width
    dy = (config.pde.y_max - config.pde.y_min) / config.pde.height
    horizontal = torch.cat(
        (
            (states[..., :, 1] - states[..., :, 0]) / dx,
            (states[..., :, -1] - states[..., :, -2]) / dx,
        ),
        dim=-1,
    )
    vertical = torch.cat(
        (
            (states[..., 1, :] - states[..., 0, :]) / dy,
            (states[..., -1, :] - states[..., -2, :]) / dy,
        ),
        dim=-1,
    )
    return float(torch.mean(torch.cat((horizontal.flatten(), vertical.flatten())).square()).item())


def _rollout_metrics(
    model: nn.Module,
    dataset: HDF5RolloutDataset,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    if config.evaluation is None:
        raise ValueError("evaluation configuration is required")
    batch = collate_rollouts([dataset[index] for index in range(len(dataset))])
    target = cast(Tensor, batch["states"]).to(device)
    times = cast(Tensor, batch["times"]).to(device)
    parameters = cast(Tensor, batch["parameters"]).to(device)
    stats = dataset.normalization
    state_mean = stats.state_mean.to(device)
    state_std = stats.state_std.to(device)
    model.eval()
    with torch.no_grad():
        prediction = cast(
            Tensor,
            continuous_rollout(
                model,
                target[:, 0],
                times,
                parameters,
                steps_per_interval=config.evaluation.steps_per_interval,
            ).states,
        )
        physical_prediction = (
            prediction * state_std[None, None, :, None, None]
            + state_mean[None, None, :, None, None]
        )
        physical_target = (
            target * state_std[None, None, :, None, None]
            + state_mean[None, None, :, None, None]
        )
        errors = rollout_nrmse(physical_prediction, physical_target)
        finite = bool(torch.isfinite(physical_prediction).all())
        bounded = bool(torch.max(torch.abs(physical_prediction)) <= 10)
    return {
        "rollout_nrmse": float(torch.median(errors).item()),
        "final_rollout_nrmse": float(torch.median(errors[:, -1]).item()),
        "boundary_normal_mse": _boundary_normal_mse(physical_prediction, config),
        "stable": finite and bounded,
    }


def _batches(loader: DataLoader[Any]) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], batch) for batch in loader]


def audit_targets(config: ProjectConfig, data_path: Path, output: Path) -> TargetAuditReport:
    dataset = HDF5VelocityDataset(
        data_path,
        "train",
        "full",
        seed=0,
        gamma=0.0,
        trajectory_ids=SMOKE_IDS,
        resample_each_epoch=False,
    )
    grid = build_grid(config.pde)
    laplacian = build_neumann_laplacian(grid)
    stats = dataset.normalization
    rows: list[dict[str, float | int | str]] = []
    target_rms: list[Tensor] = []
    spline_pde_errors: list[float] = []
    retained: dict[int, list[float]] = {mode: [] for mode in (8, 12, 16, 20, 32)}
    times: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample.target_velocity
        physical_target = target * stats.state_std[:, None, None]
        target_rms.append(torch.mean(physical_target.square(), dim=(1, 2)).sqrt())
        coefficients = dct2(physical_target)
        total_energy = float(torch.sum(coefficients.square()).item())
        mode_values: dict[int, float] = {}
        for mode in retained:
            energy = float(torch.sum(coefficients[..., :mode, :mode].square()).item())
            ratio = energy / max(total_energy, 1e-30)
            retained[mode].append(ratio)
            mode_values[mode] = ratio
        physical_state = stats.denormalize_state(sample.state)
        physical_parameters = (
            sample.parameters * stats.parameter_std + stats.parameter_mean
        )
        params = PhysicalParameters(
            du=float(physical_parameters[0]),
            dv=float(physical_parameters[1]),
            k=float(physical_parameters[2]),
        )
        flat_state = np.concatenate(
            (physical_state[0].numpy().ravel(), physical_state[1].numpy().ravel())
        ).astype(np.float64)
        rhs = reaction_diffusion_rhs(float(sample.time), flat_state, params, laplacian)
        rhs_tensor = torch.from_numpy(rhs.reshape(2, config.pde.height, config.pde.width)).float()
        normalized_rhs = rhs_tensor / stats.state_std[:, None, None]
        error = float(nrmse(normalized_rhs[None], target[None], (1, 2, 3)).item())
        spline_pde_errors.append(error)
        time_value = float(sample.time.item())
        times.append(time_value)
        rows.append(
            {
                "trajectory_id": sample.trajectory_id,
                "interval_index": sample.interval_index,
                "time": time_value,
                "spline_pde_nrmse": error,
                **{f"retained_energy_{mode}": mode_values[mode] for mode in retained},
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "target_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(spline_pde_errors, bins=20)
    axes[0].axvline(0.15, color="tab:red", linestyle="--", label="audit threshold")
    axes[0].set(xlabel="relative error", ylabel="samples", title="Spline derivative vs PDE RHS")
    axes[0].legend()
    modes = sorted(retained)
    axes[1].plot(modes, [float(np.median(retained[mode])) for mode in modes], marker="o")
    axes[1].set(
        xlabel="retained DCT modes per axis",
        ylabel="median energy fraction",
        title="Physical velocity spectral retention",
        ylim=(0.0, 1.01),
    )
    figure.savefig(output / "target_audit.png", dpi=160)
    plt.close(figure)
    rms = torch.stack(target_rms)
    time_tensor = torch.tensor(times)
    normalized_time = 2 * (time_tensor - config.pde.t_start) / (
        config.pde.t_end - config.pde.t_start
    ) - 1
    reconstructed_time = (
        (normalized_time + 1) * (config.pde.t_end - config.pde.t_start) / 2
        + config.pde.t_start
    )
    median_error = float(np.median(spline_pde_errors))
    report = TargetAuditReport(
        passed=median_error <= 0.15,
        samples=len(dataset),
        median_spline_pde_nrmse=median_error,
        target_rms_u=float(torch.median(rms[:, 0]).item()),
        target_rms_v=float(torch.median(rms[:, 1]).item()),
        retained_energy={str(mode): float(np.median(values)) for mode, values in retained.items()},
        time_min=float(time_tensor.min().item()),
        time_max=float(time_tensor.max().item()),
        maximum_time_roundtrip_error=float(torch.max(torch.abs(reconstructed_time - time_tensor))),
        output_directory=str(output),
    )
    _write_json(output / "summary.json", asdict(report))
    dataset.close()
    return report


def _plot_training(rows: list[dict[str, float]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    steps = [row["optimizer_steps"] for row in rows]
    axes[0].semilogy(steps, [row["loss"] for row in rows], label="total")
    axes[0].semilogy(steps, [row["physical_velocity_mse"] for row in rows], label="physical")
    axes[0].set(xlabel="optimizer steps", ylabel="loss", title="Velocity objective")
    axes[0].legend()
    axes[1].plot(steps, [row["velocity_nrmse"] for row in rows], label="velocity")
    rollout_rows = [row for row in rows if math.isfinite(row["rollout_nrmse"])]
    if rollout_rows:
        axes[1].plot(
            [row["optimizer_steps"] for row in rollout_rows],
            [row["rollout_nrmse"] for row in rollout_rows],
            label="rollout",
        )
    axes[1].set(xlabel="optimizer steps", ylabel="nRMSE", title="Diagnostic errors")
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_fixed_diagnostic(
    config: ProjectConfig,
    root: Path,
    data_path: Path,
    device: torch.device,
    model_name: DiagnosticModelName,
    kind: Literal["single_batch", "four_trajectory"],
    max_steps: int,
    evaluation_interval: int,
    rollout_interval: int,
) -> TrainingDiagnosticReport:
    if config.training is None or config.model is None or config.evaluation is None:
        raise ValueError("model, training, and evaluation configuration sections are required")
    seed_everything(0)
    output = root / "artifacts/diagnostics/week6" / f"{model_name}-{kind}-s0"
    output.mkdir(parents=True, exist_ok=True)
    dataset = HDF5VelocityDataset(
        data_path,
        "train",
        "full",
        seed=0,
        gamma=0.0,
        trajectory_ids=SMOKE_IDS,
        resample_each_epoch=False,
    )
    if kind == "single_batch":
        batch_list = [collate_velocity_samples([dataset[index] for index in range(16)])]
        training_loader: list[dict[str, Any]] | DataLoader[Any] = batch_list
    else:
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_velocity_samples,
            generator=torch.Generator().manual_seed(0),
        )
        batch_list = _batches(
            DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_velocity_samples)
        )
        training_loader = loader
    rollout_dataset = HDF5RolloutDataset(data_path, "train", "full", SMOKE_IDS)
    model = _model_for_diagnostic(config, model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    modes = 16 if model_name == "chronopde_modes16" else 12
    initial = _velocity_metrics(
        model,
        batch_list,
        device,
        config.training.spectral_loss_weight,
        modes,
        modes,
        dataset.normalization.state_std,
    )
    initial_row = {
        "optimizer_steps": 0.0,
        **initial,
        "gradient_norm": float("nan"),
        "rollout_nrmse": float("nan"),
        "final_rollout_nrmse": float("nan"),
        "boundary_normal_mse": float("nan"),
        "stable": 1.0,
    }
    rows: list[dict[str, float]] = [initial_row]
    best_loss = initial["loss"]
    best_step = 0
    last_gradient_norm = float("nan")
    metrics_path = output / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    metrics_path.write_text(json.dumps(initial_row, sort_keys=True) + "\n", encoding="utf-8")
    save_checkpoint(
        output / "best.pt",
        model,
        optimizer,
        scheduler,
        epoch=0,
        optimizer_steps=0,
        best_metric=best_loss,
        patience_counter=0,
        model_name=model_name,
    )
    iterator = iter(training_loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(training_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        state = cast(Tensor, batch["state"]).to(device)
        target = cast(Tensor, batch["target_velocity"]).to(device)
        time = cast(Tensor, batch["time"]).to(device)
        parameters = cast(Tensor, batch["parameters"]).to(device)
        prediction = model(state, time, parameters)
        breakdown = velocity_loss(
            prediction,
            target,
            spectral_weight=config.training.spectral_loss_weight,
            modes_y=modes,
            modes_x=modes,
        )
        if not bool(torch.isfinite(breakdown.total)):
            raise FloatingPointError("diagnostic loss became non-finite")
        breakdown.total.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        last_gradient_norm = float(gradient_norm.item())
        optimizer.step()
        if step % evaluation_interval == 0 or step == max_steps:
            metrics = _velocity_metrics(
                model,
                batch_list,
                device,
                config.training.spectral_loss_weight,
                modes,
                modes,
                dataset.normalization.state_std,
            )
            rollout = {
                "rollout_nrmse": float("nan"),
                "final_rollout_nrmse": float("nan"),
                "boundary_normal_mse": float("nan"),
                "stable": True,
            }
            if kind == "four_trajectory" and (
                step % rollout_interval == 0 or step == max_steps
            ):
                rollout = _rollout_metrics(model, rollout_dataset, config, device)
            row = {
                "optimizer_steps": float(step),
                **metrics,
                "gradient_norm": last_gradient_norm,
                "rollout_nrmse": float(rollout["rollout_nrmse"]),
                "final_rollout_nrmse": float(rollout["final_rollout_nrmse"]),
                "boundary_normal_mse": float(rollout["boundary_normal_mse"]),
                "stable": float(bool(rollout["stable"])),
            }
            rows.append(row)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"diagnostic": f"{model_name}-{kind}", **row}, sort_keys=True))
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_step = step
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch=0,
                    optimizer_steps=step,
                    best_metric=best_loss,
                    patience_counter=0,
                    model_name=model_name,
                )
    final = _velocity_metrics(
        model,
        batch_list,
        device,
        config.training.spectral_loss_weight,
        modes,
        modes,
        dataset.normalization.state_std,
    )
    rollout = (
        _rollout_metrics(model, rollout_dataset, config, device)
        if kind == "four_trajectory"
        else {
            "rollout_nrmse": float("nan"),
            "final_rollout_nrmse": float("nan"),
            "boundary_normal_mse": float("nan"),
            "stable": True,
        }
    )
    reduction = initial["loss"] / max(final["loss"], 1e-30)
    passed = (
        reduction >= 1_000 and final["velocity_nrmse"] <= 0.01
        if kind == "single_batch"
        else reduction >= 100
        and final["velocity_nrmse"] <= 0.10
        and float(rollout["rollout_nrmse"]) <= 0.15
        and bool(rollout["stable"])
    )
    save_checkpoint(
        output / "last.pt",
        model,
        optimizer,
        scheduler,
        epoch=0,
        optimizer_steps=max_steps,
        best_metric=best_loss,
        patience_counter=0,
        model_name=model_name,
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_training(rows, output / "diagnostic_curves.png")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    _write_json(output / "environment.json", environment_metadata(root, 0))
    _write_json(output / "best_step.json", {"optimizer_steps": best_step})
    report = TrainingDiagnosticReport(
        passed=passed,
        model=model_name,
        kind=kind,
        optimizer_steps=max_steps,
        initial_loss=initial["loss"],
        final_loss=final["loss"],
        loss_reduction=reduction,
        velocity_nrmse=final["velocity_nrmse"],
        velocity_nrmse_u=final["velocity_nrmse_u"],
        velocity_nrmse_v=final["velocity_nrmse_v"],
        physical_velocity_mse=final["physical_velocity_mse"],
        spectral_relative_error=final["spectral_relative_error"],
        rollout_nrmse=float(rollout["rollout_nrmse"]),
        final_rollout_nrmse=float(rollout["final_rollout_nrmse"]),
        boundary_normal_mse=float(rollout["boundary_normal_mse"]),
        stable=bool(rollout["stable"]),
        parameter_count=trainable_parameter_count(model),
        artifact_directory=str(output),
    )
    _write_json(output / "summary.json", asdict(report))
    dataset.close()
    rollout_dataset.close()
    return report


def run_continuous_gate_suite(
    config: ProjectConfig,
    root: Path,
    data_path: Path,
    *,
    device_name: str = "auto",
    single_batch_steps: int = 2_000,
    four_trajectory_steps: int = 10_000,
    evaluation_interval: int = 250,
    rollout_interval: int = 1_000,
) -> ContinuousGateSuiteReport:
    if min(single_batch_steps, four_trajectory_steps, evaluation_interval, rollout_interval) < 1:
        raise ValueError("diagnostic step counts and intervals must be positive")
    output = root / "artifacts/diagnostics/week6"
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    _write_json(output / "environment.json", environment_metadata(root, 0))
    audit = audit_targets(config, data_path, output / "target_audit")
    single: dict[str, TrainingDiagnosticReport] = {}
    four: dict[str, TrainingDiagnosticReport] = {}
    variants: dict[str, TrainingDiagnosticReport] = {}
    route = "repair_target"
    passed = False
    if audit.passed:
        device = resolve_device(device_name)
        for name in ("fno_ct", "chronopde"):
            single[name] = train_fixed_diagnostic(
                config,
                root,
                data_path,
                device,
                name,
                "single_batch",
                single_batch_steps,
                evaluation_interval,
                rollout_interval,
            )
        if all(report.passed for report in single.values()):
            for backbone in ("fno_ct", "chronopde"):
                four[backbone] = train_fixed_diagnostic(
                    config,
                    root,
                    data_path,
                    device,
                    backbone,
                    "four_trajectory",
                    four_trajectory_steps,
                    evaluation_interval,
                    rollout_interval,
                )
            control = four["fno_ct"]
            candidate = four["chronopde"]
            matched = (
                candidate.velocity_nrmse <= 1.10 * control.velocity_nrmse
                and candidate.rollout_nrmse <= 1.10 * control.rollout_nrmse
            )
            passed = candidate.passed and matched
            if passed:
                route = "proceed_week7"
            elif control.passed:
                route = "test_dct_variants"
                for name in ("chronopde_residual", "chronopde_modes16"):
                    variants[name] = train_fixed_diagnostic(
                        config,
                        root,
                        data_path,
                        device,
                        cast(DiagnosticModelName, name),
                        "four_trajectory",
                        four_trajectory_steps,
                        evaluation_interval,
                        rollout_interval,
                    )
                    if variants[name].passed:
                        route = f"retrain_full_with_{name}"
                        break
            else:
                route = "repair_continuous_gate_metric"
        else:
            route = "debug_model_loss_optimizer"
    report = ContinuousGateSuiteReport(
        passed=passed,
        route=route,
        target_audit=audit,
        single_batch=single,
        four_trajectory=four,
        variants=variants,
        output_directory=str(output),
    )
    _write_json(output / "suite_summary.json", asdict(report))
    return report
