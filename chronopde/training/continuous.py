"""Velocity-matching training for continuous-time neural operators."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chronopde.config import ProjectConfig, RegimeName
from chronopde.data.datasets import (
    HDF5RolloutDataset,
    HDF5VelocityDataset,
    NormalizationStats,
    collate_rollouts,
    collate_velocity_samples,
)
from chronopde.evaluation.metrics import nrmse, rollout_nrmse
from chronopde.evaluation.rollout import continuous_rollout, persistence_rollout
from chronopde.experiment import experiment_id
from chronopde.models import (
    DCTContinuousVectorField,
    FFTContinuousVectorField,
    trainable_parameter_count,
)
from chronopde.reproducibility import environment_metadata, seed_everything
from chronopde.training.losses import velocity_loss
from chronopde.training.trainer import (
    _plot_history,
    _scheduler,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


@dataclass(frozen=True)
class ContinuousTrainingReport:
    passed: bool
    model: str
    experiment_id: str
    epochs: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    velocity_nrmse: float
    best_validation_nrmse: float
    persistence_validation_nrmse: float
    parameter_count: int
    peak_device_memory_bytes: int
    artifact_directory: str
    message: str


ContinuousModelName = Literal["chronopde", "fno_ct"]


def build_continuous_model(
    config: ProjectConfig, model_name: ContinuousModelName = "fno_ct"
) -> nn.Module:
    if config.model is None:
        raise ValueError("model configuration is required")
    if model_name == "fno_ct":
        return FFTContinuousVectorField(
            width=config.model.fno_width,
            modes_y=config.model.spectral_modes_y,
            modes_x=config.model.spectral_modes_x,
            blocks=config.model.blocks,
            film_hidden_width=config.model.film_hidden_width,
            time_range=(config.pde.t_start, config.pde.t_end),
        )
    if model_name == "chronopde":
        return DCTContinuousVectorField(
            width=config.model.width,
            modes_y=config.model.spectral_modes_y,
            modes_x=config.model.spectral_modes_x,
            blocks=config.model.blocks,
            film_hidden_width=config.model.film_hidden_width,
            time_range=(config.pde.t_start, config.pde.t_end),
        )
    raise ValueError(f"unsupported continuous-time model: {model_name}")


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


@torch.no_grad()
def evaluate_continuous_rollouts(
    model: nn.Module,
    dataset: HDF5RolloutDataset,
    device: torch.device,
    batch_size: int,
    steps_per_interval: int,
) -> tuple[float, float, float, bool]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rollouts)
    stats = _stats_to(dataset.normalization, device)
    errors: list[Tensor] = []
    final_errors: list[Tensor] = []
    persistence_errors: list[Tensor] = []
    stable = True
    model.eval()
    for batch in loader:
        target = cast(Tensor, batch["states"]).to(device)
        times = cast(Tensor, batch["times"]).to(device)
        parameters = cast(Tensor, batch["parameters"]).to(device)
        integration = continuous_rollout(
            model,
            target[:, 0],
            times,
            parameters,
            steps_per_interval=steps_per_interval,
        )
        prediction = cast(Tensor, integration.states)
        physical_prediction = _denormalize(prediction, stats)
        physical_target = _denormalize(target, stats)
        batch_errors = rollout_nrmse(physical_prediction, physical_target)
        persistence = persistence_rollout(physical_target[:, 0], physical_target.shape[1])
        persistence_errors.append(rollout_nrmse(persistence, physical_target))
        errors.append(batch_errors)
        final_errors.append(batch_errors[:, -1])
        stable = stable and bool(torch.isfinite(physical_prediction).all())
        stable = stable and bool(torch.max(torch.abs(physical_prediction)) <= 10)
    return (
        float(torch.median(torch.cat(errors)).item()),
        float(torch.median(torch.cat(final_errors)).item()),
        float(torch.median(torch.cat(persistence_errors)).item()),
        stable,
    )


@torch.no_grad()
def evaluate_velocity(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    spectral_weight: float,
    modes_y: int,
    modes_x: int,
) -> tuple[float, float]:
    losses: list[float] = []
    errors: list[Tensor] = []
    model.eval()
    for batch in loader:
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
        losses.append(float(breakdown.total.item()))
        errors.append(nrmse(prediction, target, (1, 2, 3)))
    return float(np.mean(losses)), float(torch.median(torch.cat(errors)).item())


def train_continuous_time(
    config: ProjectConfig,
    root: Path,
    regime: RegimeName,
    seed: int,
    *,
    model_name: ContinuousModelName = "fno_ct",
    data_path: Path | None = None,
    device_name: str = "auto",
    smoke_overfit: bool = False,
    resume: bool = False,
    max_epochs: int | None = None,
    learning_rate_override: float | None = None,
    steps_per_interval: int | None = None,
) -> ContinuousTrainingReport:
    if config.training is None or config.model is None or config.evaluation is None:
        raise ValueError("model, training, and evaluation configuration sections are required")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if learning_rate_override is not None and learning_rate_override <= 0:
        raise ValueError("learning rate must be positive")
    seed_everything(seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset_path = data_path or root / config.data.output_path
    smoke_ids = tuple(f"train-{index:04d}" for index in range(4))
    selected_ids = smoke_ids if smoke_overfit else None
    velocity_dataset = HDF5VelocityDataset(
        dataset_path,
        "train",
        regime,
        seed,
        config.model.perturbation_gamma,
        selected_ids,
    )
    validation_velocity = HDF5VelocityDataset(
        dataset_path,
        "train" if smoke_overfit else "validation",
        regime,
        seed + 1,
        0.0,
        selected_ids,
    )
    rollout_dataset = HDF5RolloutDataset(
        dataset_path,
        "train" if smoke_overfit else "validation",
        regime,
        selected_ids,
    )
    batch_size = config.training.smoke_batch_size if smoke_overfit else config.training.batch_size
    workers = 0 if smoke_overfit else config.training.num_workers
    validation_loader = DataLoader(
        validation_velocity,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_velocity_samples,
    )
    model = build_continuous_model(config, model_name).to(device)
    learning_rate = learning_rate_override or (
        config.training.smoke_learning_rate if smoke_overfit else config.training.learning_rate
    )
    weight_decay = 0.0 if smoke_overfit else config.training.weight_decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = max_epochs or config.training.max_epochs
    scheduler = _scheduler(
        optimizer,
        0 if smoke_overfit else config.training.warmup_epochs,
        epochs,
        config.training.minimum_learning_rate / learning_rate,
    )
    integration_steps = (
        steps_per_interval
        if steps_per_interval is not None
        else config.evaluation.steps_per_interval
    )
    if integration_steps < 1:
        raise ValueError("steps_per_interval must be positive")
    run_id = experiment_id(model_name, regime, "train", seed)
    if smoke_overfit:
        run_id += "-smoke"
    artifact_directory = root / config.project.artifact_root / run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    (artifact_directory / "environment.json").write_text(
        json.dumps(environment_metadata(root, seed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parameter_count = trainable_parameter_count(model)
    (artifact_directory / "parameter_count.json").write_text(
        json.dumps({"real_trainable_parameters": parameter_count}, indent=2) + "\n",
        encoding="utf-8",
    )
    initial_loss, _ = evaluate_velocity(
        model,
        validation_loader,
        device,
        config.training.spectral_loss_weight,
        config.model.spectral_modes_y,
        config.model.spectral_modes_x,
    )
    start_epoch = 0
    optimizer_steps = 0
    best_metric = float("inf")
    persistence_metric = float("inf")
    patience_counter = 0
    if resume and (artifact_directory / "last.pt").is_file():
        payload = load_checkpoint(
            artifact_directory / "last.pt",
            model,
            optimizer,
            scheduler,
            expected_model_name=model_name,
        )
        start_epoch = int(payload["epoch"]) + 1
        optimizer_steps = int(payload["optimizer_steps"])
        best_metric = float(payload["best_metric"])
        patience_counter = int(payload["patience_counter"])

    history: list[dict[str, float]] = []
    metrics_path = artifact_directory / "metrics.jsonl"
    passed = False
    message = "maximum training budget reached"
    final_velocity_nrmse = float("inf")
    last_stable = False
    for epoch in range(start_epoch, epochs):
        velocity_dataset.set_epoch(epoch)
        loader = DataLoader(
            velocity_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            collate_fn=collate_velocity_samples,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
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
                modes_y=config.model.spectral_modes_y,
                modes_x=config.model.spectral_modes_x,
            )
            if not bool(torch.isfinite(breakdown.total)):
                raise FloatingPointError("training loss became non-finite")
            breakdown.total.backward()  # type: ignore[no-untyped-call]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("gradient norm became non-finite")
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(breakdown.total.item()))
            if smoke_overfit and optimizer_steps >= config.training.smoke_max_steps:
                break
        scheduler.step()
        velocity_value, final_velocity_nrmse = evaluate_velocity(
            model,
            validation_loader,
            device,
            config.training.spectral_loss_weight,
            config.model.spectral_modes_y,
            config.model.spectral_modes_x,
        )
        if smoke_overfit:
            should_rollout = optimizer_steps % config.training.smoke_evaluation_interval < max(
                len(loader), 1
            )
        else:
            should_rollout = (epoch + 1) % config.training.continuous_validation_interval == 0
        validation_metric = float("nan")
        final_metric = float("nan")
        current_persistence = float("nan")
        stable = True
        if should_rollout:
            validation_metric, final_metric, current_persistence, stable = (
                evaluate_continuous_rollouts(
                    model,
                    rollout_dataset,
                    device,
                    batch_size=min(batch_size, 8),
                    steps_per_interval=integration_steps,
                )
            )
            persistence_metric = current_persistence
            last_stable = stable
        row = {
            "epoch": float(epoch),
            "optimizer_steps": float(optimizer_steps),
            "train_loss": float(np.mean(epoch_losses)),
            "velocity_loss": velocity_value,
            "velocity_nrmse": final_velocity_nrmse,
            "validation_nrmse": validation_metric,
            "final_nrmse": final_metric,
            "persistence_nrmse": current_persistence,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        improved = math.isfinite(validation_metric) and validation_metric < best_metric
        if improved:
            best_metric = validation_metric
            patience_counter = 0
            save_checkpoint(
                artifact_directory / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                best_metric=best_metric,
                patience_counter=patience_counter,
                model_name=model_name,
            )
        elif should_rollout and not smoke_overfit:
            patience_counter += config.training.continuous_validation_interval
        save_checkpoint(
            artifact_directory / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            best_metric=best_metric,
            patience_counter=patience_counter,
            model_name=model_name,
        )
        if smoke_overfit:
            loss_ok = (
                initial_loss / max(velocity_value, 1e-30) >= config.training.smoke_loss_reduction
            )
            if (
                loss_ok
                and final_velocity_nrmse <= config.training.smoke_one_step_nrmse
                and validation_metric <= config.training.smoke_rollout_nrmse
                and stable
            ):
                passed = True
                message = "four-trajectory continuous-time overfit gate passed"
                break
            if optimizer_steps >= config.training.smoke_max_steps:
                break
        elif (
            patience_counter >= config.training.patience
            and epoch + 1 >= config.training.minimum_epochs
        ):
            passed = math.isfinite(best_metric) and best_metric < persistence_metric and stable
            message = (
                "early stopping completed"
                if passed
                else "validation rollout did not beat persistence"
            )
            break
    else:
        passed = (
            math.isfinite(best_metric)
            and best_metric < persistence_metric
            and bool(history)
            and int(history[-1]["epoch"]) + 1 >= config.training.minimum_epochs
            and last_stable
        )
        message = "configured epoch budget completed" if passed else "Week 5 gate not reached"

    final_loss, final_velocity_nrmse = evaluate_velocity(
        model,
        validation_loader,
        device,
        config.training.spectral_loss_weight,
        config.model.spectral_modes_y,
        config.model.spectral_modes_x,
    )
    _plot_history(history, artifact_directory / "training_curves.png")
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    report = ContinuousTrainingReport(
        passed=passed,
        model=model_name,
        experiment_id=run_id,
        epochs=int(history[-1]["epoch"]) + 1 if history else start_epoch,
        optimizer_steps=optimizer_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        velocity_nrmse=final_velocity_nrmse,
        best_validation_nrmse=best_metric,
        persistence_validation_nrmse=persistence_metric,
        parameter_count=parameter_count,
        peak_device_memory_bytes=peak_memory,
        artifact_directory=str(artifact_directory),
        message=message,
    )
    (artifact_directory / "summary.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    velocity_dataset.close()
    validation_velocity.close()
    rollout_dataset.close()
    return report
