"""Reproducible training and checkpointing for autoregressive baselines."""

from __future__ import annotations

import json
import math
import random
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

from chronopde.config import ProjectConfig, RegimeName
from chronopde.data.datasets import (
    HDF5AutoregressiveDataset,
    HDF5RolloutDataset,
    NormalizationStats,
    collate_autoregressive_pairs,
    collate_rollouts,
)
from chronopde.evaluation.metrics import nrmse, rollout_nrmse
from chronopde.evaluation.rollout import autoregressive_rollout, persistence_rollout
from chronopde.experiment import experiment_id
from chronopde.models import FNOAutoregressive, UNetAutoregressive, trainable_parameter_count
from chronopde.reproducibility import environment_metadata, seed_everything

AutoregressiveModelName = Literal["unet_ar", "fno_ar"]


@dataclass(frozen=True)
class TrainingReport:
    passed: bool
    model: str
    experiment_id: str
    epochs: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    best_validation_nrmse: float
    persistence_validation_nrmse: float
    parameter_count: int
    artifact_directory: str
    message: str


def build_autoregressive_model(name: AutoregressiveModelName, config: ProjectConfig) -> nn.Module:
    if config.model is None:
        raise ValueError("model configuration is required")
    if name == "unet_ar":
        return UNetAutoregressive(config.model.unet_channels)
    if name == "fno_ar":
        return FNOAutoregressive(
            width=config.model.fno_width,
            modes_y=config.model.spectral_modes_y,
            modes_x=config.model.spectral_modes_x,
        )
    raise ValueError(f"unsupported autoregressive model: {name}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def _stats_to(stats: NormalizationStats, device: torch.device) -> NormalizationStats:
    return NormalizationStats(
        state_mean=stats.state_mean.to(device),
        state_std=stats.state_std.to(device),
        parameter_mean=stats.parameter_mean.to(device),
        parameter_std=stats.parameter_std.to(device),
    )


def _denormalize_rollout(states: Tensor, stats: NormalizationStats) -> Tensor:
    return (
        states * stats.state_std[None, None, :, None, None]
        + stats.state_mean[None, None, :, None, None]
    )


@torch.no_grad()
def evaluate_rollouts(
    model: nn.Module,
    dataset: HDF5RolloutDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float, bool]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rollouts)
    stats = _stats_to(dataset.normalization, device)
    errors: list[Tensor] = []
    final_errors: list[Tensor] = []
    persistence_errors: list[Tensor] = []
    finite = True
    model.eval()
    for batch in loader:
        target = cast(Tensor, batch["states"]).to(device)
        times = cast(Tensor, batch["times"]).to(device)
        parameters = cast(Tensor, batch["parameters"]).to(device)
        prediction = autoregressive_rollout(
            model,
            target[:, 0],
            times,
            parameters,
            reference_dt=dataset.reference_dt if hasattr(dataset, "reference_dt") else 0.5,
        )
        physical_prediction = _denormalize_rollout(prediction, stats)
        physical_target = _denormalize_rollout(target, stats)
        batch_errors = rollout_nrmse(physical_prediction, physical_target)
        persistence = persistence_rollout(physical_target[:, 0], physical_target.shape[1])
        persistence_errors.append(rollout_nrmse(persistence, physical_target))
        errors.append(batch_errors)
        final_errors.append(batch_errors[:, -1])
        finite = finite and bool(torch.isfinite(physical_prediction).all())
        finite = finite and bool(torch.max(torch.abs(physical_prediction)) <= 10)
    all_errors = torch.cat(errors)
    return (
        float(torch.median(all_errors).item()),
        float(torch.median(torch.cat(final_errors)).item()),
        float(torch.median(torch.cat(persistence_errors)).item()),
        finite,
    )


def _average_loss(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            current = cast(Tensor, batch["current_state"]).to(device)
            target = cast(Tensor, batch["residual_target"]).to(device)
            delta_t = cast(Tensor, batch["delta_t"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            prediction = model(current, delta_t, parameters)
            losses.append(float(torch.mean(torch.square(prediction - target)).item()))
    return float(np.mean(losses))


def _one_step_nrmse(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    stats: NormalizationStats,
) -> float:
    errors: list[Tensor] = []
    stats = _stats_to(stats, device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            current = cast(Tensor, batch["current_state"]).to(device)
            target_residual = cast(Tensor, batch["residual_target"]).to(device)
            delta_t = cast(Tensor, batch["delta_t"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            prediction = current + model(current, delta_t, parameters)
            target = current + target_residual
            prediction_physical = (
                prediction * stats.state_std[None, :, None, None]
                + stats.state_mean[None, :, None, None]
            )
            target_physical = (
                target * stats.state_std[None, :, None, None]
                + stats.state_mean[None, :, None, None]
            )
            errors.append(nrmse(prediction_physical, target_physical, (1, 2, 3)))
    return float(torch.median(torch.cat(errors)).item())


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    *,
    epoch: int,
    optimizer_steps: int,
    best_metric: float,
    patience_counter: int,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "best_metric": best_metric,
            "patience_counter": patience_counter,
            "model_name": model_name,
            "rng_state": _rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if "rng_state" in payload:
        _restore_rng(payload["rng_state"])
    return cast(dict[str, Any], payload)


def _scheduler(
    optimizer: AdamW, warmup_epochs: int, total_epochs: int, minimum_ratio: float
) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), minimum_ratio)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        return minimum_ratio + 0.5 * (1 - minimum_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def _plot_history(history: list[dict[str, float]], output: Path) -> None:
    if not history:
        return
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history])
    axes[0].set(xlabel="epoch", ylabel="residual MSE", yscale="log", title="Training loss")
    validation = [row for row in history if math.isfinite(row["validation_nrmse"])]
    if validation:
        axes[1].plot(
            [row["epoch"] for row in validation],
            [row["validation_nrmse"] for row in validation],
        )
    axes[1].set(xlabel="epoch", ylabel="median rollout nRMSE", title="Validation rollout")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_autoregressive(
    config: ProjectConfig,
    root: Path,
    model_name: AutoregressiveModelName,
    regime: RegimeName,
    seed: int,
    *,
    data_path: Path | None = None,
    device_name: str = "auto",
    smoke_overfit: bool = False,
    resume: bool = False,
    max_epochs: int | None = None,
) -> TrainingReport:
    if config.training is None or config.model is None:
        raise ValueError("model and training configuration sections are required")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    seed_everything(seed)
    device = resolve_device(device_name)
    dataset_path = data_path or root / config.data.output_path
    smoke_ids = tuple(f"train-{index:04d}" for index in range(4))
    pair_dataset = HDF5AutoregressiveDataset(
        dataset_path,
        "train",
        regime,
        smoke_ids if smoke_overfit else None,
    )
    rollout_dataset = HDF5RolloutDataset(
        dataset_path,
        "train" if smoke_overfit else "validation",
        regime,
        smoke_ids if smoke_overfit else None,
    )
    batch_size = config.training.smoke_batch_size if smoke_overfit else config.training.batch_size
    learning_rate = (
        config.training.smoke_learning_rate if smoke_overfit else config.training.learning_rate
    )
    weight_decay = 0.0 if smoke_overfit else config.training.weight_decay
    epochs = max_epochs or config.training.max_epochs
    workers = 0 if smoke_overfit else config.training.num_workers
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_autoregressive_pairs,
        generator=generator,
    )
    evaluation_loader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_autoregressive_pairs,
    )
    model = build_autoregressive_model(model_name, config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = _scheduler(
        optimizer,
        0 if smoke_overfit else config.training.warmup_epochs,
        epochs,
        config.training.minimum_learning_rate / learning_rate,
    )
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
    initial_loss = _average_loss(model, evaluation_loader, device)
    start_epoch = 0
    optimizer_steps = 0
    best_metric = float("inf")
    persistence_metric = float("inf")
    patience_counter = 0
    if resume and (artifact_directory / "last.pt").is_file():
        payload = load_checkpoint(artifact_directory / "last.pt", model, optimizer, scheduler)
        start_epoch = int(payload["epoch"]) + 1
        optimizer_steps = int(payload["optimizer_steps"])
        best_metric = float(payload["best_metric"])
        patience_counter = int(payload["patience_counter"])

    history: list[dict[str, float]] = []
    metrics_path = artifact_directory / "metrics.jsonl"
    passed = False
    message = "maximum training budget reached"
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            current = cast(Tensor, batch["current_state"]).to(device)
            target = cast(Tensor, batch["residual_target"]).to(device)
            delta_t = cast(Tensor, batch["delta_t"]).to(device)
            parameters = cast(Tensor, batch["parameters"]).to(device)
            prediction = model(current, delta_t, parameters)
            loss = torch.mean(torch.square(prediction - target))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss became non-finite")
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("gradient norm became non-finite")
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(loss.item()))
            if smoke_overfit and optimizer_steps >= config.training.smoke_max_steps:
                break
        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        should_evaluate = not smoke_overfit or (
            optimizer_steps % config.training.smoke_evaluation_interval < max(len(loader), 1)
        )
        validation_metric = float("nan")
        final_metric = float("nan")
        current_persistence = float("nan")
        stable = True
        one_step_loss = _average_loss(model, evaluation_loader, device)
        one_step_nrmse = _one_step_nrmse(
            model, evaluation_loader, device, pair_dataset.normalization
        )
        if should_evaluate:
            validation_metric, final_metric, current_persistence, stable = evaluate_rollouts(
                model, rollout_dataset, device, batch_size=min(batch_size, 8)
            )
            persistence_metric = current_persistence
        row = {
            "epoch": float(epoch),
            "optimizer_steps": float(optimizer_steps),
            "train_loss": train_loss,
            "one_step_loss": one_step_loss,
            "one_step_nrmse": one_step_nrmse,
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
        elif not smoke_overfit:
            patience_counter += 1
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
                initial_loss / max(one_step_loss, 1e-30) >= config.training.smoke_loss_reduction
            )
            if (
                loss_ok
                and one_step_nrmse <= config.training.smoke_one_step_nrmse
                and validation_metric <= config.training.smoke_rollout_nrmse
                and stable
            ):
                passed = True
                message = "four-trajectory overfit gate passed"
                break
            if optimizer_steps >= config.training.smoke_max_steps:
                break
        elif (
            patience_counter >= config.training.patience
            and epoch + 1 >= config.training.minimum_epochs
        ):
            passed = math.isfinite(best_metric) and best_metric < persistence_metric
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
            and len(history) >= config.training.minimum_epochs
        )
        message = (
            "configured epoch budget completed" if passed else "full-training gate not reached"
        )

    final_loss = _average_loss(model, evaluation_loader, device)
    _plot_history(history, artifact_directory / "training_curves.png")
    report = TrainingReport(
        passed=passed,
        model=model_name,
        experiment_id=run_id,
        epochs=(int(history[-1]["epoch"]) + 1 if history else start_epoch),
        optimizer_steps=optimizer_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_validation_nrmse=best_metric,
        persistence_validation_nrmse=persistence_metric,
        parameter_count=parameter_count,
        artifact_directory=str(artifact_directory),
        message=message,
    )
    (artifact_directory / "summary.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pair_dataset.close()
    rollout_dataset.close()
    return report
