"""Development-only exact-RHS feasibility training for ChronoPDE V2 Phase 4."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from chronopde.evaluation.rollout import continuous_rollout
from chronopde.models import DCTContinuousVectorField, FFTContinuousVectorField
from chronopde.training.losses import full_field_relative_velocity_loss
from chronopde.v2.checkpoints import (
    CheckpointIdentity,
    load_v2_checkpoint,
    save_v2_checkpoint,
)
from chronopde.v2.common import atomic_write_bytes, atomic_write_json, canonical_json_bytes
from chronopde.v2.protocol import Phase4Protocol
from chronopde.v2.recovery import create_recovery_package
from chronopde.v2.registry import (
    ExecutionStatus,
    RunManifest,
    ScientificOutcome,
    create_run_manifest,
    transition_run_manifest,
)

ModelName = Literal["fft", "dct"]
FloatArray = NDArray[np.float64]


class DevelopmentExactRHSDataset:
    """Read only training/validation data; reject any confirmatory payload."""

    def __init__(self, path: Path, protocol: Phase4Protocol) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"development dataset not found: {path}")
        self.path = path
        self.handle = h5py.File(path, "r")
        try:
            if self.handle.attrs.get("schema_version") != "chronopde_v2_development_1":
                raise ValueError("development dataset schema mismatch")
            if (
                self.handle.attrs.get("logical_content_sha256")
                != protocol.development_dataset_sha256
            ):
                raise ValueError("development dataset identity mismatch")
            if self.handle.attrs.get("confirmatory_status") != "sealed_not_generated":
                raise ValueError("confirmatory data are not sealed")
            if "confirmatory" in self.handle["splits"]:
                raise ValueError("Phase 4 refuses datasets containing confirmatory states")
            normalization = self.handle["normalization"]
            self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
            self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
            self.parameter_mean = np.asarray(normalization["parameter_mean"], dtype=np.float32)
            self.parameter_std = np.asarray(normalization["parameter_std"], dtype=np.float32)
            payload = {
                name: np.asarray(normalization[name], dtype=np.float64).tolist()
                for name in ("state_mean", "state_std", "parameter_mean", "parameter_std")
            }
            import hashlib

            digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            if digest != protocol.normalization_sha256:
                raise ValueError("development normalization identity mismatch")
            if np.any(self.state_std <= 0) or np.any(self.parameter_std <= 0):
                raise ValueError("normalization scales must be positive")
        except Exception:
            self.handle.close()
            raise

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> DevelopmentExactRHSDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_count(self, split: Literal["training", "validation"]) -> int:
        shape = self.handle[f"splits/{split}/states"].shape
        return int(shape[0] * shape[1])

    def _read_pairs(
        self, split: Literal["training", "validation"], flat_indices: NDArray[np.int64]
    ) -> dict[str, Tensor]:
        group = self.handle[f"splits/{split}"]
        time_count = int(group["states"].shape[1])
        states: list[NDArray[np.float32]] = []
        targets: list[NDArray[np.float32]] = []
        times: list[float] = []
        parameters: list[NDArray[np.float32]] = []
        for flat in flat_indices.tolist():
            trajectory, time_index = divmod(int(flat), time_count)
            states.append(np.asarray(group["states"][trajectory, time_index], dtype=np.float32))
            targets.append(np.asarray(group["rhs"][trajectory, time_index], dtype=np.float32))
            times.append(float(group["times"][trajectory, time_index]))
            parameters.append(np.asarray(group["parameters"][trajectory], dtype=np.float32))
        state = (np.stack(states) - self.state_mean[None, :, None, None]) / self.state_std[
            None, :, None, None
        ]
        target = np.stack(targets) / self.state_std[None, :, None, None]
        parameter = (np.stack(parameters) - self.parameter_mean) / self.parameter_std
        return {
            "state": torch.from_numpy(state),
            "target": torch.from_numpy(target),
            "time": torch.tensor(times, dtype=torch.float32),
            "parameters": torch.from_numpy(parameter),
        }

    def read_rollouts(
        self, trajectory_indices: NDArray[np.int64]
    ) -> dict[str, Tensor]:
        group = self.handle["splits/validation"]
        states = np.stack(
            [
                np.asarray(group["states"][int(index)], dtype=np.float32)
                for index in trajectory_indices
            ]
        )
        times = np.stack(
            [
                np.asarray(group["times"][int(index)], dtype=np.float32)
                for index in trajectory_indices
            ]
        )
        parameters = np.stack(
            [
                np.asarray(group["parameters"][int(index)], dtype=np.float32)
                for index in trajectory_indices
            ]
        )
        normalized = (states - self.state_mean[None, None, :, None, None]) / self.state_std[
            None, None, :, None, None
        ]
        normalized_parameters = (parameters - self.parameter_mean) / self.parameter_std
        return {
            "states": torch.from_numpy(normalized),
            "physical_states": torch.from_numpy(states),
            "times": torch.from_numpy(times),
            "parameters": torch.from_numpy(normalized_parameters),
        }


class ReplacementSampler:
    """Deterministic, checkpointable uniform sample stream."""

    def __init__(self, sample_count: int, seed: int, state: dict[str, Any] | None = None) -> None:
        if sample_count < 1:
            raise ValueError("sample count must be positive")
        self.sample_count = sample_count
        self.rng = np.random.default_rng(seed)
        if state is not None:
            if int(state.get("sample_count", -1)) != sample_count:
                raise ValueError("sampler sample count mismatch")
            self.rng.bit_generator.state = cast(dict[str, Any], state["bit_generator"])

    def next(self, batch_size: int) -> NDArray[np.int64]:
        return self.rng.integers(0, self.sample_count, size=batch_size, dtype=np.int64)

    def state_dict(self) -> dict[str, Any]:
        return {"sample_count": self.sample_count, "bit_generator": self.rng.bit_generator.state}


def build_phase4_model(protocol: Phase4Protocol, model_name: ModelName) -> nn.Module:
    if model_name == "fft":
        return FFTContinuousVectorField(
            width=protocol.models.width,
            modes_y=protocol.models.fft_modes[0],
            modes_x=protocol.models.fft_modes[1],
            blocks=protocol.models.blocks,
            film_hidden_width=protocol.models.film_hidden_width,
            time_range=(0.0, 50.0),
        )
    if model_name == "dct":
        return DCTContinuousVectorField(
            width=protocol.models.width,
            modes_y=protocol.models.dct_modes[0],
            modes_x=protocol.models.dct_modes[1],
            blocks=protocol.models.blocks,
            film_hidden_width=protocol.models.film_hidden_width,
            time_range=(0.0, 50.0),
            residual_skip=protocol.models.residual_skip,
        )
    raise ValueError(f"unsupported Phase 4 model: {model_name}")


def fixed_validation_indices(
    sample_count: int, trajectory_count: int, protocol: Phase4Protocol
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(20260922 + 4000 + protocol.training.seed)
    samples = np.sort(
        rng.choice(sample_count, size=protocol.training.validation_velocity_samples, replace=False)
    ).astype(np.int64)
    trajectories = np.linspace(
        0, trajectory_count - 1, protocol.training.validation_rollout_trajectories, dtype=np.int64
    )
    return samples, trajectories


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.inference_mode()
def evaluate_velocity(
    model: nn.Module,
    dataset: DevelopmentExactRHSDataset,
    flat_indices: NDArray[np.int64],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    ratios: list[Tensor] = []
    channel_ratios: list[Tensor] = []
    physical_sse = 0.0
    physical_count = 0
    boundary_sse = boundary_energy = interior_sse = interior_energy = 0.0
    derivative_sse = derivative_energy = 0.0
    std = torch.from_numpy(dataset.state_std).to(device)[None, :, None, None]
    for start in range(0, len(flat_indices), batch_size):
        batch = _move(
            dataset._read_pairs("validation", flat_indices[start : start + batch_size]),
            device,
        )
        prediction = model(batch["state"], batch["time"], batch["parameters"])
        target = batch["target"]
        error = prediction - target
        numerator = error.square().sum(dim=(1, 2, 3))
        denominator = target.square().sum(dim=(1, 2, 3)).clamp_min(1e-8)
        ratios.append(numerator / denominator)
        channel_ratios.append(
            error.square().sum(dim=(2, 3)) / target.square().sum(dim=(2, 3)).clamp_min(1e-8)
        )
        physical_error = error * std
        physical_sse += float(physical_error.square().sum().item())
        physical_count += physical_error.numel()
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[..., :4, :] = True
        mask[..., -4:, :] = True
        mask[..., :, :4] = True
        mask[..., :, -4:] = True
        boundary_sse += float(error[mask].square().sum().item())
        boundary_energy += float(target[mask].square().sum().item())
        interior_sse += float(error[~mask].square().sum().item())
        interior_energy += float(target[~mask].square().sum().item())
        predicted_derivatives = torch.cat(
            (
                prediction[..., 1, :] - prediction[..., 0, :],
                prediction[..., -1, :] - prediction[..., -2, :],
                prediction[..., :, 1] - prediction[..., :, 0],
                prediction[..., :, -1] - prediction[..., :, -2],
            ),
            dim=-1,
        )
        target_derivatives = torch.cat(
            (
                target[..., 1, :] - target[..., 0, :],
                target[..., -1, :] - target[..., -2, :],
                target[..., :, 1] - target[..., :, 0],
                target[..., :, -1] - target[..., :, -2],
            ),
            dim=-1,
        )
        derivative_sse += float((predicted_derivatives - target_derivatives).square().sum())
        derivative_energy += float(target_derivatives.square().sum())
    all_ratios = torch.cat(ratios)
    channels = torch.cat(channel_ratios)
    return {
        "relative_loss": float(all_ratios.mean().item()),
        "velocity_nrmse": float(all_ratios.sqrt().median().item()),
        "per_channel_velocity_nrmse": [
            float(channels[:, index].sqrt().median().item()) for index in range(2)
        ],
        "physical_velocity_mse": physical_sse / physical_count,
        "boundary_strip_relative_l2": math.sqrt(boundary_sse / max(boundary_energy, 1e-12)),
        "interior_relative_l2": math.sqrt(interior_sse / max(interior_energy, 1e-12)),
        "first_interior_normal_derivative_relative_l2": math.sqrt(
            derivative_sse / max(derivative_energy, 1e-12)
        ),
    }


def _correlation_horizon(prediction: Tensor, target: Tensor, times: Tensor) -> Tensor:
    pred = prediction.flatten(2)
    truth = target.flatten(2)
    pred = pred - pred.mean(dim=2, keepdim=True)
    truth = truth - truth.mean(dim=2, keepdim=True)
    correlation = (pred * truth).sum(dim=2) / (
        pred.square().sum(dim=2).sqrt() * truth.square().sum(dim=2).sqrt()
    ).clamp_min(1e-12)
    horizons = []
    for row in range(correlation.shape[0]):
        failed = torch.nonzero(correlation[row] < 0.9)
        index = int(failed[0].item()) if len(failed) else correlation.shape[1] - 1
        horizons.append(times[row, index])
    return torch.stack(horizons)


@torch.inference_mode()
def evaluate_rollouts(
    model: nn.Module,
    dataset: DevelopmentExactRHSDataset,
    trajectory_indices: NDArray[np.int64],
    device: torch.device,
    *,
    batch_size: int,
    steps_per_interval: int,
) -> dict[str, Any]:
    model.eval()
    errors: list[Tensor] = []
    final_errors: list[Tensor] = []
    persistence_errors: list[Tensor] = []
    horizons: list[Tensor] = []
    divergent = 0
    total = 0
    mean = torch.from_numpy(dataset.state_mean).to(device)[None, None, :, None, None]
    std = torch.from_numpy(dataset.state_std).to(device)[None, None, :, None, None]
    for start in range(0, len(trajectory_indices), batch_size):
        batch = _move(dataset.read_rollouts(trajectory_indices[start : start + batch_size]), device)
        integration = continuous_rollout(
            model,
            batch["states"][:, 0],
            batch["times"],
            batch["parameters"],
            steps_per_interval=steps_per_interval,
        )
        physical = cast(Tensor, integration.states) * std + mean
        target = batch["physical_states"]
        finite = torch.isfinite(physical).flatten(1).all(dim=1)
        bounded = physical.abs().flatten(1).amax(dim=1) <= 10
        stable = finite & bounded
        divergent += int((~stable).sum().item())
        total += len(stable)
        if bool(stable.any()):
            predicted = physical[stable]
            expected = target[stable]
            relative = (predicted - expected).flatten(1).norm(dim=1) / expected.flatten(1).norm(
                dim=1
            ).clamp_min(1e-12)
            final = (predicted[:, -1] - expected[:, -1]).flatten(1).norm(dim=1) / expected[
                :, -1
            ].flatten(1).norm(dim=1).clamp_min(1e-12)
            persistence = expected[:, :1].expand_as(expected)
            persistence_relative = (
                (persistence - expected).flatten(1).norm(dim=1)
                / expected.flatten(1).norm(dim=1).clamp_min(1e-12)
            )
            errors.append(relative)
            final_errors.append(final)
            persistence_errors.append(persistence_relative)
            horizons.append(_correlation_horizon(predicted, expected, batch["times"][stable]))
    if not errors:
        return {
            "rollout_relative_l2": None,
            "final_time_relative_l2": None,
            "persistence_relative_l2": None,
            "correlation_horizon": None,
            "divergence_fraction": 1.0,
        }
    return {
        "rollout_relative_l2": float(torch.cat(errors).median().item()),
        "final_time_relative_l2": float(torch.cat(final_errors).median().item()),
        "persistence_relative_l2": float(torch.cat(persistence_errors).median().item()),
        "correlation_horizon": float(torch.cat(horizons).median().item()),
        "divergence_fraction": divergent / total,
    }


def phase4_gate(rows: list[dict[str, Any]], protocol: Phase4Protocol) -> dict[str, Any]:
    """Apply the declared feasibility gate without comparing model superiority."""

    if not rows or rows[0].get("optimizer_steps") != 0:
        raise ValueError("Phase 4 metrics require a step-zero baseline")
    eligible = [
        row
        for row in rows
        if row.get("rollout_relative_l2") is not None
        and row.get("persistence_relative_l2") is not None
    ]
    if not eligible:
        return {"passed": False, "reason": "no stable eligible validation rollout"}
    selected = min(
        eligible,
        key=lambda row: (row["rollout_relative_l2"], row["velocity_nrmse"]),
    )
    reduction = float(rows[0]["validation_relative_loss"]) / max(
        float(selected["validation_relative_loss"]), 1e-30
    )
    conditions = {
        "loss_reduction": reduction >= protocol.gate.minimum_validation_loss_reduction,
        "velocity_nrmse": selected["velocity_nrmse"] <= protocol.gate.maximum_velocity_nrmse,
        "rollout_beats_persistence": (
            selected["rollout_relative_l2"] < selected["persistence_relative_l2"]
        ),
        "zero_divergence": (
            selected["divergence_fraction"] <= protocol.gate.maximum_divergence_fraction
        ),
        "full_budget_completed": max(int(row["optimizer_steps"]) for row in rows)
        >= protocol.training.maximum_steps,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "validation_loss_reduction": reduction,
        "selected_step": selected["optimizer_steps"],
        "selected_metrics": selected,
    }


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scheduler(optimizer: AdamW, protocol: Phase4Protocol) -> LambdaLR:
    training = protocol.training
    floor = training.minimum_learning_rate / training.learning_rate

    def factor(step: int) -> float:
        if step < training.warmup_steps:
            return max((step + 1) / max(training.warmup_steps, 1), floor)
        progress = (step - training.warmup_steps) / max(
            training.maximum_steps - training.warmup_steps, 1
        )
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return LambdaLR(optimizer, factor)


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _evaluated_row(
    model: nn.Module,
    dataset: DevelopmentExactRHSDataset,
    validation_samples: NDArray[np.int64],
    rollout_trajectories: NDArray[np.int64],
    device: torch.device,
    protocol: Phase4Protocol,
    optimizer_steps: int,
) -> dict[str, Any]:
    velocity = evaluate_velocity(
        model,
        dataset,
        validation_samples,
        device,
        batch_size=protocol.training.batch_size,
    )
    rollout = evaluate_rollouts(
        model,
        dataset,
        rollout_trajectories,
        device,
        batch_size=min(protocol.training.batch_size, 4),
        steps_per_interval=protocol.training.rollout_steps_per_interval,
    )
    return {
        "optimizer_steps": optimizer_steps,
        "validation_relative_loss": velocity.pop("relative_loss"),
        **velocity,
        **rollout,
        "budget_completed": optimizer_steps >= protocol.training.maximum_steps,
    }


def train_phase4_model(
    root: Path,
    protocol: Phase4Protocol,
    data_path: Path,
    model_name: ModelName,
    device: torch.device,
    *,
    resume: bool = False,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Train one frozen matched model and return its development-only result."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_all(protocol.training.seed)
    run_id = (
        f"chronopde_v2-p4-reaction_diffusion-exact_rhs-{model_name}-feasibility-"
        f"s{protocol.training.seed}-{protocol.digest[:8]}"
    )
    run_root = root / protocol.outputs.artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = run_root / "resolved_config.yaml"
    atomic_write_bytes(resolved_config, canonical_json_bytes(protocol.model_dump(mode="json")))
    manifest_path = run_root / "run_manifest.json"
    manifest = create_run_manifest(
        manifest_path,
        phase=4,
        pde="reaction_diffusion",
        task="exact_rhs",
        model=model_name,
        regime="feasibility",
        seed=protocol.training.seed,
        config_hash=protocol.digest,
        code_commit=code_commit or _git_commit(root),
        checkpoint_selection=protocol.checkpoint_selection,
        dataset_hash=protocol.development_dataset_sha256,
        normalization_hash=protocol.normalization_sha256,
    )
    if manifest.execution_status == ExecutionStatus.COMPLETED:
        return cast(dict[str, Any], json.loads((run_root / "summary.json").read_text()))
    if manifest.execution_status == ExecutionStatus.RUNNING and not resume:
        raise ValueError(f"run is already active; pass --resume: {run_id}")
    if manifest.execution_status == ExecutionStatus.PLANNED:
        transition_run_manifest(manifest_path, ExecutionStatus.RUNNING)
    identity = CheckpointIdentity(
        study_id="chronopde_v2",
        run_id=run_id,
        model_name=model_name,
        config_hash=protocol.digest,
        dataset_hash=protocol.development_dataset_sha256,
        normalization_hash=protocol.normalization_sha256,
    )
    metrics_path = run_root / "metrics.jsonl"
    model = build_phase4_model(protocol, model_name).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=protocol.training.learning_rate,
        weight_decay=protocol.training.weight_decay,
    )
    scheduler = _scheduler(optimizer, protocol)
    start_step = 0
    sampler_state: dict[str, Any] | None = None
    best_metric: float | None = None
    best_velocity: float | None = None
    best_selection: dict[str, Any] = {
        "criterion": protocol.checkpoint_selection,
        "selected_step": None,
    }
    if resume:
        payload = load_v2_checkpoint(
            run_root / "last.pt",
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_step = int(payload["optimizer_steps"])
        sampler_state = cast(dict[str, Any], payload["sampler_state"])
        best_metric = cast(float | None, payload["best_metric"])
        saved_selection = cast(dict[str, Any], payload["checkpoint_selection"])
        best_selection = saved_selection
        if saved_selection.get("velocity_nrmse") is not None:
            best_velocity = float(saved_selection["velocity_nrmse"])
    try:
        with DevelopmentExactRHSDataset(data_path, protocol) as dataset:
            sampler = ReplacementSampler(
                dataset.sample_count("training"), protocol.training.seed, sampler_state
            )
            validation_samples, rollout_trajectories = fixed_validation_indices(
                dataset.sample_count("validation"),
                int(dataset.handle["splits/validation/states"].shape[0]),
                protocol,
            )
            rows = _read_metrics(metrics_path)
            if resume:
                rows = [row for row in rows if int(row["optimizer_steps"]) <= start_step]
                atomic_write_bytes(
                    metrics_path,
                    b"".join(
                        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                        for row in rows
                    ),
                )
            if start_step == 0 and not rows:
                row = _evaluated_row(
                    model,
                    dataset,
                    validation_samples,
                    rollout_trajectories,
                    device,
                    protocol,
                    0,
                )
                row["learning_rate"] = optimizer.param_groups[0]["lr"]
                _append_metric(metrics_path, row)
                rows.append(row)
                save_v2_checkpoint(
                    run_root / "last.pt",
                    identity=identity,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=0,
                    optimizer_steps=0,
                    sampler_state=sampler.state_dict(),
                    best_metric=None,
                    patience_counter=0,
                    checkpoint_selection=best_selection,
                )
            for step in range(start_step + 1, protocol.training.maximum_steps + 1):
                model.train()
                batch = _move(
                    dataset._read_pairs(
                        "training", sampler.next(protocol.training.batch_size)
                    ),
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["state"], batch["time"], batch["parameters"])
                loss = full_field_relative_velocity_loss(prediction, batch["target"])
                torch.autograd.backward(loss)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), protocol.training.gradient_clip_norm
                    ).item()
                )
                optimizer.step()
                scheduler.step()
                should_evaluate = (
                    step % protocol.training.evaluation_interval == 0
                    or step == protocol.training.maximum_steps
                )
                if should_evaluate:
                    row = _evaluated_row(
                        model,
                        dataset,
                        validation_samples,
                        rollout_trajectories,
                        device,
                        protocol,
                        step,
                    )
                    row.update(
                        {
                            "training_relative_loss": float(loss.item()),
                            "gradient_norm": gradient_norm,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    _append_metric(metrics_path, row)
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "model": model_name,
                                "step": step,
                                "validation_relative_loss": row["validation_relative_loss"],
                                "velocity_nrmse": row["velocity_nrmse"],
                                "rollout_relative_l2": row["rollout_relative_l2"],
                                "persistence_relative_l2": row["persistence_relative_l2"],
                                "divergence_fraction": row["divergence_fraction"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    rollout_metric = row["rollout_relative_l2"]
                    candidate_rollout = (
                        float(rollout_metric) if rollout_metric is not None else None
                    )
                    candidate_velocity = float(row["velocity_nrmse"])
                    improved = candidate_rollout is not None and (
                        best_metric is None
                        or candidate_rollout < best_metric
                        or (
                            math.isclose(candidate_rollout, best_metric)
                            and (best_velocity is None or candidate_velocity < best_velocity)
                        )
                    )
                    if improved and candidate_rollout is not None:
                        best_metric = candidate_rollout
                        best_velocity = candidate_velocity
                        best_selection = {
                            "criterion": protocol.checkpoint_selection,
                            "selected_step": step,
                            "rollout_relative_l2": best_metric,
                            "velocity_nrmse": row["velocity_nrmse"],
                        }
                        save_v2_checkpoint(
                            run_root / "best.pt",
                            identity=identity,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=(step * protocol.training.batch_size)
                            // dataset.sample_count("training"),
                            optimizer_steps=step,
                            sampler_state=sampler.state_dict(),
                            best_metric=best_metric,
                            patience_counter=0,
                            checkpoint_selection=best_selection,
                        )
                if (
                    step % protocol.training.checkpoint_interval == 0
                    or step == protocol.training.maximum_steps
                ):
                    save_v2_checkpoint(
                        run_root / "last.pt",
                        identity=identity,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=(step * protocol.training.batch_size)
                        // dataset.sample_count("training"),
                        optimizer_steps=step,
                        sampler_state=sampler.state_dict(),
                        best_metric=best_metric,
                        patience_counter=0,
                        checkpoint_selection=best_selection,
                    )
        gate = phase4_gate(rows, protocol)
        summary = {
            "schema_version": 1,
            "study_id": protocol.study_id,
            "phase": 4,
            "run_id": run_id,
            "model": model_name,
            "development_only": True,
            "confirmatory_accessed": False,
            "optimizer_steps": protocol.training.maximum_steps,
            "gate": gate,
        }
        atomic_write_json(run_root / "summary.json", summary)
        transition_run_manifest(
            manifest_path,
            ExecutionStatus.COMPLETED,
            outcome=ScientificOutcome.PASSED if gate["passed"] else ScientificOutcome.FAILED,
        )
        create_recovery_package(
            run_root / "recovery.zip",
            run_manifest=manifest_path,
            resolved_config=resolved_config,
            metrics=metrics_path,
            checkpoint=run_root / "last.pt",
        )
        return summary
    except Exception as error:
        current = RunManifest.from_dict(
            cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        )
        if current.execution_status == ExecutionStatus.RUNNING:
            transition_run_manifest(
                manifest_path,
                ExecutionStatus.FAILED,
                outcome=ScientificOutcome.INCONCLUSIVE,
                failure_reason=f"{type(error).__name__}: {error}",
            )
        raise


def iter_progress(metrics_path: Path) -> Iterator[dict[str, Any]]:
    """Yield saved progress records for notebook-side status displays."""

    yield from _read_metrics(metrics_path)
