"""Inference-only audit of the Week 6 fixed-batch mechanics checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
import yaml
from torch import Tensor, nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronopde.config import ProjectConfig
from chronopde.data.datasets import HDF5VelocityDataset, collate_velocity_samples
from chronopde.diagnostics.continuous_gate import SMOKE_IDS, _model_for_diagnostic, _write_json
from chronopde.numerics import dct2


@dataclass(frozen=True)
class CheckpointAudit:
    run: str
    checkpoint: str
    checkpoint_step: int
    checkpoint_sha256: str
    recorded_step_found: bool
    replay_within_tolerance: bool
    median_nrmse: float
    pooled_nrmse: float
    median_nrmse_u: float
    median_nrmse_v: float
    physical_pooled_nrmse: float
    normalized_rmse: float
    physical_rmse: float
    spectral_relative_error: float
    denominator_clamp_count: int


@dataclass(frozen=True)
class VelocityAuditReport:
    audit_completed: bool
    original_gate_passed: bool
    route: str
    checkpoint_count: int
    exact_batch_size: int
    exact_batch_trajectory_ids: list[str]
    all_fixed_sample_count: int
    normalization_sha256: str
    pooled_metric_sensitivity: bool
    output_directory: str
    checkpoints: list[CheckpointAudit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensors: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _quantiles(values: Tensor) -> dict[str, float]:
    values = values.double().cpu()
    return {
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(torch.median(values)),
        "p90": float(torch.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def velocity_audit_metrics(
    prediction: Tensor,
    target: Tensor,
    state_std: Tensor,
    *,
    modes: int = 12,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Return explicit sample, pooled, channel, physical, and spectral error metrics."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("velocity tensors must have matching [B,C,H,W] shapes")
    if state_std.shape != (prediction.shape[1],):
        raise ValueError("state_std must contain one value per channel")
    error = (prediction - target).double()
    target64 = target.double()
    sample_sse = error.square().sum(dim=(1, 2, 3))
    sample_energy = target64.square().sum(dim=(1, 2, 3))
    channel_sse = error.square().sum(dim=(2, 3))
    channel_energy = target64.square().sum(dim=(2, 3))
    sample_count = target[0].numel()
    clamp_threshold = sample_count * 1e-24
    sample_nrmse = torch.sqrt(sample_sse / sample_energy.clamp_min(clamp_threshold))
    channel_threshold = target.shape[-1] * target.shape[-2] * 1e-24
    channel_nrmse = torch.sqrt(channel_sse / channel_energy.clamp_min(channel_threshold))
    scale = state_std.double()[None, :, None, None]
    physical_error = error * scale
    physical_target = target64 * scale
    physical_sse = physical_error.square().sum()
    physical_energy = physical_target.square().sum()
    physical_sample_sse = physical_error.square().sum(dim=(1, 2, 3))
    physical_sample_energy = physical_target.square().sum(dim=(1, 2, 3))
    physical_channel_sse = physical_error.square().sum(dim=(2, 3))
    physical_channel_energy = physical_target.square().sum(dim=(2, 3))
    physical_sample_nrmse = torch.sqrt(
        physical_sample_sse / physical_sample_energy.clamp_min(clamp_threshold)
    )
    physical_channel_nrmse = torch.sqrt(
        physical_channel_sse / physical_channel_energy.clamp_min(channel_threshold)
    )
    predicted_coefficients = dct2(prediction.float())[..., :modes, :modes].double()
    target_coefficients = dct2(target.float())[..., :modes, :modes].double()
    spectral_numerator = (predicted_coefficients - target_coefficients).square().sum(
        dim=(1, 2, 3)
    )
    spectral_denominator = target_coefficients.square().sum(dim=(1, 2, 3))
    spectral_relative = spectral_numerator / spectral_denominator.clamp_min(1e-8)
    pooled = torch.sqrt(sample_sse.sum() / sample_energy.sum().clamp_min(clamp_threshold))
    physical_pooled = torch.sqrt(physical_sse / physical_energy.clamp_min(clamp_threshold))
    energy_quartiles = torch.quantile(
        sample_energy, sample_energy.new_tensor([0.25, 0.5, 0.75])
    )
    quartile = torch.bucketize(sample_energy, energy_quartiles)
    rows: list[dict[str, float | int]] = []
    for index in range(len(target)):
        rows.append(
            {
                "sample_index": index,
                "target_energy": float(sample_energy[index]),
                "normalized_nrmse": float(sample_nrmse[index]),
                "normalized_nrmse_u": float(channel_nrmse[index, 0]),
                "normalized_nrmse_v": float(channel_nrmse[index, 1]),
                "normalized_rmse": float(torch.sqrt(sample_sse[index] / sample_count)),
                "physical_nrmse": float(physical_sample_nrmse[index]),
                "physical_nrmse_u": float(physical_channel_nrmse[index, 0]),
                "physical_nrmse_v": float(physical_channel_nrmse[index, 1]),
                "physical_rmse": float(
                    torch.sqrt(physical_sample_sse[index] / sample_count)
                ),
                "spectral_relative_error": float(spectral_relative[index]),
                "target_energy_quartile": int(quartile[index]) + 1,
            }
        )
    quartile_summary = {}
    for index in range(4):
        selected = quartile == index
        quartile_summary[str(index + 1)] = {
            "samples": int(selected.sum()),
            "median_nrmse": float(torch.median(sample_nrmse[selected])),
            "error_energy_fraction": float(
                sample_sse[selected].sum() / sample_sse.sum().clamp_min(1e-30)
            ),
            "target_energy_fraction": float(
                sample_energy[selected].sum() / sample_energy.sum().clamp_min(1e-30)
            ),
        }
    metrics: dict[str, Any] = {
        "median_nrmse": float(torch.median(sample_nrmse)),
        "pooled_nrmse": float(pooled),
        "median_nrmse_u": float(torch.median(channel_nrmse[:, 0])),
        "median_nrmse_v": float(torch.median(channel_nrmse[:, 1])),
        "physical_pooled_nrmse": float(physical_pooled),
        "normalized_rmse": float(torch.sqrt(sample_sse.sum() / target.numel())),
        "physical_rmse": float(torch.sqrt(physical_sse / target.numel())),
        "spectral_relative_error": float(torch.mean(spectral_relative)),
        "denominator_clamp_count": int((sample_energy <= clamp_threshold).sum()),
        "sample_nrmse_distribution": _quantiles(sample_nrmse),
        "target_energy_distribution": _quantiles(sample_energy),
        "target_energy_quartiles": quartile_summary,
    }
    return metrics, rows


def _load_checkpoint_model(
    config: ProjectConfig, path: Path, model_name: str
) -> tuple[nn.Module, dict[str, Any]]:
    if model_name not in {"chronopde", "fno_ct"}:
        raise ValueError(f"unsupported mechanics model: {model_name}")
    # These are project-generated checkpoints and include NumPy RNG objects, so the
    # weights-only loader cannot deserialize their existing format.
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if payload.get("model_name") != model_name:
        raise ValueError(f"checkpoint model mismatch in {path}")
    model = _model_for_diagnostic(config, cast(Any, model_name)).cpu()
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _find_runs(mechanics_root: Path) -> list[Path]:
    runs = []
    for protocol in mechanics_root.rglob("diagnostic_protocol.json"):
        directory = protocol.parent
        if (directory / "summary.json").is_file() and any(
            (directory / name).is_file() for name in ("best.pt", "last.pt")
        ):
            runs.append(directory)
    unique = {path.resolve(): path for path in runs}
    return sorted(unique.values(), key=lambda path: str(path))


def run_velocity_checkpoint_audit(
    config: ProjectConfig,
    data_path: Path,
    mechanics_root: Path,
    output: Path,
) -> VelocityAuditReport:
    """Replay saved mechanics checkpoints on the exact fixed batch, without training."""
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
        samples = [dataset[index] for index in range(len(dataset))]
        exact = samples[:16]
        batch = collate_velocity_samples(exact)
        state = cast(Tensor, batch["state"])
        target = cast(Tensor, batch["target_velocity"])
        time = cast(Tensor, batch["time"])
        parameters = cast(Tensor, batch["parameters"])
        output.mkdir(parents=True, exist_ok=True)
        sample_manifest: list[dict[str, Any]] = []
        for index, sample in enumerate(exact):
            sample_manifest.append(
                {
                    "sample_index": index,
                    "trajectory_id": sample.trajectory_id,
                    "interval_index": sample.interval_index,
                    "time": float(sample.time),
                    "sample_sha256": _tensor_digest(
                        [sample.state, sample.target_velocity, sample.time, sample.parameters]
                    ),
                }
            )
        _write_json(output / "exact_batch_manifest.json", sample_manifest)
        all_energy = torch.tensor(
            [float(sample.target_velocity.double().square().sum()) for sample in samples]
        )
        target_report = {
            "samples": len(samples),
            "trajectory_counts": {
                trajectory_id: sum(s.trajectory_id == trajectory_id for s in samples)
                for trajectory_id in SMOKE_IDS
            },
            "target_energy_distribution": _quantiles(all_energy),
            "exact_batch_target_energy_distribution": _quantiles(all_energy[:16]),
        }
        _write_json(output / "fixed_target_distribution.json", target_report)
        normalization_hash = _tensor_digest(
            [
                dataset.normalization.state_mean,
                dataset.normalization.state_std,
                dataset.normalization.parameter_mean,
                dataset.normalization.parameter_std,
            ]
        )
        audits: list[CheckpointAudit] = []
        sample_rows: list[dict[str, Any]] = []
        for run_directory in _find_runs(mechanics_root):
            protocol = json.loads((run_directory / "diagnostic_protocol.json").read_text())
            summary = json.loads((run_directory / "summary.json").read_text())
            model_name = str(summary["model"])
            modes = 12
            metrics_rows = [
                json.loads(line)
                for line in (run_directory / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            rows_by_step = {int(row["optimizer_steps"]): row for row in metrics_rows}
            for checkpoint_name in ("best.pt", "best_velocity.pt", "last.pt"):
                checkpoint_path = run_directory / checkpoint_name
                if not checkpoint_path.is_file():
                    continue
                model, payload = _load_checkpoint_model(config, checkpoint_path, model_name)
                with torch.no_grad():
                    prediction = model(state, time, parameters)
                metrics, per_sample = velocity_audit_metrics(
                    prediction, target, dataset.normalization.state_std, modes=modes
                )
                step = int(payload["optimizer_steps"])
                recorded = rows_by_step.get(step)
                replay_ok = False
                if recorded is not None:
                    replay_ok = bool(
                        np.isclose(
                            metrics["median_nrmse"],
                            float(recorded["velocity_nrmse"]),
                            rtol=1e-4,
                            atol=1e-7,
                        )
                    )
                audit = CheckpointAudit(
                    run=run_directory.name,
                    checkpoint=checkpoint_name,
                    checkpoint_step=step,
                    checkpoint_sha256=_sha256(checkpoint_path),
                    recorded_step_found=recorded is not None,
                    replay_within_tolerance=replay_ok,
                    median_nrmse=metrics["median_nrmse"],
                    pooled_nrmse=metrics["pooled_nrmse"],
                    median_nrmse_u=metrics["median_nrmse_u"],
                    median_nrmse_v=metrics["median_nrmse_v"],
                    physical_pooled_nrmse=metrics["physical_pooled_nrmse"],
                    normalized_rmse=metrics["normalized_rmse"],
                    physical_rmse=metrics["physical_rmse"],
                    spectral_relative_error=metrics["spectral_relative_error"],
                    denominator_clamp_count=metrics["denominator_clamp_count"],
                )
                audits.append(audit)
                _write_json(
                    output / f"{run_directory.name}-{checkpoint_name}.json",
                    {**asdict(audit), **metrics, "protocol": protocol},
                )
                for row, manifest in zip(per_sample, sample_manifest, strict=True):
                    sample_rows.append(
                        {
                            "run": run_directory.name,
                            "checkpoint": checkpoint_name,
                            **manifest,
                            **row,
                        }
                    )
        if not audits:
            raise FileNotFoundError(f"no mechanics checkpoints found under {mechanics_root}")
        with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        labels = [f"{item.run}\n{item.checkpoint}" for item in audits]
        positions = np.arange(len(audits))
        axis.plot(positions, [item.median_nrmse for item in audits], "o", label="median")
        axis.plot(positions, [item.pooled_nrmse for item in audits], "x", label="pooled")
        axis.axhline(0.01, color="tab:red", linestyle="--", label="original gate")
        axis.set_xticks(positions, labels, rotation=60, ha="right")
        axis.set(ylabel="velocity nRMSE", title="Checkpoint metric sensitivity")
        axis.legend()
        figure.savefig(output / "checkpoint_metric_comparison.png", dpi=160)
        plt.close(figure)
        replay_passed = all(item.replay_within_tolerance for item in audits)
        sensitivity = any(
            item.pooled_nrmse <= 0.01 < item.median_nrmse for item in audits
        )
        report = VelocityAuditReport(
            audit_completed=replay_passed,
            original_gate_passed=False,
            route=(
                "review_metric_sensitivity_without_gate_change"
                if replay_passed and sensitivity
                else "audit_complete_no_gate_change"
                if replay_passed
                else "replay_mismatch"
            ),
            checkpoint_count=len(audits),
            exact_batch_size=16,
            exact_batch_trajectory_ids=sorted({sample.trajectory_id for sample in exact}),
            all_fixed_sample_count=len(samples),
            normalization_sha256=normalization_hash,
            pooled_metric_sensitivity=sensitivity,
            output_directory=str(output),
            checkpoints=audits,
        )
        _write_json(output / "summary.json", asdict(report))
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )
        return report
    finally:
        dataset.close()
