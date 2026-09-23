"""Phase 4B evaluation-only integration and instability audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from chronopde.evaluation.rollout import continuous_rollout
from chronopde.v2.checkpoints import CheckpointIdentity, load_v2_checkpoint, verify_checkpoint
from chronopde.v2.common import atomic_write_json, read_json_object, sha256_file
from chronopde.v2.phase4 import validate_phase4_dataset
from chronopde.v2.phase4_training import (
    DevelopmentExactRHSDataset,
    ModelName,
    build_phase4_model,
)
from chronopde.v2.protocol import (
    Phase4BProtocol,
    Phase4Protocol,
    load_phase4_protocol,
    load_phase4b_protocol,
)
from chronopde.v2.registry import ExecutionStatus, RunManifest, ScientificOutcome

MODEL_NAMES: tuple[ModelName, ModelName] = ("fft", "dct")


def _safe_archive(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Phase 4 archive contains duplicate members")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Phase 4 archive contains an unsafe path")
    if sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
        raise ValueError("Phase 4 archive expands beyond the declared size limit")


def _find_runs_root(root: Path) -> Path:
    direct = root / "artifacts/chronopde_v2/runs"
    if direct.is_dir():
        return direct
    matches = list(root.glob("**/artifacts/chronopde_v2/runs"))
    if len(matches) != 1:
        raise FileNotFoundError("could not identify one Phase 4 runs directory")
    return matches[0]


def prepare_phase4_results(
    source: Path, extraction_root: Path, protocol: Phase4BProtocol
) -> tuple[Path, str]:
    """Safely resolve either the original results ZIP or its Kaggle-expanded contents."""

    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError("Phase 4 results must be a ZIP or an extracted directory")
        digest = sha256_file(source)
        if digest != protocol.checkpoints.source_archive_sha256:
            raise ValueError("Phase 4 source archive SHA-256 mismatch")
        with zipfile.ZipFile(source) as archive:
            _safe_archive(archive)
            archive.extractall(extraction_root)
        return _find_runs_root(extraction_root), "archive_sha256_verified"
    if source.is_dir():
        return _find_runs_root(source), "expanded_contents_verified"
    raise FileNotFoundError(source)


def _model_run_directory(runs_root: Path, model_name: ModelName) -> Path:
    matches = list(runs_root.glob(f"*-{model_name}-feasibility-s0-f03756e7"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one Phase 4 {model_name} run, found {len(matches)}")
    return matches[0]


def validate_phase4_checkpoint(
    runs_root: Path,
    phase4: Phase4Protocol,
    phase4b: Phase4BProtocol,
    model_name: ModelName,
) -> tuple[Path, RunManifest]:
    run_root = _model_run_directory(runs_root, model_name)
    manifest = RunManifest.from_dict(read_json_object(run_root / "run_manifest.json"))
    expected_outcome = (
        ScientificOutcome.FAILED if model_name == "fft" else ScientificOutcome.PASSED
    )
    if manifest.execution_status != ExecutionStatus.COMPLETED:
        raise ValueError(f"Phase 4 {model_name} run is incomplete")
    if manifest.scientific_outcome != expected_outcome:
        raise ValueError(f"Phase 4 {model_name} outcome conflicts with frozen evidence")
    if manifest.code_commit != phase4b.checkpoints.source_code_commit:
        raise ValueError("Phase 4 checkpoint code commit mismatch")
    expected_identity = (
        phase4.digest,
        phase4b.development_dataset_sha256,
        phase4b.normalization_sha256,
        model_name,
    )
    actual_identity = (
        manifest.config_hash,
        manifest.dataset_hash,
        manifest.normalization_hash,
        manifest.model,
    )
    if actual_identity != expected_identity:
        raise ValueError(f"Phase 4 {model_name} run identity mismatch")
    if sha256_file(run_root / "resolved_config.yaml") != phase4.digest:
        raise ValueError(f"Phase 4 {model_name} resolved configuration mismatch")
    checkpoint = run_root / "best.pt"
    digest = verify_checkpoint(checkpoint)
    expected_digest = (
        phase4b.checkpoints.fft_best_sha256
        if model_name == "fft"
        else phase4b.checkpoints.dct_best_sha256
    )
    if digest != expected_digest:
        raise ValueError(f"Phase 4 {model_name} best checkpoint hash mismatch")
    expected_step = (
        phase4b.checkpoints.fft_best_step
        if model_name == "fft"
        else phase4b.checkpoints.dct_best_step
    )
    summary = read_json_object(run_root / "summary.json")
    if cast(dict[str, Any], summary["gate"])["selected_step"] != expected_step:
        raise ValueError(f"Phase 4 {model_name} selected step mismatch")
    return checkpoint, manifest


def load_frozen_model(
    checkpoint: Path,
    manifest: RunManifest,
    phase4: Phase4Protocol,
    device: torch.device,
) -> nn.Module:
    model_name = cast(ModelName, manifest.model)
    model = build_phase4_model(phase4, model_name)
    identity = CheckpointIdentity(
        study_id=manifest.study_id,
        run_id=manifest.run_id,
        model_name=model_name,
        config_hash=manifest.config_hash,
        dataset_hash=manifest.dataset_hash,
        normalization_hash=manifest.normalization_hash,
    )
    load_v2_checkpoint(
        checkpoint,
        expected_identity=identity,
        model=model,
        optimizer=None,
        scheduler=None,
        restore_rng=False,
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _correlation_horizon(prediction: Tensor, target: Tensor, times: Tensor) -> float:
    pred = prediction.flatten(1) - prediction.flatten(1).mean(dim=1, keepdim=True)
    truth = target.flatten(1) - target.flatten(1).mean(dim=1, keepdim=True)
    correlation = (pred * truth).sum(dim=1) / (
        pred.square().sum(dim=1).sqrt() * truth.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)
    failed = torch.nonzero(correlation < 0.9)
    index = int(failed[0].item()) if len(failed) else len(times) - 1
    return float(times[index].item())


def _rollout_rows(
    model: nn.Module,
    model_name: ModelName,
    dataset: DevelopmentExactRHSDataset,
    trajectory_indices: NDArray[np.int64],
    steps_per_interval: int,
    device: torch.device,
    *,
    batch_size: int = 4,
) -> tuple[list[dict[str, Any]], dict[int, Tensor]]:
    rows: list[dict[str, Any]] = []
    stable_predictions: dict[int, Tensor] = {}
    state_mean = torch.from_numpy(dataset.state_mean).to(device)[None, None, :, None, None]
    state_std = torch.from_numpy(dataset.state_std).to(device)[None, None, :, None, None]
    ids = dataset.handle["splits/validation/trajectory_ids"]

    def evaluate(indices: NDArray[np.int64]) -> None:
        batch = {name: value.to(device) for name, value in dataset.read_rollouts(indices).items()}
        with torch.inference_mode():
            integration = continuous_rollout(
                model,
                batch["states"][:, 0],
                batch["times"],
                batch["parameters"],
                steps_per_interval=steps_per_interval,
                method="rk4",
            )
            physical = cast(Tensor, integration.states) * state_std + state_mean
        for position, raw_index in enumerate(indices.tolist()):
            index = int(raw_index)
            prediction = physical[position]
            target = batch["physical_states"][position]
            finite = bool(torch.isfinite(prediction).all())
            maximum = float(prediction.abs().max().item()) if finite else None
            stable = finite and cast(float, maximum) <= 10.0
            trajectory_id_raw = ids[index]
            trajectory_id = (
                trajectory_id_raw.decode("utf-8")
                if isinstance(trajectory_id_raw, bytes)
                else str(trajectory_id_raw)
            )
            row: dict[str, Any] = {
                "model": model_name,
                "steps_per_interval": steps_per_interval,
                "trajectory_index": index,
                "trajectory_id": trajectory_id,
                "stable": stable,
                "maximum_absolute_state": maximum,
                "rollout_relative_l2": None,
                "final_time_relative_l2": None,
                "persistence_relative_l2": None,
                "correlation_horizon": None,
            }
            if finite:
                row["rollout_relative_l2"] = float(
                    ((prediction - target).flatten().norm() / target.flatten().norm()).item()
                )
                row["final_time_relative_l2"] = float(
                    (
                        (prediction[-1] - target[-1]).flatten().norm()
                        / target[-1].flatten().norm()
                    ).item()
                )
                persistence = target[:1].expand_as(target)
                row["persistence_relative_l2"] = float(
                    ((persistence - target).flatten().norm() / target.flatten().norm()).item()
                )
                row["correlation_horizon"] = _correlation_horizon(
                    prediction, target, batch["times"][position]
                )
            if stable:
                stable_predictions[index] = prediction.detach().cpu()
            rows.append(row)

    for start in range(0, len(trajectory_indices), batch_size):
        indices = trajectory_indices[start : start + batch_size]
        try:
            evaluate(indices)
        except (RuntimeError, ValueError):
            for index in indices:
                try:
                    evaluate(np.asarray([index], dtype=np.int64))
                except (RuntimeError, ValueError):
                    raw = ids[int(index)]
                    rows.append(
                        {
                            "model": model_name,
                            "steps_per_interval": steps_per_interval,
                            "trajectory_index": int(index),
                            "trajectory_id": raw.decode("utf-8")
                            if isinstance(raw, bytes)
                            else str(raw),
                            "stable": False,
                            "maximum_absolute_state": None,
                            "rollout_relative_l2": None,
                            "final_time_relative_l2": None,
                            "persistence_relative_l2": None,
                            "correlation_horizon": None,
                        }
                    )
    return rows, stable_predictions


def _aggregate_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in rows if row["stable"]]
    finite = [row for row in rows if row["rollout_relative_l2"] is not None]

    def stable_median(name: str) -> float | None:
        values = [float(row[name]) for row in stable if row[name] is not None]
        return median(values) if values else None

    persistence_values = [
        float(row["persistence_relative_l2"])
        for row in finite
        if row["persistence_relative_l2"] is not None
    ]
    return {
        "model": rows[0]["model"],
        "steps_per_interval": rows[0]["steps_per_interval"],
        "trajectories": len(rows),
        "stable_trajectories": len(stable),
        "divergence_fraction": 1 - len(stable) / len(rows),
        "rollout_relative_l2": stable_median("rollout_relative_l2"),
        "final_time_relative_l2": stable_median("final_time_relative_l2"),
        "correlation_horizon": stable_median("correlation_horizon"),
        "persistence_relative_l2_all": median(persistence_values)
        if persistence_values
        else None,
    }


def _velocity_attribution(
    model: nn.Module,
    model_name: ModelName,
    dataset: DevelopmentExactRHSDataset,
    trajectory_indices: NDArray[np.int64],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    time_count = int(dataset.handle["splits/validation/states"].shape[1])
    ids = dataset.handle["splits/validation/trajectory_ids"]
    for raw_index in trajectory_indices.tolist():
        index = int(raw_index)
        sample_indices = np.arange(index * time_count, (index + 1) * time_count, dtype=np.int64)
        total_sse = total_energy = 0.0
        boundary_sse = boundary_energy = 0.0
        interior_sse = interior_energy = 0.0
        derivative_sse = derivative_energy = 0.0
        for start in range(0, time_count, batch_size):
            batch = {
                name: value.to(device)
                for name, value in dataset._read_pairs(
                    "validation", sample_indices[start : start + batch_size]
                ).items()
            }
            with torch.inference_mode():
                prediction = model(batch["state"], batch["time"], batch["parameters"])
            target = batch["target"]
            error = prediction - target
            total_sse += float(error.square().sum())
            total_energy += float(target.square().sum())
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[..., :4, :] = True
            mask[..., -4:, :] = True
            mask[..., :, :4] = True
            mask[..., :, -4:] = True
            boundary_sse += float(error[mask].square().sum())
            boundary_energy += float(target[mask].square().sum())
            interior_sse += float(error[~mask].square().sum())
            interior_energy += float(target[~mask].square().sum())
            pred_derivative = torch.cat(
                (
                    prediction[..., 1, :] - prediction[..., 0, :],
                    prediction[..., -1, :] - prediction[..., -2, :],
                    prediction[..., :, 1] - prediction[..., :, 0],
                    prediction[..., :, -1] - prediction[..., :, -2],
                ),
                dim=-1,
            )
            target_derivative = torch.cat(
                (
                    target[..., 1, :] - target[..., 0, :],
                    target[..., -1, :] - target[..., -2, :],
                    target[..., :, 1] - target[..., :, 0],
                    target[..., :, -1] - target[..., :, -2],
                ),
                dim=-1,
            )
            derivative_sse += float((pred_derivative - target_derivative).square().sum())
            derivative_energy += float(target_derivative.square().sum())
        raw_id = ids[index]
        rows.append(
            {
                "model": model_name,
                "trajectory_index": index,
                "trajectory_id": raw_id.decode("utf-8")
                if isinstance(raw_id, bytes)
                else str(raw_id),
                "velocity_relative_l2": math.sqrt(total_sse / max(total_energy, 1e-12)),
                "boundary_strip_relative_l2": math.sqrt(
                    boundary_sse / max(boundary_energy, 1e-12)
                ),
                "interior_relative_l2": math.sqrt(
                    interior_sse / max(interior_energy, 1e-12)
                ),
                "first_interior_normal_derivative_relative_l2": math.sqrt(
                    derivative_sse / max(derivative_energy, 1e-12)
                ),
            }
        )
    return rows


def _convergence_rows(
    model_name: ModelName,
    coarse: dict[int, Tensor],
    fine: dict[int, Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for index in sorted(set(coarse) | set(fine)):
        comparable = index in coarse and index in fine
        relative = None
        if comparable:
            difference = (coarse[index] - fine[index]).flatten().norm()
            reference = fine[index].flatten().norm()
            relative = float(
                (difference / reference).item()
            )
        rows.append(
            {
                "model": model_name,
                "trajectory_index": index,
                "coarse_steps_per_interval": 4,
                "fine_steps_per_interval": 8,
                "comparable": comparable,
                "prediction_relative_l2": relative,
            }
        )
    return rows


def phase4b_decision(
    aggregates: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
    protocol: Phase4BProtocol,
) -> dict[str, Any]:
    by_model = {
        model: {int(row["steps_per_interval"]): row for row in aggregates if row["model"] == model}
        for model in ("fft", "dct")
    }
    classifications: dict[str, Any] = {}
    for model in ("fft", "dct"):
        rows = by_model[model]
        values = [
            float(row["prediction_relative_l2"])
            for row in convergence
            if row["model"] == model and row["prediction_relative_l2"] is not None
        ]
        p95 = float(np.percentile(values, 95)) if values else None
        converged = (
            p95 is not None
            and len(values) == len(protocol.validation_trajectory_indices)
            and p95 <= protocol.integrator.maximum_convergence_relative_l2_p95
        )
        divergence_one = float(rows[1]["divergence_fraction"])
        divergence_eight = float(rows[8]["divergence_fraction"])
        rollout_one = rows[1]["rollout_relative_l2"]
        rollout_eight = rows[8]["rollout_relative_l2"]
        improved = (
            rollout_one is not None
            and rollout_eight is not None
            and float(rollout_eight)
            <= (1 - protocol.integrator.material_rollout_improvement_fraction)
            * float(rollout_one)
        )
        if divergence_eight > 0:
            label = "learned_vector_field_instability"
        elif divergence_one > 0 and improved:
            label = "integration_limited_at_one_step"
        elif all(float(rows[step]["divergence_fraction"]) == 0 for step in rows):
            label = "stable_across_refinement"
        else:
            label = "refinement_sensitive"
        classifications[model] = {
            "classification": label,
            "converged_4_vs_8": converged,
            "convergence_relative_l2_p95": p95,
            "divergence_fraction_step_1": divergence_one,
            "divergence_fraction_step_8": divergence_eight,
            "material_rollout_improvement": improved,
        }
    dct = classifications["dct"]
    fft = classifications["fft"]
    if dct["classification"] != "stable_across_refinement" or not dct["converged_4_vs_8"]:
        decision = "phase4b_inconclusive_dct_evaluation_control_failed"
        phase5_allowed = False
    elif fft["classification"] == "learned_vector_field_instability":
        decision = "phase4b_complete_phase5_allowed_with_fft_instability_label"
        phase5_allowed = True
    elif fft["converged_4_vs_8"]:
        decision = "phase4b_complete_phase5_allowed_with_eight_step_integrator"
        phase5_allowed = True
    else:
        decision = "phase4b_inconclusive_additional_integration_audit_required"
        phase5_allowed = False
    return {
        "decision": decision,
        "phase5_multiseed_development_allowed": phase5_allowed,
        "classifications": classifications,
        "frozen_future_steps_per_interval": 8 if phase5_allowed else None,
        "confirmatory_accessed": False,
        "superiority_claim_authorized": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evidence table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_audit(path: Path, aggregates: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for model, color in (("fft", "tab:blue"), ("dct", "tab:orange")):
        rows = sorted(
            (row for row in aggregates if row["model"] == model),
            key=lambda row: int(row["steps_per_interval"]),
        )
        steps = [int(row["steps_per_interval"]) for row in rows]
        rollout = [row["rollout_relative_l2"] for row in rows]
        divergence = [row["divergence_fraction"] for row in rows]
        axes[0].plot(steps, rollout, marker="o", label=model.upper(), color=color)
        axes[1].plot(steps, divergence, marker="o", label=model.upper(), color=color)
    axes[0].set(xlabel="RK4 steps per stored interval", ylabel="median rollout relative L2")
    axes[1].set(
        xlabel="RK4 steps per stored interval",
        ylabel="divergence fraction",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _package(output: Path, files: list[Path], root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        _safe_archive(archive)
        if archive.testzip() is not None:
            raise ValueError("Phase 4B output package failed CRC validation")


def run_phase4b(
    root: Path,
    protocol: Phase4BProtocol,
    data_path: Path,
    phase4_results: Path,
    device: torch.device,
    extraction_root: Path,
) -> dict[str, Any]:
    phase4 = load_phase4_protocol(root / protocol.phase4_descriptor)
    if phase4.digest != protocol.phase4_protocol_sha256:
        raise ValueError("Phase 4B references the wrong Phase 4 protocol")
    validate_phase4_dataset(root, phase4, data_path)
    runs_root, source_verification = prepare_phase4_results(
        phase4_results, extraction_root, protocol
    )
    validated = {
        model: validate_phase4_checkpoint(runs_root, phase4, protocol, model)
        for model in MODEL_NAMES
    }
    run_id = f"chronopde_v2-p4b-reaction_diffusion-integration-audit-s0-{protocol.digest[:8]}"
    artifact_root = root / protocol.outputs.artifact_root / run_id
    report_root = root / protocol.outputs.report_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    trajectory_indices = np.asarray(protocol.validation_trajectory_indices, dtype=np.int64)
    rollout_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    with DevelopmentExactRHSDataset(data_path, phase4) as dataset:
        for model_name in MODEL_NAMES:
            checkpoint, manifest = validated[model_name]
            model = load_frozen_model(checkpoint, manifest, phase4, device)
            predictions: dict[int, dict[int, Tensor]] = {}
            for steps in protocol.integrator.steps_per_interval:
                rows, stable_predictions = _rollout_rows(
                    model,
                    model_name,
                    dataset,
                    trajectory_indices,
                    steps,
                    device,
                )
                rollout_rows.extend(rows)
                aggregate = _aggregate_rollouts(rows)
                aggregate_rows.append(aggregate)
                if steps in protocol.integrator.comparison_pair:
                    predictions[steps] = stable_predictions
                print(json.dumps(aggregate, sort_keys=True), flush=True)
            velocity_rows.extend(
                _velocity_attribution(model, model_name, dataset, trajectory_indices, device)
            )
            convergence_rows.extend(
                _convergence_rows(
                    model_name,
                    predictions[protocol.integrator.comparison_pair[0]],
                    predictions[protocol.integrator.comparison_pair[1]],
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    decision = phase4b_decision(aggregate_rows, convergence_rows, protocol)
    report = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "phase": protocol.phase,
        "protocol_sha256": protocol.digest,
        "source_verification": source_verification,
        "training_performed": False,
        "checkpoint_modified": False,
        "development_only": True,
        "aggregates": aggregate_rows,
        **decision,
    }
    paths = {
        "decision": report_root / "decision_report.json",
        "rollouts": report_root / "per_trajectory_rollouts.csv",
        "velocity": report_root / "per_trajectory_velocity.csv",
        "convergence": report_root / "integration_convergence.csv",
        "plot": report_root / "integration_audit.png",
        "protocol": report_root / "protocol_snapshot.json",
    }
    atomic_write_json(paths["decision"], report)
    atomic_write_json(paths["protocol"], protocol.model_dump(mode="json"))
    _write_csv(paths["rollouts"], rollout_rows)
    _write_csv(paths["velocity"], velocity_rows)
    _write_csv(paths["convergence"], convergence_rows)
    _plot_audit(paths["plot"], aggregate_rows)
    atomic_write_json(artifact_root / "runtime_summary.json", report)
    package_files = [*paths.values(), artifact_root / "runtime_summary.json"]
    _package(artifact_root / "chronopde_v2_phase4b_outputs.zip", package_files, root)
    return report


def _discover(root: Path, filename: str) -> Path | None:
    local = list(root.glob(f"artifacts/**/{filename}"))
    kaggle = Path("/kaggle/input")
    remote = list(kaggle.rglob(filename)) if kaggle.is_dir() else []
    matches = [path for path in [*local, *remote] if path.is_file()]
    return matches[0] if len(matches) == 1 else None


def _discover_results() -> Path | None:
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        archives = list(kaggle.rglob("chronopde_v2_phase4_outputs.zip"))
        if len(archives) == 1:
            return archives[0]
        roots = [path for path in kaggle.iterdir() if path.is_dir()]
        containing = [root for root in roots if list(root.glob("**/*-fft-feasibility-*"))]
        if len(containing) == 1:
            return containing[0]
    return None


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 4B execution requires CUDA; use --check-only locally")
        return torch.device("cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase4b",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/chronopde_v2/phase4b.yaml")
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--phase4-results", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        protocol = load_phase4b_protocol(config_path.resolve())
        phase4 = load_phase4_protocol(root / protocol.phase4_descriptor)
        if phase4.digest != protocol.phase4_protocol_sha256:
            raise ValueError("Phase 4 protocol hash mismatch")
        data_path = args.data_path or _discover(root, "chronopde_v2_development.h5")
        results_path = args.phase4_results or _discover_results()
        if data_path is None or results_path is None:
            raise FileNotFoundError("attach/pass both Phase 3 data and Phase 4 results")
        temporary_parent = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else None
        with tempfile.TemporaryDirectory(prefix="chronopde-phase4b-", dir=temporary_parent) as tmp:
            extraction = Path(tmp)
            if args.check_only:
                validate_phase4_dataset(root, phase4, data_path.resolve())
                runs_root, source_verification = prepare_phase4_results(
                    results_path.resolve(), extraction, protocol
                )
                checkpoints = {
                    model: str(
                        validate_phase4_checkpoint(runs_root, phase4, protocol, model)[0]
                    )
                    for model in MODEL_NAMES
                }
                report: dict[str, Any] = {
                    "passed": True,
                    "protocol_sha256": protocol.digest,
                    "source_verification": source_verification,
                    "checkpoints": checkpoints,
                    "training_performed": False,
                    "confirmatory_accessed": False,
                }
            else:
                report = run_phase4b(
                    root,
                    protocol,
                    data_path.resolve(),
                    results_path.resolve(),
                    _resolve_device(args.device),
                    extraction,
                )
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        print(str(error))
        return 2
    except RuntimeError as error:
        print(str(error))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
