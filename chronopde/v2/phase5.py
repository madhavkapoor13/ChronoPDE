"""Phase 5 multi-seed development training and model-freeze decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from chronopde.models import DCTContinuousVectorField, FFTContinuousVectorField
from chronopde.models.common import trainable_parameter_count
from chronopde.training.losses import full_field_relative_velocity_loss
from chronopde.v2.checkpoints import CheckpointIdentity, load_v2_checkpoint, save_v2_checkpoint
from chronopde.v2.common import atomic_write_bytes, atomic_write_json, canonical_json_bytes
from chronopde.v2.phase4 import validate_phase4_dataset
from chronopde.v2.phase4_training import (
    DevelopmentExactRHSDataset,
    ModelName,
    ReplacementSampler,
    evaluate_velocity,
    fixed_validation_indices,
)
from chronopde.v2.phase4b import _aggregate_rollouts, _rollout_rows
from chronopde.v2.protocol import (
    Phase4Protocol,
    Phase5Protocol,
    load_phase4_protocol,
    load_phase4b_protocol,
    load_phase5_protocol,
)
from chronopde.v2.recovery import create_recovery_package
from chronopde.v2.registry import (
    ExecutionStatus,
    RunManifest,
    ScientificOutcome,
    create_run_manifest,
    transition_run_manifest,
)

MODEL_NAMES: tuple[ModelName, ModelName] = ("fft", "dct")


def build_phase5_model(protocol: Phase5Protocol, model_name: ModelName) -> nn.Module:
    if model_name == "fft":
        return FFTContinuousVectorField(
            width=protocol.models.width,
            modes_y=protocol.models.fft_modes[0],
            modes_x=protocol.models.fft_modes[1],
            blocks=protocol.models.blocks,
            film_hidden_width=protocol.models.film_hidden_width,
            time_range=(0.0, 50.0),
        )
    return DCTContinuousVectorField(
        width=protocol.models.width,
        modes_y=protocol.models.dct_modes[0],
        modes_x=protocol.models.dct_modes[1],
        blocks=protocol.models.blocks,
        film_hidden_width=protocol.models.film_hidden_width,
        time_range=(0.0, 50.0),
        residual_skip=protocol.models.residual_skip,
    )


def validate_phase5_contract(root: Path, protocol: Phase5Protocol) -> dict[str, Any]:
    phase4 = load_phase4_protocol(root / protocol.phase4_descriptor)
    phase4b = load_phase4b_protocol(root / protocol.phase4b_descriptor)
    if phase4b.digest != protocol.phase4b_protocol_sha256:
        raise ValueError("Phase 5 references the wrong Phase 4B protocol")
    evidence = json.loads(
        (root / "reports/chronopde_v2/phase4b/result_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if evidence.get("output_archive_sha256") != protocol.phase4b_output_sha256:
        raise ValueError("Phase 5 Phase 4B evidence hash mismatch")
    if not evidence.get("phase5_multiseed_development_allowed"):
        raise ValueError("Phase 4B did not authorize Phase 5")
    if evidence.get("frozen_future_steps_per_interval") != 8:
        raise ValueError("Phase 4B did not freeze the required integrator")
    counts = {
        model: trainable_parameter_count(build_phase5_model(protocol, model))
        for model in MODEL_NAMES
    }
    if counts != {"fft": 1_973_657, "dct": 1_973_657}:
        raise ValueError(f"Phase 5 comparator identity changed: {counts}")
    return {
        "passed": True,
        "phase4": phase4,
        "phase4b_protocol_sha256": phase4b.digest,
        "parameter_counts": counts,
        "seeds": list(protocol.training.seeds),
        "rollout_steps_per_interval": protocol.training.rollout_steps_per_interval,
        "confirmatory_access_allowed": False,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scheduler(optimizer: AdamW, protocol: Phase5Protocol) -> LambdaLR:
    training = protocol.training
    floor = training.minimum_learning_rate / training.learning_rate

    def factor(step: int) -> float:
        if step < training.warmup_steps:
            return max((step + 1) / training.warmup_steps, floor)
        progress = (step - training.warmup_steps) / (
            training.maximum_steps - training.warmup_steps
        )
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return LambdaLR(optimizer, factor)


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_bytes(
        path,
        b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            for row in rows
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evidence table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_id(protocol: Phase5Protocol, model_name: ModelName, seed: int) -> str:
    return (
        f"chronopde_v2-p5-reaction_diffusion-exact_rhs-{model_name}-multiseed-"
        f"s{seed}-{protocol.digest[:8]}"
    )


def _save_checkpoint(
    path: Path,
    *,
    identity: CheckpointIdentity,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    sample_count: int,
    batch_size: int,
    sampler: ReplacementSampler,
    best_metric: float | None,
    selection: dict[str, Any],
) -> None:
    save_v2_checkpoint(
        path,
        identity=identity,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=(step * batch_size) // sample_count,
        optimizer_steps=step,
        sampler_state=sampler.state_dict(),
        best_metric=best_metric,
        patience_counter=0,
        checkpoint_selection=selection,
    )


def train_phase5_run(
    root: Path,
    protocol: Phase5Protocol,
    phase4: Phase4Protocol,
    data_path: Path,
    model_name: ModelName,
    seed: int,
    device: torch.device,
    *,
    resume: bool = False,
    code_commit: str | None = None,
) -> dict[str, Any]:
    if seed not in protocol.training.seeds:
        raise ValueError(f"seed {seed} is outside the frozen Phase 5 seed set")
    _seed_all(seed)
    run_id = _run_id(protocol, model_name, seed)
    run_root = root / protocol.outputs.artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = run_root / "resolved_config.yaml"
    atomic_write_bytes(resolved_config, canonical_json_bytes(protocol.model_dump(mode="json")))
    manifest_path = run_root / "run_manifest.json"
    manifest = create_run_manifest(
        manifest_path,
        phase=5,
        pde="reaction_diffusion",
        task="exact_rhs",
        model=model_name,
        regime="multiseed",
        seed=seed,
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
        study_id=protocol.study_id,
        run_id=run_id,
        model_name=model_name,
        config_hash=protocol.digest,
        dataset_hash=protocol.development_dataset_sha256,
        normalization_hash=protocol.normalization_sha256,
    )
    model = build_phase5_model(protocol, model_name).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=protocol.training.learning_rate,
        weight_decay=protocol.training.weight_decay,
    )
    scheduler = _scheduler(optimizer, protocol)
    metrics_path = run_root / "metrics.jsonl"
    start_step = 0
    sampler_state: dict[str, Any] | None = None
    best_velocity: float | None = None
    best_relative: float | None = None
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
        best_velocity = cast(float | None, payload["best_metric"])
        best_selection = cast(dict[str, Any], payload["checkpoint_selection"])
        if best_selection.get("validation_relative_loss") is not None:
            best_relative = float(best_selection["validation_relative_loss"])
    try:
        with DevelopmentExactRHSDataset(data_path, phase4) as dataset:
            sampler = ReplacementSampler(
                dataset.sample_count("training"), seed, sampler_state
            )
            validation_samples, rollout_trajectories = fixed_validation_indices(
                dataset.sample_count("validation"),
                int(dataset.handle["splits/validation/states"].shape[0]),
                phase4,
            )
            rows = _read_jsonl(metrics_path)
            if resume:
                rows = [row for row in rows if int(row["optimizer_steps"]) <= start_step]
                _write_jsonl(metrics_path, rows)

            def evaluate(step: int) -> dict[str, Any]:
                result = evaluate_velocity(
                    model,
                    dataset,
                    validation_samples,
                    device,
                    batch_size=protocol.training.batch_size,
                )
                relative = float(result.pop("relative_loss"))
                return {
                    "optimizer_steps": step,
                    "validation_relative_loss": relative,
                    **result,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "budget_completed": step >= protocol.training.maximum_steps,
                }

            if start_step == 0 and not rows:
                row = evaluate(0)
                _append_jsonl(metrics_path, row)
                rows.append(row)
                best_velocity = float(row["velocity_nrmse"])
                best_relative = float(row["validation_relative_loss"])
                best_selection = {
                    "criterion": protocol.checkpoint_selection,
                    "selected_step": 0,
                    "velocity_nrmse": best_velocity,
                    "validation_relative_loss": best_relative,
                }
                for name in ("best.pt", "last.pt"):
                    _save_checkpoint(
                        run_root / name,
                        identity=identity,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=0,
                        sample_count=dataset.sample_count("training"),
                        batch_size=protocol.training.batch_size,
                        sampler=sampler,
                        best_metric=best_velocity,
                        selection=best_selection,
                    )
            gradient_norm = 0.0
            training_loss = 0.0
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
                training_loss = float(loss.item())
                if (
                    step % protocol.training.evaluation_interval == 0
                    or step == protocol.training.maximum_steps
                ):
                    row = evaluate(step)
                    row["training_relative_loss"] = training_loss
                    row["gradient_norm"] = gradient_norm
                    _append_jsonl(metrics_path, row)
                    rows.append(row)
                    candidate_velocity = float(row["velocity_nrmse"])
                    candidate_relative = float(row["validation_relative_loss"])
                    improved = (
                        best_velocity is None
                        or candidate_velocity < best_velocity
                        or (
                            math.isclose(candidate_velocity, best_velocity)
                            and (best_relative is None or candidate_relative < best_relative)
                        )
                    )
                    if improved:
                        best_velocity = candidate_velocity
                        best_relative = candidate_relative
                        best_selection = {
                            "criterion": protocol.checkpoint_selection,
                            "selected_step": step,
                            "velocity_nrmse": best_velocity,
                            "validation_relative_loss": best_relative,
                        }
                        _save_checkpoint(
                            run_root / "best.pt",
                            identity=identity,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            sample_count=dataset.sample_count("training"),
                            batch_size=protocol.training.batch_size,
                            sampler=sampler,
                            best_metric=best_velocity,
                            selection=best_selection,
                        )
                    print(
                        json.dumps(
                            {
                                "model": model_name,
                                "seed": seed,
                                "step": step,
                                "velocity_nrmse": row["velocity_nrmse"],
                                "validation_relative_loss": row[
                                    "validation_relative_loss"
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if (
                    step % protocol.training.checkpoint_interval == 0
                    or step == protocol.training.maximum_steps
                ):
                    _save_checkpoint(
                        run_root / "last.pt",
                        identity=identity,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        sample_count=dataset.sample_count("training"),
                        batch_size=protocol.training.batch_size,
                        sampler=sampler,
                        best_metric=best_velocity,
                        selection=best_selection,
                    )
            load_v2_checkpoint(
                run_root / "best.pt",
                expected_identity=identity,
                model=model,
                optimizer=None,
                scheduler=None,
                restore_rng=False,
            )
            rollout_rows, _ = _rollout_rows(
                model,
                model_name,
                dataset,
                rollout_trajectories,
                protocol.training.rollout_steps_per_interval,
                device,
            )
            rollout = _aggregate_rollouts(rollout_rows)
            finite_rollouts = [
                float(row["rollout_relative_l2"])
                for row in rollout_rows
                if row["rollout_relative_l2"] is not None
            ]
            rollout["rollout_relative_l2_all_finite"] = (
                median(finite_rollouts) if finite_rollouts else None
            )
            _write_csv(run_root / "per_trajectory_rollouts.csv", rollout_rows)
        gate_conditions = {
            "full_budget_completed": max(int(row["optimizer_steps"]) for row in rows)
            >= protocol.training.maximum_steps,
            "velocity_nrmse": cast(float, best_velocity)
            <= protocol.gate.maximum_median_dct_velocity_nrmse,
            "zero_divergence": rollout["divergence_fraction"] == 0,
            "rollout_beats_persistence": (
                rollout["rollout_relative_l2"] is not None
                and rollout["persistence_relative_l2_all"] is not None
                and rollout["rollout_relative_l2"]
                < rollout["persistence_relative_l2_all"]
            ),
        }
        run_passed = all(gate_conditions.values())
        summary = {
            "schema_version": 1,
            "study_id": protocol.study_id,
            "phase": 5,
            "run_id": run_id,
            "model": model_name,
            "seed": seed,
            "development_only": True,
            "confirmatory_accessed": False,
            "optimizer_steps": protocol.training.maximum_steps,
            "checkpoint_selection": best_selection,
            "rollout": rollout,
            "run_gate": {
                "passed": run_passed,
                "conditions": gate_conditions,
            },
        }
        atomic_write_json(run_root / "summary.json", summary)
        transition_run_manifest(
            manifest_path,
            ExecutionStatus.COMPLETED,
            outcome=(
                ScientificOutcome.PASSED
                if run_passed
                else ScientificOutcome.FAILED
            ),
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


def phase5_decision(
    summaries: dict[tuple[str, int], dict[str, Any]], protocol: Phase5Protocol
) -> dict[str, Any]:
    expected = {(model, seed) for model in MODEL_NAMES for seed in protocol.training.seeds}
    complete = set(summaries) == expected
    if not complete:
        missing = sorted(f"{model}:s{seed}" for model, seed in expected - set(summaries))
        return {
            "complete": False,
            "decision": "phase5_incomplete",
            "missing_runs": missing,
            "phase6_confirmatory_generation_allowed": False,
            "superiority_claim_authorized": False,
        }
    dct_velocity = [
        float(summaries[("dct", seed)]["checkpoint_selection"]["velocity_nrmse"])
        for seed in protocol.training.seeds
    ]
    dct_zero_divergence = all(
        summaries[("dct", seed)]["rollout"]["divergence_fraction"] == 0
        for seed in protocol.training.seeds
    )
    dct_beats_persistence = True
    for seed in protocol.training.seeds:
        rollout = summaries[("dct", seed)]["rollout"]["rollout_relative_l2"]
        persistence = summaries[("dct", seed)]["rollout"][
            "persistence_relative_l2_all"
        ]
        dct_beats_persistence = dct_beats_persistence and (
            rollout is not None
            and persistence is not None
            and float(rollout) < float(persistence)
        )
    rollout_wins = 0
    velocity_wins = 0
    divergence_no_worse = True
    paired_rows = []
    for seed in protocol.training.seeds:
        fft = summaries[("fft", seed)]
        dct = summaries[("dct", seed)]
        fft_rollout = fft["rollout"]["rollout_relative_l2_all_finite"]
        dct_rollout = dct["rollout"]["rollout_relative_l2_all_finite"]
        rollout_win = (
            dct_rollout is not None
            and (fft_rollout is None or float(dct_rollout) < float(fft_rollout))
        )
        velocity_win = (
            float(dct["checkpoint_selection"]["velocity_nrmse"])
            < float(fft["checkpoint_selection"]["velocity_nrmse"])
        )
        divergence_win = (
            float(dct["rollout"]["divergence_fraction"])
            <= float(fft["rollout"]["divergence_fraction"])
        )
        rollout_wins += int(rollout_win)
        velocity_wins += int(velocity_win)
        divergence_no_worse = divergence_no_worse and divergence_win
        paired_rows.append(
            {
                "seed": seed,
                "fft_velocity_nrmse": fft["checkpoint_selection"]["velocity_nrmse"],
                "dct_velocity_nrmse": dct["checkpoint_selection"]["velocity_nrmse"],
                "fft_rollout_relative_l2": fft_rollout,
                "dct_rollout_relative_l2": dct_rollout,
                "fft_divergence_fraction": fft["rollout"]["divergence_fraction"],
                "dct_divergence_fraction": dct["rollout"]["divergence_fraction"],
                "dct_rollout_win": rollout_win,
                "dct_velocity_win": velocity_win,
            }
        )
    conditions = {
        "all_runs_complete": complete,
        "dct_zero_divergence_all_seeds": dct_zero_divergence,
        "median_dct_velocity_nrmse": median(dct_velocity)
        <= protocol.gate.maximum_median_dct_velocity_nrmse,
        "dct_beats_persistence_all_seeds": dct_beats_persistence,
        "dct_rollout_wins": rollout_wins >= protocol.gate.minimum_dct_rollout_wins,
        "dct_velocity_wins": velocity_wins >= protocol.gate.minimum_dct_velocity_wins,
        "dct_divergence_no_worse_all_seeds": divergence_no_worse,
    }
    passed = all(conditions.values())
    if passed:
        decision = "phase5_complete_freeze_dct_for_phase6_confirmatory_generation"
    elif dct_zero_divergence and dct_beats_persistence:
        decision = "phase5_inconclusive_no_confirmatory_generation"
    else:
        decision = "phase5_failed_no_confirmatory_generation"
    return {
        "complete": True,
        "passed": passed,
        "decision": decision,
        "conditions": conditions,
        "dct_rollout_wins": rollout_wins,
        "dct_velocity_wins": velocity_wins,
        "median_dct_velocity_nrmse": median(dct_velocity),
        "paired_results": paired_rows,
        "frozen_candidate_model": "dct" if passed else None,
        "frozen_rollout_steps_per_interval": 8 if passed else None,
        "phase6_confirmatory_generation_allowed": passed,
        "superiority_claim_authorized": False,
        "confirmatory_accessed": False,
    }


def _collect_summaries(
    root: Path, protocol: Phase5Protocol
) -> dict[tuple[str, int], dict[str, Any]]:
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for model in MODEL_NAMES:
        for seed in protocol.training.seeds:
            run_root = root / protocol.outputs.artifact_root / _run_id(protocol, model, seed)
            summary_path = run_root / "summary.json"
            manifest_path = run_root / "run_manifest.json"
            if not summary_path.is_file() or not manifest_path.is_file():
                continue
            manifest = RunManifest.from_dict(
                cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
            )
            if manifest.execution_status != ExecutionStatus.COMPLETED:
                continue
            summaries[(model, seed)] = cast(
                dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8"))
            )
    return summaries


def _plot_multiseed(path: Path, paired: list[dict[str, Any]]) -> None:
    seeds = [row["seed"] for row in paired]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for model, color in (("fft", "tab:blue"), ("dct", "tab:orange")):
        axes[0].plot(
            seeds,
            [row[f"{model}_velocity_nrmse"] for row in paired],
            marker="o",
            label=model.upper(),
            color=color,
        )
        axes[1].plot(
            seeds,
            [row[f"{model}_rollout_relative_l2"] for row in paired],
            marker="o",
            label=model.upper(),
            color=color,
        )
        axes[2].plot(
            seeds,
            [row[f"{model}_divergence_fraction"] for row in paired],
            marker="o",
            label=model.upper(),
            color=color,
        )
    axes[0].set(ylabel="velocity nRMSE", xlabel="seed")
    axes[1].set(ylabel="rollout relative L2", xlabel="seed")
    axes[2].set(ylabel="divergence fraction", xlabel="seed", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.set_xticks(seeds)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _curated_package(
    root: Path,
    protocol: Phase5Protocol,
    output: Path,
    report_paths: list[Path],
) -> None:
    files = list(report_paths)
    for model in MODEL_NAMES:
        for seed in protocol.training.seeds:
            run_root = root / protocol.outputs.artifact_root / _run_id(protocol, model, seed)
            files.extend(
                run_root / name
                for name in (
                    "best.pt",
                    "best.pt.sha256",
                    "metrics.jsonl",
                    "per_trajectory_rollouts.csv",
                    "resolved_config.yaml",
                    "run_manifest.json",
                    "summary.json",
                )
            )
    if any(not path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise FileNotFoundError(f"Phase 5 package is missing files: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Phase 5 package failed CRC validation")


def collect_phase5(root: Path, protocol: Phase5Protocol) -> dict[str, Any]:
    summaries = _collect_summaries(root, protocol)
    decision = phase5_decision(summaries, protocol)
    report = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "phase": 5,
        "protocol_sha256": protocol.digest,
        "development_only": True,
        "runs_found": len(summaries),
        **decision,
    }
    report_root = root / protocol.outputs.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    decision_path = report_root / "decision_report.json"
    protocol_path = report_root / "protocol_snapshot.json"
    atomic_write_json(decision_path, report)
    atomic_write_json(protocol_path, protocol.model_dump(mode="json"))
    report_paths = [decision_path, protocol_path]
    if decision.get("paired_results"):
        paired_path = report_root / "paired_seed_results.csv"
        plot_path = report_root / "multiseed_comparison.png"
        _write_csv(paired_path, cast(list[dict[str, Any]], decision["paired_results"]))
        _plot_multiseed(plot_path, cast(list[dict[str, Any]], decision["paired_results"]))
        report_paths.extend((paired_path, plot_path))
    if decision["complete"]:
        package_root = (
            root
            / protocol.outputs.artifact_root
            / f"chronopde_v2-p5-multiseed-study-{protocol.digest[:8]}"
        )
        _curated_package(
            root,
            protocol,
            package_root / "chronopde_v2_phase5_outputs.zip",
            report_paths,
        )
    return report


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 5 training requires CUDA; use --check-only locally")
        return torch.device("cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _discover_data(root: Path) -> Path | None:
    candidates = list(root.glob("artifacts/**/chronopde_v2_development.h5"))
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.extend(kaggle.rglob("chronopde_v2_development.h5"))
    files = [path for path in candidates if path.is_file()]
    return files[0] if len(files) == 1 else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase5",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/chronopde_v2/phase5.yaml")
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--model", choices=("fft", "dct"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        protocol = load_phase5_protocol(config_path.resolve())
        contract = validate_phase5_contract(root, protocol)
        phase4 = cast(Phase4Protocol, contract.pop("phase4"))
        if args.collect:
            report = collect_phase5(root, protocol)
        else:
            data_path = args.data_path or _discover_data(root)
            if data_path is None:
                raise FileNotFoundError("attach/pass chronopde_v2_development.h5")
            dataset = validate_phase4_dataset(root, phase4, data_path.resolve())
            if args.check_only:
                report = {
                    **contract,
                    "dataset": dataset,
                    "training_performed": False,
                }
            else:
                if args.model is None or args.seed is None:
                    raise ValueError("Phase 5 training requires --model and --seed")
                report = train_phase5_run(
                    root,
                    protocol,
                    phase4,
                    data_path.resolve(),
                    cast(ModelName, args.model),
                    args.seed,
                    _resolve_device(args.device),
                    resume=args.resume,
                )
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 2
    except RuntimeError as error:
        print(str(error))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
