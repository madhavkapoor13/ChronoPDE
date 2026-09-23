"""Phase 6 sealed-data generation and one-shot confirmatory evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any, cast

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from chronopde.config import SolverConfig
from chronopde.contracts import PhysicalParameters
from chronopde.data.simulator import simulate_trajectory
from chronopde.evaluation.rollout import continuous_rollout
from chronopde.v2.checkpoints import CheckpointIdentity, load_v2_checkpoint, verify_checkpoint
from chronopde.v2.common import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from chronopde.v2.numerical import exact_discrete_rhs, relative_l2
from chronopde.v2.phase4_training import ModelName
from chronopde.v2.phase5 import build_phase5_model
from chronopde.v2.pilot import (
    confirmatory_rows,
    exact_rhs_series,
    future_manifest,
    generate_v2_initial_condition,
    manifest_bytes,
    manifest_hash,
)
from chronopde.v2.protocol import (
    Phase2Protocol,
    Phase5Protocol,
    Phase6Protocol,
    load_phase2_protocol,
    load_phase5_protocol,
    load_phase6_protocol,
)
from chronopde.v2.registry import (
    ExecutionStatus,
    RunManifest,
    ScientificOutcome,
    create_run_manifest,
    transition_run_manifest,
)

Float32 = NDArray[np.float32]
Float64 = NDArray[np.float64]
MODEL_NAMES: tuple[ModelName, ModelName] = ("fft", "dct")


def _evaluation_run_id(protocol: Phase6Protocol, model: str, seed: int) -> str:
    return (
        "chronopde_v2-p6-reaction_diffusion-confirmatory-"
        f"{model}-frozen-s{seed}-{protocol.digest[:8]}"
    )


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _safe_archive(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Phase 5 archive contains duplicate members")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Phase 5 archive contains an unsafe path")
    if sum(item.file_size for item in archive.infolist()) > 300 * 1024 * 1024:
        raise ValueError("Phase 5 archive expands beyond the frozen limit")


def _run_pattern(model: str, seed: int) -> str:
    return f"chronopde_v2-p5-reaction_diffusion-exact_rhs-{model}-multiseed-s{seed}-73d7ad05"


@contextmanager
def verified_phase5_archive(
    source: Path, protocol: Phase6Protocol, phase5: Phase5Protocol
) -> Iterator[dict[tuple[str, int], tuple[Path, RunManifest]]]:
    """Verify the frozen Phase 5 study and expose its ten selected checkpoints."""

    if not source.is_file() or source.suffix.lower() != ".zip":
        raise FileNotFoundError("Phase 6 requires the curated Phase 5 ZIP")
    if sha256_file(source) != protocol.checkpoints.source_archive_sha256:
        raise ValueError("Phase 5 archive SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="chronopde-phase6-p5-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(source) as archive:
            _safe_archive(archive)
            archive.extractall(root)
        decision_paths = list(root.glob("**/reports/chronopde_v2/phase5/decision_report.json"))
        if len(decision_paths) != 1:
            raise ValueError("Phase 5 archive lacks one decision report")
        decision = json.loads(decision_paths[0].read_text(encoding="utf-8"))
        if (
            decision.get("decision")
            != "phase5_complete_freeze_dct_for_phase6_confirmatory_generation"
            or decision.get("protocol_sha256") != protocol.phase5_protocol_sha256
            or decision.get("phase6_confirmatory_generation_allowed") is not True
            or decision.get("confirmatory_accessed") is not False
        ):
            raise ValueError("Phase 5 decision does not authorize Phase 6")
        runs_roots = list(root.glob("**/artifacts/chronopde_v2/runs"))
        if len(runs_roots) != 1:
            raise ValueError("Phase 5 archive lacks one runs directory")
        runs_root = runs_roots[0]
        result: dict[tuple[str, int], tuple[Path, RunManifest]] = {}
        for model in MODEL_NAMES:
            expected_hashes = getattr(protocol.checkpoints, model)
            for seed in protocol.evaluation.seeds:
                run_root = runs_root / _run_pattern(model, seed)
                manifest = RunManifest.from_dict(
                    cast(
                        dict[str, Any],
                        json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8")),
                    )
                )
                expected_outcome = (
                    ScientificOutcome.PASSED if model == "dct" else ScientificOutcome.FAILED
                )
                if (
                    manifest.execution_status != ExecutionStatus.COMPLETED
                    or manifest.scientific_outcome != expected_outcome
                    or manifest.code_commit != protocol.checkpoints.source_code_commit
                    or manifest.config_hash != phase5.digest
                    or manifest.dataset_hash != phase5.development_dataset_sha256
                    or manifest.normalization_hash != protocol.normalization_sha256
                    or manifest.model != model
                    or manifest.seed != seed
                ):
                    raise ValueError(f"Phase 5 {model} seed {seed} identity mismatch")
                checkpoint = run_root / "best.pt"
                if verify_checkpoint(checkpoint) != expected_hashes[seed]:
                    raise ValueError(f"Phase 5 {model} seed {seed} checkpoint mismatch")
                if sha256_file(run_root / "resolved_config.yaml") != phase5.digest:
                    raise ValueError("Phase 5 resolved configuration mismatch")
                result[(model, seed)] = (checkpoint, manifest)
        yield result


def _normalization(root: Path, protocol: Phase6Protocol) -> dict[str, list[float]]:
    summary = json.loads((root / protocol.phase3_summary).read_text(encoding="utf-8"))
    values = cast(dict[str, list[float]], summary["normalization"])
    digest = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
    if digest != protocol.normalization_sha256:
        raise ValueError("Phase 3 normalization identity mismatch")
    return values


def validate_phase6_contract(
    root: Path, protocol: Phase6Protocol, archive: Path
) -> tuple[Phase2Protocol, Phase5Protocol, list[dict[str, str | int | float]]]:
    phase2 = load_phase2_protocol(root / protocol.phase2_descriptor)
    phase5 = load_phase5_protocol(root / protocol.phase5_descriptor)
    if phase5.digest != protocol.phase5_protocol_sha256:
        raise ValueError("Phase 6 references the wrong Phase 5 protocol")
    all_rows = future_manifest(phase2)
    if manifest_hash(all_rows) != protocol.frozen_future_manifest_sha256:
        raise ValueError("Phase 6 future-manifest identity mismatch")
    rows = confirmatory_rows(all_rows, frozen_model=True)
    if len(rows) != protocol.generation.trajectories:
        raise ValueError("Phase 6 confirmatory trajectory count mismatch")
    if manifest_hash(rows) != protocol.confirmatory_manifest_sha256:
        raise ValueError("Phase 6 confirmatory-manifest identity mismatch")
    committed = root / "reports/chronopde_v2/phase2/future_manifest.csv"
    if committed.read_bytes() != manifest_bytes(all_rows):
        raise ValueError("committed future manifest differs from Phase 2")
    _normalization(root, protocol)
    with verified_phase5_archive(archive, protocol, phase5) as checkpoints:
        if len(checkpoints) != 10:
            raise ValueError("Phase 6 requires all ten frozen checkpoints")
    return phase2, phase5, rows


def _cache_path(root: Path, row: dict[str, str | int | float]) -> Path:
    return root / f"{row['trajectory_id']}.npz"


def _trajectory_digest(
    row: dict[str, str | int | float],
    states: NDArray[np.float32],
    rhs: NDArray[np.float32],
    times: NDArray[np.float64],
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(row))
    for name, values in (("states", states), ("rhs", rhs), ("times", times)):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generate_one(
    row: dict[str, str | int | float],
    cache_root: Path,
    phase2: Phase2Protocol,
    protocol: Phase6Protocol,
) -> dict[str, Any]:
    path = _cache_path(cache_root, row)
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            if (
                str(cached["trajectory_id"].item()) != row["trajectory_id"]
                or str(cached["protocol_hash"].item()) != protocol.digest
                or str(cached["row_json"].item()) != canonical_json_bytes(row).decode("utf-8")
            ):
                raise ValueError(f"invalid Phase 6 cache identity: {path}")
            states = np.asarray(cached["states"], dtype=np.float32)
            rhs = np.asarray(cached["rhs"], dtype=np.float32)
            times = np.asarray(cached["times"], dtype=np.float64)
            nfev = int(cached["nfev"].item())
            maximum = float(cached["maximum"].item())
            stored_digest = str(cached["logical_sha256"].item())
        if stored_digest != _trajectory_digest(row, states, rhs, times):
            raise ValueError(f"corrupt Phase 6 cache payload: {path}")
        reused = True
    else:
        params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
        initial = generate_v2_initial_condition(phase2, int(row["ic_seed"]))
        result = simulate_trajectory(
            initial,
            params,
            phase2.pde,
            divergence_threshold=phase2.tolerances.divergence_threshold,
        )
        if not result.diagnostics.success:
            raise RuntimeError(f"confirmatory trajectory failed: {row['trajectory_id']}")
        states = result.states
        rhs = exact_rhs_series(states, params, phase2.pde)
        times = result.times
        nfev = result.diagnostics.nfev
        maximum = result.diagnostics.max_abs_state
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(
                temporary,
                trajectory_id=row["trajectory_id"],
                protocol_hash=protocol.digest,
                row_json=canonical_json_bytes(row).decode("utf-8"),
                states=states,
                rhs=rhs,
                times=times,
                nfev=nfev,
                maximum=maximum,
                logical_sha256=_trajectory_digest(row, states, rhs, times),
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        reused = False
    expected_shape = (phase2.pde.stored_times, 2, phase2.pde.height, phase2.pde.width)
    if states.shape != expected_shape or rhs.shape != expected_shape:
        raise ValueError("Phase 6 cached trajectory shape mismatch")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(rhs)):
        raise ValueError("Phase 6 cached trajectory is nonfinite")
    return {**row, "cached": reused, "nfev": nfev, "max_abs_state": maximum}


def _logical_hash(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    group = handle["confirmatory"]
    for name in ("states", "rhs", "times", "parameters", "ic_seeds", "trajectory_ids"):
        dataset = group[name]
        digest.update(name.encode())
        digest.update(str(dataset.dtype).encode())
        digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
        for index in range(dataset.shape[0]):
            value = dataset[index]
            if name == "trajectory_ids":
                digest.update(str(value.decode() if isinstance(value, bytes) else value).encode())
            else:
                digest.update(np.asarray(value).tobytes())
    for name in ("state_mean", "state_std", "parameter_mean", "parameter_std"):
        digest.update(name.encode())
        digest.update(np.asarray(handle[f"normalization/{name}"]).tobytes())
    return digest.hexdigest()


def _assemble_dataset(
    path: Path,
    rows: list[dict[str, str | int | float]],
    cache_root: Path,
    phase2: Phase2Protocol,
    protocol: Phase6Protocol,
    normalization: dict[str, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = "chronopde_v2_confirmatory_1"
            handle.attrs["phase6_protocol_sha256"] = protocol.digest
            handle.attrs["confirmatory_manifest_sha256"] = manifest_hash(rows)
            handle.attrs["normalization_sha256"] = protocol.normalization_sha256
            group = handle.create_group("confirmatory")
            shape = (len(rows), phase2.pde.stored_times, 2, phase2.pde.height, phase2.pde.width)
            states_ds = group.create_dataset(
                "states", shape=shape, dtype=np.float32, chunks=(1, 1, 2, 64, 64), compression="lzf"
            )
            rhs_ds = group.create_dataset(
                "rhs", shape=shape, dtype=np.float32, chunks=(1, 1, 2, 64, 64), compression="lzf"
            )
            times_ds = group.create_dataset("times", shape=(len(rows), 101), dtype=np.float64)
            parameters_ds = group.create_dataset(
                "parameters", shape=(len(rows), 3), dtype=np.float64
            )
            seeds_ds = group.create_dataset("ic_seeds", shape=(len(rows),), dtype=np.int64)
            ids_ds = group.create_dataset(
                "trajectory_ids", shape=(len(rows),), dtype=h5py.string_dtype("utf-8")
            )
            for index, row in enumerate(rows):
                with np.load(_cache_path(cache_root, row), allow_pickle=False) as cached:
                    states_ds[index] = cached["states"]
                    rhs_ds[index] = cached["rhs"]
                    times_ds[index] = cached["times"]
                parameters_ds[index] = [row["du"], row["dv"], row["k"]]
                seeds_ds[index] = int(row["ic_seed"])
                ids_ds[index] = str(row["trajectory_id"])
            normal = handle.create_group("normalization")
            normal.attrs["source_split"] = "phase3_training_only"
            for name, values in normalization.items():
                normal.create_dataset(name, data=np.asarray(values, dtype=np.float64))
            handle.attrs["logical_content_sha256"] = _logical_hash(handle)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_confirmatory_dataset(
    path: Path,
    rows: list[dict[str, str | int | float]],
    phase2: Phase2Protocol,
    protocol: Phase6Protocol,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"confirmatory dataset checksum sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    file_digest = sha256_file(path)
    if len(fields) != 2 or fields[0] != file_digest or fields[1] != path.name:
        raise ValueError("confirmatory dataset file checksum mismatch")
    with h5py.File(path, "r") as handle:
        if (
            handle.attrs.get("schema_version") != "chronopde_v2_confirmatory_1"
            or handle.attrs.get("phase6_protocol_sha256") != protocol.digest
            or handle.attrs.get("confirmatory_manifest_sha256") != manifest_hash(rows)
            or handle.attrs.get("normalization_sha256") != protocol.normalization_sha256
        ):
            raise ValueError("confirmatory dataset identity mismatch")
        group = handle["confirmatory"]
        shape = (256, 101, 2, 64, 64)
        if group["states"].shape != shape or group["rhs"].shape != shape:
            raise ValueError("confirmatory dataset shape mismatch")
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in group["trajectory_ids"]
        ]
        if ids != [str(row["trajectory_id"]) for row in rows]:
            raise ValueError("confirmatory trajectory identities mismatch")
        expected_parameters = np.asarray(
            [[row["du"], row["dv"], row["k"]] for row in rows], dtype=np.float64
        )
        expected_seeds = np.asarray([row["ic_seed"] for row in rows], dtype=np.int64)
        if not np.array_equal(group["parameters"][:], expected_parameters) or not np.array_equal(
            group["ic_seeds"][:], expected_seeds
        ):
            raise ValueError("confirmatory parameters or seeds differ from the sealed manifest")
        normalization_payload = {
            name: np.asarray(handle[f"normalization/{name}"], dtype=np.float64).tolist()
            for name in ("state_mean", "state_std", "parameter_mean", "parameter_std")
        }
        if (
            hashlib.sha256(canonical_json_bytes(normalization_payload)).hexdigest()
            != protocol.normalization_sha256
        ):
            raise ValueError("confirmatory normalization payload mismatch")
        if _logical_hash(handle) != handle.attrs.get("logical_content_sha256"):
            raise ValueError("confirmatory logical checksum mismatch")
        maximum = 0.0
        for trajectory in range(256):
            states = np.asarray(group["states"][trajectory], dtype=np.float32)
            rhs = np.asarray(group["rhs"][trajectory], dtype=np.float32)
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(rhs)):
                raise ValueError("confirmatory dataset contains nonfinite values")
            maximum = max(maximum, float(np.max(np.abs(states))))
        if maximum > phase2.tolerances.divergence_threshold:
            raise ValueError("confirmatory dataset exceeds the frozen state limit")
        rng = np.random.default_rng(20260923)
        chosen = rng.choice(256 * 101, size=protocol.generation.rhs_spot_checks, replace=False)
        errors = []
        for flat in chosen:
            trajectory, time_index = divmod(int(flat), 101)
            row = rows[trajectory]
            params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
            state = np.asarray(group["states"][trajectory, time_index], dtype=np.float64)
            stored = np.asarray(group["rhs"][trajectory, time_index], dtype=np.float64)
            errors.append(relative_l2(stored, exact_discrete_rhs(state, params, phase2.pde)))
        return {
            "logical_content_sha256": str(handle.attrs["logical_content_sha256"]),
            "file_sha256": file_digest,
            "trajectory_count": 256,
            "maximum_absolute_state": maximum,
            "rhs_spot_checks": len(errors),
            "maximum_rhs_spot_check_relative_l2": max(errors),
        }


def generate_confirmatory(root: Path, protocol: Phase6Protocol, archive: Path) -> dict[str, Any]:
    phase2, _, rows = validate_phase6_contract(root, protocol, archive)
    run_id = (
        "chronopde_v2-p6-reaction_diffusion-data-reference-"
        f"confirmatory-s20260922-{protocol.digest[:8]}"
    )
    artifact = root / protocol.outputs.artifact_root / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact / "run_manifest.json"
    manifest = create_run_manifest(
        manifest_path,
        phase=6,
        pde="reaction_diffusion",
        task="data",
        model="reference",
        regime="confirmatory",
        seed=20260922,
        config_hash=protocol.digest,
        code_commit=_git_commit(root),
        checkpoint_selection="not_applicable_exact_simulator",
        dataset_hash=protocol.confirmatory_manifest_sha256,
        normalization_hash=protocol.normalization_sha256,
    )
    if manifest.run_id != run_id:
        raise ValueError("Phase 6 generation run identity mismatch")
    if manifest.execution_status == ExecutionStatus.COMPLETED:
        return cast(dict[str, Any], json.loads((artifact / "generation_summary.json").read_text()))
    if manifest.execution_status == ExecutionStatus.PLANNED:
        transition_run_manifest(manifest_path, ExecutionStatus.RUNNING)
    elif manifest.execution_status != ExecutionStatus.RUNNING:
        raise ValueError(f"Phase 6 generation is finalized as {manifest.execution_status}")
    cache = artifact / "trajectory_cache"
    with ThreadPoolExecutor(max_workers=protocol.generation.workers) as executor:
        qa = list(executor.map(lambda row: _generate_one(row, cache, phase2, protocol), rows))
    maximum = max(float(row["max_abs_state"]) for row in qa)
    if maximum > phase2.tolerances.divergence_threshold:
        raise ValueError("confirmatory trajectory exceeds the divergence threshold")
    tight_pde = phase2.pde.model_copy(
        update={"solver": SolverConfig(method="DOP853", rtol=1e-10, atol=1e-12)}
    )
    convergence: list[float] = []
    final_errors: list[float] = []
    for index in protocol.generation.tight_solver_indices:
        row = rows[index]
        with np.load(_cache_path(cache, row), allow_pickle=False) as cached:
            standard = np.asarray(cached["states"], dtype=np.float32)
        params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
        tight = simulate_trajectory(
            generate_v2_initial_condition(phase2, int(row["ic_seed"])),
            params,
            tight_pde,
            divergence_threshold=phase2.tolerances.divergence_threshold,
        )
        if not tight.diagnostics.success:
            raise RuntimeError("tight confirmatory solver control failed")
        convergence.extend(relative_l2(a, b) for a, b in zip(standard, tight.states, strict=True))
        final_errors.append(relative_l2(standard[-1], tight.states[-1]))
    p95 = float(np.percentile(convergence, 95))
    final_max = max(final_errors)
    if (
        p95 > phase2.tolerances.solver_state_p95_relative_l2
        or final_max > phase2.tolerances.solver_final_relative_l2
    ):
        raise ValueError("confirmatory solver convergence control failed")
    data_path = artifact / "chronopde_v2_confirmatory.h5"
    if not data_path.exists():
        _assemble_dataset(data_path, rows, cache, phase2, protocol, _normalization(root, protocol))
    data_sidecar = data_path.with_suffix(data_path.suffix + ".sha256")
    atomic_write_bytes(data_sidecar, f"{sha256_file(data_path)}  {data_path.name}\n".encode())
    validated = validate_confirmatory_dataset(data_path, rows, phase2, protocol)
    summary = {
        "passed": True,
        "run_id": run_id,
        "phase6_protocol_sha256": protocol.digest,
        "phase5_archive_sha256": protocol.checkpoints.source_archive_sha256,
        "confirmatory_manifest_sha256": protocol.confirmatory_manifest_sha256,
        "successful_trajectories": len(qa),
        "maximum_absolute_state": maximum,
        "solver_convergence": {
            "tight_trajectory_count": 12,
            "state_relative_l2_p95": p95,
            "maximum_final_state_relative_l2": final_max,
        },
        **validated,
    }
    atomic_write_json(artifact / "generation_summary.json", summary)
    _write_csv(artifact / "trajectory_qa.csv", qa)
    evidence = artifact / "chronopde_v2_phase6_generation_evidence.zip"
    with zipfile.ZipFile(evidence, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr(
            "protocol_snapshot.json", canonical_json_bytes(protocol.model_dump(mode="json"))
        )
        output.writestr("confirmatory_manifest.csv", manifest_bytes(rows))
        output.writestr("generation_summary.json", canonical_json_bytes(summary))
        output.write(artifact / "trajectory_qa.csv", "trajectory_qa.csv")
        output.write(data_sidecar, data_sidecar.name)
    if zipfile.ZipFile(evidence).testzip() is not None:
        raise ValueError("Phase 6 generation evidence failed CRC validation")
    transition_run_manifest(manifest_path, ExecutionStatus.COMPLETED)
    with zipfile.ZipFile(evidence, "a", zipfile.ZIP_DEFLATED) as output:
        output.write(manifest_path, "run_manifest.json")
    if zipfile.ZipFile(evidence).testzip() is not None:
        raise ValueError("Phase 6 finalized generation evidence failed CRC validation")
    return {**summary, "data_path": str(data_path), "evidence_path": str(evidence)}


class ConfirmatoryDataset:
    """Read the sealed confirmatory set using the frozen Phase 3 normalization."""

    def __init__(self, path: Path) -> None:
        self.handle = h5py.File(path, "r")
        normal = self.handle["normalization"]
        self.state_mean = np.asarray(normal["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(normal["state_std"], dtype=np.float32)
        self.parameter_mean = np.asarray(normal["parameter_mean"], dtype=np.float32)
        self.parameter_std = np.asarray(normal["parameter_std"], dtype=np.float32)

    def __enter__(self) -> ConfirmatoryDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.handle.close()

    def _read_pairs(self, _split: str, indices: NDArray[np.int64]) -> dict[str, Tensor]:
        group = self.handle["confirmatory"]
        states, targets, times, parameters = [], [], [], []
        for raw in indices.tolist():
            trajectory, time_index = divmod(int(raw), 101)
            states.append(np.asarray(group["states"][trajectory, time_index], dtype=np.float32))
            targets.append(np.asarray(group["rhs"][trajectory, time_index], dtype=np.float32))
            times.append(float(group["times"][trajectory, time_index]))
            parameters.append(np.asarray(group["parameters"][trajectory], dtype=np.float32))
        state = (np.stack(states) - self.state_mean[None, :, None, None]) / self.state_std[
            None, :, None, None
        ]
        target = np.stack(targets) / self.state_std[None, :, None, None]
        params = (np.stack(parameters) - self.parameter_mean) / self.parameter_std
        return {
            "state": torch.from_numpy(state),
            "target": torch.from_numpy(target),
            "time": torch.tensor(times, dtype=torch.float32),
            "parameters": torch.from_numpy(params),
        }

    def read_rollouts(self, indices: NDArray[np.int64]) -> dict[str, Tensor]:
        group = self.handle["confirmatory"]
        states = np.stack([np.asarray(group["states"][int(i)], dtype=np.float32) for i in indices])
        times = np.stack([np.asarray(group["times"][int(i)], dtype=np.float32) for i in indices])
        params = np.stack(
            [np.asarray(group["parameters"][int(i)], dtype=np.float32) for i in indices]
        )
        normalized = (states - self.state_mean[None, None, :, None, None]) / self.state_std[
            None, None, :, None, None
        ]
        normalized_params = (params - self.parameter_mean) / self.parameter_std
        return {
            "states": torch.from_numpy(normalized),
            "physical_states": torch.from_numpy(states),
            "times": torch.from_numpy(times),
            "parameters": torch.from_numpy(normalized_params),
        }


def _load_model(
    checkpoint: Path,
    manifest: RunManifest,
    phase5: Phase5Protocol,
    device: torch.device,
) -> nn.Module:
    model_name = cast(ModelName, manifest.model)
    model = build_phase5_model(phase5, model_name)
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


def _rollouts(
    model: nn.Module,
    model_name: ModelName,
    dataset: ConfirmatoryDataset,
    protocol: Phase6Protocol,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = dataset.handle["confirmatory/trajectory_ids"]
    for start in range(0, 256, protocol.evaluation.rollout_batch_size):
        indices = np.arange(
            start, min(start + protocol.evaluation.rollout_batch_size, 256), dtype=np.int64
        )
        batch = {name: value.to(device) for name, value in dataset.read_rollouts(indices).items()}
        try:
            with torch.inference_mode():
                result = continuous_rollout(
                    model,
                    batch["states"][:, 0],
                    batch["times"],
                    batch["parameters"],
                    steps_per_interval=protocol.evaluation.steps_per_interval,
                    method="rk4",
                )
                mean = torch.from_numpy(dataset.state_mean).to(device)[None, None, :, None, None]
                std = torch.from_numpy(dataset.state_std).to(device)[None, None, :, None, None]
                physical = cast(Tensor, result.states) * std + mean
        except (RuntimeError, ValueError):
            recovered = []
            for position in range(len(indices)):
                try:
                    with torch.inference_mode():
                        result = continuous_rollout(
                            model,
                            batch["states"][position : position + 1, 0],
                            batch["times"][position : position + 1],
                            batch["parameters"][position : position + 1],
                            steps_per_interval=protocol.evaluation.steps_per_interval,
                            method="rk4",
                        )
                        mean = torch.from_numpy(dataset.state_mean).to(device)[
                            None, None, :, None, None
                        ]
                        std = torch.from_numpy(dataset.state_std).to(device)[
                            None, None, :, None, None
                        ]
                        recovered.append(cast(Tensor, result.states) * std + mean)
                except (RuntimeError, ValueError):
                    recovered.append(
                        torch.full_like(
                            batch["physical_states"][position : position + 1], float("nan")
                        )
                    )
            physical = torch.cat(recovered)
        for position, index in enumerate(indices.tolist()):
            prediction = physical[position]
            target = batch["physical_states"][position]
            finite = bool(torch.isfinite(prediction).all())
            maximum = float(prediction.abs().max()) if finite else None
            stable = finite and cast(float, maximum) <= 10.0
            raw_id = ids[index]
            row: dict[str, Any] = {
                "model": model_name,
                "trajectory_index": index,
                "trajectory_id": raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id),
                "finite": finite,
                "stable": stable,
                "maximum_absolute_state": maximum,
                "rollout_relative_l2": None,
                "final_time_relative_l2": None,
                "persistence_relative_l2": None,
                "correlation_horizon": None,
            }
            if finite:
                row["rollout_relative_l2"] = float((prediction - target).norm() / target.norm())
                row["final_time_relative_l2"] = float(
                    (prediction[-1] - target[-1]).norm() / target[-1].norm()
                )
                persistence = target[:1].expand_as(target)
                row["persistence_relative_l2"] = float(
                    (persistence - target).norm() / target.norm()
                )
                row["correlation_horizon"] = _correlation_horizon(
                    prediction, target, batch["times"][position]
                )
            rows.append(row)
    return rows


def _attribution(
    model: nn.Module,
    model_name: ModelName,
    dataset: ConfirmatoryDataset,
    protocol: Phase6Protocol,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    ids = dataset.handle["confirmatory/trajectory_ids"]
    physical_std = torch.from_numpy(dataset.state_std).to(device)[None, :, None, None]
    for trajectory in range(256):
        finite = True
        total_sse = total_energy = boundary_sse = boundary_energy = 0.0
        interior_sse = interior_energy = derivative_sse = derivative_energy = 0.0
        channel_sse = [0.0, 0.0]
        channel_energy = [0.0, 0.0]
        physical_sse = 0.0
        physical_count = 0
        indices = np.arange(trajectory * 101, (trajectory + 1) * 101, dtype=np.int64)
        for start in range(0, 101, protocol.evaluation.velocity_batch_size):
            batch = {
                name: value.to(device)
                for name, value in dataset._read_pairs(
                    "confirmatory", indices[start : start + protocol.evaluation.velocity_batch_size]
                ).items()
            }
            with torch.inference_mode():
                prediction = model(batch["state"], batch["time"], batch["parameters"])
            target = batch["target"]
            if not bool(torch.isfinite(prediction).all()):
                finite = False
                break
            error = prediction - target
            total_sse += float(error.square().sum())
            total_energy += float(target.square().sum())
            for channel in range(2):
                channel_sse[channel] += float(error[:, channel].square().sum())
                channel_energy[channel] += float(target[:, channel].square().sum())
            physical_error = error * physical_std
            physical_sse += float(physical_error.square().sum())
            physical_count += physical_error.numel()
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[..., :4, :] = mask[..., -4:, :] = True
            mask[..., :, :4] = mask[..., :, -4:] = True
            boundary_sse += float(error[mask].square().sum())
            boundary_energy += float(target[mask].square().sum())
            interior_sse += float(error[~mask].square().sum())
            interior_energy += float(target[~mask].square().sum())
            pred_d = torch.cat(
                (
                    prediction[..., 1, :] - prediction[..., 0, :],
                    prediction[..., -1, :] - prediction[..., -2, :],
                    prediction[..., :, 1] - prediction[..., :, 0],
                    prediction[..., :, -1] - prediction[..., :, -2],
                ),
                dim=-1,
            )
            target_d = torch.cat(
                (
                    target[..., 1, :] - target[..., 0, :],
                    target[..., -1, :] - target[..., -2, :],
                    target[..., :, 1] - target[..., :, 0],
                    target[..., :, -1] - target[..., :, -2],
                ),
                dim=-1,
            )
            derivative_sse += float((pred_d - target_d).square().sum())
            derivative_energy += float(target_d.square().sum())
        raw_id = ids[trajectory]
        rows.append(
            {
                "model": model_name,
                "trajectory_index": trajectory,
                "trajectory_id": raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id),
                "finite": finite,
                "velocity_relative_l2": (
                    math.sqrt(total_sse / max(total_energy, 1e-12)) if finite else None
                ),
                "channel_u_velocity_relative_l2": (
                    math.sqrt(channel_sse[0] / max(channel_energy[0], 1e-12)) if finite else None
                ),
                "channel_v_velocity_relative_l2": (
                    math.sqrt(channel_sse[1] / max(channel_energy[1], 1e-12)) if finite else None
                ),
                "physical_velocity_mse": (
                    physical_sse / max(physical_count, 1) if finite else None
                ),
                "boundary_strip_relative_l2": (
                    math.sqrt(boundary_sse / max(boundary_energy, 1e-12)) if finite else None
                ),
                "interior_relative_l2": (
                    math.sqrt(interior_sse / max(interior_energy, 1e-12)) if finite else None
                ),
                "first_interior_normal_derivative_relative_l2": (
                    math.sqrt(derivative_sse / max(derivative_energy, 1e-12)) if finite else None
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_confirmatory(
    root: Path,
    protocol: Phase6Protocol,
    archive: Path,
    data_path: Path,
    model_name: ModelName,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    phase2, phase5, rows = validate_phase6_contract(root, protocol, archive)
    validate_confirmatory_dataset(data_path, rows, phase2, protocol)
    if seed not in protocol.evaluation.seeds:
        raise ValueError("seed is outside the frozen Phase 6 set")
    run_id = _evaluation_run_id(protocol, model_name, seed)
    run_root = root / protocol.outputs.artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    with h5py.File(data_path, "r") as handle:
        dataset_hash = str(handle.attrs["logical_content_sha256"])
    manifest = create_run_manifest(
        manifest_path,
        phase=6,
        pde="reaction_diffusion",
        task="confirmatory",
        model=model_name,
        regime="frozen",
        seed=seed,
        config_hash=protocol.digest,
        code_commit=_git_commit(root),
        checkpoint_selection="frozen_phase5_best_no_selection",
        dataset_hash=dataset_hash,
        normalization_hash=protocol.normalization_sha256,
    )
    if manifest.execution_status == ExecutionStatus.COMPLETED:
        return cast(dict[str, Any], json.loads((run_root / "summary.json").read_text()))
    if manifest.execution_status == ExecutionStatus.PLANNED:
        transition_run_manifest(manifest_path, ExecutionStatus.RUNNING)
    elif manifest.execution_status != ExecutionStatus.RUNNING:
        raise ValueError(f"Phase 6 run is finalized as {manifest.execution_status}")
    try:
        with verified_phase5_archive(archive, protocol, phase5) as checkpoints:
            checkpoint, source_manifest = checkpoints[(model_name, seed)]
            model = _load_model(checkpoint, source_manifest, phase5, device)
            with ConfirmatoryDataset(data_path) as dataset:
                rollout_rows = _rollouts(model, model_name, dataset, protocol, device)
                attribution_rows = _attribution(model, model_name, dataset, protocol, device)
        finite = [
            float(row["rollout_relative_l2"])
            for row in rollout_rows
            if row["rollout_relative_l2"] is not None
        ]
        persistence = [
            float(row["persistence_relative_l2"])
            for row in rollout_rows
            if row["persistence_relative_l2"] is not None
        ]
        finite_attribution = [row for row in attribution_rows if row["finite"]]

        def attribution_median(name: str) -> float | None:
            values = [float(row[name]) for row in finite_attribution if row[name] is not None]
            return median(values) if values else None

        velocity = {
            "available_trajectories": len(finite_attribution),
            "velocity_nrmse": attribution_median("velocity_relative_l2"),
            "per_channel_velocity_nrmse": [
                attribution_median("channel_u_velocity_relative_l2"),
                attribution_median("channel_v_velocity_relative_l2"),
            ],
            "physical_velocity_mse": attribution_median("physical_velocity_mse"),
        }
        summary = {
            "schema_version": 1,
            "study_id": protocol.study_id,
            "phase": 6,
            "run_id": run_id,
            "model": model_name,
            "seed": seed,
            "checkpoint_sha256": getattr(protocol.checkpoints, model_name)[seed],
            "trajectories": 256,
            "finite_trajectories": len(finite),
            "divergence_fraction": 1 - sum(bool(row["stable"]) for row in rollout_rows) / 256,
            "rollout_relative_l2_all_finite": median(finite) if finite else None,
            "persistence_relative_l2_all": median(persistence) if persistence else None,
            "velocity": velocity,
            "attribution_medians": {
                name: attribution_median(name)
                for name in (
                    "velocity_relative_l2",
                    "boundary_strip_relative_l2",
                    "interior_relative_l2",
                    "first_interior_normal_derivative_relative_l2",
                )
            },
            "training_performed": False,
            "checkpoint_written": False,
        }
        _write_csv(run_root / "per_trajectory_rollouts.csv", rollout_rows)
        _write_csv(run_root / "per_trajectory_attribution.csv", attribution_rows)
        atomic_write_json(run_root / "summary.json", summary)
        atomic_write_json(run_root / "resolved_config.json", protocol.model_dump(mode="json"))
        transition_run_manifest(manifest_path, ExecutionStatus.COMPLETED)
        return summary
    except Exception as error:
        transition_run_manifest(
            manifest_path,
            ExecutionStatus.FAILED,
            outcome=ScientificOutcome.INCONCLUSIVE,
            failure_reason=f"{type(error).__name__}: {error}",
        )
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def phase6_decision(
    summaries: dict[tuple[str, int], dict[str, Any]], protocol: Phase6Protocol
) -> dict[str, Any]:
    expected = {(model, seed) for model in MODEL_NAMES for seed in protocol.evaluation.seeds}
    if set(summaries) != expected:
        return {
            "complete": False,
            "decision": "phase6_incomplete_identical_rerun_only",
            "missing_runs": sorted(f"{model}:s{seed}" for model, seed in expected - set(summaries)),
            "superiority_claim_authorized": False,
        }
    paired = []
    rollout_wins = velocity_wins = persistence_wins = divergence_wins = 0
    relative_improvements = []
    boundary_wins = derivative_wins = 0

    def lower(left: Any, right: Any) -> bool:
        return left is not None and right is not None and float(left) < float(right)

    for seed in protocol.evaluation.seeds:
        fft, dct = summaries[("fft", seed)], summaries[("dct", seed)]
        fft_rollout = fft["rollout_relative_l2_all_finite"]
        dct_rollout = dct["rollout_relative_l2_all_finite"]
        rollout_win = lower(dct_rollout, fft_rollout)
        improvement = (
            (float(fft_rollout) - float(dct_rollout)) / float(fft_rollout)
            if rollout_win
            else float("-inf")
        )
        velocity_win = lower(dct["velocity"]["velocity_nrmse"], fft["velocity"]["velocity_nrmse"])
        persistence_win = lower(dct_rollout, dct["persistence_relative_l2_all"])
        divergence_win = dct["divergence_fraction"] <= fft["divergence_fraction"]
        boundary_win = lower(
            dct["attribution_medians"]["boundary_strip_relative_l2"],
            fft["attribution_medians"]["boundary_strip_relative_l2"],
        )
        derivative_win = lower(
            dct["attribution_medians"]["first_interior_normal_derivative_relative_l2"],
            fft["attribution_medians"]["first_interior_normal_derivative_relative_l2"],
        )
        rollout_wins += int(rollout_win)
        velocity_wins += int(velocity_win)
        persistence_wins += int(persistence_win)
        divergence_wins += int(divergence_win)
        boundary_wins += int(boundary_win)
        derivative_wins += int(derivative_win)
        relative_improvements.append(improvement)
        paired.append(
            {
                "seed": seed,
                "fft_rollout_relative_l2": fft_rollout,
                "dct_rollout_relative_l2": dct_rollout,
                "relative_improvement": improvement if math.isfinite(improvement) else None,
                "fft_velocity_nrmse": fft["velocity"]["velocity_nrmse"],
                "dct_velocity_nrmse": dct["velocity"]["velocity_nrmse"],
                "fft_divergence_fraction": fft["divergence_fraction"],
                "dct_divergence_fraction": dct["divergence_fraction"],
            }
        )
    median_improvement = (
        median(relative_improvements)
        if all(math.isfinite(value) for value in relative_improvements)
        else None
    )
    conditions = {
        "all_runs_complete": True,
        "dct_zero_divergence": all(
            summaries[("dct", seed)]["divergence_fraction"] == 0
            for seed in protocol.evaluation.seeds
        ),
        "dct_rollout_wins_all_seeds": rollout_wins == 5,
        "median_relative_improvement_at_least_10_percent": median_improvement is not None
        and median_improvement >= protocol.gate.minimum_median_relative_improvement,
        "dct_velocity_wins_all_seeds": velocity_wins == 5,
        "dct_beats_persistence_all_seeds": persistence_wins == 5,
        "dct_divergence_no_worse_all_seeds": divergence_wins == 5,
    }
    passed = all(conditions.values())
    return {
        "complete": True,
        "passed": passed,
        "decision": "phase6_confirmatory_superiority_supported"
        if passed
        else "phase6_confirmatory_claim_failed_no_retuning",
        "conditions": conditions,
        "paired_results": paired,
        "rollout_wins": rollout_wins,
        "velocity_wins": velocity_wins,
        "median_relative_improvement": median_improvement,
        "one_sided_exact_sign_test_p": sum(math.comb(5, k) for k in range(rollout_wins, 6)) / 32,
        "boundary_wording_authorized": boundary_wins == 5 and derivative_wins == 5,
        "permitted_boundary_claim": (
            "Boundary-strip and first-interior normal-derivative errors were lower for DCT "
            "in all five frozen seeds; this does not establish boundary enforcement or "
            "physical wall-flux correctness."
            if passed and boundary_wins == 5 and derivative_wins == 5
            else None
        ),
        "superiority_claim_authorized": passed,
        "permitted_claim": (
            "For this reaction-diffusion PDE, 64x64 grid, central parameter range and frozen "
            "training protocol, the matched DCT operator had lower rollout and exact-velocity "
            "error than FFT across all five seeds on 256 sealed in-distribution trajectories."
            if passed
            else None
        ),
    }


def _bootstrap(
    root: Path, protocol: Phase6Protocol, summaries: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    rng = np.random.default_rng(protocol.evaluation.bootstrap_seed)
    pairs: dict[
        int, tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]
    ] = {}
    for seed in protocol.evaluation.seeds:
        values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for model in MODEL_NAMES:
            run_id = summaries[(model, seed)]["run_id"]
            rows = _read_csv(
                root / protocol.outputs.artifact_root / run_id / "per_trajectory_rollouts.csv"
            )
            values.append(
                np.asarray(
                    [
                        float(row["rollout_relative_l2"]) if row["rollout_relative_l2"] else np.nan
                        for row in rows
                    ],
                    dtype=np.float64,
                )
            )
        pairs[seed] = (values[0], values[1])
    effects = []
    seeds = np.asarray(protocol.evaluation.seeds)
    for _ in range(protocol.evaluation.bootstrap_resamples):
        sampled_seeds = rng.choice(seeds, size=5, replace=True)
        seed_effects = []
        for raw_seed in sampled_seeds:
            fft, dct = pairs[int(raw_seed)]
            indices = rng.integers(0, len(fft), size=len(fft))
            selected_fft, selected_dct = fft[indices], dct[indices]
            available = np.isfinite(selected_fft) & np.isfinite(selected_dct)
            if not np.any(available):
                continue
            fft_median = np.median(selected_fft[available])
            dct_median = np.median(selected_dct[available])
            seed_effects.append((fft_median - dct_median) / fft_median)
        if seed_effects:
            effects.append(float(np.median(seed_effects)))
    if not effects:
        return {
            "resamples": protocol.evaluation.bootstrap_resamples,
            "available_resamples": 0,
            "relative_improvement_ci_low": None,
            "relative_improvement_ci_high": None,
        }
    low, high = np.percentile(effects, [2.5, 97.5])
    return {
        "resamples": protocol.evaluation.bootstrap_resamples,
        "available_resamples": len(effects),
        "relative_improvement_ci_low": float(low),
        "relative_improvement_ci_high": float(high),
    }


def collect_phase6(
    root: Path, protocol: Phase6Protocol, archive: Path, data_path: Path
) -> dict[str, Any]:
    phase2, _, rows = validate_phase6_contract(root, protocol, archive)
    dataset = validate_confirmatory_dataset(data_path, rows, phase2, protocol)
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for model in MODEL_NAMES:
        for seed in protocol.evaluation.seeds:
            run_id = _evaluation_run_id(protocol, model, seed)
            run_root = root / protocol.outputs.artifact_root / run_id
            if (run_root / "summary.json").is_file():
                manifest = RunManifest.from_dict(
                    cast(dict[str, Any], json.loads((run_root / "run_manifest.json").read_text()))
                )
                if manifest.execution_status == ExecutionStatus.COMPLETED:
                    summaries[(model, seed)] = cast(
                        dict[str, Any], json.loads((run_root / "summary.json").read_text())
                    )
    decision = phase6_decision(summaries, protocol)
    if decision["complete"]:
        decision["hierarchical_bootstrap"] = _bootstrap(root, protocol, summaries)
    report = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "phase": 6,
        "protocol_sha256": protocol.digest,
        "dataset": dataset,
        **decision,
    }
    report_root = root / protocol.outputs.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_root / "decision_report.json", report)
    atomic_write_json(report_root / "protocol_snapshot.json", protocol.model_dump(mode="json"))
    if decision.get("paired_results"):
        _write_csv(
            report_root / "paired_seed_results.csv",
            cast(list[dict[str, Any]], decision["paired_results"]),
        )
    output = (
        root
        / protocol.outputs.artifact_root
        / f"chronopde_v2-p6-confirmatory-study-{protocol.digest[:8]}"
        / "chronopde_v2_phase6_confirmatory_results.zip"
    )
    if decision["complete"]:
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
            for path in report_root.glob("*"):
                if path.is_file():
                    package.write(path, path.relative_to(root))
            for summary in summaries.values():
                run_root = root / protocol.outputs.artifact_root / str(summary["run_id"])
                for name in (
                    "summary.json",
                    "run_manifest.json",
                    "resolved_config.json",
                    "per_trajectory_rollouts.csv",
                    "per_trajectory_attribution.csv",
                ):
                    package.write(run_root / name, (run_root / name).relative_to(root))
        if zipfile.ZipFile(output).testzip() is not None:
            raise ValueError("Phase 6 results package failed CRC validation")
        report["results_archive"] = str(output)
    return report


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase6",))
    parser.add_argument("action", choices=("generate", "evaluate", "collect"))
    parser.add_argument("--config", type=Path, default=Path("configs/chronopde_v2/phase6.yaml"))
    parser.add_argument("--phase5-archive", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        protocol = load_phase6_protocol(config.resolve())
        archive = args.phase5_archive.resolve()
        if args.action == "generate":
            report = generate_confirmatory(root, protocol, archive)
        elif args.action == "evaluate":
            if args.data_path is None or args.model is None or args.seed is None:
                raise ValueError("evaluate requires --data-path, --model and --seed")
            report = evaluate_confirmatory(
                root,
                protocol,
                archive,
                args.data_path.resolve(),
                cast(ModelName, args.model),
                args.seed,
                _device(args.device),
            )
        else:
            if args.data_path is None:
                raise ValueError("collect requires --data-path")
            report = collect_phase6(root, protocol, archive, args.data_path.resolve())
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 2
    except RuntimeError as error:
        print(str(error))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
