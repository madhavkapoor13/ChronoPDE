"""Resumable exact-RHS development dataset generation for V2 Phase 3."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from chronopde.contracts import PhysicalParameters
from chronopde.data.simulator import simulate_trajectory
from chronopde.v2.common import atomic_write_json, canonical_json_bytes
from chronopde.v2.numerical import exact_discrete_rhs, relative_l2
from chronopde.v2.pilot import (
    development_rows,
    exact_rhs_series,
    future_manifest,
    generate_v2_initial_condition,
    manifest_hash,
)
from chronopde.v2.protocol import Phase2Protocol, Phase3Protocol

Float32 = NDArray[np.float32]
Float64 = NDArray[np.float64]


def _cache_path(root: Path, row: dict[str, str | int | float]) -> Path:
    return root / str(row["split"]) / f"{row['trajectory_id']}.npz"


def _logical_trajectory_hash(
    row: dict[str, str | int | float],
    states: Float32,
    rhs: Float32,
    times: Float64,
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(row))
    for name, value in (("states", states), ("rhs", rhs), ("times", times)):
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_cache(
    path: Path,
    row: dict[str, str | int | float],
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
) -> tuple[Float32, Float32, Float64, dict[str, int | float]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float32)
            rhs = np.asarray(data["rhs"], dtype=np.float32)
            times = np.asarray(data["times"], dtype=np.float64)
            expected_shape = (
                phase2.pde.stored_times,
                2,
                phase2.pde.height,
                phase2.pde.width,
            )
            if states.shape != expected_shape or rhs.shape != expected_shape:
                raise ValueError("cached trajectory shape mismatch")
            if times.shape != (phase2.pde.stored_times,):
                raise ValueError("cached time shape mismatch")
            if str(data["trajectory_id"].item()) != row["trajectory_id"]:
                raise ValueError("cached trajectory identity mismatch")
            if str(data["phase2_hash"].item()) != phase2.digest:
                raise ValueError("cached Phase 2 identity mismatch")
            if str(data["phase3_hash"].item()) != phase3.digest:
                raise ValueError("cached Phase 3 identity mismatch")
            if int(data["ic_seed"].item()) != int(row["ic_seed"]):
                raise ValueError("cached initial-condition identity mismatch")
            parameters = np.asarray(data["parameters"], dtype=np.float64)
            expected_parameters = np.asarray(
                [row["du"], row["dv"], row["k"]], dtype=np.float64
            )
            if not np.array_equal(parameters, expected_parameters):
                raise ValueError("cached parameters mismatch")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(rhs)):
                raise ValueError("cached trajectory contains non-finite values")
            digest = _logical_trajectory_hash(row, states, rhs, times)
            if str(data["logical_sha256"].item()) != digest:
                raise ValueError("cached trajectory checksum mismatch")
            diagnostics: dict[str, int | float] = {
                "nfev": int(data["nfev"].item()),
                "max_abs_state": float(data["max_abs_state"].item()),
            }
            return states, rhs, times, diagnostics
    except (EOFError, KeyError, OSError, ValueError) as error:
        raise ValueError(f"invalid trajectory cache {path}: {error}") from error


def _write_cache(
    path: Path,
    row: dict[str, str | int | float],
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
    states: Float32,
    rhs: Float32,
    times: Float64,
    nfev: int,
    max_abs_state: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _logical_trajectory_hash(row, states, rhs, times)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            trajectory_id=row["trajectory_id"],
            split=row["split"],
            index=row["index"],
            ic_seed=row["ic_seed"],
            parameters=np.asarray([row["du"], row["dv"], row["k"]], dtype=np.float64),
            states=states,
            rhs=rhs,
            times=times,
            nfev=nfev,
            max_abs_state=max_abs_state,
            phase2_hash=phase2.digest,
            phase3_hash=phase3.digest,
            logical_sha256=digest,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_one(
    row: dict[str, str | int | float],
    cache_root: Path,
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
) -> dict[str, Any]:
    path = _cache_path(cache_root, row)
    cached = _load_cache(path, row, phase2, phase3)
    reused = cached is not None
    if cached is None:
        params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
        initial = generate_v2_initial_condition(phase2, int(row["ic_seed"]))
        result = simulate_trajectory(
            initial,
            params,
            phase2.pde,
            divergence_threshold=phase2.tolerances.divergence_threshold,
        )
        if not result.diagnostics.success:
            raise RuntimeError(f"development trajectory failed: {row['trajectory_id']}")
        states = result.states
        rhs = exact_rhs_series(states, params, phase2.pde)
        times = result.times
        diagnostics: dict[str, int | float] = {
            "nfev": result.diagnostics.nfev,
            "max_abs_state": result.diagnostics.max_abs_state,
        }
        _write_cache(
            path,
            row,
            phase2,
            phase3,
            states,
            rhs,
            times,
            int(diagnostics["nfev"]),
            float(diagnostics["max_abs_state"]),
        )
    else:
        states, rhs, times, diagnostics = cached
    return {
        **row,
        "cached": reused,
        "nfev": int(diagnostics["nfev"]),
        "max_abs_state": float(diagnostics["max_abs_state"]),
        "final_mean_u": float(np.mean(states[-1, 0])),
        "final_mean_v": float(np.mean(states[-1, 1])),
        "final_std_u": float(np.std(states[-1, 0])),
        "final_std_v": float(np.std(states[-1, 1])),
    }


def _training_normalization(
    rows: list[dict[str, str | int | float]], cache_root: Path, phase2: Phase2Protocol,
    phase3: Phase3Protocol,
) -> tuple[Float64, Float64, Float64, Float64]:
    total = np.zeros(2, dtype=np.float64)
    total_square = np.zeros(2, dtype=np.float64)
    count = 0
    parameters: list[Float64] = []
    for row in rows:
        if row["split"] != "training":
            continue
        cached = _load_cache(_cache_path(cache_root, row), row, phase2, phase3)
        if cached is None:
            raise ValueError("training cache disappeared before normalization")
        states = cached[0].astype(np.float64)
        total += states.sum(axis=(0, 2, 3))
        total_square += np.square(states).sum(axis=(0, 2, 3))
        count += states.shape[0] * states.shape[2] * states.shape[3]
        parameters.append(np.asarray([row["du"], row["dv"], row["k"]], dtype=np.float64))
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    parameter_array = np.stack(parameters)
    return mean, np.sqrt(variance), parameter_array.mean(axis=0), parameter_array.std(axis=0)


def _dataset_logical_hash(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for split in ("training", "validation"):
        group = handle[f"splits/{split}"]
        for name in ("states", "rhs", "times", "parameters", "ic_seeds"):
            dataset = group[name]
            digest.update(f"{split}/{name}".encode("ascii"))
            digest.update(str(dataset.dtype).encode("ascii"))
            digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
            if dataset.ndim > 0:
                for index in range(dataset.shape[0]):
                    digest.update(np.asarray(dataset[index]).tobytes())
            else:
                digest.update(np.asarray(dataset[()]).tobytes())
    normalization = handle["normalization"]
    for name in ("state_mean", "state_std", "parameter_mean", "parameter_std"):
        digest.update(f"normalization/{name}".encode("ascii"))
        digest.update(np.asarray(normalization[name]).tobytes())
    return digest.hexdigest()


def _assemble_hdf5(
    output: Path,
    rows: list[dict[str, str | int | float]],
    cache_root: Path,
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
) -> None:
    state_mean, state_std, parameter_mean, parameter_std = _training_normalization(
        rows, cache_root, phase2, phase3
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = "chronopde_v2_development_1"
            handle.attrs["phase2_protocol_sha256"] = phase2.digest
            handle.attrs["phase3_protocol_sha256"] = phase3.digest
            handle.attrs["development_manifest_sha256"] = manifest_hash(rows)
            handle.attrs["frozen_future_manifest_sha256"] = (
                phase3.frozen_future_manifest_sha256
            )
            handle.attrs["confirmatory_status"] = "sealed_not_generated"
            splits = handle.create_group("splits")
            for split, expected_count in (
                ("training", phase3.development_splits.training),
                ("validation", phase3.development_splits.validation),
            ):
                selected = [row for row in rows if row["split"] == split]
                if len(selected) != expected_count:
                    raise ValueError(f"incorrect {split} manifest count")
                group = splits.create_group(split)
                shape = (
                    expected_count,
                    phase2.pde.stored_times,
                    2,
                    phase2.pde.height,
                    phase2.pde.width,
                )
                states_ds = group.create_dataset(
                    "states",
                    shape=shape,
                    dtype=np.float32,
                    chunks=(1, 1, 2, phase2.pde.height, phase2.pde.width),
                    compression="lzf",
                )
                rhs_ds = group.create_dataset(
                    "rhs",
                    shape=shape,
                    dtype=np.float32,
                    chunks=(1, 1, 2, phase2.pde.height, phase2.pde.width),
                    compression="lzf",
                )
                times_ds = group.create_dataset(
                    "times", shape=(expected_count, phase2.pde.stored_times), dtype=np.float64
                )
                parameters_ds = group.create_dataset(
                    "parameters", shape=(expected_count, 3), dtype=np.float64
                )
                seeds_ds = group.create_dataset("ic_seeds", shape=(expected_count,), dtype=np.int64)
                string_type = h5py.string_dtype(encoding="utf-8")
                ids_ds = group.create_dataset(
                    "trajectory_ids", shape=(expected_count,), dtype=string_type
                )
                for index, row in enumerate(selected):
                    cached = _load_cache(_cache_path(cache_root, row), row, phase2, phase3)
                    if cached is None:
                        raise ValueError("trajectory cache disappeared during assembly")
                    states, rhs, times, _ = cached
                    states_ds[index] = states
                    rhs_ds[index] = rhs
                    times_ds[index] = times
                    parameters_ds[index] = np.asarray(
                        [row["du"], row["dv"], row["k"]], dtype=np.float64
                    )
                    seeds_ds[index] = int(row["ic_seed"])
                    ids_ds[index] = str(row["trajectory_id"])
            normalization = handle.create_group("normalization")
            normalization.attrs["source_split"] = "training"
            normalization.create_dataset("state_mean", data=state_mean)
            normalization.create_dataset("state_std", data=state_std)
            normalization.create_dataset("parameter_mean", data=parameter_mean)
            normalization.create_dataset("parameter_std", data=parameter_std)
            handle.attrs["logical_content_sha256"] = _dataset_logical_hash(handle)
            handle.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_development_dataset(
    path: Path,
    rows: list[dict[str, str | int | float]],
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
) -> dict[str, Any]:
    """Validate schema, identities, normalization, hashes, and exact RHS labels."""

    if not path.is_file():
        raise FileNotFoundError(f"development dataset not found: {path}")
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != "chronopde_v2_development_1":
            raise ValueError("development dataset schema mismatch")
        if handle.attrs.get("phase2_protocol_sha256") != phase2.digest:
            raise ValueError("development dataset Phase 2 identity mismatch")
        if handle.attrs.get("phase3_protocol_sha256") != phase3.digest:
            raise ValueError("development dataset Phase 3 identity mismatch")
        if handle.attrs.get("development_manifest_sha256") != manifest_hash(rows):
            raise ValueError("development dataset manifest mismatch")
        if handle.attrs.get("confirmatory_status") != "sealed_not_generated":
            raise ValueError("confirmatory split must remain sealed and absent")
        if "confirmatory" in handle["splits"]:
            raise ValueError("confirmatory data must not be generated in Phase 3")
        for split, expected_count in (
            ("training", phase3.development_splits.training),
            ("validation", phase3.development_splits.validation),
        ):
            group = handle[f"splits/{split}"]
            expected_shape = (
                expected_count,
                phase2.pde.stored_times,
                2,
                phase2.pde.height,
                phase2.pde.width,
            )
            if group["states"].shape != expected_shape or group["rhs"].shape != expected_shape:
                raise ValueError(f"incorrect {split} development shape")
            if group["states"].dtype != np.float32 or group["rhs"].dtype != np.float32:
                raise ValueError("development states and RHS must use float32")
            selected = [row for row in rows if row["split"] == split]
            actual_ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(group["trajectory_ids"])
            ]
            expected_ids = [str(row["trajectory_id"]) for row in selected]
            if actual_ids != expected_ids:
                raise ValueError(f"{split} trajectory identities do not match the manifest")
            expected_parameters = np.asarray(
                [[row["du"], row["dv"], row["k"]] for row in selected],
                dtype=np.float64,
            )
            if not np.array_equal(np.asarray(group["parameters"]), expected_parameters):
                raise ValueError(f"{split} parameters do not match the manifest")
            expected_seeds = np.asarray([row["ic_seed"] for row in selected], dtype=np.int64)
            if not np.array_equal(np.asarray(group["ic_seeds"]), expected_seeds):
                raise ValueError(f"{split} seeds do not match the manifest")
        state_std = np.asarray(handle["normalization/state_std"], dtype=np.float64)
        parameter_std = np.asarray(handle["normalization/parameter_std"], dtype=np.float64)
        if np.any(state_std <= 0) or np.any(parameter_std <= 0):
            raise ValueError("development normalization scales must be positive")
        training_states = handle["splits/training/states"]
        total = np.zeros(2, dtype=np.float64)
        total_square = np.zeros(2, dtype=np.float64)
        count = 0
        for index in range(training_states.shape[0]):
            state = np.asarray(training_states[index], dtype=np.float64)
            total += state.sum(axis=(0, 2, 3))
            total_square += np.square(state).sum(axis=(0, 2, 3))
            count += state.shape[0] * state.shape[2] * state.shape[3]
        expected_state_mean = total / count
        expected_state_std = np.sqrt(
            np.maximum(total_square / count - np.square(expected_state_mean), 0.0)
        )
        training_parameters = np.asarray(
            handle["splits/training/parameters"], dtype=np.float64
        )
        expected_normalization = {
            "state_mean": expected_state_mean,
            "state_std": expected_state_std,
            "parameter_mean": training_parameters.mean(axis=0),
            "parameter_std": training_parameters.std(axis=0),
        }
        for name, expected in expected_normalization.items():
            actual = np.asarray(handle[f"normalization/{name}"], dtype=np.float64)
            if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"training-only normalization mismatch: {name}")
        selected_rows = [row for row in rows if row["split"] == "training"]
        rng = np.random.default_rng(phase2.data.master_seed + 3)
        chosen = rng.choice(
            len(selected_rows) * phase2.pde.stored_times,
            size=phase3.rhs_spot_checks,
            replace=False,
        )
        rhs_errors: list[float] = []
        for flat_index in chosen:
            trajectory_index, time_index = divmod(int(flat_index), phase2.pde.stored_times)
            row = selected_rows[trajectory_index]
            state = np.asarray(
                handle["splits/training/states"][trajectory_index, time_index],
                dtype=np.float64,
            )
            stored_rhs = np.asarray(
                handle["splits/training/rhs"][trajectory_index, time_index],
                dtype=np.float64,
            )
            params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
            expected_rhs = exact_discrete_rhs(state, params, phase2.pde)
            rhs_errors.append(relative_l2(stored_rhs, expected_rhs))
        maximum_rhs_error = max(rhs_errors)
        if maximum_rhs_error > phase2.tolerances.float32_relative:
            raise ValueError("exact-RHS spot check exceeds the frozen tolerance")
        logical_hash = _dataset_logical_hash(handle)
        if logical_hash != handle.attrs.get("logical_content_sha256"):
            raise ValueError("development dataset logical checksum mismatch")
        normalization_payload = {
            name: np.asarray(handle[f"normalization/{name}"], dtype=np.float64).tolist()
            for name in ("state_mean", "state_std", "parameter_mean", "parameter_std")
        }
        normalization_hash = hashlib.sha256(
            canonical_json_bytes(normalization_payload)
        ).hexdigest()
    return {
        "logical_content_sha256": logical_hash,
        "normalization_sha256": normalization_hash,
        "normalization": normalization_payload,
        "rhs_spot_checks": len(rhs_errors),
        "maximum_rhs_spot_check_relative_l2": maximum_rhs_error,
        "training_trajectories": phase3.development_splits.training,
        "validation_trajectories": phase3.development_splits.validation,
        "confirmatory_trajectories_generated": 0,
    }


def generate_development_dataset(
    phase2: Phase2Protocol,
    phase3: Phase3Protocol,
    artifact_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate caches, assemble the dataset, and return deterministic QA evidence."""

    full_rows = future_manifest(phase2)
    if manifest_hash(full_rows) != phase3.frozen_future_manifest_sha256:
        raise ValueError("Phase 2 future manifest hash does not match Phase 3")
    rows = development_rows(full_rows)
    if len(rows) != (
        phase3.development_splits.training + phase3.development_splits.validation
    ):
        raise ValueError("development manifest count mismatch")
    cache_root = artifact_directory / "trajectory_cache"
    with ThreadPoolExecutor(max_workers=phase3.workers) as executor:
        qa_rows = list(
            executor.map(
                lambda row: _generate_one(row, cache_root, phase2, phase3),
                rows,
            )
        )
    output = artifact_directory / "chronopde_v2_development.h5"
    if not output.exists():
        _assemble_hdf5(output, rows, cache_root, phase2, phase3)
    validated = validate_development_dataset(output, rows, phase2, phase3)
    maximum_state = max(float(row["max_abs_state"]) for row in qa_rows)
    if maximum_state > phase2.tolerances.divergence_threshold:
        raise ValueError("development dataset exceeds the divergence threshold")
    summary = {
        "passed": True,
        **validated,
        "development_manifest_sha256": manifest_hash(rows),
        "frozen_future_manifest_sha256": manifest_hash(full_rows),
        "successful_trajectories": len(qa_rows),
        "trajectory_cache_entries": len(qa_rows),
        "maximum_absolute_state": maximum_state,
        "target": "exact_discrete_rhs_per_physical_time",
        "normalization_source": "training_split_only",
        "confirmatory_status": "sealed_not_generated",
    }
    atomic_write_json(artifact_directory / "dataset_runtime_summary.json", summary)
    return summary, qa_rows


def read_runtime_summary(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
