"""Resumable full-dataset generation, HDF5 assembly, and QA reporting."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from chronopde.config import ProjectConfig
from chronopde.contracts import GenerationDiagnostics, TrajectoryManifestEntry
from chronopde.data.initial_conditions import generate_initial_condition
from chronopde.data.manifest import (
    build_dataset_manifest,
    manifest_hash,
    write_frozen_manifest,
)
from chronopde.data.masks import MaskRegime, observation_mask
from chronopde.data.pilot import configuration_hash
from chronopde.data.simulator import simulate_trajectory
from chronopde.numerics.laplacian import build_grid
from chronopde.reproducibility import git_commit

SPLITS = ("train", "validation", "id", "oodparam", "oodic")
MASK_REGIMES: tuple[MaskRegime, ...] = ("full", "irregular_50", "irregular_25")


@dataclass(frozen=True)
class DatasetGenerationReport:
    passed: bool
    total: int
    successful: int
    cached: int
    manifest_hash: str
    config_hash: str
    output_path: str
    diagnostics_path: str


def _absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def prepare_manifest(
    config: ProjectConfig, root: Path
) -> tuple[list[TrajectoryManifestEntry], str]:
    entries = build_dataset_manifest(config)
    path = _absolute(root, config.data.manifest_path)
    digest = write_frozen_manifest(path, entries)
    return entries, digest


def _trajectory_path(directory: Path, entry: TrajectoryManifestEntry) -> Path:
    return directory / "trajectories" / entry.split / f"{entry.trajectory_id}.npz"


def _cache_valid(
    path: Path,
    entry: TrajectoryManifestEntry,
    config: ProjectConfig,
    config_digest: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_shape = (
                config.pde.stored_times,
                2,
                config.pde.height,
                config.pde.width,
            )
            return bool(
                str(data["trajectory_id"].item()) == entry.trajectory_id
                and str(data["config_hash"].item()) == config_digest
                and int(data["ic_seed"].item()) == entry.ic_seed
                and bool(data["success"].item())
                and data["states"].shape == expected_shape
                and data["times"].shape == (config.pde.stored_times,)
                and np.array_equal(data["params"], entry.params.as_array())
                and np.all(np.isfinite(data["states"]))
                and np.all(np.isfinite(data["times"]))
                and float(data["max_abs_state"].item()) <= config.data.pilot.divergence_threshold
            )
    except (KeyError, OSError, ValueError, EOFError):
        return False


def _save_trajectory(
    path: Path,
    entry: TrajectoryManifestEntry,
    config_digest: str,
    result: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        trajectory_id=entry.trajectory_id,
        split=entry.split,
        ic_seed=entry.ic_seed,
        ic_regime=entry.ic_regime,
        parameter_band=entry.parameter_band,
        params=entry.params.as_array(),
        states=result.states,
        times=result.times,
        success=result.diagnostics.success,
        runtime_seconds=result.diagnostics.runtime_seconds,
        nfev=result.diagnostics.nfev,
        max_abs_state=result.diagnostics.max_abs_state,
        message=result.diagnostics.message,
        config_hash=config_digest,
    )
    temporary.replace(path)


def _generate_one(
    entry: TrajectoryManifestEntry,
    config: ProjectConfig,
    directory: Path,
    config_digest: str,
) -> tuple[GenerationDiagnostics, bool]:
    path = _trajectory_path(directory, entry)
    if _cache_valid(path, entry, config, config_digest):
        with np.load(path, allow_pickle=False) as data:
            return (
                GenerationDiagnostics(
                    trajectory_id=entry.trajectory_id,
                    success=True,
                    runtime_seconds=float(data["runtime_seconds"].item()),
                    nfev=int(data["nfev"].item()),
                    max_abs_state=float(data["max_abs_state"].item()),
                    message=str(data["message"].item()),
                ),
                True,
            )
    grid = build_grid(config.pde)
    initial = generate_initial_condition(
        grid, entry.ic_seed, entry.ic_regime, config.data.initial_conditions
    )
    result = simulate_trajectory(
        initial,
        entry.params,
        config.pde,
        divergence_threshold=config.data.pilot.divergence_threshold,
    )
    _save_trajectory(path, entry, config_digest, result)
    return (
        GenerationDiagnostics(
            trajectory_id=entry.trajectory_id,
            success=result.diagnostics.success,
            runtime_seconds=result.diagnostics.runtime_seconds,
            nfev=result.diagnostics.nfev,
            max_abs_state=result.diagnostics.max_abs_state,
            message=result.diagnostics.message,
        ),
        False,
    )


def _write_diagnostics(
    path: Path,
    entries: list[TrajectoryManifestEntry],
    results: list[tuple[GenerationDiagnostics, bool]],
    directory: Path,
) -> list[dict[str, Any]]:
    by_id = {entry.trajectory_id: entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for diagnostic, cached in results:
        entry = by_id[diagnostic.trajectory_id]
        with np.load(_trajectory_path(directory, entry), allow_pickle=False) as trajectory:
            final_state = np.asarray(trajectory["states"][-1], dtype=np.float64)
        rows.append(
            {
                "trajectory_id": entry.trajectory_id,
                "split": entry.split,
                "index": entry.index,
                "du": entry.params.du,
                "dv": entry.params.dv,
                "k": entry.params.k,
                "ic_seed": entry.ic_seed,
                "ic_regime": entry.ic_regime,
                "parameter_band": entry.parameter_band,
                "success": diagnostic.success,
                "cached": cached,
                "runtime_seconds": diagnostic.runtime_seconds,
                "nfev": diagnostic.nfev,
                "max_abs_state": diagnostic.max_abs_state,
                "final_mean_u": float(np.mean(final_state[0])),
                "final_mean_v": float(np.mean(final_state[1])),
                "final_std_u": float(np.std(final_state[0])),
                "final_std_v": float(np.std(final_state[1])),
                "message": diagnostic.message,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _training_normalization(
    entries: list[TrajectoryManifestEntry], directory: Path
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    total = np.zeros(2, dtype=np.float64)
    total_square = np.zeros(2, dtype=np.float64)
    count = 0
    parameters: list[NDArray[np.float64]] = []
    for entry in entries:
        if entry.split != "train":
            continue
        with np.load(_trajectory_path(directory, entry), allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float64)
        total += states.sum(axis=(0, 2, 3))
        total_square += np.square(states).sum(axis=(0, 2, 3))
        count += states.shape[0] * states.shape[2] * states.shape[3]
        parameters.append(entry.params.as_array())
    state_mean = total / count
    state_variance = np.maximum(total_square / count - np.square(state_mean), 0)
    parameter_array = np.stack(parameters)
    return state_mean, np.sqrt(state_variance), parameter_array.mean(0), parameter_array.std(0)


def _write_hdf5(
    output: Path,
    entries: list[TrajectoryManifestEntry],
    directory: Path,
    config: ProjectConfig,
    config_digest: str,
    manifest_digest: str,
    commit: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    grid = build_grid(config.pde)
    state_mean, state_std, parameter_mean, parameter_std = _training_normalization(
        entries, directory
    )
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["config_hash"] = config_digest
        handle.attrs["manifest_hash"] = manifest_digest
        handle.attrs["source_commit"] = commit or "uncommitted"
        grid_group = handle.create_group("grid")
        grid_group.create_dataset("x", data=grid.x)
        grid_group.create_dataset("y", data=grid.y)
        split_group = handle.create_group("splits")
        string_type = h5py.string_dtype(encoding="utf-8")
        for split in SPLITS:
            split_entries = [entry for entry in entries if entry.split == split]
            count = len(split_entries)
            group = split_group.create_group(split)
            states_ds = group.create_dataset(
                "states",
                shape=(count, config.pde.stored_times, 2, config.pde.height, config.pde.width),
                dtype=np.float32,
                chunks=(1, 1, 2, config.pde.height, config.pde.width),
                compression="lzf",
            )
            times_ds = group.create_dataset(
                "times", shape=(count, config.pde.stored_times), dtype=np.float64
            )
            group.create_dataset(
                "parameters", data=np.stack([entry.params.as_array() for entry in split_entries])
            )
            group.create_dataset("ic_seeds", data=[entry.ic_seed for entry in split_entries])
            group.create_dataset(
                "trajectory_ids",
                data=np.asarray([entry.trajectory_id for entry in split_entries], dtype=object),
                dtype=string_type,
            )
            masks = group.create_group("masks")
            for regime in MASK_REGIMES:
                masks.create_dataset(
                    regime,
                    data=np.stack(
                        [
                            observation_mask(
                                config.pde.stored_times,
                                entry.trajectory_id,
                                regime,
                                config.data.base_seed,
                            )
                            for entry in split_entries
                        ]
                    ),
                    dtype=np.bool_,
                )
            diagnostics = group.create_group("diagnostics")
            runtime = np.empty(count, dtype=np.float64)
            nfev = np.empty(count, dtype=np.int64)
            maximum = np.empty(count, dtype=np.float64)
            success = np.empty(count, dtype=np.bool_)
            for index, entry in enumerate(split_entries):
                with np.load(_trajectory_path(directory, entry), allow_pickle=False) as data:
                    states_ds[index] = data["states"]
                    times_ds[index] = data["times"]
                    runtime[index] = data["runtime_seconds"]
                    nfev[index] = data["nfev"]
                    maximum[index] = data["max_abs_state"]
                    success[index] = data["success"]
            diagnostics.create_dataset("runtime", data=runtime)
            diagnostics.create_dataset("nfev", data=nfev)
            diagnostics.create_dataset("max_abs_state", data=maximum)
            diagnostics.create_dataset("success", data=success)
        normalization = handle.create_group("normalization")
        normalization.create_dataset("state_mean", data=state_mean)
        normalization.create_dataset("state_std", data=state_std)
        normalization.create_dataset("parameter_mean", data=parameter_mean)
        normalization.create_dataset("parameter_std", data=parameter_std)
        handle.flush()
    temporary.replace(output)


def validate_hdf5(
    output: Path,
    entries: list[TrajectoryManifestEntry],
    directory: Path,
    config: ProjectConfig,
    config_digest: str,
    manifest_digest: str,
) -> None:
    with h5py.File(output, "r") as handle:
        if handle.attrs["config_hash"] != config_digest:
            raise ValueError("HDF5 configuration hash mismatch")
        if handle.attrs["manifest_hash"] != manifest_digest:
            raise ValueError("HDF5 manifest hash mismatch")
        for split in SPLITS:
            expected = [entry for entry in entries if entry.split == split]
            group = handle[f"splits/{split}"]
            states = group["states"]
            if states.shape != (
                len(expected),
                config.pde.stored_times,
                2,
                config.pde.height,
                config.pde.width,
            ):
                raise ValueError(f"incorrect state shape for {split}")
            if states.dtype != np.float32 or states.compression != "lzf":
                raise ValueError(f"incorrect state storage for {split}")
            if not bool(np.asarray(group["diagnostics/success"]).all()):
                raise ValueError(f"failed trajectory recorded in {split}")
            if (
                float(np.max(group["diagnostics/max_abs_state"]))
                > config.data.pilot.divergence_threshold
            ):
                raise ValueError(f"divergent trajectory recorded in {split}")
        rng = np.random.default_rng(config.data.base_seed)
        chosen = rng.choice(len(entries), size=min(10, len(entries)), replace=False)
        split_positions = {split: 0 for split in SPLITS}
        locations: dict[int, tuple[str, int]] = {}
        for global_index, entry in enumerate(entries):
            locations[global_index] = (entry.split, split_positions[entry.split])
            split_positions[entry.split] += 1
        for global_index in chosen:
            entry = entries[int(global_index)]
            split, local_index = locations[int(global_index)]
            with np.load(_trajectory_path(directory, entry), allow_pickle=False) as data:
                if not np.array_equal(
                    handle[f"splits/{split}/states"][local_index], data["states"]
                ):
                    raise ValueError(f"HDF5/source mismatch for {entry.trajectory_id}")


def _publish_qa(
    report_directory: Path,
    rows: list[dict[str, Any]],
    entries: list[TrajectoryManifestEntry],
    config: ProjectConfig,
    directory: Path,
    summary: dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    colours = {name: index for index, name in enumerate(SPLITS)}
    for split in SPLITS:
        selected = [entry for entry in entries if entry.split == split]
        axes[0].scatter(
            [x.params.du for x in selected],
            [x.params.dv for x in selected],
            s=8,
            label=split,
            c=[colours[split]] * len(selected),
        )
        axes[1].scatter(
            [x.params.du for x in selected],
            [x.params.k for x in selected],
            s=8,
            c=[colours[split]] * len(selected),
        )
        axes[2].scatter(
            [x.params.dv for x in selected],
            [x.params.k for x in selected],
            s=8,
            c=[colours[split]] * len(selected),
        )
    axes[0].set(xlabel="Du", ylabel="Dv")
    axes[1].set(xlabel="Du", ylabel="k")
    axes[2].set(xlabel="Dv", ylabel="k")
    axes[0].legend(fontsize=7)
    figure.savefig(report_directory / "parameter_coverage.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].hist([float(row["runtime_seconds"]) for row in rows], bins=30)
    axes[0].set_title("Runtime (seconds)")
    axes[1].hist([int(row["nfev"]) for row in rows], bins=30)
    axes[1].set_title("Function evaluations")
    axes[2].hist([float(row["max_abs_state"]) for row in rows], bins=30)
    axes[2].set_title("Maximum magnitude")
    figure.savefig(report_directory / "diagnostic_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].hist(
        [
            [float(row["final_mean_u"]) for row in rows],
            [float(row["final_mean_v"]) for row in rows],
        ],
        bins=30,
        label=("u", "v"),
    )
    axes[0].set_title("Final channel means")
    axes[1].hist(
        [
            [float(row["final_std_u"]) for row in rows],
            [float(row["final_std_v"]) for row in rows],
        ],
        bins=30,
        label=("u", "v"),
    )
    axes[1].set_title("Final channel standard deviations")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.savefig(report_directory / "state_distributions.png", dpi=160)
    plt.close(figure)

    representatives = [next(entry for entry in entries if entry.split == split) for split in SPLITS]
    figure, axes = plt.subplots(len(SPLITS), 4, figsize=(11, 13), constrained_layout=True)
    for row, entry in enumerate(representatives):
        with np.load(_trajectory_path(directory, entry), allow_pickle=False) as data:
            states = data["states"]
        for column, (time_index, channel) in enumerate(((0, 0), (0, 1), (-1, 0), (-1, 1))):
            axes[row, column].imshow(states[time_index, channel], cmap="coolwarm")
            axes[row, column].set_title(
                f"{entry.split}: {'uv'[channel]} {'initial' if time_index == 0 else 'final'}"
            )
            axes[row, column].set_axis_off()
    figure.savefig(report_directory / "state_panels.png", dpi=160)
    plt.close(figure)

    mask_examples = np.stack(
        [
            observation_mask(
                config.pde.stored_times, entry.trajectory_id, regime, config.data.base_seed
            )
            for entry in representatives
            for regime in MASK_REGIMES
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.imshow(mask_examples, aspect="auto", interpolation="nearest", cmap="Greys")
    axis.set(xlabel="stored time index", ylabel="split/regime example", title="Temporal masks")
    figure.savefig(report_directory / "observation_masks.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.bar(("full", "irregular_50", "irregular_25"), (101, 52, 27))
    axis.set(ylabel="retained stored times", title="Observation-regime sizes")
    figure.savefig(report_directory / "mask_retention.png", dpi=160)
    plt.close(figure)


def run_full_generation(config: ProjectConfig, root: Path) -> DatasetGenerationReport:
    entries, manifest_digest = prepare_manifest(config, root)
    if manifest_digest != manifest_hash(entries):
        raise RuntimeError("manifest hashing is inconsistent")
    config_digest = configuration_hash(config)
    directory = _absolute(root, config.data.generation_directory)
    report_directory = _absolute(root, config.data.report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=config.data.workers) as executor:
        results = list(
            executor.map(
                lambda entry: _generate_one(entry, config, directory, config_digest),
                entries,
            )
        )
    rows = _write_diagnostics(report_directory / "diagnostics.csv", entries, results, directory)
    successful = sum(result.success for result, _ in results)
    cached = sum(was_cached for _, was_cached in results)
    output = _absolute(root, config.data.output_path)
    passed = successful == len(entries)
    if passed:
        _write_hdf5(
            output,
            entries,
            directory,
            config,
            config_digest,
            manifest_digest,
            git_commit(root),
        )
        validate_hdf5(output, entries, directory, config, config_digest, manifest_digest)
    summary: dict[str, object] = {
        "passed": passed,
        "total": len(entries),
        "successful": successful,
        "failed": len(entries) - successful,
        "cached": cached,
        "config_hash": config_digest,
        "manifest_hash": manifest_digest,
        "git_commit": git_commit(root),
        "maximum_state_magnitude": max(float(row["max_abs_state"]) for row in rows),
        "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in rows])),
        "median_nfev": float(np.median([row["nfev"] for row in rows])),
    }
    _publish_qa(report_directory, rows, entries, config, directory, summary)
    return DatasetGenerationReport(
        passed=passed,
        total=len(entries),
        successful=successful,
        cached=cached,
        manifest_hash=manifest_digest,
        config_hash=config_digest,
        output_path=str(output),
        diagnostics_path=str(report_directory / "diagnostics.csv"),
    )
