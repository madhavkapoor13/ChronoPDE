"""Fresh deterministic CPU pilot generation for ChronoPDE V2 Phase 2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.fft import idctn
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from chronopde.config import PDEConfig
from chronopde.contracts import PhysicalParameters
from chronopde.data.simulator import reaction_diffusion_rhs, simulate_trajectory
from chronopde.numerics.laplacian import build_grid, build_neumann_laplacian
from chronopde.v2.common import atomic_write_json, sha256_file
from chronopde.v2.numerical import relative_l2
from chronopde.v2.protocol import Phase2Protocol

Float32 = NDArray[np.float32]
Float64 = NDArray[np.float64]

MANIFEST_FIELDS = (
    "trajectory_id",
    "split",
    "index",
    "du",
    "dv",
    "k",
    "ic_seed",
)


def _bounds(protocol: Phase2Protocol) -> tuple[tuple[float, float], ...]:
    ranges = protocol.data.parameter_ranges
    return (ranges.du, ranges.dv, ranges.k)


def maximin_latin_hypercube(
    count: int,
    bounds: tuple[tuple[float, float], ...],
    seed: int,
    *,
    candidates: int = 64,
) -> Float64:
    """Choose the deterministic LHS candidate with greatest minimum separation."""

    if count < 2 or candidates < 1:
        raise ValueError("maximin design requires at least two points and one candidate")
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    if np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError("Latin-hypercube bounds must be positive and increasing")
    sequence = np.random.SeedSequence(seed).spawn(candidates)
    best: NDArray[np.float64] | None = None
    best_distance = -1.0
    for child in sequence:
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        unit = qmc.LatinHypercube(d=len(bounds), seed=child_seed).random(count)
        distance = float(np.min(pdist(unit)))
        if distance > best_distance:
            best = np.asarray(unit, dtype=np.float64)
            best_distance = distance
    if best is None:
        raise RuntimeError("failed to build Latin-hypercube design")
    return np.asarray(qmc.scale(best, lower, upper), dtype=np.float64)


def _manifest_rows(
    protocol: Phase2Protocol,
    split: str,
    count: int,
    offset: int,
) -> list[dict[str, str | int | float]]:
    values = maximin_latin_hypercube(
        count,
        _bounds(protocol),
        protocol.data.master_seed + offset,
    )
    prefix = "v2-pilot" if split == "pilot" else f"v2-{split}"
    return [
        {
            "trajectory_id": f"{prefix}-{index:04d}",
            "split": split,
            "index": index,
            "du": float(row[0]),
            "dv": float(row[1]),
            "k": float(row[2]),
            "ic_seed": protocol.data.master_seed + offset + index,
        }
        for index, row in enumerate(values)
    ]


def pilot_manifest(protocol: Phase2Protocol) -> list[dict[str, str | int | float]]:
    return _manifest_rows(
        protocol,
        "pilot",
        protocol.data.pilot_trajectories,
        protocol.data.seed_offsets.pilot,
    )


def future_manifest(protocol: Phase2Protocol) -> list[dict[str, str | int | float]]:
    """Build frozen identities for future splits without generating their states."""

    result: list[dict[str, str | int | float]] = []
    specs = (
        (
            "training",
            protocol.data.training_trajectories,
            protocol.data.seed_offsets.training,
        ),
        (
            "validation",
            protocol.data.validation_trajectories,
            protocol.data.seed_offsets.validation,
        ),
        (
            "confirmatory",
            protocol.data.confirmatory_trajectories,
            protocol.data.seed_offsets.confirmatory,
        ),
    )
    for split, count, offset in specs:
        result.extend(_manifest_rows(protocol, split, count, offset))
    return result


def development_rows(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    """Return only splits permitted during model development."""

    return [row for row in rows if row["split"] != "confirmatory"]


def confirmatory_rows(
    rows: list[dict[str, str | int | float]], *, frozen_model: bool = False
) -> list[dict[str, str | int | float]]:
    """Guard the sealed split until the caller records a frozen model contract."""

    if not frozen_model:
        raise PermissionError("confirmatory access requires a frozen model contract")
    return [row for row in rows if row["split"] == "confirmatory"]


def manifest_bytes(rows: list[dict[str, str | int | float]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def manifest_hash(rows: list[dict[str, str | int | float]]) -> str:
    return hashlib.sha256(manifest_bytes(rows)).hexdigest()


def generate_v2_initial_condition(protocol: Phase2Protocol, seed: int) -> Float32:
    """Generate smooth Neumann-compatible fields with decaying DCT coefficients."""

    pde = protocol.pde
    settings = protocol.data.initial_condition
    if seed < 0:
        raise ValueError("initial-condition seed must be non-negative")
    maximum_mode = settings.maximum_mode
    ky, kx = np.meshgrid(
        np.arange(maximum_mode + 1, dtype=np.float64),
        np.arange(maximum_mode + 1, dtype=np.float64),
        indexing="ij",
    )
    decay = (1.0 + ky**2 + kx**2) ** (-settings.spectral_decay / 2.0)
    fields: list[Float32] = []
    for child in np.random.SeedSequence(seed).spawn(2):
        rng = np.random.default_rng(child)
        coefficients = np.zeros((pde.height, pde.width), dtype=np.float64)
        coefficients[: maximum_mode + 1, : maximum_mode + 1] = (
            rng.standard_normal((maximum_mode + 1, maximum_mode + 1)) * decay
        )
        coefficients[0, 0] = 0.0
        field = idctn(coefficients, type=2, norm="ortho")
        field -= np.mean(field)
        observed = float(np.std(field))
        if not np.isfinite(observed) or observed <= 0:
            raise RuntimeError("initial-condition spectrum is degenerate")
        field *= settings.standard_deviation / observed
        fields.append(np.asarray(field, dtype=np.float32))
    return np.stack(fields).astype(np.float32, copy=False)


def _rhs_series(states: Float32, params: PhysicalParameters, pde: PDEConfig) -> Float32:
    grid = build_grid(pde)
    laplacian = build_neumann_laplacian(grid)
    result = np.empty_like(states, dtype=np.float32)
    spatial_size = pde.height * pde.width
    for index, state in enumerate(states):
        flat = np.concatenate((state[0].ravel(), state[1].ravel())).astype(np.float64)
        rhs = reaction_diffusion_rhs(0.0, flat, params, laplacian)
        result[index, 0] = rhs[:spatial_size].reshape(pde.height, pde.width)
        result[index, 1] = rhs[spatial_size:].reshape(pde.height, pde.width)
    return result


def _logical_content_hash(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in ("manifest_csv", "states", "rhs", "times", "parameters", "ic_seeds"):
        value = np.asarray(handle[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_pilot_file(path: Path, protocol: Phase2Protocol) -> dict[str, Any]:
    """Validate a pilot cache and recompute its logical content identity."""

    if not path.is_file():
        raise FileNotFoundError(f"pilot dataset not found: {path}")
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != "chronopde_v2_pilot_1":
            raise ValueError("pilot schema mismatch")
        if handle.attrs.get("protocol_hash") != protocol.digest:
            raise ValueError("pilot protocol hash mismatch")
        count = protocol.data.pilot_trajectories
        shape = (count, protocol.pde.stored_times, 2, protocol.pde.height, protocol.pde.width)
        if handle["states"].shape != shape or handle["rhs"].shape != shape:
            raise ValueError("pilot state or RHS shape mismatch")
        if handle["states"].dtype != np.float32 or handle["rhs"].dtype != np.float32:
            raise ValueError("pilot state and RHS must use float32 storage")
        if not np.all(np.isfinite(handle["states"])) or not np.all(np.isfinite(handle["rhs"])):
            raise ValueError("pilot contains non-finite values")
        actual = _logical_content_hash(handle)
        if actual != handle.attrs.get("logical_content_sha256"):
            raise ValueError("pilot logical content checksum mismatch")
        return {
            "logical_content_sha256": actual,
            "file_sha256": sha256_file(path),
            "trajectory_count": count,
            "state_shape": list(shape),
        }


def _trajectory_error(standard: Float32, tight: Float32) -> tuple[list[float], float]:
    errors = [relative_l2(standard[index], tight[index]) for index in range(len(standard))]
    return errors, relative_l2(standard[-1], tight[-1])


def generate_cpu_pilot(
    protocol: Phase2Protocol,
    output: Path,
    runtime_summary: Path,
) -> dict[str, Any]:
    """Generate or verify the 24-trajectory CPU pilot and exact RHS labels."""

    if output.exists() or runtime_summary.exists():
        if not output.is_file() or not runtime_summary.is_file():
            raise ValueError("incomplete pilot cache; remove it before regeneration")
        validated = validate_pilot_file(output, protocol)
        cached = cast(
            dict[str, Any], json.loads(runtime_summary.read_text(encoding="utf-8"))
        )
        if cached.get("logical_content_sha256") != validated["logical_content_sha256"]:
            raise ValueError("pilot runtime summary checksum mismatch")
        return cached

    rows = pilot_manifest(protocol)
    states: list[Float32] = []
    rhs_values: list[Float32] = []
    times: list[Float64] = []
    parameters: list[Float64] = []
    maximum_values: list[float] = []
    convergence_errors: list[float] = []
    final_errors: list[float] = []
    tight_indices = set(protocol.data.tight_solver_indices)
    tight_pde = protocol.pde.model_copy(update={"solver": protocol.data.tight_solver})

    for row in rows:
        params = PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"]))
        initial = generate_v2_initial_condition(protocol, int(row["ic_seed"]))
        result = simulate_trajectory(
            initial,
            params,
            protocol.pde,
            divergence_threshold=protocol.tolerances.divergence_threshold,
        )
        if not result.diagnostics.success:
            raise RuntimeError(f"pilot trajectory failed: {row['trajectory_id']}")
        states.append(result.states)
        rhs_values.append(_rhs_series(result.states, params, protocol.pde))
        times.append(result.times)
        parameters.append(params.as_array())
        maximum_values.append(result.diagnostics.max_abs_state)
        if int(row["index"]) in tight_indices:
            tight = simulate_trajectory(
                initial,
                params,
                tight_pde,
                divergence_threshold=protocol.tolerances.divergence_threshold,
            )
            if not tight.diagnostics.success:
                raise RuntimeError(f"tight pilot trajectory failed: {row['trajectory_id']}")
            errors, final_error = _trajectory_error(result.states, tight.states)
            convergence_errors.extend(errors)
            final_errors.append(final_error)

    state_array = np.stack(states)
    rhs_array = np.stack(rhs_values)
    time_array = np.stack(times)
    parameter_array = np.stack(parameters)
    ic_seeds = np.asarray([int(row["ic_seed"]) for row in rows], dtype=np.int64)
    state_mean = np.mean(state_array, axis=(0, 1, 3, 4), dtype=np.float64)
    state_std = np.std(state_array, axis=(0, 1, 3, 4), dtype=np.float64)
    p95 = float(np.percentile(np.asarray(convergence_errors), 95))
    maximum_final = float(np.max(final_errors))
    passed = bool(
        len(states) == protocol.data.pilot_trajectories
        and np.all(np.isfinite(state_array))
        and np.all(np.isfinite(rhs_array))
        and max(maximum_values) <= protocol.tolerances.divergence_threshold
        and p95 <= protocol.tolerances.solver_state_p95_relative_l2
        and maximum_final <= protocol.tolerances.solver_final_relative_l2
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = "chronopde_v2_pilot_1"
            handle.attrs["protocol_hash"] = protocol.digest
            handle.attrs["manifest_hash"] = manifest_hash(rows)
            handle.attrs["normalization_status"] = "diagnostic_only"
            handle.create_dataset("manifest_csv", data=np.void(manifest_bytes(rows)))
            handle.create_dataset("states", data=state_array, compression="lzf")
            handle.create_dataset("rhs", data=rhs_array, compression="lzf")
            handle.create_dataset("times", data=time_array)
            handle.create_dataset("parameters", data=parameter_array)
            handle.create_dataset("ic_seeds", data=ic_seeds)
            normalization = handle.create_group("diagnostic_normalization")
            normalization.create_dataset("state_mean", data=state_mean)
            normalization.create_dataset("state_std", data=state_std)
            handle.attrs["logical_content_sha256"] = _logical_content_hash(handle)
            handle.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    validated = validate_pilot_file(output, protocol)
    summary = {
        "passed": passed,
        "trajectory_count": len(states),
        "successful_trajectories": len(states),
        "maximum_absolute_state": float(max(maximum_values)),
        "solver_convergence": {
            "tight_trajectory_count": len(tight_indices),
            "state_relative_l2_p95": p95,
            "maximum_final_state_relative_l2": maximum_final,
        },
        "manifest_sha256": manifest_hash(rows),
        "logical_content_sha256": validated["logical_content_sha256"],
        "file_sha256": validated["file_sha256"],
        "normalization_status": "diagnostic_only",
        "diagnostic_state_mean": state_mean.tolist(),
        "diagnostic_state_std": state_std.tolist(),
    }
    atomic_write_json(runtime_summary, summary)
    return summary


def manifest_summary(protocol: Phase2Protocol) -> dict[str, Any]:
    pilot = pilot_manifest(protocol)
    future = future_manifest(protocol)
    all_ids = [str(row["trajectory_id"]) for row in pilot + future]
    all_seeds = [int(row["ic_seed"]) for row in pilot + future]
    if len(all_ids) != len(set(all_ids)) or len(all_seeds) != len(set(all_seeds)):
        raise ValueError("pilot and future identities must be disjoint")
    counts = {
        split: sum(row["split"] == split for row in future)
        for split in ("training", "validation", "confirmatory")
    }
    return {
        "pilot_manifest_sha256": manifest_hash(pilot),
        "future_manifest_sha256": manifest_hash(future),
        "future_split_counts": counts,
        "all_trajectory_ids_unique": True,
        "all_ic_seeds_unique": True,
        "confirmatory_status": "sealed_until_model_contract_is_frozen",
    }
