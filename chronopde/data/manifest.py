"""Deterministic split manifests for the frozen ChronoPDE dataset."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from chronopde.config import ProjectConfig
from chronopde.contracts import PhysicalParameters, TrajectoryManifestEntry

SPLIT_OFFSETS = {
    "train": 10_000,
    "validation": 20_000,
    "id": 30_000,
    "oodparam": 40_000,
    "oodic": 50_000,
}
FIELDS = (
    "trajectory_id",
    "split",
    "index",
    "du",
    "dv",
    "k",
    "ic_seed",
    "ic_regime",
    "parameter_band",
)


def _scale_lhs(
    count: int, bounds: tuple[tuple[float, float], ...], seed: int
) -> NDArray[np.float64]:
    sample = qmc.LatinHypercube(d=len(bounds), seed=seed).random(count)
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    return cast(NDArray[np.float64], qmc.scale(sample, lower, upper))


def _entries(
    split: str,
    values: NDArray[np.float64],
    config: ProjectConfig,
    *,
    ic_regime: str,
    bands: list[str] | None = None,
) -> list[TrajectoryManifestEntry]:
    result: list[TrajectoryManifestEntry] = []
    for index, row in enumerate(values):
        result.append(
            TrajectoryManifestEntry(
                trajectory_id=f"{split}-{index:04d}",
                split=split,  # type: ignore[arg-type]
                index=index,
                params=PhysicalParameters(*map(float, row)),
                ic_seed=config.data.base_seed + SPLIT_OFFSETS[split] + index,
                ic_regime=ic_regime,  # type: ignore[arg-type]
                parameter_band=(bands[index] if bands else "central"),
            )
        )
    return result


def build_dataset_manifest(config: ProjectConfig) -> list[TrajectoryManifestEntry]:
    """Build all five splits using independent deterministic LHS designs."""

    train = config.data.train_ranges
    central = (train.du, train.dv, train.k)
    seed_sequences = np.random.SeedSequence(config.data.base_seed).spawn(12)
    seeds = [int(item.generate_state(1, dtype=np.uint32)[0]) for item in seed_sequences]

    result: list[TrajectoryManifestEntry] = []
    specs = (
        ("train", config.data.train_trajectories, 0),
        ("validation", config.data.validation_trajectories, 1),
        ("id", config.data.id_test_trajectories, 2),
    )
    for split, count, seed_index in specs:
        result.extend(
            _entries(
                split,
                _scale_lhs(count, central, seeds[seed_index]),
                config,
                ic_regime="train",
            )
        )

    ood_values: list[NDArray[np.float64]] = []
    ood_bands: list[str] = []
    ood = config.data.ood_ranges
    for combo_index, choices in enumerate(product((0, 1), repeat=3)):
        bounds = (ood.du[choices[0]], ood.dv[choices[1]], ood.k[choices[2]])
        values = _scale_lhs(15, bounds, seeds[3 + combo_index])
        labels = tuple("low" if choice == 0 else "high" for choice in choices)
        ood_values.append(values)
        ood_bands.extend([f"du_{labels[0]}__dv_{labels[1]}__k_{labels[2]}"] * len(values))
    combined = np.concatenate(ood_values, axis=0)
    if len(combined) != config.data.ood_parameter_trajectories:
        raise ValueError("parameter-OOD count must be 120 for the balanced 8x15 design")
    result.extend(_entries("oodparam", combined, config, ic_regime="train", bands=ood_bands))

    result.extend(
        _entries(
            "oodic",
            _scale_lhs(config.data.ood_ic_trajectories, central, seeds[11]),
            config,
            ic_regime="ood",
        )
    )
    if len(result) != config.data.trajectory_count:
        raise RuntimeError("manifest size does not match configuration")
    return result


def _row(entry: TrajectoryManifestEntry) -> dict[str, str | int | float]:
    raw = asdict(entry)
    params = raw.pop("params")
    return {
        "trajectory_id": raw["trajectory_id"],
        "split": raw["split"],
        "index": raw["index"],
        "du": params["du"],
        "dv": params["dv"],
        "k": params["k"],
        "ic_seed": raw["ic_seed"],
        "ic_regime": raw["ic_regime"],
        "parameter_band": raw["parameter_band"],
    }


def manifest_bytes(entries: list[TrajectoryManifestEntry]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_row(entry) for entry in entries)
    return stream.getvalue().encode("utf-8")


def manifest_hash(entries: list[TrajectoryManifestEntry]) -> str:
    return hashlib.sha256(manifest_bytes(entries)).hexdigest()


def write_frozen_manifest(path: Path, entries: list[TrajectoryManifestEntry]) -> str:
    """Write a new manifest, or verify that an existing manifest is unchanged."""

    payload = manifest_bytes(entries)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"existing frozen manifest differs from configuration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    hash_path = path.with_suffix(path.suffix + ".sha256")
    expected = f"{digest}  {path.name}\n"
    if hash_path.exists() and hash_path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"existing manifest hash file is inconsistent: {hash_path}")
    hash_path.write_text(expected, encoding="utf-8")
    return digest


def read_manifest(path: Path) -> list[TrajectoryManifestEntry]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: list[TrajectoryManifestEntry] = []
    for row in rows:
        result.append(
            TrajectoryManifestEntry(
                trajectory_id=row["trajectory_id"],
                split=row["split"],  # type: ignore[arg-type]
                index=int(row["index"]),
                params=PhysicalParameters(float(row["du"]), float(row["dv"]), float(row["k"])),
                ic_seed=int(row["ic_seed"]),
                ic_regime=row["ic_regime"],  # type: ignore[arg-type]
                parameter_band=row["parameter_band"],
            )
        )
    return result
