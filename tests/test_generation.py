from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from chronopde.config import load_config
from chronopde.contracts import PhysicalParameters, TrajectoryManifestEntry
from chronopde.data.generation import (
    _cache_valid,
    _save_trajectory,
    _trajectory_path,
    _write_hdf5,
    validate_hdf5,
)
from chronopde.data.pilot import configuration_hash

ROOT = Path(__file__).resolve().parents[1]


def _small_config():
    config = load_config(ROOT / "configs/project.yaml")
    return config.model_copy(
        update={
            "pde": config.pde.model_copy(
                update={"height": 8, "width": 8, "stored_times": 3, "t_end": 1.0}
            )
        }
    )


def _entries() -> list[TrajectoryManifestEntry]:
    result = []
    for index, split in enumerate(("train", "validation", "id", "oodparam", "oodic")):
        result.append(
            TrajectoryManifestEntry(
                trajectory_id=f"{split}-0000",
                split=split,  # type: ignore[arg-type]
                index=0,
                params=PhysicalParameters(0.001 + index * 1e-5, 0.005, 0.005),
                ic_seed=100 + index,
                ic_regime="ood" if split == "oodic" else "train",
                parameter_band="central",
            )
        )
    return result


def _write_fake_sources(
    tmp_path: Path, entries: list[TrajectoryManifestEntry], digest: str
) -> None:
    rng = np.random.default_rng(4)
    for entry in entries:
        states = rng.normal(size=(3, 2, 8, 8)).astype(np.float32)
        diagnostics = SimpleNamespace(
            success=True,
            runtime_seconds=0.1,
            nfev=8,
            max_abs_state=float(np.max(np.abs(states))),
            message="ok",
        )
        result = SimpleNamespace(
            states=states,
            times=np.linspace(0, 1, 3),
            diagnostics=diagnostics,
        )
        _save_trajectory(_trajectory_path(tmp_path, entry), entry, digest, result)


def test_hdf5_schema_normalization_and_source_validation(tmp_path: Path) -> None:
    config = _small_config()
    entries = _entries()
    digest = configuration_hash(config)
    _write_fake_sources(tmp_path, entries, digest)
    output = tmp_path / "dataset.h5"
    _write_hdf5(output, entries, tmp_path, config, digest, "manifest", "commit")
    validate_hdf5(output, entries, tmp_path, config, digest, "manifest")
    with np.load(_trajectory_path(tmp_path, entries[0]), allow_pickle=False) as source:
        expected_mean = source["states"].mean(axis=(0, 2, 3), dtype=np.float64)
    with h5py.File(output, "r") as handle:
        assert handle["splits/train/states"].chunks == (1, 1, 2, 8, 8)
        assert handle["splits/train/states"].compression == "lzf"
        np.testing.assert_allclose(handle["normalization/state_mean"][:], expected_mean)


def test_cache_rejects_corrupt_file(tmp_path: Path) -> None:
    config = _small_config()
    entry = _entries()[0]
    digest = configuration_hash(config)
    _write_fake_sources(tmp_path, [entry], digest)
    path = _trajectory_path(tmp_path, entry)
    assert _cache_valid(path, entry, config, digest)
    path.write_bytes(b"corrupt")
    assert not _cache_valid(path, entry, config, digest)
