from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from chronopde.config import load_config
from chronopde.data.manifest import (
    build_dataset_manifest,
    manifest_bytes,
    manifest_hash,
    write_frozen_manifest,
)
from chronopde.data.masks import observation_mask

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_deterministic_and_has_exact_split_design() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    first = build_dataset_manifest(config)
    second = build_dataset_manifest(config)
    assert manifest_bytes(first) == manifest_bytes(second)
    assert len(first) == 720
    assert Counter(entry.split for entry in first) == {
        "train": 320,
        "validation": 60,
        "id": 100,
        "oodparam": 120,
        "oodic": 120,
    }
    assert len({entry.trajectory_id for entry in first}) == 720
    assert len({entry.ic_seed for entry in first}) == 720
    ood = [entry for entry in first if entry.split == "oodparam"]
    assert set(Counter(entry.parameter_band for entry in ood).values()) == {15}
    assert len({entry.parameter_band for entry in ood}) == 8


def test_manifest_parameter_membership() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    entries = build_dataset_manifest(config)
    central = config.data.train_ranges
    for entry in entries:
        values = entry.params.as_array()
        if entry.split != "oodparam":
            bounds = (central.du, central.dv, central.k)
            assert all(
                low <= value <= high for value, (low, high) in zip(values, bounds, strict=True)
            )
        else:
            bands = (config.data.ood_ranges.du, config.data.ood_ranges.dv, config.data.ood_ranges.k)
            assert all(
                any(low <= value <= high for low, high in pair)
                for value, pair in zip(values, bands, strict=True)
            )


def test_frozen_manifest_rejects_changes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/project.yaml")
    entries = build_dataset_manifest(config)
    path = tmp_path / "manifest.csv"
    digest = write_frozen_manifest(path, entries)
    assert digest == manifest_hash(entries)
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        write_frozen_manifest(path, entries)


def test_observation_masks_have_exact_counts_and_endpoints() -> None:
    full = observation_mask(101, "train-0000", "full", 1729)
    half = observation_mask(101, "train-0000", "irregular_50", 1729)
    quarter = observation_mask(101, "train-0000", "irregular_25", 1729)
    assert (full.sum(), half.sum(), quarter.sum()) == (101, 52, 27)
    assert all(mask[0] and mask[-1] for mask in (full, half, quarter))
    assert np.array_equal(half, observation_mask(101, "train-0000", "irregular_50", 1729))
    assert not np.array_equal(half, observation_mask(101, "train-0001", "irregular_50", 1729))
