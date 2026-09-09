from pathlib import Path

import pytest
from pydantic import ValidationError

from chronopde.config import ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_loads() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    assert config.pde.height == 64
    assert config.data.trajectory_count == 720
    assert config.model is not None
    assert config.model.name == "chronopde"


def test_data_only_config_loads() -> None:
    config = load_config(ROOT / "configs/data.yaml")
    assert config.model is None
    assert config.training is None


def test_unknown_key_is_rejected() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    raw = config.model_dump(mode="python")
    raw["pde"]["misspelled_field"] = 4
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_overlapping_ood_range_is_rejected() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    raw = config.model_dump(mode="python")
    raw["data"]["ood_ranges"]["du"] = ((0.0007, 0.0009), (0.0013, 0.0014))
    with pytest.raises(ValidationError, match="must not overlap"):
        ProjectConfig.model_validate(raw)


def test_modes_cannot_exceed_grid() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    raw = config.model_dump(mode="python")
    raw["model"]["spectral_modes_x"] = 65
    with pytest.raises(ValidationError, match="cannot exceed"):
        ProjectConfig.model_validate(raw)
