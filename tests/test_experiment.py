from pathlib import Path

import pytest

from chronopde.config import load_config
from chronopde.experiment import build_dry_run_plan, experiment_id, state_shape

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_identifier() -> None:
    assert experiment_id("chronopde", "irreg25", "oodparam", 2) == ("chronopde-irreg25-oodparam-s2")


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        experiment_id("chronopde", "full", "id", -1)


def test_shape_and_storage_plan() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    assert state_shape(config) == (720, 101, 2, 64, 64)
    plan = build_dry_run_plan(config, command="generate_data", seed=1729)
    assert plan.trajectory_count == 720
    assert 2.0 < plan.estimated_state_gib < 3.0
