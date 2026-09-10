import json
from pathlib import Path

from chronopde.config import load_config
from chronopde.data.pilot import build_pilot_manifest, configuration_hash, run_pilot

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_manifest_is_frozen_and_complete() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    cases = build_pilot_manifest(config)
    assert len(cases) == 24
    assert [case.seed for case in cases] == list(range(1729, 1753))
    assert len({case.trajectory_id for case in cases}) == 24
    assert sum(case.category == "train_corner" for case in cases) == 8
    assert sum(case.category == "ood_corner" for case in cases) == 8
    assert sum(case.category.startswith("one_factor_") for case in cases) == 6
    assert sum(case.category == "ood_ic" for case in cases) == 2
    assert sum(case.ic_regime == "ood" for case in cases) == 2


def test_configuration_hash_is_stable_and_sensitive() -> None:
    config = load_config(ROOT / "configs/project.yaml")
    assert configuration_hash(config) == configuration_hash(config)
    changed = config.model_copy(
        update={"pde": config.pde.model_copy(update={"stored_times": 99})}
    )
    assert configuration_hash(config) != configuration_hash(changed)


def test_small_pilot_runs_and_resumes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/project.yaml")
    small_pde = config.pde.model_copy(
        update={"height": 18, "width": 18, "t_end": 0.1, "stored_times": 3}
    )
    small_pilot = config.data.pilot.model_copy(update={"workers": 2})
    small_data = config.data.model_copy(update={"pilot": small_pilot})
    small_config = config.model_copy(
        update={"pde": small_pde, "data": small_data, "model": None}
    )
    output = tmp_path / "raw"
    published = tmp_path / "published"
    first = run_pilot(
        small_config, ROOT, output_directory=output, published_directory=published
    )
    assert first.passed
    assert first.successful == 24
    assert (output / "manifest.csv").is_file()
    assert (published / "state_panels.png").is_file()

    second = run_pilot(
        small_config, ROOT, output_directory=output, published_directory=published
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert second.passed
    assert summary["cached_trajectories"] == 24
