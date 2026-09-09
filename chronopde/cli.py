"""Shared CLI parsing and dry-run rendering."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from chronopde.config import RegimeName, SplitName, load_config
from chronopde.experiment import build_dry_run_plan
from chronopde.reproducibility import environment_metadata, seed_everything


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repository_root() / candidate).resolve()


def print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def generate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the ChronoPDE dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("data generation is scheduled for Week 2; use --dry-run in Week 1")
    config = load_config(resolve_config_path(args.config))
    seed_everything(config.data.base_seed)
    plan = build_dry_run_plan(config, command="generate_data", seed=config.data.base_seed)
    payload = plan.to_dict()
    payload["environment"] = environment_metadata(repository_root(), config.data.base_seed)
    print_payload(payload)
    return 0


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a ChronoPDE model.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--model", choices=("chronopde", "fno_ct", "fno_ar", "unet_ar"), default="chronopde"
    )
    parser.add_argument("--regime", choices=("full", "irreg50", "irreg25"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("training is scheduled for a later phase; use --dry-run in Week 1")
    config = load_config(resolve_config_path(args.config))
    seed_everything(args.seed)
    plan = build_dry_run_plan(
        config,
        command="train",
        model=args.model,
        regime=args.regime,
        split="train",
        seed=args.seed,
    )
    print_payload(plan.to_dict())
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ChronoPDE checkpoint.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--experiment",
        choices=(
            "id_rollout",
            "sparse_50",
            "sparse_25",
            "hidden_time",
            "ood_params",
            "ood_ic",
            "basis_ablation",
            "integrator_sweep",
            "mode_sweep",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("evaluation requires a future checkpoint; use --dry-run in Week 1")
    config = load_config(resolve_config_path(args.config))
    split_by_experiment: dict[str, SplitName] = {
        "id_rollout": "id",
        "sparse_50": "id",
        "sparse_25": "id",
        "hidden_time": "hidden_time",
        "ood_params": "oodparam",
        "ood_ic": "oodic",
        "basis_ablation": "id",
        "integrator_sweep": "id",
        "mode_sweep": "id",
    }
    regime_by_experiment: dict[str, RegimeName] = {
        "sparse_50": "irreg50",
        "sparse_25": "irreg25",
    }
    plan = build_dry_run_plan(
        config,
        command=f"evaluate:{args.experiment}",
        model="chronopde",
        regime=regime_by_experiment.get(args.experiment, "full"),
        split=split_by_experiment[args.experiment],
        seed=0,
    )
    payload = plan.to_dict()
    payload["checkpoint"] = str(Path(args.checkpoint))
    print_payload(payload)
    return 0
