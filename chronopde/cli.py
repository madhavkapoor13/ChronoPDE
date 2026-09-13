"""Shared CLI parsing and dry-run rendering."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronopde.config import RegimeName, SplitName, load_config
from chronopde.experiment import build_dry_run_plan
from chronopde.reproducibility import environment_metadata, seed_everything


def repository_root() -> Path:
    candidates = [Path.cwd()]
    if sys.argv and sys.argv[0]:
        script = Path(sys.argv[0]).expanduser().resolve()
        candidates.append(script.parent.parent)
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(resolve_config_path(args.config))
    if args.pilot:
        from chronopde.data.pilot import run_pilot

        seed_everything(config.data.base_seed)
        pilot_report = run_pilot(config, repository_root())
        print_payload(asdict(pilot_report))
        return 0 if pilot_report.passed else 2
    if args.manifest_only:
        from chronopde.data.generation import prepare_manifest

        entries, digest = prepare_manifest(config, repository_root())
        print_payload(
            {
                "manifest_path": str(repository_root() / config.data.manifest_path),
                "manifest_hash": digest,
                "trajectory_count": len(entries),
            }
        )
        return 0
    if args.full:
        from chronopde.data.generation import run_full_generation

        seed_everything(config.data.base_seed)
        generation_report = run_full_generation(config, repository_root())
        print_payload(asdict(generation_report))
        return 0 if generation_report.passed else 2
    if not args.dry_run:
        parser.error("choose one of --dry-run, --pilot, --manifest-only, or --full")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-path")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--steps-per-interval", type=int)
    parser.add_argument("--smoke-overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(resolve_config_path(args.config))
    if not args.dry_run:
        report: Any
        data_path = Path(args.data_path).expanduser().resolve() if args.data_path else None
        if args.model in {"chronopde", "fno_ct"}:
            from chronopde.training import train_continuous_time

            report = train_continuous_time(
                config,
                repository_root(),
                args.regime,
                args.seed,
                model_name=args.model,
                data_path=data_path,
                device_name=args.device,
                smoke_overfit=args.smoke_overfit,
                resume=args.resume,
                max_epochs=args.max_epochs,
                learning_rate_override=args.learning_rate,
                steps_per_interval=args.steps_per_interval,
            )
        else:
            if args.model not in {"unet_ar", "fno_ar"}:
                parser.error("training supports chronopde, fno_ct, fno_ar, and unet_ar")
            if args.learning_rate is not None or args.steps_per_interval is not None:
                parser.error(
                    "learning-rate and steps-per-interval overrides are for continuous models"
                )
            from chronopde.training import train_autoregressive

            report = train_autoregressive(
                config,
                repository_root(),
                args.model,
                args.regime,
                args.seed,
                data_path=data_path,
                device_name=args.device,
                smoke_overfit=args.smoke_overfit,
                resume=args.resume,
                max_epochs=args.max_epochs,
            )
        print_payload(asdict(report))
        return 0 if report.passed else 2
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
    parser.add_argument(
        "--model", choices=("chronopde", "fno_ct", "fno_ar", "unet_ar"), default="chronopde"
    )
    parser.add_argument("--regime", choices=("full", "irreg50", "irreg25"), default="full")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-path")
    parser.add_argument("--steps-per-interval", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(resolve_config_path(args.config))
    if not args.dry_run:
        report: Any
        if args.experiment != "id_rollout":
            parser.error("current executable evaluation supports id_rollout only")
        data_path = Path(args.data_path).expanduser().resolve() if args.data_path else None
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if args.model in {"chronopde", "fno_ct"}:
            from chronopde.evaluation.continuous import evaluate_continuous_baseline

            report = evaluate_continuous_baseline(
                config,
                repository_root(),
                checkpoint,
                model_name=args.model,
                regime=args.regime,
                data_path=data_path,
                device_name=args.device,
                steps_per_interval=args.steps_per_interval,
            )
        else:
            if args.model not in {"unet_ar", "fno_ar"}:
                parser.error("evaluation supports chronopde, fno_ct, fno_ar, and unet_ar")
            if args.steps_per_interval is not None:
                parser.error("steps-per-interval applies only to continuous models")
            from chronopde.evaluation.baselines import evaluate_autoregressive_baseline

            report = evaluate_autoregressive_baseline(
                config,
                repository_root(),
                args.model,
                checkpoint,
                regime=args.regime,
                data_path=data_path,
                device_name=args.device,
            )
        print_payload(asdict(report))
        return 0
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
        model=args.model,
        regime=regime_by_experiment.get(args.experiment, args.regime),
        split=split_by_experiment[args.experiment],
        seed=0,
    )
    payload = plan.to_dict()
    payload["checkpoint"] = str(Path(args.checkpoint))
    print_payload(payload)
    return 0
