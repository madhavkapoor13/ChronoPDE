"""ChronoPDE V2 Phase 4 matched feasibility training command."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from chronopde.models import trainable_parameter_count
from chronopde.v2.common import atomic_write_json, read_json_object
from chronopde.v2.development_data import validate_development_dataset
from chronopde.v2.phase4_training import ModelName, build_phase4_model, train_phase4_model
from chronopde.v2.pilot import development_rows, future_manifest
from chronopde.v2.protocol import (
    Phase4Protocol,
    load_phase2_protocol,
    load_phase3_protocol,
    load_phase4_protocol,
)


def validate_phase4_contract(root: Path, protocol: Phase4Protocol) -> dict[str, Any]:
    phase2 = load_phase2_protocol(root / protocol.phase2_descriptor)
    phase3 = load_phase3_protocol(root / protocol.phase3_descriptor)
    if phase3.digest != protocol.phase3_protocol_sha256:
        raise ValueError("Phase 4 references the wrong Phase 3 protocol")
    phase3_report = read_json_object(root / "reports/chronopde_v2/phase3/decision_report.json")
    dataset = cast(dict[str, Any], phase3_report["dataset"])
    if phase3_report.get("decision") != "phase3_complete_phase4_feasibility_training_allowed":
        raise ValueError("Phase 3 did not authorize feasibility training")
    if dataset.get("logical_content_sha256") != protocol.development_dataset_sha256:
        raise ValueError("Phase 4 dataset declaration conflicts with Phase 3 evidence")
    if dataset.get("normalization_sha256") != protocol.normalization_sha256:
        raise ValueError("Phase 4 normalization declaration conflicts with Phase 3 evidence")
    model_names: tuple[ModelName, ModelName] = ("fft", "dct")
    counts = {
        name: trainable_parameter_count(build_phase4_model(protocol, name))
        for name in model_names
    }
    if counts != {"fft": 1_973_657, "dct": 1_973_657}:
        raise ValueError(f"matched comparator parameter count changed: {counts}")
    return {
        "passed": True,
        "phase3_protocol_sha256": phase3.digest,
        "development_dataset_sha256": protocol.development_dataset_sha256,
        "normalization_sha256": protocol.normalization_sha256,
        "parameter_counts": counts,
        "confirmatory_access_allowed": False,
        "training_budget_steps_per_model": protocol.training.maximum_steps,
        "phase2": phase2,
        "phase3": phase3,
    }


def validate_phase4_dataset(
    root: Path, protocol: Phase4Protocol, data_path: Path
) -> dict[str, Any]:
    contract = validate_phase4_contract(root, protocol)
    phase2 = contract.pop("phase2")
    phase3 = contract.pop("phase3")
    rows = development_rows(future_manifest(phase2))
    validated = validate_development_dataset(data_path, rows, phase2, phase3)
    if validated["logical_content_sha256"] != protocol.development_dataset_sha256:
        raise ValueError("validated dataset does not match the Phase 4 identity")
    if validated["normalization_sha256"] != protocol.normalization_sha256:
        raise ValueError("validated normalization does not match the Phase 4 identity")
    return {**contract, "dataset_validation": validated}


def phase4_decision(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(results) != {"fft", "dct"}:
        return {
            "complete": False,
            "decision": "awaiting_both_matched_feasibility_runs",
            "available_models": sorted(results),
        }
    passed = {name: bool(result["gate"]["passed"]) for name, result in results.items()}
    if all(passed.values()):
        decision = "phase4_passed_phase5_multiseed_validation_allowed"
    elif any(passed.values()):
        decision = "single_backbone_pass_controlled_investigation_required"
    else:
        decision = "phase4_failed_pause_continuous_time_study"
    return {
        "complete": True,
        "decision": decision,
        "passed": all(passed.values()),
        "model_gates": passed,
        "superiority_claim_authorized": False,
        "confirmatory_accessed": False,
    }


def _result_paths(root: Path, protocol: Phase4Protocol) -> dict[str, Path]:
    base = root / protocol.outputs.artifact_root
    return {
        name: base
        / (
            f"chronopde_v2-p4-reaction_diffusion-exact_rhs-{name}-feasibility-"
            f"s{protocol.training.seed}-{protocol.digest[:8]}"
        )
        / "summary.json"
        for name in ("fft", "dct")
    }


def collect_phase4_results(root: Path, protocol: Phase4Protocol) -> dict[str, Any]:
    results = {
        name: read_json_object(path)
        for name, path in _result_paths(root, protocol).items()
        if path.is_file()
    }
    report = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "phase": 4,
        "protocol_sha256": protocol.digest,
        "development_only": True,
        "results": results,
        **phase4_decision(results),
    }
    report_root = root / protocol.outputs.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_root / "runtime_decision_report.json", report)
    return report


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 4 training requires a CUDA GPU; use --check-only locally")
        return torch.device("cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _discover_data(root: Path) -> Path | None:
    candidates = [
        root
        / "artifacts/chronopde_v2/runs/"
        "chronopde_v2-p3-reaction_diffusion-data-reference-development-s20260922-f8e716c7/"
        "chronopde_v2_development.h5"
    ]
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.extend(kaggle.rglob("chronopde_v2_development.h5"))
    return next((path for path in candidates if path.is_file()), None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase4",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/chronopde_v2/phase4.yaml")
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--model", choices=("fft", "dct", "both"), default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        protocol = load_phase4_protocol(config_path.resolve())
        data_path = args.data_path or _discover_data(root)
        if args.check_only and data_path is None:
            report: dict[str, Any] = validate_phase4_contract(root, protocol)
            report.pop("phase2")
            report.pop("phase3")
            report["dataset_validation"] = "not_requested"
            report["training_performed"] = False
        else:
            if data_path is None:
                raise FileNotFoundError(
                    "chronopde_v2_development.h5 was not found; attach it or pass --data-path"
                )
            validation = validate_phase4_dataset(root, protocol, data_path.resolve())
            if args.check_only:
                report = {**validation, "training_performed": False}
            else:
                device = _resolve_device(args.device)
                selected: tuple[ModelName, ...] = (
                    ("fft", "dct") if args.model == "both" else (cast(ModelName, args.model),)
                )
                for model_name in selected:
                    train_phase4_model(
                        root,
                        protocol,
                        data_path.resolve(),
                        model_name,
                        device,
                        resume=args.resume,
                    )
                report = {"input_validation": validation, **collect_phase4_results(root, protocol)}
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 2
    except RuntimeError as error:
        print(str(error))
        return 3
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("ChronoPDE V2 Phase 4")
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
