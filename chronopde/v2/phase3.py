"""ChronoPDE V2 Phase 3 development-data generation and validation."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chronopde.v2.common import canonical_json_bytes
from chronopde.v2.development_data import generate_development_dataset
from chronopde.v2.phase2 import _package_evidence, _write_or_verify, phase2_report
from chronopde.v2.pilot import development_rows, future_manifest, manifest_bytes, manifest_hash
from chronopde.v2.protocol import load_phase2_protocol, load_phase3_protocol


def _qa_csv(rows: list[dict[str, Any]]) -> bytes:
    fields = (
        "trajectory_id",
        "split",
        "index",
        "du",
        "dv",
        "k",
        "ic_seed",
        "nfev",
        "max_abs_state",
        "final_mean_u",
        "final_mean_v",
        "final_std_u",
        "final_std_v",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in fields})
    return stream.getvalue().encode("utf-8")


def phase3_report(root: Path, config_path: Path) -> dict[str, Any]:
    """Generate and freeze the development-only exact-RHS dataset."""

    phase3 = load_phase3_protocol(config_path)
    phase2_path = root / phase3.phase2_descriptor
    phase2 = load_phase2_protocol(phase2_path)
    if phase2.digest != phase3.phase2_protocol_sha256:
        raise ValueError("Phase 3 references the wrong Phase 2 protocol")
    previous = phase2_report(root, phase2_path)
    if previous["decision"] != "phase2_complete_phase3_data_generation_allowed":
        raise ValueError("Phase 2 did not authorize development-data generation")
    all_rows = future_manifest(phase2)
    if manifest_hash(all_rows) != phase3.frozen_future_manifest_sha256:
        raise ValueError("frozen future manifest hash conflict")
    committed_future = root / "reports/chronopde_v2/phase2/future_manifest.csv"
    if committed_future.read_bytes() != manifest_bytes(all_rows):
        raise ValueError("committed future manifest differs from the frozen identities")

    config8 = phase3.digest[:8]
    run_id = (
        "chronopde_v2-p3-reaction_diffusion-data-reference-development-"
        f"s{phase2.data.master_seed}-{config8}"
    )
    artifact_directory = root / phase3.outputs.artifact_root / run_id
    summary, qa_rows = generate_development_dataset(phase2, phase3, artifact_directory)
    report = {
        "schema_version": 1,
        "study_id": phase3.study_id,
        "phase": 3,
        "protocol_sha256": phase3.digest,
        "phase2_protocol_sha256": phase2.digest,
        "run_id": run_id,
        "decision": "phase3_complete_phase4_feasibility_training_allowed",
        "passed": True,
        "gpu_training_performed": False,
        "model_training_performed": False,
        "confirmatory_data_generated": False,
        "dataset": summary,
        "next_gate": (
            "Phase 4 may train matched models on training data and select on validation "
            "data only; confirmatory identities remain sealed."
        ),
    }
    report_root = root / phase3.outputs.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    development_manifest = development_rows(all_rows)
    evidence = {
        report_root / "protocol_snapshot.json": canonical_json_bytes(
            phase3.model_dump(mode="json")
        ),
        report_root / "development_manifest.csv": manifest_bytes(development_manifest),
        report_root / "trajectory_qa.csv": _qa_csv(qa_rows),
        report_root / "dataset_summary.json": canonical_json_bytes(summary),
        report_root / "decision_report.json": canonical_json_bytes(report),
    }
    for path, contents in evidence.items():
        _write_or_verify(path, contents)
    package_paths = list(evidence)
    readme = report_root / "README.md"
    if readme.is_file():
        package_paths.append(readme)
    _package_evidence(
        artifact_directory / "chronopde_v2_phase3_recovery.zip",
        package_paths,
        root,
        phase=3,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase3",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/chronopde_v2/phase3.yaml")
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        report = phase3_report(root, config_path.resolve())
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 2
    except RuntimeError as error:
        print(str(error))
        return 3
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        dataset = report["dataset"]
        print("ChronoPDE V2 Phase 3")
        print(f"Decision: {report['decision']}")
        print(f"Development trajectories: {dataset['successful_trajectories']}")
        print(f"Dataset hash: {dataset['logical_content_sha256']}")
        print("Confirmatory trajectories generated: 0")
        print("GPU/model training performed: False")
    return 0
