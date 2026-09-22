"""ChronoPDE V2 Phase 2 numerical audit and fresh CPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from chronopde.v2.common import atomic_write_bytes, canonical_json_bytes
from chronopde.v2.comparison import design_comparator
from chronopde.v2.numerical import run_numerical_audit
from chronopde.v2.phase1 import phase1_report
from chronopde.v2.pilot import (
    future_manifest,
    generate_cpu_pilot,
    manifest_bytes,
    manifest_summary,
    pilot_manifest,
)
from chronopde.v2.protocol import load_phase2_protocol


def _write_or_verify(path: Path, contents: bytes) -> None:
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"frozen Phase 2 evidence differs: {path}")
        return
    atomic_write_bytes(path, contents)


def _package_evidence(
    package: Path, paths: list[Path], root: Path, *, phase: int = 2
) -> dict[str, Any]:
    entries: dict[str, bytes] = {}
    checksums: dict[str, str] = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        entries[relative] = payload
        checksums[relative] = hashlib.sha256(payload).hexdigest()
    package_manifest = {
        "schema_version": 1,
        "study_id": "chronopde_v2",
        "phase": phase,
        "partial": False,
        "files": checksums,
    }
    entries["package_manifest.json"] = canonical_json_bytes(package_manifest)
    package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{package.name}.", suffix=".tmp", dir=package.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload)
        atomic_write_bytes(package, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    return verify_evidence_package(package, expected_phase=phase)


def verify_evidence_package(package: Path, *, expected_phase: int) -> dict[str, Any]:
    """Verify member safety, identity, uniqueness, and checksums in a V2 package."""

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Phase 2 package contains duplicate members")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Phase 2 package contains an unsafe member path")
        manifest = cast(dict[str, Any], json.loads(archive.read("package_manifest.json")))
        if (
            manifest.get("study_id") != "chronopde_v2"
            or manifest.get("phase") != expected_phase
        ):
            raise ValueError(f"Phase {expected_phase} package identity mismatch")
        for name, expected in manifest["files"].items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"Phase 2 package checksum mismatch: {name}")
    return manifest


def verify_phase2_package(package: Path) -> dict[str, Any]:
    """Verify a Phase 2 recovery package."""

    return verify_evidence_package(package, expected_phase=2)


def _decision(
    numerical: dict[str, Any], comparator: dict[str, Any], pilot: dict[str, Any]
) -> tuple[str, bool]:
    if not numerical["passed"]:
        return "stop_and_repair_numerical_implementation", False
    if not comparator["passed"]:
        return "stop_no_fair_comparator", False
    if not pilot["passed"]:
        return "stop_and_revise_simulator_or_data_protocol", False
    return "phase2_complete_phase3_data_generation_allowed", True


def phase2_report(root: Path, config_path: Path) -> dict[str, Any]:
    """Execute the frozen Phase 2 audit and pilot, then freeze lightweight evidence."""

    phase1 = phase1_report(root)
    if phase1["decision"] != "phase1_complete":
        raise ValueError("Phase 1 provenance must validate before Phase 2")
    protocol = load_phase2_protocol(config_path)
    parent = root / protocol.parent_descriptor
    if not parent.is_file():
        raise ValueError("Phase 2 parent descriptor is missing")

    numerical = run_numerical_audit(protocol.pde, protocol.tolerances)
    comparison = design_comparator(
        protocol.comparison,
        domain_height=protocol.pde.y_max - protocol.pde.y_min,
        domain_width=protocol.pde.x_max - protocol.pde.x_min,
    )
    identities = manifest_summary(protocol)
    config8 = protocol.digest[:8]
    run_id = (
        "chronopde_v2-p2-reaction_diffusion-data-reference-pilot-"
        f"s{protocol.data.master_seed}-{config8}"
    )
    artifact_directory = root / protocol.outputs.artifact_root / run_id
    pilot = generate_cpu_pilot(
        protocol,
        artifact_directory / "pilot.h5",
        artifact_directory / "pilot_runtime_summary.json",
    )
    decision, passed = _decision(numerical, comparison, pilot)
    report = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "phase": 2,
        "protocol_sha256": protocol.digest,
        "run_id": run_id,
        "decision": decision,
        "passed": passed,
        "gpu_training_performed": False,
        "full_dataset_generated": False,
        "scientific_model_result_produced": False,
        "numerical_audit": numerical,
        "comparator": comparison,
        "identities": identities,
        "pilot": pilot,
        "claim_boundary": (
            "The DCT operator is a Neumann-aligned inductive bias. Phase 2 does not "
            "establish model superiority or physical-wall flux enforcement."
        ),
    }

    report_root = root / protocol.outputs.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    evidence = {
        report_root / "protocol_snapshot.json": canonical_json_bytes(
            protocol.model_dump(mode="json")
        ),
        report_root / "numerical_audit.json": canonical_json_bytes(numerical),
        report_root / "comparator.json": canonical_json_bytes(comparison),
        report_root / "pilot_manifest.csv": manifest_bytes(pilot_manifest(protocol)),
        report_root / "future_manifest.csv": manifest_bytes(future_manifest(protocol)),
        report_root / "pilot_summary.json": canonical_json_bytes(pilot),
        report_root / "decision_report.json": canonical_json_bytes(report),
    }
    for path, contents in evidence.items():
        _write_or_verify(path, contents)
    package = artifact_directory / "chronopde_v2_phase2_recovery.zip"
    package_paths = list(evidence)
    readme = report_root / "README.md"
    if readme.is_file():
        package_paths.append(readme)
    package_report = _package_evidence(package, package_paths, root)
    if package_report["partial"] is not False:
        raise ValueError("completed Phase 2 package cannot be partial")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase2",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/chronopde_v2/phase2.yaml")
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        report = phase2_report(root, config_path.resolve())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(str(error))
        return 4
    if not report["numerical_audit"]["passed"]:
        code = 2
    elif not report["comparator"]["passed"]:
        code = 3
    elif not report["pilot"]["passed"]:
        code = 4
    else:
        code = 0
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("ChronoPDE V2 Phase 2")
        print(f"Decision: {report['decision']}")
        print(f"Numerical audit passed: {report['numerical_audit']['passed']}")
        print(f"Comparator passed: {report['comparator']['passed']}")
        print(f"CPU pilot passed: {report['pilot']['passed']}")
        print("GPU training performed: False")
    return code
