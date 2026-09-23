"""Evidence, claim, and study-contract validation for ChronoPDE V2 Phase 1."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml

from chronopde.analysis.week6_recovery import analyze_week6_failure
from chronopde.v2.common import sha256_file
from chronopde.v2.registry import STUDY_ID

PARENT_COMMIT = "ebaa66781c646eb85cdf270d474fff3f05611ada"
EVIDENCE_MANIFEST = Path("reports/diagnostics/week6/evidence_manifest.json")
V2_REPORT_ROOT = Path("reports/chronopde_v2/phase1")
ALLOWED_CLAIM_STATUSES = {"verified", "exploratory", "unsupported", "contradicted", "not_run"}
ALLOWED_CLAIM_CATEGORIES = {"scientific", "software"}
FORBIDDEN_EVIDENCE_SUFFIXES = {".h5", ".hdf5", ".pt", ".pth", ".zip"}


def _read_object(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object: {path}")
    return payload


def _git_values(root: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return [line for line in completed.stdout.splitlines() if line]


def _release_tags(root: Path) -> list[dict[str, str]]:
    return [
        {"name": tag, "commit": _git_values(root, "rev-list", "-n", "1", tag)[0]}
        for tag in _git_values(root, "tag", "--list")
    ]


def _safe_archive_filename(filename: str) -> None:
    candidate = Path(filename)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise ValueError(f"unsafe archive filename: {filename}")


def discover_archives(
    declarations: list[dict[str, Any]], archive_root: Path | None
) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, str]] = []
    if archive_root is not None:
        resolved_root = archive_root.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(resolved_root)
        candidates = [(path, sha256_file(path)) for path in sorted(resolved_root.rglob("*.zip"))]
    records: list[dict[str, Any]] = []
    for declaration in declarations:
        archive_id = declaration.get("id")
        filename = declaration.get("filename")
        expected = declaration.get("sha256")
        commit = declaration.get("git_commit")
        if not all(isinstance(value, str) for value in (archive_id, filename, expected, commit)):
            raise ValueError("archive declarations require string identity fields")
        _safe_archive_filename(cast(str, filename))
        if not re.fullmatch(r"[0-9a-f]{64}", cast(str, expected)):
            raise ValueError(f"invalid archive digest: {archive_id}")
        if not re.fullmatch(r"[0-9a-f]{40}", cast(str, commit)):
            raise ValueError(f"invalid source commit: {archive_id}")
        matches = [str(path.resolve()) for path, digest in candidates if digest == expected]
        exact = [] if archive_root is None else [
            path for path, digest in candidates if path.name == filename and digest != expected
        ]
        if archive_root is None:
            status = "declared_only"
        elif matches:
            status = "verified_locally"
        elif exact:
            status = "hash_mismatch"
        else:
            status = "missing"
        records.append(
            {
                "id": archive_id,
                "historical_filename": filename,
                "source_commit": commit,
                "expected_sha256": expected,
                "status": status,
                "matching_paths": matches,
                "duplicate_matches": max(len(matches) - 1, 0),
            }
        )
    return records


def lightweight_evidence(root: Path) -> list[dict[str, object]]:
    evidence_root = root / "reports/diagnostics/week6/evidence"
    files = sorted(path for path in evidence_root.rglob("*") if path.is_file())
    forbidden = [path for path in files if path.suffix.lower() in FORBIDDEN_EVIDENCE_SUFFIXES]
    if forbidden:
        raise ValueError(f"large or binary artifacts found in committed evidence: {forbidden}")
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def validate_claim_ledger(root: Path, ledger_path: Path) -> list[dict[str, Any]]:
    payload = _read_object(ledger_path)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("claim ledger must contain claims")
    identifiers: set[str] = set()
    claims: list[dict[str, Any]] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise ValueError("claim entries must be objects")
        identifier = raw.get("id")
        wording = raw.get("wording")
        category = raw.get("category", "scientific")
        status = raw.get("status")
        evidence = raw.get("evidence")
        limitations = raw.get("limitations")
        public_wording = raw.get("permitted_public_wording")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise ValueError("claim identifiers must be unique strings")
        if not isinstance(wording, str) or not wording.strip():
            raise ValueError(f"claim wording is missing: {identifier}")
        if status not in ALLOWED_CLAIM_STATUSES:
            raise ValueError(f"invalid claim status: {identifier}")
        if category not in ALLOWED_CLAIM_CATEGORIES:
            raise ValueError(f"invalid claim category: {identifier}")
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise ValueError(f"claim evidence must be a list of paths: {identifier}")
        if status != "not_run" and not evidence:
            raise ValueError(f"claim requires evidence: {identifier}")
        for item in evidence:
            if not (root / item).is_file():
                raise ValueError(f"claim evidence does not exist: {item}")
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) and item.strip() for item in limitations
        ):
            raise ValueError(f"claim limitations must be non-empty strings: {identifier}")
        if not isinstance(public_wording, str) or not public_wording.strip():
            raise ValueError(f"permitted public wording is missing: {identifier}")
        if (
            category == "scientific"
            and status == "verified"
            and any("test" in item.lower() for item in evidence)
            and not any("report" in item or "evidence" in item for item in evidence)
        ):
            raise ValueError(f"tests alone cannot verify a scientific claim: {identifier}")
        identifiers.add(identifier)
        claims.append(raw)
    return claims


def validate_study_descriptor(root: Path, descriptor_path: Path) -> dict[str, Any]:
    payload = _read_object(descriptor_path)
    expected_manifest_hash = sha256_file(root / EVIDENCE_MANIFEST)
    required = {
        "study_id": STUDY_ID,
        "parent_study": "chronopde_v1",
        "parent_commit": PARENT_COMMIT,
        "phase": 1,
        "protocol_version": 1,
        "legacy_evidence_manifest_sha256": expected_manifest_hash,
        "legacy_id_test_status": "development_exposed",
        "confirmatory_reuse_allowed": False,
        "artifact_root": "artifacts/chronopde_v2/runs",
        "report_root": "reports/chronopde_v2",
        "phase_status": "complete",
        "scientific_experiment_performed": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid V2 study descriptor field: {key}")
    return payload


def build_inventory(root: Path, archive_root: Path | None = None) -> dict[str, Any]:
    manifest = _read_object(root / EVIDENCE_MANIFEST)
    archives = manifest.get("archives")
    if not isinstance(archives, list):
        raise ValueError("legacy evidence manifest has no archive declarations")
    with tempfile.TemporaryDirectory(prefix="chronopde-v2-phase1-") as temporary:
        output = Path(temporary)
        frozen = analyze_week6_failure(root, root / EVIDENCE_MANIFEST, output)
    protocol = _read_object(
        root / "reports/diagnostics/week6/evidence/loss_alignment/protocol.json"
    )
    selections = {
        model: _read_object(
            root
            / f"reports/diagnostics/week6/evidence/loss_alignment/{model}/checkpoint_selection.json"
        )
        for model in ("chronopde", "fno_ct")
    }
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "parent_commit": PARENT_COMMIT,
        "repository_head_at_freeze": PARENT_COMMIT,
        "release_tags": _release_tags(root),
        "legacy_evidence_manifest": str(EVIDENCE_MANIFEST),
        "legacy_evidence_manifest_sha256": sha256_file(root / EVIDENCE_MANIFEST),
        "lightweight_evidence": lightweight_evidence(root),
        "archives": discover_archives(cast(list[dict[str, Any]], archives), archive_root),
        "recorded_results": {
            "classification": {
                "loss_alignment": "training_diagnostic",
                "legacy_id": "exploratory_evaluation",
                "confirmatory_evidence": "none",
            },
            "seed": 0,
            "sample_count": frozen["gate"]["sample_count"],
            "objective": protocol["objective"],
            "threshold": frozen["gate"]["velocity_nrmse_threshold"],
            "training_budget": {
                "max_optimizer_steps": protocol["max_steps"],
                "evaluation_interval": protocol["evaluation_interval"],
                "batch_size": protocol["batch_size"],
            },
            "checkpoint_selection": selections,
            "models": frozen["models"],
            "dct_wins": frozen["dct_wins"],
            "decision": frozen["decision"],
        },
    }
