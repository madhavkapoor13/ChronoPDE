"""Phase 1 audit entrypoint for preserving V1 evidence and validating V2 contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chronopde.v2.common import canonical_json_bytes
from chronopde.v2.provenance import (
    V2_REPORT_ROOT,
    build_inventory,
    validate_claim_ledger,
    validate_study_descriptor,
)


def phase1_report(root: Path, archive_root: Path | None = None) -> dict[str, Any]:
    report_root = root / V2_REPORT_ROOT
    committed_inventory = json.loads(
        (report_root / "evidence_inventory.json").read_text(encoding="utf-8")
    )
    current_inventory = build_inventory(root, archive_root)
    frozen_tags = {
        item["name"]: item["commit"] for item in committed_inventory["release_tags"]
    }
    current_tags = {item["name"]: item["commit"] for item in current_inventory["release_tags"]}
    if any(current_tags.get(name) != commit for name, commit in frozen_tags.items()):
        raise ValueError("a frozen Phase 1 release tag is missing or changed")
    comparable = dict(current_inventory)
    comparable["repository_head_at_freeze"] = committed_inventory["repository_head_at_freeze"]
    # Preserve the Phase 1 tag snapshot while permitting later, additive releases.
    comparable["release_tags"] = committed_inventory["release_tags"]
    comparable["archives"] = [
        {
            **archive,
            "status": "declared_only",
            "matching_paths": [],
            "duplicate_matches": 0,
        }
        for archive in current_inventory["archives"]
    ]
    if canonical_json_bytes(comparable) != canonical_json_bytes(committed_inventory):
        raise ValueError("committed Phase 1 evidence inventory is stale")
    claims = validate_claim_ledger(root, report_root / "claim_ledger.yaml")
    study = validate_study_descriptor(root, root / "configs/chronopde_v2/study.yaml")
    statuses = [archive["status"] for archive in current_inventory["archives"]]
    return {
        "study_id": study["study_id"],
        "phase": 1,
        "decision": "phase1_complete",
        "scientific_experiment_performed": False,
        "claims_validated": len(claims),
        "lightweight_evidence_files": len(current_inventory["lightweight_evidence"]),
        "archives": current_inventory["archives"],
        "archive_status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "frozen_diagnostic": current_inventory["recorded_results"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase1",))
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        report = phase1_report(root, args.archive_root.resolve() if args.archive_root else None)
    except FileNotFoundError as error:
        print(str(error))
        return 2
    except ValueError as error:
        message = str(error)
        print(message)
        return 3 if "archive" in message or "checksum" in message else 4
    statuses = set(report["archive_status_counts"])
    if args.archive_root and "hash_mismatch" in statuses:
        code = 3
    elif args.archive_root and "missing" in statuses:
        code = 2
    else:
        code = 0
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("ChronoPDE V2 Phase 1")
        print(f"Decision: {report['decision']}")
        print(f"Claims validated: {report['claims_validated']}")
        print(f"Scientific experiment performed: {report['scientific_experiment_performed']}")
        print(f"Archive statuses: {report['archive_status_counts']}")
    return code
