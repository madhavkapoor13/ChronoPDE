"""Verify and print the committed ChronoPDE V2 confirmatory claim."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_release_evidence(root: Path) -> Any:
    """Load the stdlib-only evidence module without importing package extras."""

    module_path = root / "chronopde/v2/publication.py"
    spec = importlib.util.spec_from_file_location("chronopde_v2_publication", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load release verifier: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_release_evidence(root)


def verified_summary(root: Path = ROOT) -> dict[str, Any]:
    """Return the release-safe headline after validating frozen evidence."""

    evidence = _load_release_evidence(root)
    return {
        "decision": evidence.decision,
        "passed": True,
        "confirmatory_trajectories": 256,
        "checkpoint_pairs": len(evidence.seed_results),
        "rollout_wins": sum(
            row.dct_rollout_relative_l2 < row.fft_rollout_relative_l2
            for row in evidence.seed_results
        ),
        "velocity_wins": sum(
            row.dct_velocity_nrmse < row.fft_velocity_nrmse
            for row in evidence.seed_results
        ),
        "median_paired_rollout_improvement": evidence.median_relative_improvement,
        "hierarchical_bootstrap_95_ci": [
            evidence.bootstrap_ci_low,
            evidence.bootstrap_ci_high,
        ],
        "one_sided_exact_sign_test_p": evidence.sign_test_p,
        "dct_divergent_rollouts": evidence.dct_divergent_rollouts,
        "total_dct_rollouts": 256 * len(evidence.seed_results),
        "confirmatory_dataset_sha256": evidence.dataset_sha256,
        "results_archive_sha256": evidence.results_sha256,
        "protocol_sha256": evidence.protocol_sha256,
        "evidence": [
            "reports/chronopde_v2/phase6/decision_report.json",
            "reports/chronopde_v2/phase6/paired_seed_results.csv",
            "reports/chronopde_v2/phase6/protocol_snapshot.json",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    summary = verified_summary()
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    low, high = summary["hierarchical_bootstrap_95_ci"]
    print("ChronoPDE V2 sealed confirmatory evidence: PASS")
    print(
        f"Paired wins: rollout {summary['rollout_wins']}/5; "
        f"exact velocity {summary['velocity_wins']}/5"
    )
    print(f"Confirmatory trajectories: {summary['confirmatory_trajectories']}")
    print(
        "Median paired rollout improvement: "
        f"{100 * summary['median_paired_rollout_improvement']:.2f}%"
    )
    print(f"Hierarchical-bootstrap 95% CI: {100 * low:.2f}% to {100 * high:.2f}%")
    print(
        "DCT divergence: "
        f"{summary['dct_divergent_rollouts']}/{summary['total_dct_rollouts']} rollouts"
    )
    print(f"One-sided exact sign test: p={summary['one_sided_exact_sign_test_p']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
