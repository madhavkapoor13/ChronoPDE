"""Validated lightweight evidence used by the ChronoPDE V2 public release."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

EXPECTED_DATASET_SHA256 = "0a432d24d34605274a4e4bfa06d97831602062b016b44f4d5b81f91cc14e64e3"
EXPECTED_RESULTS_SHA256 = "151e0fb628735b21f7d63a159abfa9695253f2a210147c29555ced06d587cfc1"
EXPECTED_PROTOCOL_SHA256 = "7a869166876c6bd2a6143bd334a71128f692b7dc77cc9554d69e510b09c26464"


@dataclass(frozen=True)
class SeedResult:
    """One frozen FFT-DCT confirmatory comparison."""

    seed: int
    fft_rollout_relative_l2: float
    dct_rollout_relative_l2: float
    relative_improvement: float
    fft_velocity_nrmse: float
    dct_velocity_nrmse: float
    fft_divergence_fraction: float
    dct_divergence_fraction: float


@dataclass(frozen=True)
class ReleaseEvidence:
    """Release-safe subset of the frozen Phase 6 evidence."""

    decision: str
    permitted_claim: str
    permitted_boundary_claim: str
    median_relative_improvement: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    sign_test_p: float
    dataset_sha256: str
    results_sha256: str
    protocol_sha256: str
    seed_results: tuple[SeedResult, ...]

    @property
    def fft_divergent_rollouts(self) -> int:
        return round(sum(row.fft_divergence_fraction * 256 for row in self.seed_results))

    @property
    def dct_divergent_rollouts(self) -> int:
        return round(sum(row.dct_divergence_fraction * 256 for row in self.seed_results))


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def load_release_evidence(root: Path) -> ReleaseEvidence:
    """Load and validate the committed Phase 6 decision and paired seed table."""

    report = root / "reports/chronopde_v2/phase6"
    decision = _object(report / "decision_report.json")
    with (report / "paired_seed_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(
            SeedResult(
                seed=int(row["seed"]),
                fft_rollout_relative_l2=float(row["fft_rollout_relative_l2"]),
                dct_rollout_relative_l2=float(row["dct_rollout_relative_l2"]),
                relative_improvement=float(row["relative_improvement"]),
                fft_velocity_nrmse=float(row["fft_velocity_nrmse"]),
                dct_velocity_nrmse=float(row["dct_velocity_nrmse"]),
                fft_divergence_fraction=float(row["fft_divergence_fraction"]),
                dct_divergence_fraction=float(row["dct_divergence_fraction"]),
            )
            for row in csv.DictReader(stream)
        )

    dataset = cast(dict[str, Any], decision["dataset"])
    bootstrap = cast(dict[str, Any], decision["hierarchical_bootstrap"])
    evidence = ReleaseEvidence(
        decision=str(decision["decision"]),
        permitted_claim=str(decision["permitted_claim"]),
        permitted_boundary_claim=str(decision["permitted_boundary_claim"]),
        median_relative_improvement=float(decision["median_relative_improvement"]),
        bootstrap_ci_low=float(bootstrap["relative_improvement_ci_low"]),
        bootstrap_ci_high=float(bootstrap["relative_improvement_ci_high"]),
        sign_test_p=float(decision["one_sided_exact_sign_test_p"]),
        dataset_sha256=str(dataset["file_sha256"]),
        results_sha256=EXPECTED_RESULTS_SHA256,
        protocol_sha256=str(decision["protocol_sha256"]),
        seed_results=rows,
    )
    if (
        decision.get("passed") is not True
        or decision.get("superiority_claim_authorized") is not True
    ):
        raise ValueError("Phase 6 evidence does not authorize the public confirmatory claim")
    if tuple(row.seed for row in rows) != (0, 1, 2, 3, 4):
        raise ValueError("Phase 6 evidence must contain exactly seeds 0 through 4")
    if evidence.dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("confirmatory dataset hash conflicts with the frozen release")
    if evidence.protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Phase 6 protocol hash conflicts with the frozen release")
    if not all(
        row.dct_rollout_relative_l2 < row.fft_rollout_relative_l2
        and row.dct_velocity_nrmse < row.fft_velocity_nrmse
        and row.dct_divergence_fraction == 0.0
        for row in rows
    ):
        raise ValueError("paired seed table conflicts with the passing decision")
    return evidence
