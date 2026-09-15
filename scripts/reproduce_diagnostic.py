"""Recompute the frozen Week 6 headline result without retraining."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from chronopde.analysis.week6_recovery import analyze_week6_failure

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = {
    "fno_ct": "fno_ct",
    "chronopde_dct": "chronopde",
}
MODEL_LABELS = {
    "fno_ct": "CT-FFT",
    "chronopde_dct": "ChronoPDE DCT",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("both", "fno_ct", "chronopde_dct"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective",
        choices=("full_field_relative",),
        default="full_field_relative",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--archive-root", type=Path)
    return parser


def _selected_models(model: str) -> tuple[str, ...]:
    return ("fno_ct", "chronopde_dct") if model == "both" else (model,)


def _payload(report: dict[str, Any], model: str) -> dict[str, Any]:
    raw_models = cast(dict[str, dict[str, Any]], report["models"])
    selected: dict[str, Any] = {}
    for public_name in _selected_models(model):
        result = raw_models[MODEL_KEYS[public_name]]
        selected[public_name] = {
            "best_step": result["best_eligible_step"],
            "gate_nrmse": result["historical_gate_median_nrmse"],
            "gate_threshold": report["gate"]["velocity_nrmse_threshold"],
            "loss_reduction": result["loss_reduction"],
            "result": "PASS" if result["passed"] else "FAIL",
        }
    return {
        "decision": report["decision"],
        "dct_wins": report["dct_wins"],
        "models": selected,
        "objective": "full_field_relative",
        "sample_count": report["gate"]["sample_count"],
        "seed": 0,
    }


def _text(payload: dict[str, Any]) -> str:
    lines = ["ChronoPDE Week 6 frozen diagnostic", ""]
    models = cast(dict[str, dict[str, Any]], payload["models"])
    for name, result in models.items():
        lines.extend(
            [
                MODEL_LABELS[name],
                f"  Best step: {result['best_step']}",
                f"  Loss reduction: {result['loss_reduction']:.0f}x",
                f"  Median velocity nRMSE: {result['gate_nrmse']:.5f}",
                f"  Gate: {result['gate_threshold']:.5f}",
                f"  Result: {result['result']}",
                "",
            ]
        )
    if payload["dct_wins"] is not None:
        lines.append(
            f"Matched samples: DCT lower error on {payload['dct_wins']} / {payload['sample_count']}"
        )
    lines.append(f"Decision: {payload['decision']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed != 0:
        raise SystemExit("only the frozen seed-0 diagnostic is available")
    manifest = ROOT / "reports/diagnostics/week6/evidence_manifest.json"
    with tempfile.TemporaryDirectory(prefix="chronopde-reproduce-") as temporary:
        report = analyze_week6_failure(
            ROOT,
            manifest,
            Path(temporary),
            archive_root=args.archive_root.resolve() if args.archive_root else None,
        )
    payload = _payload(report, args.model)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
