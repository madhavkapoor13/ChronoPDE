"""Reproducible evidence analysis for the closed Week 6 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_ROUTE = "stop_and_document_model_or_conditioning_limitation"
EXPECTED_IDENTITIES = tuple(
    (f"train-{trajectory:04d}", interval) for trajectory in range(4) for interval in (0, 33, 66, 99)
)
BOOTSTRAP_SEED = 1729
BOOTSTRAP_RESAMPLES = 10_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(cast(dict[str, Any], value))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    raw_inputs = manifest.get("analysis_inputs")
    if not isinstance(raw_inputs, dict):
        raise ValueError("evidence manifest must contain analysis_inputs")
    resolved: dict[str, Path] = {}
    for key, relative in raw_inputs.items():
        if not isinstance(key, str) or not isinstance(relative, str):
            raise ValueError("analysis input keys and paths must be strings")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"analysis input escapes repository root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[key] = path
    return resolved


def _archive_declarations(manifest: dict[str, Any]) -> dict[str, str]:
    raw_archives = manifest.get("archives")
    if not isinstance(raw_archives, list):
        raise ValueError("evidence manifest must contain archives")
    declared: dict[str, str] = {}
    for raw in raw_archives:
        if not isinstance(raw, dict):
            raise ValueError("archive entries must be objects")
        archive_id = raw.get("id")
        filename = raw.get("filename")
        expected = raw.get("sha256")
        commit = raw.get("git_commit")
        if not all(isinstance(item, str) for item in (archive_id, filename, expected, commit)):
            raise ValueError("archive id, filename, sha256, and git_commit must be strings")
        if cast(str, archive_id) in declared:
            raise ValueError(f"duplicate archive id: {archive_id}")
        if SHA256_PATTERN.fullmatch(cast(str, expected)) is None:
            raise ValueError(f"invalid archive sha256 for {archive_id}")
        if GIT_COMMIT_PATTERN.fullmatch(cast(str, commit)) is None:
            raise ValueError(f"invalid Git commit for {archive_id}")
        declared[cast(str, archive_id)] = cast(str, expected)
    return declared


def verify_archives(manifest: dict[str, Any], archive_root: Path) -> dict[str, str]:
    declarations = _archive_declarations(manifest)
    raw_archives = cast(list[dict[str, Any]], manifest["archives"])
    root = archive_root.resolve()
    verified: dict[str, str] = {}
    for raw in raw_archives:
        archive_id = cast(str, raw["id"])
        filename = cast(str, raw["filename"])
        path = (root / filename).resolve()
        if root not in path.parents:
            raise ValueError(f"archive path escapes declared root: {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != declarations[archive_id]:
            raise ValueError(f"archive checksum mismatch for {filename}: {actual}")
        verified[archive_id] = actual
    return verified


def _metric_row(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    matches = [row for row in rows if int(float(row["optimizer_steps"])) == step]
    if len(matches) != 1:
        raise ValueError(f"expected one metric row at optimizer step {step}")
    return matches[0]


def _sample_rows(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["optimizer_steps"]) == step]
    identities = [(str(row["trajectory_id"]), int(row["interval_index"])) for row in selected]
    if len(selected) != len(EXPECTED_IDENTITIES) or len(set(identities)) != len(identities):
        raise ValueError(f"step {step} does not contain 16 unique sample identities")
    if set(identities) != set(EXPECTED_IDENTITIES):
        raise ValueError(f"step {step} sample identities do not match the frozen protocol")
    return selected


def _best_step(selection: dict[str, Any]) -> int:
    raw = selection.get("best_velocity.pt")
    if not isinstance(raw, dict) or raw.get("optimizer_steps") is None:
        raise ValueError("checkpoint selection is missing best_velocity.pt")
    return int(raw["optimizer_steps"])


def _paired_bootstrap(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _model_result(
    model: str,
    metrics: list[dict[str, str]],
    samples: list[dict[str, Any]],
    selection: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step = _best_step(selection)
    row = _metric_row(metrics, step)
    selected = _sample_rows(samples, step)
    values = np.asarray([float(sample["velocity_nrmse"]) for sample in selected])
    initial_loss = float(_metric_row(metrics, 0)["loss"])
    selected_loss = float(row["loss"])
    historical_median = float(row["velocity_nrmse"])
    loss_reduction = initial_loss / max(selected_loss, 1e-30)
    result = {
        "best_eligible_step": step,
        "conventional_sample_median_nrmse": float(np.median(values)),
        "historical_gate_median_nrmse": historical_median,
        "loss_reduction": loss_reduction,
        "model": model,
        "parameter_count": None,
        "passed": loss_reduction >= 1_000 and historical_median <= threshold,
        "sample_max_nrmse": float(np.max(values)),
        "sample_mean_nrmse": float(np.mean(values)),
        "sample_min_nrmse": float(np.min(values)),
        "velocity_nrmse_u": float(row["velocity_nrmse_u"]),
        "velocity_nrmse_v": float(row["velocity_nrmse_v"]),
    }
    return result, selected


def _plot_convergence(
    model_metrics: dict[str, list[dict[str, str]]], threshold: float, output: Path
) -> None:
    colors = {"fno_ct": "#2563eb", "chronopde": "#dc2626"}
    labels = {"fno_ct": "CT-FFT", "chronopde": "ChronoPDE DCT"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model, rows in model_metrics.items():
        steps = [float(row["optimizer_steps"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]
        errors = [float(row["velocity_nrmse"]) for row in rows]
        axes[0].plot(steps, losses, color=colors[model], label=labels[model])
        axes[1].plot(steps, errors, color=colors[model], label=labels[model])
    axes[0].set_yscale("log")
    axes[0].set(title="Aligned optimization loss", xlabel="optimizer step", ylabel="loss")
    axes[1].set_yscale("log")
    axes[1].axhline(threshold, color="#111827", linestyle="--", label="0.01 gate")
    axes[1].set(title="Fixed-sample velocity error", xlabel="optimizer step", ylabel="nRMSE")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_pairs(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5), constrained_layout=True)
    intervals = sorted({int(row["interval_index"]) for row in rows})
    colors = dict(zip(intervals, ("#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"), strict=True))
    maximum = max(max(float(row["fno_ct_nrmse"]), float(row["chronopde_nrmse"])) for row in rows)
    for interval in intervals:
        subset = [row for row in rows if int(row["interval_index"]) == interval]
        axis.scatter(
            [float(row["fno_ct_nrmse"]) for row in subset],
            [float(row["chronopde_nrmse"]) for row in subset],
            color=colors[interval],
            label=f"interval {interval}",
            s=48,
        )
    axis.plot((0, maximum * 1.08), (0, maximum * 1.08), color="#111827", linestyle="--")
    axis.set(
        xlim=(0, maximum * 1.08),
        ylim=(0, maximum * 1.08),
        xlabel="CT-FFT velocity nRMSE",
        ylabel="ChronoPDE DCT velocity nRMSE",
        title="Matched diagnostic samples (below line favors DCT)",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_conditioning(rows: list[dict[str, Any]], output: Path) -> None:
    colors = {"fno_ct": "#2563eb", "chronopde": "#dc2626"}
    labels = {"fno_ct": "CT-FFT", "chronopde": "ChronoPDE DCT"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model in ("fno_ct", "chronopde"):
        key = f"{model}_nrmse"
        axes[0].scatter(
            [float(row["target_energy"]) for row in rows],
            [float(row[key]) for row in rows],
            color=colors[model],
            label=labels[model],
            alpha=0.8,
        )
        axes[1].scatter(
            [float(row["time"]) for row in rows],
            [float(row[key]) for row in rows],
            color=colors[model],
            label=labels[model],
            alpha=0.8,
        )
    axes[0].set(title="Error versus target energy", xlabel="target energy", ylabel="velocity nRMSE")
    axes[1].set(title="Error versus physical time", xlabel="time", ylabel="velocity nRMSE")
    for axis in axes:
        axis.axhline(0.01, color="#111827", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_objectives(
    balanced: dict[str, Any], aligned_models: dict[str, dict[str, Any]], output: Path
) -> None:
    x = np.arange(2)
    width = 0.34
    balanced_runs = cast(dict[str, Any], balanced["runs"])
    old = [float(balanced_runs[name]["best_velocity_nrmse"]) for name in ("fno_ct", "chronopde")]
    new = [aligned_models[name]["historical_gate_median_nrmse"] for name in ("fno_ct", "chronopde")]
    figure, axis = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    axis.bar(x - width / 2, old, width, label="MSE + 12-mode spectral", color="#94a3b8")
    axis.bar(x + width / 2, new, width, label="full-field relative", color="#7c3aed")
    axis.axhline(0.01, color="#111827", linestyle="--", label="0.01 gate")
    axis.set(
        xticks=x,
        xticklabels=("CT-FFT", "ChronoPDE DCT"),
        ylabel="best fixed-sample velocity nRMSE",
        title="Objective alignment improves both models but does not pass",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_decision_markdown(path: Path, report: dict[str, Any]) -> None:
    models = cast(dict[str, dict[str, Any]], report["models"])
    bootstrap = cast(dict[str, Any], report["bootstrap"])
    interval = cast(list[float], bootstrap["confidence_interval_95"])
    id_results = cast(dict[str, Any], report["exploratory_id"])
    fft_row = (
        f"| CT-FFT | {models['fno_ct']['best_eligible_step']} | "
        f"{models['fno_ct']['loss_reduction']:.0f}x | "
        f"{models['fno_ct']['historical_gate_median_nrmse']:.5f} | Fail |"
    )
    dct_row = (
        f"| ChronoPDE DCT | {models['chronopde']['best_eligible_step']} | "
        f"{models['chronopde']['loss_reduction']:.0f}x | "
        f"{models['chronopde']['historical_gate_median_nrmse']:.5f} | Fail |"
    )
    text = f"""# Week 6 decision: valid negative result

## Decision

The fixed 5,000-step loss-alignment experiment completed, but neither matched
continuous-time model reached the unchanged `0.01` velocity-nRMSE gate. The
frozen route is `{EXPECTED_ROUTE}`. Production retraining, four-trajectory
comparison, sparse-time experiments, and OOD evaluation are not authorized.

## Gate result

| Model | Best eligible step | Loss reduction | Historical gate nRMSE | Result |
| --- | ---: | ---: | ---: | --- |
{fft_row}
{dct_row}

The historical gate used `torch.median`, which returns the lower middle value
for an even sample count. Conventional medians are
`{models["fno_ct"]["conventional_sample_median_nrmse"]:.5f}` for CT-FFT and
`{models["chronopde"]["conventional_sample_median_nrmse"]:.5f}` for DCT. Both
definitions preserve the failed decision.

## What was learned

- The 400-sample target audit passed; median spline/PDE nRMSE was
  `{report["target_audit"]["median_spline_pde_nrmse"]:.5f}`.
- Full-field relative loss reduced the balanced error by
  `{100 * report["objective_alignment_improvement"]["fno_ct"]["fraction"]:.1f}%`
  for CT-FFT and `{100 * report["objective_alignment_improvement"]["chronopde"]["fraction"]:.1f}%`
  for DCT.
- DCT had lower error on all `{report["dct_wins"]}` paired samples. The paired
  mean DCT-minus-FFT difference was `{bootstrap["paired_mean_difference_dct_minus_fft"]:.5f}`
  with a deterministic bootstrap 95% interval of
  `[{interval[0]:.5f}, {interval[1]:.5f}]`. This is a diagnostic result, not a
  generalization claim.
- The exploratory 100-trajectory ID runs were stable and beat persistence, but
  used unequal training histories. Their final nRMSE values were
  `{id_results["fno_ct"]["median_final_nrmse"]:.4f}` for CT-FFT and
  `{id_results["chronopde"]["median_final_nrmse"]:.4f}` for DCT and are retained
  only as context.

## Claim boundary

The project may claim a reproducible objective-metric alignment finding and a
matched diagnostic DCT advantage. It may not claim that ChronoPDE passed Week 6,
is generally superior to FFT, or has validated sparse-time or OOD performance.
"""
    path.write_text(text, encoding="utf-8")


def analyze_week6_failure(
    root: Path,
    manifest_path: Path,
    output: Path,
    *,
    archive_root: Path | None = None,
) -> dict[str, Any]:
    """Validate frozen evidence and write the deterministic recovery report."""
    manifest = _read_json(manifest_path)
    archive_declarations = _archive_declarations(manifest)
    inputs = _resolve_inputs(root, manifest)
    archive_hashes = verify_archives(manifest, archive_root) if archive_root else {}
    imported_hashes = {key: _sha256(path) for key, path in sorted(inputs.items())}

    protocol = _read_json(inputs["loss_alignment_protocol"])
    suite = _read_json(inputs["loss_alignment_suite"])
    threshold = float(protocol["median_velocity_nrmse_threshold"])
    if suite.get("budget_complete") is not True or suite.get("route") != EXPECTED_ROUTE:
        raise ValueError("loss-alignment suite is incomplete or has an unexpected route")
    protocol_identities = tuple(
        (str(row["trajectory_id"]), int(row["interval_index"]))
        for row in cast(list[dict[str, Any]], protocol["sample_identities"])
    )
    if protocol_identities != EXPECTED_IDENTITIES:
        raise ValueError("loss-alignment protocol identities changed")

    model_metrics: dict[str, list[dict[str, str]]] = {}
    model_samples: dict[str, list[dict[str, Any]]] = {}
    models: dict[str, dict[str, Any]] = {}
    selected_samples: dict[str, list[dict[str, Any]]] = {}
    for model in ("fno_ct", "chronopde"):
        metrics = _read_csv(inputs[f"loss_alignment_{model}_metrics"])
        samples = _read_jsonl(inputs[f"loss_alignment_{model}_samples"])
        selection = _read_json(inputs[f"loss_alignment_{model}_checkpoint_selection"])
        result, selected = _model_result(model, metrics, samples, selection, threshold)
        suite_run = cast(dict[str, Any], cast(dict[str, Any], suite["runs"])[model])
        training = cast(dict[str, Any], suite_run["training"])
        result["parameter_count"] = int(training["parameter_count"])
        if (
            abs(result["historical_gate_median_nrmse"] - float(suite_run["best_velocity_nrmse"]))
            > 1e-9
        ):
            raise ValueError(f"{model} suite summary disagrees with selected metrics")
        model_metrics[model] = metrics
        model_samples[model] = samples
        models[model] = result
        selected_samples[model] = selected

    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for model, rows in selected_samples.items():
        indexed[model] = {
            (str(row["trajectory_id"]), int(row["interval_index"])): row for row in rows
        }
    paired_rows: list[dict[str, Any]] = []
    for trajectory_id, interval in EXPECTED_IDENTITIES:
        fft = indexed["fno_ct"][(trajectory_id, interval)]
        dct = indexed["chronopde"][(trajectory_id, interval)]
        if abs(float(fft["time"]) - float(dct["time"])) > 1e-6:
            raise ValueError("paired samples have inconsistent times")
        if abs(float(fft["target_energy"]) - float(dct["target_energy"])) > 1e-6:
            raise ValueError("paired samples have inconsistent target energy")
        fft_error = float(fft["velocity_nrmse"])
        dct_error = float(dct["velocity_nrmse"])
        paired_rows.append(
            {
                "chronopde_nrmse": dct_error,
                "difference_dct_minus_fft": dct_error - fft_error,
                "fno_ct_nrmse": fft_error,
                "interval_index": interval,
                "target_energy": float(fft["target_energy"]),
                "time": float(fft["time"]),
                "trajectory_id": trajectory_id,
            }
        )
    differences = np.asarray([float(row["difference_dct_minus_fft"]) for row in paired_rows])
    confidence_interval = _paired_bootstrap(differences)

    balanced = _read_json(inputs["balanced_suite"])
    balanced_runs = cast(dict[str, Any], balanced["runs"])
    improvements = {
        model: {
            "factor": float(balanced_runs[model]["best_velocity_nrmse"])
            / models[model]["historical_gate_median_nrmse"],
            "fraction": 1.0
            - models[model]["historical_gate_median_nrmse"]
            / float(balanced_runs[model]["best_velocity_nrmse"]),
        }
        for model in ("fno_ct", "chronopde")
    }
    target_audit = _read_json(inputs["target_audit"])
    exploratory_id = {
        "chronopde": _read_json(inputs["chronopde_id"]),
        "fno_ar": _read_json(inputs["week4_fno_ar_id"]),
        "fno_ct": _read_json(inputs["ct_fft_id"]),
        "status": "context_only_not_confirmatory",
        "unet_ar": _read_json(inputs["week4_unet_ar_id"]),
    }
    result = {
        "archive_hashes_declared": archive_declarations,
        "archive_hashes_verified": archive_hashes,
        "bootstrap": {
            "confidence_interval_95": list(confidence_interval),
            "paired_mean_difference_dct_minus_fft": float(np.mean(differences)),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "claim_status": "valid_negative_result",
        "decision": EXPECTED_ROUTE,
        "dct_wins": int(np.sum(differences < 0)),
        "exploratory_id": exploratory_id,
        "gate": {
            "loss_reduction_threshold": float(protocol["minimum_loss_reduction"]),
            "median_definition": "torch.median lower-middle value for an even sample count",
            "passed": False,
            "sample_count": len(EXPECTED_IDENTITIES),
            "velocity_nrmse_threshold": threshold,
        },
        "imported_evidence_sha256": imported_hashes,
        "models": models,
        "objective_alignment_improvement": improvements,
        "target_audit": {
            "median_spline_pde_nrmse": float(target_audit["median_spline_pde_nrmse"]),
            "passed": bool(target_audit["passed"]),
            "samples": int(target_audit["samples"]),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "failure_analysis.json", result)
    _write_json(output / "final_summary.json", result)
    _write_json(output / "demo_payload.json", {"paired_samples": paired_rows})
    with (output / "paired_sample_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(paired_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(paired_rows)
    _plot_convergence(model_metrics, threshold, output / "loss_alignment_convergence.png")
    _plot_pairs(paired_rows, output / "paired_model_comparison.png")
    _plot_conditioning(paired_rows, output / "conditioning_analysis.png")
    _plot_objectives(balanced, models, output / "objective_alignment.png")
    _write_decision_markdown(output / "week6_negative_result.md", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("reports/diagnostics/week6/evidence_manifest.json"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("reports/final"))
    parser.add_argument("--archive-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    manifest = args.evidence_manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    output = args.output_directory
    if not output.is_absolute():
        output = root / output
    report = analyze_week6_failure(
        root,
        manifest,
        output,
        archive_root=args.archive_root.resolve() if args.archive_root else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
