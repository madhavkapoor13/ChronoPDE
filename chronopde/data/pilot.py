"""Deterministic, resumable Week 2 simulator pilot and report generation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronopde.config import ProjectConfig
from chronopde.contracts import PhysicalParameters, SimulationResult
from chronopde.data.initial_conditions import InitialConditionRegime, generate_initial_condition
from chronopde.data.simulator import simulate_trajectory
from chronopde.numerics.laplacian import build_grid
from chronopde.reproducibility import git_commit


@dataclass(frozen=True)
class PilotCase:
    trajectory_id: str
    seed: int
    category: str
    ic_regime: InitialConditionRegime
    params: PhysicalParameters


@dataclass(frozen=True)
class PilotReport:
    passed: bool
    successful: int
    failed: int
    total: int
    allowed_failures: int
    config_hash: str
    output_directory: str
    published_directory: str


def configuration_hash(config: ProjectConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _midpoint(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2


def build_pilot_manifest(config: ProjectConfig) -> list[PilotCase]:
    """Construct the frozen 8 + 8 + 6 + 2 Week 2 pilot design."""

    train = config.data.train_ranges
    ood = config.data.ood_ranges
    cases: list[tuple[str, InitialConditionRegime, PhysicalParameters]] = []

    for values in product(train.du, train.dv, train.k):
        cases.append(("train_corner", "train", PhysicalParameters(*values)))

    global_extremes = (
        (ood.du[0][0], ood.du[1][1]),
        (ood.dv[0][0], ood.dv[1][1]),
        (ood.k[0][0], ood.k[1][1]),
    )
    for values in product(*global_extremes):
        cases.append(("ood_corner", "train", PhysicalParameters(*values)))

    centre = PhysicalParameters(_midpoint(train.du), _midpoint(train.dv), _midpoint(train.k))
    one_factor_values = {
        "du_low": PhysicalParameters(ood.du[0][0], centre.dv, centre.k),
        "du_high": PhysicalParameters(ood.du[1][1], centre.dv, centre.k),
        "dv_low": PhysicalParameters(centre.du, ood.dv[0][0], centre.k),
        "dv_high": PhysicalParameters(centre.du, ood.dv[1][1], centre.k),
        "k_low": PhysicalParameters(centre.du, centre.dv, ood.k[0][0]),
        "k_high": PhysicalParameters(centre.du, centre.dv, ood.k[1][1]),
    }
    for label, params in one_factor_values.items():
        cases.append((f"one_factor_{label}", "train", params))
    cases.extend(("ood_ic", "ood", centre) for _ in range(2))

    if len(cases) != config.data.pilot.trajectories:
        raise RuntimeError("pilot design does not match configured trajectory count")
    return [
        PilotCase(
            trajectory_id=f"pilot-{index:02d}",
            seed=config.data.base_seed + index,
            category=category,
            ic_regime=ic_regime,
            params=params,
        )
        for index, (category, ic_regime, params) in enumerate(cases)
    ]


def _manifest_row(case: PilotCase, config_hash: str, commit: str | None) -> dict[str, Any]:
    return {
        "trajectory_id": case.trajectory_id,
        "seed": case.seed,
        "category": case.category,
        "ic_regime": case.ic_regime,
        "du": case.params.du,
        "dv": case.params.dv,
        "k": case.params.k,
        "config_hash": config_hash,
        "git_commit": commit or "uncommitted",
    }


def write_manifest(
    path: Path, cases: list[PilotCase], config_hash: str, commit: str | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_manifest_row(case, config_hash, commit) for case in cases]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw_path(output_directory: Path, case: PilotCase) -> Path:
    return output_directory / "trajectories" / f"{case.trajectory_id}.npz"


def _load_cached_result(path: Path, config_hash: str) -> tuple[SimulationResult, bool] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["config_hash"].item()) != config_hash or not bool(data["success"].item()):
            return None
        params = PhysicalParameters(*np.asarray(data["params"], dtype=np.float64).tolist())
        from chronopde.contracts import SimulationDiagnostics

        diagnostics = SimulationDiagnostics(
            success=True,
            message=str(data["message"].item()),
            nfev=int(data["nfev"].item()),
            runtime_seconds=float(data["runtime_seconds"].item()),
            max_abs_state=float(data["max_abs_state"].item()),
        )
        result = SimulationResult(
            states=np.asarray(data["states"], dtype=np.float32),
            times=np.asarray(data["times"], dtype=np.float64),
            params=params,
            diagnostics=diagnostics,
        )
    return result, True


def _save_result(path: Path, result: SimulationResult, config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        states=result.states,
        times=result.times,
        params=result.params.as_array(),
        success=result.diagnostics.success,
        message=result.diagnostics.message,
        nfev=result.diagnostics.nfev,
        runtime_seconds=result.diagnostics.runtime_seconds,
        max_abs_state=result.diagnostics.max_abs_state,
        config_hash=config_hash,
    )
    temporary.replace(path)


def _execute_case(
    case: PilotCase,
    config: ProjectConfig,
    output_directory: Path,
    config_hash: str,
) -> tuple[PilotCase, SimulationResult, bool]:
    path = _raw_path(output_directory, case)
    cached = _load_cached_result(path, config_hash)
    if cached is not None:
        result, was_cached = cached
        return case, result, was_cached
    grid = build_grid(config.pde)
    initial_state = generate_initial_condition(
        grid, case.seed, case.ic_regime, config.data.initial_conditions
    )
    result = simulate_trajectory(
        initial_state,
        case.params,
        config.pde,
        divergence_threshold=config.data.pilot.divergence_threshold,
    )
    _save_result(path, result, config_hash)
    return case, result, False


def _diagnostic_rows(
    results: list[tuple[PilotCase, SimulationResult, bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, result, cached in results:
        states = result.states
        final_mean_u = float(np.mean(states[-1, 0])) if states.size else float("nan")
        final_mean_v = float(np.mean(states[-1, 1])) if states.size else float("nan")
        final_std_u = float(np.std(states[-1, 0])) if states.size else float("nan")
        final_std_v = float(np.std(states[-1, 1])) if states.size else float("nan")
        rows.append(
            {
                "trajectory_id": case.trajectory_id,
                "seed": case.seed,
                "category": case.category,
                "ic_regime": case.ic_regime,
                "du": case.params.du,
                "dv": case.params.dv,
                "k": case.params.k,
                "success": result.diagnostics.success,
                "cached": cached,
                "runtime_seconds": result.diagnostics.runtime_seconds,
                "nfev": result.diagnostics.nfev,
                "max_abs_state": result.diagnostics.max_abs_state,
                "final_mean_u": final_mean_u,
                "final_mean_v": final_mean_v,
                "final_std_u": final_std_u,
                "final_std_v": final_std_v,
                "message": result.diagnostics.message,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_state_panels(
    results: list[tuple[PilotCase, SimulationResult, bool]], output: Path
) -> None:
    selected = [
        next(item for item in results if item[0].category == "train_corner"),
        next(item for item in results if item[0].category == "ood_ic"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for row, (case, result, _) in enumerate(selected):
        for column, (time_index, channel, title) in enumerate(
            ((0, 0, "initial u"), (0, 1, "initial v"), (-1, 0, "final u"), (-1, 1, "final v"))
        ):
            image = axes[row, column].imshow(result.states[time_index, channel], cmap="coolwarm")
            axes[row, column].set_title(f"{case.ic_regime}: {title}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.savefig(output / "state_panels.png", dpi=160)
    plt.close(figure)


def _plot_maximums(
    results: list[tuple[PilotCase, SimulationResult, bool]],
    output: Path,
    divergence_threshold: float,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case, result, _ in results:
        if not result.states.size:
            continue
        maximum = np.max(np.abs(result.states), axis=(1, 2, 3))
        axis.plot(result.times, maximum, alpha=0.65, label=case.trajectory_id)
    axis.set(xlabel="time", ylabel="maximum absolute state", title="Pilot stability")
    axis.axhline(
        divergence_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label="threshold",
    )
    axis.grid(alpha=0.25)
    figure.savefig(output / "max_abs_over_time.png", dpi=160)
    plt.close(figure)


def _plot_distributions(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].hist([row["runtime_seconds"] for row in rows], bins=10)
    axes[0, 0].set_title("Runtime (seconds)")
    axes[0, 1].hist([row["nfev"] for row in rows], bins=10)
    axes[0, 1].set_title("Function evaluations")
    axes[1, 0].hist(
        [[row["final_mean_u"] for row in rows], [row["final_mean_v"] for row in rows]],
        bins=10,
        label=("u", "v"),
    )
    axes[1, 0].set_title("Final channel means")
    axes[1, 0].legend()
    axes[1, 1].hist(
        [[row["final_std_u"] for row in rows], [row["final_std_v"] for row in rows]],
        bins=10,
        label=("u", "v"),
    )
    axes[1, 1].set_title("Final channel standard deviations")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / "diagnostic_distributions.png", dpi=160)
    plt.close(figure)


def _plot_parameter_coverage(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    pairs = (("du", "dv"), ("du", "k"), ("dv", "k"))
    categories = sorted({str(row["category"]) for row in rows})
    for axis, (x_name, y_name) in zip(axes, pairs, strict=True):
        for category in categories:
            subset = [row for row in rows if row["category"] == category]
            axis.scatter(
                [row[x_name] for row in subset],
                [row[y_name] for row in subset],
                label=category,
                alpha=0.8,
            )
        axis.set(xlabel=x_name, ylabel=y_name)
        axis.grid(alpha=0.2)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.2, 1, 1))
    figure.savefig(output / "parameter_coverage.png", dpi=160)
    plt.close(figure)


def _publish_report(
    output_directory: Path,
    published_directory: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    published_directory.mkdir(parents=True, exist_ok=True)
    for filename in (
        "diagnostics.csv",
        "summary.json",
        "state_panels.png",
        "max_abs_over_time.png",
        "diagnostic_distributions.png",
        "parameter_coverage.png",
    ):
        shutil.copy2(output_directory / filename, published_directory / filename)
    adjustment_status = "required" if not summary["passed"] else "not required"
    lines = [
        "# Week 2 Simulator Pilot",
        "",
        f"- Status: **{'PASS' if summary['passed'] else 'FAIL'}**",
        f"- Successful trajectories: {summary['successful']}/{summary['total']}",
        f"- Maximum state magnitude: {summary['maximum_state_magnitude']:.6f}",
        f"- Median runtime: {summary['median_runtime_seconds']:.3f} seconds",
        f"- Median function evaluations: {summary['median_nfev']:.0f}",
        f"- Parameter-range adjustment: **{adjustment_status}**",
        f"- Configuration SHA-256: `{summary['config_hash']}`",
        "",
        "The pilot covers all training-range corners, global OOD extremes,",
        "one-factor OOD extremes, and two high-frequency OOD initial conditions.",
        "Raw trajectories remain under the ignored artifacts directory.",
        "",
        "## Figures",
        "",
        "- `state_panels.png`",
        "- `max_abs_over_time.png`",
        "- `diagnostic_distributions.png`",
        "- `parameter_coverage.png`",
    ]
    (published_directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    config: ProjectConfig,
    repo_root: Path,
    *,
    output_directory: Path | None = None,
    published_directory: Path | None = None,
) -> PilotReport:
    """Run or resume the pilot and publish deterministic diagnostics."""

    output = output_directory or repo_root / config.data.pilot.output_directory
    published = published_directory or repo_root / "reports/pilot/week2"
    output.mkdir(parents=True, exist_ok=True)
    cases = build_pilot_manifest(config)
    digest = configuration_hash(config)
    commit = git_commit(repo_root)
    write_manifest(output / "manifest.csv", cases, digest, commit)

    with ThreadPoolExecutor(max_workers=config.data.pilot.workers) as executor:
        results = list(
            executor.map(
                lambda case: _execute_case(case, config, output, digest),
                cases,
            )
        )
    rows = _diagnostic_rows(results)
    _write_csv(output / "diagnostics.csv", rows)

    successful = sum(bool(row["success"]) for row in rows)
    total = len(rows)
    failed = total - successful
    allowed_failures = int(np.floor(config.data.pilot.maximum_failure_fraction * total))
    passed = failed <= allowed_failures
    summary = {
        "passed": passed,
        "successful": successful,
        "failed": failed,
        "total": total,
        "allowed_failures": allowed_failures,
        "maximum_state_magnitude": max(float(row["max_abs_state"]) for row in rows),
        "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in rows])),
        "median_nfev": float(np.median([row["nfev"] for row in rows])),
        "cached_trajectories": sum(bool(row["cached"]) for row in rows),
        "config_hash": digest,
        "git_commit": commit,
        "range_adjustment_required": not passed,
        "range_adjustments": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_state_panels(results, output)
    _plot_maximums(results, output, config.data.pilot.divergence_threshold)
    _plot_distributions(rows, output)
    _plot_parameter_coverage(rows, output)
    _publish_report(output, published, rows, summary)

    return PilotReport(
        passed=passed,
        successful=successful,
        failed=failed,
        total=total,
        allowed_failures=allowed_failures,
        config_hash=digest,
        output_directory=str(output),
        published_directory=str(published),
    )
