"""Render the frozen exploratory FFT/DCT rollout comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronopde.config import load_config
from chronopde.data.datasets import HDF5RolloutDataset, NormalizationStats
from chronopde.evaluation.rollout import continuous_rollout
from chronopde.training.continuous import build_continuous_model
from chronopde.training.trainer import load_checkpoint, resolve_device

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_ID = "id-0042"
TIME_INDICES = (20, 60, 100)
EXPECTED_DATA_SHA256 = "907aa0d79e604e68ce2d4f5cccfd93ffc64eb68c473caf3edae4be438472caec"
EXPECTED_CHECKPOINT_SHA256 = {
    "fno_ct": "8c2ccbf0e6dad3fcd7c279bdb8861b440336a70d850ccc97dbbab69f2b76e229",
    "chronopde": "2d09db1df2b4c2513af852e7848f8267aa3d989378882e85e59b1b3626b62735",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: {actual}")
    return actual


def _stats_to(stats: NormalizationStats, device: torch.device) -> NormalizationStats:
    return NormalizationStats(
        state_mean=stats.state_mean.to(device),
        state_std=stats.state_std.to(device),
        parameter_mean=stats.parameter_mean.to(device),
        parameter_std=stats.parameter_std.to(device),
    )


def _denormalize(states: Tensor, stats: NormalizationStats) -> Tensor:
    return (
        states * stats.state_std[None, None, :, None, None]
        + stats.state_mean[None, None, :, None, None]
    )


def _rollout(
    model_name: str,
    checkpoint: Path,
    sample: Any,
    stats: NormalizationStats,
    device: torch.device,
) -> Tensor:
    config = load_config(ROOT / "configs/project.yaml")
    model = build_continuous_model(config, cast(Any, model_name)).to(device)
    load_checkpoint(checkpoint, model, expected_model_name=model_name)
    model.eval()
    with torch.no_grad():
        result = continuous_rollout(
            model,
            sample.states[0][None].to(device),
            sample.times[None].to(device),
            sample.parameters[None].to(device),
            steps_per_interval=2,
        )
    return _denormalize(cast(Tensor, result.states), stats)[0].cpu()


def _nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sqrt(np.mean(np.square(prediction - target)))
    denominator = max(float(np.sqrt(np.mean(np.square(target)))), 1e-12)
    return float(numerator / denominator)


def _plot(
    target: np.ndarray,
    fft: np.ndarray,
    dct: np.ndarray,
    times: np.ndarray,
    output: Path,
) -> None:
    columns = ("Ground truth", "CT-FFT", "ChronoPDE DCT", "|FFT error|", "|DCT error|")
    figure, axes = plt.subplots(6, 5, figsize=(14, 14), constrained_layout=True)
    for channel, channel_name in enumerate(("u", "v")):
        values = np.concatenate(
            [
                target[list(TIME_INDICES), channel],
                fft[list(TIME_INDICES), channel],
                dct[list(TIME_INDICES), channel],
            ]
        )
        field_min, field_max = float(values.min()), float(values.max())
        errors = np.concatenate(
            [
                np.abs(fft[list(TIME_INDICES), channel] - target[list(TIME_INDICES), channel]),
                np.abs(dct[list(TIME_INDICES), channel] - target[list(TIME_INDICES), channel]),
            ]
        )
        error_max = max(float(errors.max()), 1e-12)
        for time_position, time_index in enumerate(TIME_INDICES):
            row = channel * len(TIME_INDICES) + time_position
            truth = target[time_index, channel]
            fft_field = fft[time_index, channel]
            dct_field = dct[time_index, channel]
            panels = (
                truth,
                fft_field,
                dct_field,
                np.abs(fft_field - truth),
                np.abs(dct_field - truth),
            )
            for column, panel in enumerate(panels):
                error_panel = column >= 3
                axes[row, column].imshow(
                    panel,
                    cmap="magma" if error_panel else "viridis",
                    vmin=0.0 if error_panel else field_min,
                    vmax=error_max if error_panel else field_max,
                )
                axes[row, column].set_axis_off()
                if row == 0:
                    axes[row, column].set_title(columns[column], fontsize=11)
            axes[row, 0].text(
                -0.14,
                0.5,
                f"{channel_name}  t={times[time_index]:.0f}",
                rotation=90,
                va="center",
                ha="center",
                transform=axes[row, 0].transAxes,
                fontsize=10,
                fontweight="bold",
            )
    figure.suptitle(
        "Exploratory held-out rollout: id-0042\n"
        "Frozen full-run checkpoints; unequal training histories",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--fft-checkpoint", type=Path, required=True)
    parser.add_argument("--dct-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures/qualitative_rollout.png",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        "data": args.data_path.resolve(),
        "fno_ct": args.fft_checkpoint.resolve(),
        "chronopde": args.dct_checkpoint.resolve(),
    }
    hashes = {
        "data": _verify(paths["data"], EXPECTED_DATA_SHA256, "dataset"),
        "fno_ct": _verify(paths["fno_ct"], EXPECTED_CHECKPOINT_SHA256["fno_ct"], "FFT checkpoint"),
        "chronopde": _verify(
            paths["chronopde"],
            EXPECTED_CHECKPOINT_SHA256["chronopde"],
            "DCT checkpoint",
        ),
    }
    device = resolve_device(args.device)
    dataset = HDF5RolloutDataset(paths["data"], "id", "full", trajectory_ids=(REPRESENTATIVE_ID,))
    sample = dataset[0]
    stats = _stats_to(dataset.normalization, device)
    target = _denormalize(sample.states[None].to(device), stats)[0].cpu()
    fft = _rollout("fno_ct", paths["fno_ct"], sample, stats, device)
    dct = _rollout("chronopde", paths["chronopde"], sample, stats, device)
    target_array = target.numpy()
    fft_array = fft.numpy()
    dct_array = dct.numpy()
    times = sample.times.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(target_array, fft_array, dct_array, times, args.output)
    metadata = {
        "artifact_status": "exploratory_context_only",
        "checkpoint_sha256": {
            "chronopde": hashes["chronopde"],
            "fno_ct": hashes["fno_ct"],
        },
        "dataset_sha256": hashes["data"],
        "device": str(device),
        "nrmse": {
            model: {
                str(int(times[index])): {
                    channel_name: _nrmse(prediction[index, channel], target_array[index, channel])
                    for channel, channel_name in enumerate(("u", "v"))
                }
                for index in TIME_INDICES
            }
            for model, prediction in (("fno_ct", fft_array), ("chronopde", dct_array))
        },
        "selection_rule": (
            "closest to the median combined percentile rank of mean rollout nRMSE across "
            "the two archived 100-trajectory ID evaluations; lexicographic ID tie-break"
        ),
        "source_commits": {
            "chronopde": "47fb45711391b34f5d06e6e5dc3808f18ef60665",
            "fno_ct": "7d15658c9229e989a85feb4807c37e65daa71485",
        },
        "time_indices": list(TIME_INDICES),
        "times": [float(times[index]) for index in TIME_INDICES],
        "training_histories": {
            "chronopde": "85 epochs; early-stopped exploratory run",
            "fno_ct": "150 epochs; completed exploratory run",
        },
        "trajectory_id": REPRESENTATIVE_ID,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset.close()
    print(args.output)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
