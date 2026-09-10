"""Deterministic seeding and environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> dict[str, bool]:
    """Seed Python, NumPy, and PyTorch when installed."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
    except ImportError:
        pass
    return {"python": True, "numpy": True, "torch": torch_seeded}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_metadata(repo_root: Path, seed: int) -> dict[str, Any]:
    """Return JSON-serialisable provenance for a scientific run."""

    torch_details: dict[str, Any] = {"installed": False}
    try:
        import torch

        torch_details = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except ImportError:
        pass

    return {
        "project": "chronopde",
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "git_commit": git_commit(repo_root),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "h5py", "pydantic", "pyyaml", "torch")
        },
        "torch": torch_details,
    }


def write_environment_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
