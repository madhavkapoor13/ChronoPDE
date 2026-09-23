"""Versioned, provenance-checked checkpoints for ChronoPDE V2."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from chronopde.v2.common import atomic_write_bytes, sha256_file
from chronopde.v2.registry import HASH_PATTERN, STUDY_ID

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointIdentity:
    study_id: str
    run_id: str
    model_name: str
    config_hash: str
    dataset_hash: str | None
    normalization_hash: str | None

    def validate(self) -> None:
        if self.study_id != STUDY_ID:
            raise ValueError("checkpoint has an unsupported study identity")
        for field, value in (
            ("config_hash", self.config_hash),
            ("dataset_hash", self.dataset_hash),
            ("normalization_hash", self.normalization_hash),
        ):
            if value is None or not HASH_PATTERN.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _write_sidecar(path: Path) -> None:
    atomic_write_bytes(_sidecar_path(path), f"{sha256_file(path)}  {path.name}\n".encode())


def verify_checkpoint(path: Path) -> str:
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("checkpoint and SHA-256 sidecar are both required")
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name or not HASH_PATTERN.fullmatch(parts[0]):
        raise ValueError("invalid checkpoint SHA-256 sidecar")
    actual = sha256_file(path)
    if actual != parts[0]:
        raise ValueError("checkpoint SHA-256 mismatch")
    return actual


def save_v2_checkpoint(
    path: Path,
    *,
    identity: CheckpointIdentity,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    optimizer_steps: int,
    sampler_state: dict[str, Any],
    best_metric: float | None,
    patience_counter: int,
    checkpoint_selection: dict[str, Any],
) -> str:
    identity.validate()
    if epoch < 0 or optimizer_steps < 0 or patience_counter < 0:
        raise ValueError("checkpoint progress values must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": asdict(identity),
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "sampler_state": sampler_state,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "checkpoint_selection": checkpoint_selection,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _write_sidecar(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _read_identity(payload: dict[str, Any]) -> CheckpointIdentity:
    raw = payload.get("identity")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint identity is missing")
    identity = CheckpointIdentity(
        study_id=str(raw.get("study_id")),
        run_id=str(raw.get("run_id")),
        model_name=str(raw.get("model_name")),
        config_hash=str(raw.get("config_hash")),
        dataset_hash=cast(str | None, raw.get("dataset_hash")),
        normalization_hash=cast(str | None, raw.get("normalization_hash")),
    )
    identity.validate()
    return identity


def read_checkpoint_identity(path: Path) -> CheckpointIdentity:
    """Verify a checkpoint and return its identity without restoring any state."""
    verify_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported V2 checkpoint schema")
    return _read_identity(cast(dict[str, Any], payload))


def load_v2_checkpoint(
    path: Path,
    *,
    expected_identity: CheckpointIdentity,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    actual_identity = read_checkpoint_identity(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported V2 checkpoint schema")
    expected_identity.validate()
    if actual_identity != expected_identity:
        differing = [
            field
            for field in asdict(expected_identity)
            if getattr(actual_identity, field) != getattr(expected_identity, field)
        ]
        raise ValueError(f"checkpoint identity mismatch: {', '.join(differing)}")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng(payload["rng_state"])
    return cast(dict[str, Any], payload)
