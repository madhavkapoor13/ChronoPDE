"""Verified local recovery packages for completed and interrupted V2 runs."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from chronopde.v2.checkpoints import read_checkpoint_identity, verify_checkpoint
from chronopde.v2.common import canonical_json_bytes, read_json_object, sha256_file
from chronopde.v2.registry import ExecutionStatus, RunManifest

RECOVERY_SCHEMA_VERSION = 1


def _safe_name(path: Path) -> str:
    if path.name.startswith(".") or path.suffix == ".tmp" or ".tmp." in path.name:
        raise ValueError(f"temporary files cannot enter recovery packages: {path}")
    return path.name


def create_recovery_package(
    output: Path,
    *,
    run_manifest: Path,
    resolved_config: Path,
    metrics: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    manifest = RunManifest.from_dict(read_json_object(run_manifest))
    for path in (resolved_config, metrics, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
        _safe_name(path)
    checkpoint_hash = verify_checkpoint(checkpoint)
    checkpoint_identity = read_checkpoint_identity(checkpoint)
    expected_identity = (
        manifest.run_id,
        manifest.model,
        manifest.config_hash,
        manifest.dataset_hash,
        manifest.normalization_hash,
    )
    actual_identity = (
        checkpoint_identity.run_id,
        checkpoint_identity.model_name,
        checkpoint_identity.config_hash,
        checkpoint_identity.dataset_hash,
        checkpoint_identity.normalization_hash,
    )
    if actual_identity != expected_identity:
        raise ValueError("checkpoint identity does not match the recovery run manifest")
    if sha256_file(resolved_config) != manifest.config_hash:
        raise ValueError("resolved configuration hash does not match the run manifest")
    partial = manifest.execution_status in {ExecutionStatus.FAILED, ExecutionStatus.INTERRUPTED}
    entries = {
        "run_manifest.json": run_manifest,
        "resolved_config.yaml": resolved_config,
        "metrics.jsonl": metrics,
        "last.pt": checkpoint,
        "last.pt.sha256": checkpoint.with_suffix(checkpoint.suffix + ".sha256"),
    }
    recovery_manifest: dict[str, Any] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "partial": partial,
        "execution_status": manifest.execution_status.value,
        "files": {name: sha256_file(path) for name, path in entries.items()},
        "checkpoint_sha256": checkpoint_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in entries.items():
                archive.write(path, name)
            archive.writestr("recovery_manifest.json", canonical_json_bytes(recovery_manifest))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    verify_recovery_package(output)
    return recovery_manifest


def verify_recovery_package(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise ValueError("unsafe recovery package member")
        raw_manifest = json.loads(archive.read("recovery_manifest.json"))
        if not isinstance(raw_manifest, dict) or raw_manifest.get("schema_version") != 1:
            raise ValueError("unsupported recovery package manifest")
        declared = raw_manifest.get("files")
        if not isinstance(declared, dict):
            raise ValueError("recovery package file declarations are missing")
        import hashlib

        for name, expected in declared.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError("invalid recovery package declaration")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"recovery package checksum mismatch: {name}")
    return raw_manifest
