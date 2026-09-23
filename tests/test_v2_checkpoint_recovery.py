from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from chronopde.v2.checkpoints import (
    CheckpointIdentity,
    load_v2_checkpoint,
    save_v2_checkpoint,
    verify_checkpoint,
)
from chronopde.v2.common import sha256_file
from chronopde.v2.recovery import create_recovery_package, verify_recovery_package
from chronopde.v2.registry import ExecutionStatus, create_run_manifest, transition_run_manifest

CONFIG_HASH = "a" * 64
DATASET_HASH = "c" * 64
NORMALIZATION_HASH = "d" * 64
COMMIT = "b" * 40


def _parts():
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def _identity(model: str = "toy") -> CheckpointIdentity:
    return CheckpointIdentity(
        study_id="chronopde_v2",
        run_id="chronopde_v2-p2-reaction_diffusion-boundary-toy-full-s0-aaaaaaaa",
        model_name=model,
        config_hash=CONFIG_HASH,
        dataset_hash=DATASET_HASH,
        normalization_hash=NORMALIZATION_HASH,
    )


def _save(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    identity: CheckpointIdentity | None = None,
) -> None:
    save_v2_checkpoint(
        path,
        identity=identity or _identity(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        optimizer_steps=2,
        sampler_state={"position": 4, "order": [2, 0, 3, 1]},
        best_metric=0.5,
        patience_counter=0,
        checkpoint_selection={"criterion": "minimum_validation_rollout"},
    )


def test_identity_mismatch_precedes_state_mutation(tmp_path: Path) -> None:
    model, optimizer, scheduler = _parts()
    path = tmp_path / "last.pt"
    _save(path, model, optimizer, scheduler)
    target, target_optimizer, target_scheduler = _parts()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="model_name"):
        load_v2_checkpoint(
            path,
            expected_identity=_identity("wrong"),
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
        )
    assert all(torch.equal(before[name], value) for name, value in target.state_dict().items())


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    model, optimizer, scheduler = _parts()
    path = tmp_path / "last.pt"
    _save(path, model, optimizer, scheduler)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path)


def test_atomic_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer, scheduler = _parts()
    path = tmp_path / "last.pt"
    _save(path, model, optimizer, scheduler)
    original_hash = verify_checkpoint(path)
    import chronopde.v2.checkpoints as checkpoints

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        _save(path, model, optimizer, scheduler)
    assert verify_checkpoint(path) == original_hash


def test_rng_and_sampler_state_resume_deterministically(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler = _parts()
    path = tmp_path / "last.pt"
    _save(path, model, optimizer, scheduler)
    expected = (random.random(), float(np.random.random()), torch.rand(2))

    restored, restored_optimizer, restored_scheduler = _parts()
    payload = load_v2_checkpoint(
        path,
        expected_identity=_identity(),
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    observed = (random.random(), float(np.random.random()), torch.rand(2))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
    assert payload["sampler_state"] == {"position": 4, "order": [2, 0, 3, 1]}


def test_resumed_toy_training_matches_uninterrupted_training(tmp_path: Path) -> None:
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0], [3.0, 0.25]])
    targets = torch.tensor([[0.5], [1.0], [-1.5], [2.0]])
    order = [2, 0, 3, 1]

    def train_step(model: nn.Module, optimizer: AdamW, index: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model(inputs[index]) - targets[index]).mean()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    torch.manual_seed(23)
    full_model, full_optimizer, full_scheduler = _parts()
    full_metrics = [train_step(full_model, full_optimizer, index) for index in order]
    full_scheduler.step()

    torch.manual_seed(23)
    first_model, first_optimizer, first_scheduler = _parts()
    resumed_metrics = [train_step(first_model, first_optimizer, index) for index in order[:2]]
    checkpoint = tmp_path / "resume.pt"
    save_v2_checkpoint(
        checkpoint,
        identity=_identity(),
        model=first_model,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        epoch=0,
        optimizer_steps=2,
        sampler_state={"position": 2, "order": order},
        best_metric=min(resumed_metrics),
        patience_counter=0,
        checkpoint_selection={"criterion": "toy_loss"},
    )

    resumed_model, resumed_optimizer, resumed_scheduler = _parts()
    payload = load_v2_checkpoint(
        checkpoint,
        expected_identity=_identity(),
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    sampler = payload["sampler_state"]
    resumed_metrics.extend(
        train_step(resumed_model, resumed_optimizer, index)
        for index in sampler["order"][sampler["position"] :]
    )
    resumed_scheduler.step()

    assert payload["optimizer_steps"] + len(order[sampler["position"] :]) == 4
    assert resumed_metrics == pytest.approx(full_metrics)
    assert all(
        torch.equal(full_model.state_dict()[name], value)
        for name, value in resumed_model.state_dict().items()
    )
    assert (
        full_optimizer.state_dict()["state"].keys()
        == resumed_optimizer.state_dict()["state"].keys()
    )


def test_interrupted_run_creates_verified_partial_package(tmp_path: Path) -> None:
    config = tmp_path / "resolved_config.yaml"
    metrics = tmp_path / "metrics.jsonl"
    config.write_text("study: chronopde_v2\n")
    metrics.write_text(json.dumps({"optimizer_steps": 2}) + "\n")
    config_hash = sha256_file(config)
    manifest_path = tmp_path / "run.json"
    manifest = create_run_manifest(
        manifest_path,
        phase=2,
        pde="reaction_diffusion",
        task="boundary",
        model="toy",
        regime="full",
        seed=0,
        config_hash=config_hash,
        code_commit=COMMIT,
        checkpoint_selection="minimum_validation_rollout",
        dataset_hash=DATASET_HASH,
        normalization_hash=NORMALIZATION_HASH,
        timestamp="2026-09-20T00:00:00+00:00",
    )
    transition_run_manifest(manifest_path, ExecutionStatus.RUNNING)
    transition_run_manifest(manifest_path, ExecutionStatus.INTERRUPTED)
    model, optimizer, scheduler = _parts()
    checkpoint = tmp_path / "last.pt"
    identity = CheckpointIdentity(
        study_id="chronopde_v2",
        run_id=manifest.run_id,
        model_name="toy",
        config_hash=config_hash,
        dataset_hash=DATASET_HASH,
        normalization_hash=NORMALIZATION_HASH,
    )
    _save(checkpoint, model, optimizer, scheduler, identity)
    package = tmp_path / "recovery.zip"
    report = create_recovery_package(
        package,
        run_manifest=manifest_path,
        resolved_config=config,
        metrics=metrics,
        checkpoint=checkpoint,
    )
    assert report["run_id"] == manifest.run_id
    assert report["partial"] is True
    assert verify_recovery_package(package)["partial"] is True
