from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from chronopde.v2.phase4b import phase4b_decision, prepare_phase4_results
from chronopde.v2.protocol import Phase4BProtocol, load_phase4_protocol, load_phase4b_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chronopde_v2/phase4b.yaml"


def test_phase4b_protocol_is_evaluation_only_and_linked() -> None:
    protocol = load_phase4b_protocol(CONFIG)
    phase4 = load_phase4_protocol(ROOT / protocol.phase4_descriptor)
    assert phase4.digest == protocol.phase4_protocol_sha256
    assert protocol.restrictions.training is False
    assert protocol.restrictions.optimizer_updates is False
    assert protocol.restrictions.checkpoint_writes is False
    assert protocol.restrictions.inspect_confirmatory is False
    assert protocol.integrator.steps_per_interval == (1, 2, 4, 8)
    assert len(protocol.validation_trajectory_indices) == 16


def _protocol_for_archive(path: Path) -> Phase4BProtocol:
    protocol = load_phase4b_protocol(CONFIG)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoints = protocol.checkpoints.model_copy(update={"source_archive_sha256": digest})
    return protocol.model_copy(update={"checkpoints": checkpoints})


def test_phase4b_archive_extraction_is_hash_bound_and_safe(tmp_path: Path) -> None:
    archive_path = tmp_path / "phase4.zip"
    member = "artifacts/chronopde_v2/runs/example/summary.json"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "{}")
    protocol = _protocol_for_archive(archive_path)
    runs, status = prepare_phase4_results(archive_path, tmp_path / "extract", protocol)
    assert runs.name == "runs"
    assert status == "archive_sha256_verified"

    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(corrupted, "w") as archive:
        archive.writestr(member, "changed")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_phase4_results(corrupted, tmp_path / "other", protocol)


def test_phase4b_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape", "unsafe")
    protocol = _protocol_for_archive(archive_path)
    with pytest.raises(ValueError, match="unsafe path"):
        prepare_phase4_results(archive_path, tmp_path / "extract", protocol)


def _aggregate(
    model: str, step: int, divergence: float, rollout: float
) -> dict[str, object]:
    return {
        "model": model,
        "steps_per_interval": step,
        "divergence_fraction": divergence,
        "rollout_relative_l2": rollout,
    }


def _convergence(model: str, value: float) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "trajectory_index": index,
            "prediction_relative_l2": value,
        }
        for index in range(16)
    ]


def test_phase4b_routes_persistent_fft_instability_to_phase5() -> None:
    protocol = load_phase4b_protocol(CONFIG)
    aggregates = []
    for step in (1, 2, 4, 8):
        aggregates.append(_aggregate("dct", step, 0.0, 0.9))
        aggregates.append(_aggregate("fft", step, 0.5, 1.1))
    result = phase4b_decision(
        aggregates,
        [*_convergence("dct", 0.01), *_convergence("fft", 0.2)],
        protocol,
    )
    assert result["phase5_multiseed_development_allowed"] is True
    assert (
        result["classifications"]["fft"]["classification"]
        == "learned_vector_field_instability"
    )
    assert result["superiority_claim_authorized"] is False


def test_phase4b_blocks_phase5_when_dct_control_is_not_converged() -> None:
    protocol = load_phase4b_protocol(CONFIG)
    aggregates = []
    for step in (1, 2, 4, 8):
        aggregates.append(_aggregate("dct", step, 0.0, 0.9))
        aggregates.append(_aggregate("fft", step, 0.5, 1.1))
    result = phase4b_decision(
        aggregates,
        [*_convergence("dct", 0.2), *_convergence("fft", 0.2)],
        protocol,
    )
    assert result["phase5_multiseed_development_allowed"] is False
    assert result["decision"] == "phase4b_inconclusive_dct_evaluation_control_failed"


def test_phase4b_notebook_code_cells_compile() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/chronopde_v2_phase4b_kaggle.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "phase4b_notebook", "exec")
