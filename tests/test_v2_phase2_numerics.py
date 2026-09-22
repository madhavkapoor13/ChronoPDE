from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chronopde.models import trainable_parameter_count
from chronopde.models.ct_dct import DCTContinuousVectorField
from chronopde.models.ct_fno import FFTContinuousVectorField
from chronopde.v2.comparison import design_comparator
from chronopde.v2.numerical import (
    normalized_physical_rhs,
    run_numerical_audit,
)
from chronopde.v2.protocol import load_phase2_protocol

ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return load_phase2_protocol(ROOT / "configs/chronopde_v2/phase2.yaml")


def test_phase2_protocol_freezes_target_splits_and_isolation() -> None:
    protocol = _protocol()
    assert protocol.parent_phase_commit.startswith("58b0bd8")
    assert protocol.target.kind == "exact_discrete_rhs"
    assert protocol.target.time_units == "physical"
    assert protocol.target.spline_targets is False
    assert protocol.data.training_trajectories == 512
    assert protocol.data.validation_trajectories == 128
    assert protocol.data.confirmatory_trajectories == 256
    assert str(protocol.outputs.artifact_root) == "artifacts/chronopde_v2/runs"
    assert protocol.restrictions.gpu_training is False


def test_numerical_audit_validates_all_independent_oracles() -> None:
    protocol = _protocol()
    audit = run_numerical_audit(protocol.pde, protocol.tolerances)
    assert audit["passed"] is True
    assert audit["structural"]["sparse_vs_dct_laplacian_relative_error"] <= 1e-11
    assert audit["rhs_and_units"]["sparse_vs_stencil_relative_error"] <= 1e-10
    assert audit["rhs_and_units"]["float32_state_rhs_relative_error"] <= 1e-5
    assert "basis-aligned, not boundary-enforcing" in audit["basis_statement"]


def test_physical_rhs_normalization_formula_and_validation() -> None:
    rhs = np.arange(24, dtype=np.float64).reshape(3, 2, 2, 2)
    std = np.asarray([2.0, 4.0])
    normalized = normalized_physical_rhs(rhs, std)
    np.testing.assert_allclose(normalized * std[None, :, None, None], rhs)
    with pytest.raises(ValueError, match="positive"):
        normalized_physical_rhs(rhs, np.asarray([1.0, 0.0]))
    invalid = rhs.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalized_physical_rhs(invalid, std)


def test_expected_comparator_is_exactly_parameter_and_dof_matched() -> None:
    protocol = _protocol()
    report = design_comparator(
        protocol.comparison,
        domain_height=protocol.pde.y_max - protocol.pde.y_min,
        domain_width=protocol.pde.x_max - protocol.pde.x_min,
    )
    assert report["passed"] is True
    assert report["selection"] == "predeclared_expected_pair"
    assert report["fft"]["modes_y"] == 12
    assert report["dct"]["modes_y"] == 24
    assert report["mismatch"]["total_parameter_relative_mismatch"] == 0
    assert report["mismatch"]["spectral_dof_relative_mismatch"] == 0
    assert report["mismatch"]["physical_cutoff_relative_mismatch"] <= 0.10

    fft = FFTContinuousVectorField(width=29, modes_y=12, modes_x=12)
    dct = DCTContinuousVectorField(width=29, modes_y=24, modes_x=24)
    assert trainable_parameter_count(fft) == report["fft"]["total_real_parameters"]
    assert trainable_parameter_count(dct) == report["dct"]["total_real_parameters"]


def test_comparator_reports_failure_when_tolerances_are_impossible() -> None:
    protocol = _protocol()
    impossible = protocol.comparison.model_copy(
        update={
            "total_parameter_tolerance": 1e-15,
            "spectral_dof_tolerance": 1e-15,
            "physical_cutoff_tolerance": 1e-15,
            "search_modes_min": 24,
            "search_modes_max": 24,
            "search_width_min": 29,
            "search_width_max": 29,
        }
    )
    report = design_comparator(impossible, domain_height=2.0, domain_width=2.0)
    assert report["passed"] is False
    assert report["selection"] == "no_feasible_pair"
