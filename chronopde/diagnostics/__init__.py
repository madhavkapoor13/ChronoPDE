"""Scientific diagnostics that do not alter production training runs."""

from chronopde.diagnostics.balanced_batch import BalancedBatchReport, run_balanced_batch_suite
from chronopde.diagnostics.continuous_gate import (
    ContinuousGateSuiteReport,
    run_continuous_gate_suite,
)
from chronopde.diagnostics.loss_alignment import LossAlignmentReport, run_loss_alignment_suite
from chronopde.diagnostics.mechanics import MechanicsSuiteReport, run_single_batch_mechanics_suite
from chronopde.diagnostics.velocity_audit import VelocityAuditReport, run_velocity_checkpoint_audit

__all__ = [
    "BalancedBatchReport",
    "ContinuousGateSuiteReport",
    "LossAlignmentReport",
    "MechanicsSuiteReport",
    "VelocityAuditReport",
    "run_balanced_batch_suite",
    "run_continuous_gate_suite",
    "run_loss_alignment_suite",
    "run_single_batch_mechanics_suite",
    "run_velocity_checkpoint_audit",
]
