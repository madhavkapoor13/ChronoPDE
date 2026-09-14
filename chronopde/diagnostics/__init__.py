"""Scientific diagnostics that do not alter production training runs."""

from chronopde.diagnostics.continuous_gate import (
    ContinuousGateSuiteReport,
    run_continuous_gate_suite,
)

__all__ = ["ContinuousGateSuiteReport", "run_continuous_gate_suite"]
