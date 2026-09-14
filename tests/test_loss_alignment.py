import pytest

from chronopde.diagnostics.loss_alignment import loss_alignment_route


@pytest.mark.parametrize(
    ("fft_passed", "dct_passed", "budget_complete", "expected"),
    [
        (True, True, True, "proceed_aligned_fixed_four_trajectory_comparison"),
        (True, False, True, "test_predeclared_dct_variants_with_aligned_loss"),
        (False, True, True, "stop_and_investigate_control_anomaly"),
        (False, False, True, "stop_and_document_model_or_conditioning_limitation"),
        (True, True, False, "incomplete_budget_no_decision"),
    ],
)
def test_loss_alignment_routing(
    fft_passed: bool, dct_passed: bool, budget_complete: bool, expected: str
) -> None:
    assert (
        loss_alignment_route(
            fft_passed=fft_passed,
            dct_passed=dct_passed,
            budget_complete=budget_complete,
        )
        == expected
    )
