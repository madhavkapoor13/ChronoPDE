import pytest
import torch

from chronopde.diagnostics.velocity_audit import velocity_audit_metrics


def test_velocity_audit_metrics_separates_median_and_pooled() -> None:
    target = torch.ones(3, 2, 4, 4)
    target[0] *= 0.01
    prediction = target.clone()
    prediction[0] += 0.01
    prediction[1] += 0.04
    prediction[2] += 0.06
    metrics, rows = velocity_audit_metrics(prediction, target, torch.tensor([1.0, 2.0]))
    assert metrics["median_nrmse"] == rows[2]["normalized_nrmse"]
    assert metrics["pooled_nrmse"] < metrics["median_nrmse"]
    assert metrics["denominator_clamp_count"] == 0
    assert len(rows) == 3


def test_velocity_audit_metrics_handles_zero_target_and_channels() -> None:
    target = torch.zeros(2, 2, 4, 4)
    target[1] = 1.0
    prediction = target.clone()
    metrics, rows = velocity_audit_metrics(prediction, target, torch.tensor([1.0, 3.0]))
    assert metrics["pooled_nrmse"] == 0.0
    assert metrics["physical_pooled_nrmse"] == 0.0
    assert metrics["denominator_clamp_count"] == 1
    assert rows[0]["normalized_nrmse_u"] == 0.0
    assert rows[0]["normalized_nrmse_v"] == 0.0


def test_velocity_audit_pooled_metric_is_batch_partition_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(4, 2, 4, 4, generator=generator)
    prediction = target + 0.1 * torch.randn(4, 2, 4, 4, generator=generator)
    full, _ = velocity_audit_metrics(prediction, target, torch.ones(2))
    first, _ = velocity_audit_metrics(prediction[:2], target[:2], torch.ones(2))
    second, _ = velocity_audit_metrics(prediction[2:], target[2:], torch.ones(2))
    error_sum = (prediction - target).double().square().sum()
    target_sum = target.double().square().sum()
    expected = float(torch.sqrt(error_sum / target_sum))
    assert full["pooled_nrmse"] == pytest.approx(expected)
    assert first["pooled_nrmse"] != second["pooled_nrmse"]
