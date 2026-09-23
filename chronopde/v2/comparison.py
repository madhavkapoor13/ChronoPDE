"""Deterministic matched-comparator design for ChronoPDE V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from chronopde.v2.protocol import ComparatorContract


@dataclass(frozen=True)
class OperatorAccounting:
    basis: Literal["fft", "dct"]
    width: int
    modes_y: int
    modes_x: int
    blocks: int
    film_hidden_width: int
    total_real_parameters: int
    active_real_spectral_dof_per_block: int
    maximum_wavenumber_y: float
    maximum_wavenumber_x: float


def operator_accounting(
    basis: Literal["fft", "dct"],
    *,
    width: int,
    modes_y: int,
    modes_x: int,
    blocks: int,
    film_hidden_width: int,
    domain_height: float,
    domain_width: float,
) -> OperatorAccounting:
    """Count real parameters, spectral degrees of freedom, and physical cutoffs."""

    if basis == "fft":
        spectral_dof = 4 * width * width * modes_y * modes_x
        maximum_y = 2.0 * 3.141592653589793 * modes_y / domain_height
        maximum_x = 2.0 * 3.141592653589793 * (modes_x - 1) / domain_width
    else:
        spectral_dof = width * width * modes_y * modes_x
        maximum_y = 3.141592653589793 * (modes_y - 1) / domain_height
        maximum_x = 3.141592653589793 * (modes_x - 1) / domain_width
    lift_parameters = 3 * width
    conditioner_parameters = (
        5 * film_hidden_width
        + film_hidden_width * 2 * blocks * width
        + 2 * blocks * width
    )
    pointwise_parameters = blocks * (width * width + width)
    projection_parameters = 2 * width * width + 6 * width + 2
    total_parameters = (
        lift_parameters
        + conditioner_parameters
        + pointwise_parameters
        + blocks * spectral_dof
        + projection_parameters
    )
    return OperatorAccounting(
        basis=basis,
        width=width,
        modes_y=modes_y,
        modes_x=modes_x,
        blocks=blocks,
        film_hidden_width=film_hidden_width,
        total_real_parameters=total_parameters,
        active_real_spectral_dof_per_block=spectral_dof,
        maximum_wavenumber_y=maximum_y,
        maximum_wavenumber_x=maximum_x,
    )


def _relative_mismatch(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-30)


def _pair_metrics(
    fft: OperatorAccounting, dct: OperatorAccounting
) -> dict[str, float]:
    return {
        "total_parameter_relative_mismatch": _relative_mismatch(
            fft.total_real_parameters, dct.total_real_parameters
        ),
        "spectral_dof_relative_mismatch": _relative_mismatch(
            fft.active_real_spectral_dof_per_block,
            dct.active_real_spectral_dof_per_block,
        ),
        "physical_cutoff_relative_mismatch": max(
            _relative_mismatch(fft.maximum_wavenumber_y, dct.maximum_wavenumber_y),
            _relative_mismatch(fft.maximum_wavenumber_x, dct.maximum_wavenumber_x),
        ),
    }


def _eligible(metrics: dict[str, float], contract: ComparatorContract) -> bool:
    return bool(
        metrics["total_parameter_relative_mismatch"]
        <= contract.total_parameter_tolerance
        and metrics["spectral_dof_relative_mismatch"] <= contract.spectral_dof_tolerance
        and metrics["physical_cutoff_relative_mismatch"]
        <= contract.physical_cutoff_tolerance
    )


def design_comparator(
    contract: ComparatorContract,
    *,
    domain_height: float,
    domain_width: float,
) -> dict[str, Any]:
    """Select the predeclared pair or a deterministic feasible fallback."""

    fft = operator_accounting(
        "fft",
        width=contract.width,
        modes_y=contract.fft_modes_y,
        modes_x=contract.fft_modes_x,
        blocks=contract.blocks,
        film_hidden_width=contract.film_hidden_width,
        domain_height=domain_height,
        domain_width=domain_width,
    )

    candidates: list[tuple[tuple[float, float, float, int, int], OperatorAccounting]] = []
    expected: OperatorAccounting | None = None
    for width in range(contract.search_width_min, contract.search_width_max + 1):
        for modes in range(contract.search_modes_min, contract.search_modes_max + 1):
            dct = operator_accounting(
                "dct",
                width=width,
                modes_y=modes,
                modes_x=modes,
                blocks=contract.blocks,
                film_hidden_width=contract.film_hidden_width,
                domain_height=domain_height,
                domain_width=domain_width,
            )
            if (
                width == contract.width
                and modes == contract.expected_dct_modes_y
                and modes == contract.expected_dct_modes_x
            ):
                expected = dct
            metrics = _pair_metrics(fft, dct)
            if _eligible(metrics, contract):
                rank = (
                    metrics["physical_cutoff_relative_mismatch"],
                    metrics["spectral_dof_relative_mismatch"],
                    metrics["total_parameter_relative_mismatch"],
                    width,
                    modes,
                )
                candidates.append((rank, dct))

    if expected is None:
        raise ValueError("expected DCT comparator lies outside the configured search")
    expected_metrics = _pair_metrics(fft, expected)
    if _eligible(expected_metrics, contract):
        selected = expected
        selection = "predeclared_expected_pair"
    elif candidates:
        selected = min(candidates, key=lambda item: item[0])[1]
        selection = "deterministic_fallback_search"
    else:
        return {
            "passed": False,
            "selection": "no_feasible_pair",
            "fft": asdict(fft),
            "expected_dct": asdict(expected),
            "mismatch": expected_metrics,
        }

    metrics = _pair_metrics(fft, selected)
    return {
        "passed": True,
        "selection": selection,
        "fft": asdict(fft),
        "dct": asdict(selected),
        "mismatch": metrics,
        "tolerances": {
            "total_parameters": contract.total_parameter_tolerance,
            "spectral_dof": contract.spectral_dof_tolerance,
            "physical_cutoff": contract.physical_cutoff_tolerance,
        },
        "shared_architecture": {
            "blocks": contract.blocks,
            "conditioner": "FiLMConditioner",
            "film_hidden_width": contract.film_hidden_width,
            "lift": "1x1 convolution",
            "projection": "1x1 convolution, GELU, 1x1 convolution",
            "pointwise_path": "1x1 convolution in every block",
            "residual_skip": False,
            "activation": "GELU except final spectral block",
            "spectral_weight_initialization": "normal with scale 1 / width",
        },
    }
