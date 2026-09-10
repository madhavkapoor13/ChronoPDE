import numpy as np
import pytest
import torch
from scipy.fft import dctn

from chronopde.numerics.dct import dct2, dct_1d, idct2, idct_1d


@pytest.mark.parametrize("length", [1, 2, 3, 8, 15])
def test_dct_round_trip(length: int) -> None:
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(2, 3, length, generator=generator)
    assert torch.max(torch.abs(idct_1d(dct_1d(values)) - values)).item() < 1e-5


def test_dct2_matches_scipy_and_preserves_shape_dtype() -> None:
    values = torch.randn(2, 3, 7, 8, dtype=torch.float64)
    actual = dct2(values)
    expected = dctn(values.numpy(), type=2, axes=(-2, -1), norm="ortho")
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(idct2(actual), values, rtol=1e-12, atol=1e-12)


def test_dct_is_differentiable() -> None:
    values = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: idct2(dct2(value)), (values,))


def test_dct_rejects_integer_input() -> None:
    with pytest.raises(TypeError):
        dct_1d(torch.ones(3, dtype=torch.int64))
