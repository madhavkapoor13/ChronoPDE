import random

import numpy as np
import pytest

from chronopde.reproducibility import seed_everything


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first_python = random.random()
    first_numpy = np.random.random(4)
    seed_everything(123)
    assert random.random() == first_python
    np.testing.assert_array_equal(np.random.random(4), first_numpy)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)

