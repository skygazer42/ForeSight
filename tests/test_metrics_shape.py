import numpy as np
import pytest

from foresight.metrics import mae


def test_metrics_raise_on_shape_mismatch():
    with pytest.raises(ValueError):
        mae(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
