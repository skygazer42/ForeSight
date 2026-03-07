import numpy as np
import pytest

from foresight import BaseForecaster, make_forecaster_object
from foresight.models.registry import make_forecaster


def test_local_forecaster_object_requires_fit_before_predict() -> None:
    f = make_forecaster_object("naive-last")
    assert isinstance(f, BaseForecaster)

    with pytest.raises(RuntimeError):
        f.predict(2)


def test_local_forecaster_object_supports_fit_then_predict() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    f = make_forecaster_object("moving-average", window=3)
    assert f.fit(y) is f
    assert f.model_key == "moving-average"
    assert f.model_params["window"] == 3

    yhat = f.predict(2)
    expected = make_forecaster("moving-average", window=3)(y, 2)

    assert yhat.shape == (2,)
    assert np.allclose(yhat, expected)


def test_local_forecaster_object_preserves_registry_defaults() -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)

    f = make_forecaster_object("naive-last")
    f.fit(y)

    yhat = f.predict(3)
    expected = make_forecaster("naive-last")(y, 3)

    assert np.allclose(yhat, expected)
