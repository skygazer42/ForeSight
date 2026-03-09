import numpy as np
import pytest

from foresight.transforms import (
    BoxCoxTransformer,
    Differencer,
    MissingValueImputer,
    StandardScaler,
)


def test_missing_value_imputer_ffill_resolves_internal_nans() -> None:
    t = MissingValueImputer(strategy="ffill")
    y = np.array([1.0, np.nan, 3.0], dtype=float)

    out = t.fit(y).transform(y)

    assert np.allclose(out, np.array([1.0, 1.0, 3.0]))


def test_standard_scaler_round_trips_values() -> None:
    t = StandardScaler()
    y = np.array([1.0, 2.0, 4.0], dtype=float)

    yt = t.fit(y).transform(y)
    back = t.inverse_transform(yt)

    assert np.allclose(back, y)
    assert yt.shape == y.shape


def test_differencer_inverse_transform_recovers_future_levels_from_differences() -> None:
    t = Differencer(order=1)
    y = np.array([1.0, 3.0, 6.0], dtype=float)

    yt = t.fit(y).transform(y)
    future = t.inverse_transform(np.array([1.0, 2.0], dtype=float))

    assert np.allclose(yt, np.array([2.0, 3.0]))
    assert np.allclose(future, np.array([7.0, 9.0]))


def test_boxcox_transformer_round_trips_positive_values() -> None:
    t = BoxCoxTransformer(lmbda=0.5)
    y = np.array([1.0, 2.0, 4.0], dtype=float)

    yt = t.fit(y).transform(y)
    back = t.inverse_transform(yt)

    assert np.allclose(back, y)


def test_boxcox_transformer_rejects_non_positive_values() -> None:
    t = BoxCoxTransformer(lmbda=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        t.fit(np.array([0.0, 1.0, 2.0], dtype=float))
