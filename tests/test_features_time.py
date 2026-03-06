import numpy as np
import pytest

from foresight.features.time import build_fourier_features


def test_build_fourier_features_basic_values() -> None:
    t = np.asarray([0, 1, 2], dtype=int)
    X, names = build_fourier_features(t, periods=4, orders=1)
    assert X.shape == (3, 2)
    assert names == ["fourier_4_sin_1", "fourier_4_cos_1"]

    # sin(2πt/4), cos(2πt/4)
    assert X[0, 0] == pytest.approx(0.0)
    assert X[0, 1] == pytest.approx(1.0)
    assert X[1, 0] == pytest.approx(1.0)
    assert X[1, 1] == pytest.approx(0.0, abs=1e-8)
    assert X[2, 0] == pytest.approx(0.0, abs=1e-8)
    assert X[2, 1] == pytest.approx(-1.0)


def test_build_fourier_features_empty_periods_returns_empty_matrix() -> None:
    t = np.asarray([0, 1, 2], dtype=int)
    X, names = build_fourier_features(t, periods=())
    assert X.shape == (3, 0)
    assert names == []


def test_build_fourier_features_rejects_invalid_specs() -> None:
    t = np.asarray([0, 1, 2], dtype=int)

    with pytest.raises(ValueError):
        build_fourier_features(t, periods=1, orders=1)

    with pytest.raises(ValueError):
        build_fourier_features(t, periods=4, orders=0)

