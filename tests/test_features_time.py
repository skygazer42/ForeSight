import warnings

import numpy as np
import pandas as pd
import pytest

from foresight.features.time import build_fourier_features, build_time_features


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


def test_build_time_features_parses_datetimes_and_returns_finite() -> None:
    ds = pd.date_range("2020-01-01", periods=4, freq="D")

    X, names = build_time_features(ds)

    assert X.shape == (4, 9)
    assert names == [
        "time_idx",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
    ]
    assert np.all(np.isfinite(X))
    assert X[:, 0].tolist() == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert np.allclose(X[:, -2], 0.0)  # hour_sin
    assert np.allclose(X[:, -1], 1.0)  # hour_cos


def test_build_time_features_falls_back_to_zeros_for_non_datetime_like_inputs() -> None:
    ds = ["a", "b", "c"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X, names = build_time_features(ds)

    assert X.shape == (3, 9)
    assert names[0] == "time_idx"
    assert np.all(np.isfinite(X))
    assert X[:, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert np.allclose(X[:, 1:], 0.0)


def test_build_fourier_features_broadcasts_single_string_order_across_periods() -> None:
    t = np.asarray([0, 1], dtype=int)

    X, names = build_fourier_features(t, periods="4,8", orders="2")

    assert X.shape == (2, 8)
    assert names == [
        "fourier_4_sin_1",
        "fourier_4_cos_1",
        "fourier_4_sin_2",
        "fourier_4_cos_2",
        "fourier_8_sin_1",
        "fourier_8_cos_1",
        "fourier_8_sin_2",
        "fourier_8_cos_2",
    ]
    assert np.all(np.isfinite(X))


def test_build_time_features_normalizes_timezone_aware_inputs() -> None:
    ds = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")

    X, names = build_time_features(
        ds,
        add_time_idx=False,
        add_dow=False,
        add_month=False,
        add_doy=False,
        add_hour=True,
    )

    assert names == ["hour_sin", "hour_cos"]
    assert X.shape == (3, 2)
    assert X[:, 0].tolist() == pytest.approx(
        [
            0.0,
            np.sin(2.0 * np.pi / 24.0),
            np.sin(2.0 * 2.0 * np.pi / 24.0),
        ]
    )
