import numpy as np
import pytest

from foresight.features.tabular import build_lag_derived_features


def test_build_lag_derived_features_values_and_names() -> None:
    X = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    F, names = build_lag_derived_features(
        X,
        roll_windows=(2, 4),
        roll_stats=("mean", "std", "min", "max", "median", "slope", "iqr", "mad", "skew", "kurt"),
        diff_lags=(1, 3),
    )
    assert F.shape == (1, len(names))
    assert set(names) >= {
        "roll_mean_2",
        "roll_std_2",
        "roll_min_2",
        "roll_max_2",
        "roll_median_2",
        "roll_iqr_2",
        "roll_mad_2",
        "roll_skew_2",
        "roll_kurt_2",
        "roll_slope_2",
        "roll_mean_4",
        "roll_iqr_4",
        "roll_mad_4",
        "roll_skew_4",
        "roll_kurt_4",
        "roll_slope_4",
        "diff_1",
        "diff_3",
    }

    idx = {n: i for i, n in enumerate(names)}
    assert F[0, idx["roll_mean_2"]] == pytest.approx((3.0 + 4.0) / 2.0)
    assert F[0, idx["roll_std_2"]] == pytest.approx(0.5)
    assert F[0, idx["roll_min_2"]] == pytest.approx(3.0)
    assert F[0, idx["roll_max_2"]] == pytest.approx(4.0)
    assert F[0, idx["roll_median_2"]] == pytest.approx(3.5)
    assert F[0, idx["roll_iqr_2"]] == pytest.approx(0.5)
    assert F[0, idx["roll_mad_2"]] == pytest.approx(0.5)
    assert F[0, idx["roll_skew_2"]] == pytest.approx(0.0)
    assert F[0, idx["roll_kurt_2"]] == pytest.approx(-2.0)
    assert F[0, idx["roll_slope_2"]] == pytest.approx(1.0)
    assert F[0, idx["roll_mean_4"]] == pytest.approx(2.5)
    assert F[0, idx["roll_iqr_4"]] == pytest.approx(1.5)
    assert F[0, idx["roll_mad_4"]] == pytest.approx(1.0)
    assert F[0, idx["roll_skew_4"]] == pytest.approx(0.0)
    assert F[0, idx["roll_kurt_4"]] == pytest.approx(-1.36, abs=1e-2)
    assert F[0, idx["roll_slope_4"]] == pytest.approx(1.0)
    assert F[0, idx["diff_1"]] == pytest.approx(1.0)  # 4 - 3
    assert F[0, idx["diff_3"]] == pytest.approx(3.0)  # 4 - 1


def test_build_lag_derived_features_empty_config_returns_empty_matrix() -> None:
    X = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    F, names = build_lag_derived_features(X)
    assert F.shape == (1, 0)
    assert names == []


def test_build_lag_derived_features_rejects_invalid_specs() -> None:
    X = np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    with pytest.raises(ValueError):
        build_lag_derived_features(X, roll_windows=(10,), roll_stats=("mean",))

    with pytest.raises(ValueError):
        build_lag_derived_features(X, roll_windows=(2,), roll_stats=("unknown_stat",))

    with pytest.raises(ValueError):
        build_lag_derived_features(X, diff_lags=(3,))  # must be < lags (=3)


def test_build_lag_derived_features_keeps_constant_moment_stats_finite() -> None:
    X = np.asarray([[2.0, 2.0, 2.0, 2.0]], dtype=float)

    F, names = build_lag_derived_features(
        X,
        roll_windows=(4,),
        roll_stats=("skew", "kurt"),
    )

    assert names == ["roll_skew_4", "roll_kurt_4"]
    assert np.all(np.isfinite(F))
    assert F[0, 0] == pytest.approx(0.0)
    assert F[0, 1] == pytest.approx(0.0)
