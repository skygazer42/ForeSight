import numpy as np
import pytest

from foresight.features.tabular import (
    _append_diff_features,
    _append_roll_stat_features,
    _as_2d_float_array,
    _build_lag_feature_blocks,
    _finalize_feature_matrix,
    _prepare_lag_feature_inputs,
    _resolve_column_names,
    _roll_kurt_feature,
    _roll_skew_feature,
    _roll_slope_feature,
    _validate_roll_window,
    build_column_lag_features,
    build_lag_derived_features,
    normalize_int_tuple,
    normalize_lag_steps,
    normalize_str_tuple,
)


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


def test_tabular_normalizers_cover_empty_and_scalar_paths() -> None:
    assert normalize_int_tuple(None) == ()
    assert normalize_int_tuple(" ") == ()
    assert normalize_int_tuple([1, None, "2"]) == (1, 2)
    assert normalize_int_tuple(3.0) == (3,)

    assert normalize_lag_steps(None) == ()
    assert normalize_lag_steps(" ") == ()
    assert normalize_lag_steps((2, 1), name="demo") == (2, 1)
    assert normalize_lag_steps(3, allow_zero=True) == (2, 1, 0)
    with pytest.raises(ValueError, match="demo must be >= 1"):
        normalize_lag_steps((0,), name="demo")
    with pytest.raises(ValueError, match="demo must be >= 0"):
        normalize_lag_steps((-1, 0), allow_zero=True, name="demo")
    with pytest.raises(ValueError, match="demo must be >= 1"):
        normalize_lag_steps(0, name="demo")

    assert normalize_str_tuple(None) == ()
    assert normalize_str_tuple(" ") == ()
    assert normalize_str_tuple([" a ", "", 2]) == ("a", "2")
    assert normalize_str_tuple(0) == ("0",)


def test_build_column_lag_feature_helpers_cover_validation_paths() -> None:
    arr, tt = _prepare_lag_feature_inputs(np.array([1.0, 2.0, 3.0]), [1, 2])
    assert arr.shape == (3, 1)
    assert tt.tolist() == [1, 2]

    empty_arr, empty_t = _prepare_lag_feature_inputs(np.array([1.0, 2.0]), [])
    assert empty_arr.shape == (2, 1)
    assert empty_t.size == 0

    with pytest.raises(ValueError, match="Expected 1D or 2D array"):
        _prepare_lag_feature_inputs(np.zeros((1, 1, 1)), [0])

    with pytest.raises(ValueError, match="Non-finite values in lag feature source"):
        _prepare_lag_feature_inputs(np.array([[1.0, np.nan]]), [0])

    with pytest.raises(ValueError, match="t indices must be >= 0"):
        _prepare_lag_feature_inputs(np.array([[1.0], [2.0]]), [-1])

    with pytest.raises(ValueError, match="t indices must be < len\\(values\\)"):
        _prepare_lag_feature_inputs(np.array([[1.0], [2.0]]), [2])

    assert _resolve_column_names((), width=2) == ["col0", "col1"]
    with pytest.raises(ValueError, match="column_names must match values.shape\\[1\\]"):
        _resolve_column_names(("only_one",), width=2)

    with pytest.raises(ValueError, match="demo_lags require t >= lag"):
        _build_lag_feature_blocks(
            np.arange(6.0).reshape(-1, 1),
            np.array([1]),
            (2,),
            ["value"],
            prefix="demo",
        )


def test_build_column_lag_features_covers_empty_and_error_paths() -> None:
    feats, names = build_column_lag_features(np.arange(4.0), t=[], lags=(1,))
    assert feats.shape == (0, 0)
    assert names == []

    feats, names = build_column_lag_features(np.arange(4.0), t=[1], lags=())
    assert feats.shape == (1, 0)
    assert names == []

    with pytest.raises(ValueError, match="non-finite features"):
        _finalize_feature_matrix(
            [np.array([[np.nan]])],
            ["broken"],
            rows=1,
            error_message="non-finite features",
        )


def test_build_lag_derived_helpers_cover_guard_paths() -> None:
    with pytest.raises(ValueError, match="Expected 2D array"):
        _as_2d_float_array(np.array([1.0, 2.0]))
    assert _as_2d_float_array(np.empty((0, 0))).shape == (0, 0)
    with pytest.raises(ValueError, match="Non-finite values in lag matrix"):
        _as_2d_float_array(np.array([[1.0, np.nan]]))

    with pytest.raises(ValueError, match="roll_windows must be >= 1"):
        _validate_roll_window(0, lags=3)

    with pytest.raises(RuntimeError, match="slope denom must be > 0"):
        _roll_slope_feature(np.empty((1, 1)))

    assert _roll_skew_feature(np.ones((1, 2)), None).tolist() == [[0.0]]
    assert _roll_kurt_feature(np.ones((1, 2)), None).tolist() == [[0.0]]

    feats: list[np.ndarray] = []
    names: list[str] = []
    _append_roll_stat_features(np.ones((1, 2)), 2, (), feats, names)
    assert feats == []
    assert names == []

    with pytest.raises(ValueError, match="diff_lags must be >= 1"):
        _append_diff_features(np.ones((1, 3)), (0,), [], [], lags=3)

    out, derived_names = build_lag_derived_features(np.empty((0, 0)))
    assert out.shape == (0, 0)
    assert derived_names == []
