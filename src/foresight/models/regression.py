from __future__ import annotations

from typing import Any

import numpy as np

from ..features.lag import build_seasonal_lag_features, make_lagged_xy
from ..features.tabular import build_lag_derived_features, normalize_int_tuple, normalize_lag_steps
from ..features.time import build_fourier_features

TARGET_LAGS_MIN_ERROR = "target_lags must be >= 1"
HORIZON_MIN_ERROR = "horizon must be >= 1"
LAGS_MIN_ERROR = "lags must be >= 1"
N_ESTIMATORS_MIN_ERROR = "n_estimators must be >= 1"
MAX_DEPTH_MIN_ERROR = "max_depth must be >= 1"
MAX_DEPTH_MIN_OR_NONE_ERROR = "max_depth must be >= 1 or None"
MAX_ITER_MIN_ERROR = "max_iter must be >= 1"
ALPHA_NON_NEGATIVE_ERROR = "alpha must be >= 0"
STEP_SCALE_OPTIONS_ERROR = "step_scale must be one of: one_based, zero_based, unit"
LEARNING_RATE_POSITIVE_ERROR = "learning_rate must be > 0"
SUBSAMPLE_RANGE_ERROR = "subsample must be in (0,1]"
COLSAMPLE_BYTREE_RANGE_ERROR = "colsample_bytree must be in (0,1]"
N_JOBS_NON_ZERO_ERROR = "n_jobs must be non-zero"
REG_LAMBDA_NON_NEGATIVE_ERROR = "reg_lambda must be >= 0"
MIN_CHILD_WEIGHT_NON_NEGATIVE_ERROR = "min_child_weight must be >= 0"
GAMMA_NON_NEGATIVE_ERROR = "gamma must be >= 0"
SVR_C_ERROR = "C must be > 0"
SVR_EPSILON_ERROR = "epsilon must be >= 0"
XGB_INSTALL_ERROR = 'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
XGB_OBJECTIVE_EMPTY_ERROR = "objective must be non-empty"
XGB_OBJECTIVE_SQUAREDERROR = "reg:squarederror"
XGB_OBJECTIVE_SQUAREDLOGERROR = "reg:squaredlogerror"
XGB_OBJECTIVE_LOGISTIC = "reg:logistic"
XGB_OBJECTIVE_POISSON = "count:poisson"
XGB_OBJECTIVE_GAMMA = "reg:gamma"
XGB_OBJECTIVE_TWEEDIE = "reg:tweedie"


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _wants_lag_derived_features(*, roll_windows: Any, roll_stats: Any, diff_lags: Any) -> bool:
    return bool(roll_windows) or bool(roll_stats) or bool(diff_lags)


def _wants_seasonal_or_fourier_features(
    *,
    seasonal_lags: Any,
    seasonal_diff_lags: Any,
    fourier_periods: Any,
) -> bool:
    return bool(seasonal_lags) or bool(seasonal_diff_lags) or bool(fourier_periods)


def _compute_feature_start_t(*, lags: Any, seasonal_lags: Any, seasonal_diff_lags: Any) -> int:
    """
    Compute the earliest target index t such that all requested lag-based features are available.

    For contiguous lag windows we need t >= lags.
    For seasonal_lags (y[t-p]) we need t >= max(p).
    For seasonal_diff_lags (y[t-1]-y[t-1-p]) we need t >= 1 + max(p).
    """
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if not lag_steps:
        raise ValueError(TARGET_LAGS_MIN_ERROR)
    start_t = int(max(lag_steps))

    seas = normalize_int_tuple(seasonal_lags)
    diffs = normalize_int_tuple(seasonal_diff_lags)
    if any(int(p) <= 0 for p in seas):
        raise ValueError("seasonal_lags must be >= 1")
    if any(int(p) <= 0 for p in diffs):
        raise ValueError("seasonal_diff_lags must be >= 1")

    if seas:
        start_t = max(start_t, int(max(seas)))
    if diffs:
        start_t = max(start_t, 1 + int(max(diffs)))
    return start_t


def _make_target_feat_row(history: Any, *, lags: Any) -> np.ndarray:
    x = _as_1d_float_array(history)
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if not lag_steps:
        raise ValueError(TARGET_LAGS_MIN_ERROR)

    t_next = int(x.size)
    max_lag = int(max(lag_steps))
    if t_next <= max_lag - 1:
        raise ValueError(f"Need > max target lag points (max_lag={max_lag}), got {t_next}")
    return np.asarray([[x[t_next - lag] for lag in lag_steps]], dtype=float)


def _augment_lag_matrix(
    lag_matrix: np.ndarray,
    *,
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    t_index: np.ndarray | None = None,
    series: np.ndarray | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = [lag_matrix]

    if _wants_lag_derived_features(
        roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    ):
        derived, _ = build_lag_derived_features(
            lag_matrix,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )
        if derived.shape[1] > 0:
            parts.append(derived)

    if _wants_seasonal_or_fourier_features(
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
    ):
        if t_index is None:
            raise ValueError("t_index is required for seasonal/fourier features")
        if int(t_index.shape[0]) != int(lag_matrix.shape[0]):
            raise ValueError("t_index must have the same number of rows as X_base")

    if bool(seasonal_lags) or bool(seasonal_diff_lags):
        if series is None:
            raise ValueError("series is required for seasonal lag features")
        seasonal, _ = build_seasonal_lag_features(
            series,
            t=t_index,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
        )
        if seasonal.shape[1] > 0:
            parts.append(seasonal)

    if bool(fourier_periods):
        fourier, _ = build_fourier_features(t_index, periods=fourier_periods, orders=fourier_orders)
        if fourier.shape[1] > 0:
            parts.append(fourier)

    if len(parts) == 1:
        return lag_matrix
    return np.concatenate(parts, axis=1)


def _augment_lag_feat_row(
    lag_row: np.ndarray,
    *,
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    t_next: int | None = None,
    history: np.ndarray | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = [lag_row]

    if _wants_lag_derived_features(
        roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    ):
        derived, _ = build_lag_derived_features(
            lag_row,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )
        if derived.shape[1] > 0:
            parts.append(derived)

    if _wants_seasonal_or_fourier_features(
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
    ) and (t_next is None or history is None):
        raise ValueError("t_next and history are required for seasonal/fourier features")

    if bool(seasonal_lags) or bool(seasonal_diff_lags):
        seasonal, _ = build_seasonal_lag_features(
            history,
            t=[int(t_next)],
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
        )
        if seasonal.shape[1] > 0:
            parts.append(seasonal)

    if bool(fourier_periods):
        fourier, _ = build_fourier_features(
            [int(t_next)], periods=fourier_periods, orders=fourier_orders
        )
        if fourier.shape[1] > 0:
            parts.append(fourier)

    if len(parts) == 1:
        return lag_row
    return np.concatenate(parts, axis=1)


def lr_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: Any,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Fit an OLS linear regression on lag features and forecast recursively.

    This is intentionally lightweight (pure numpy) and suitable as a baseline.
    """
    x = _as_1d_float_array(train)
    target_lags = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    max_target_lag = int(max(target_lags))
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size <= max_target_lag:
        raise ValueError(
            f"lr_lag_forecast requires > lags points (lags={max_target_lag}), got {x.size}"
        )

    start_t = _compute_feature_start_t(
        lags=target_lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=target_lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    x_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)

    intercept = float(coef[0])
    w = coef[1:].astype(float, copy=False)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(horizon):
        feat_base = _make_target_feat_row(np.asarray(history, dtype=float), lags=target_lags)
        feat_aug = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        ).reshape(-1)
        yhat = intercept + float(np.dot(w, feat_aug))
        out.append(float(yhat))
        history.append(float(yhat))
    return np.asarray(out, dtype=float)


def _make_lagged_xy_multi(
    x: np.ndarray, *, lags: Any, horizon: int, start_t: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.size)
    h = int(horizon)
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if not lag_steps:
        raise ValueError(TARGET_LAGS_MIN_ERROR)

    t0 = int(max(lag_steps)) if start_t is None else max(int(max(lag_steps)), int(start_t))
    if n < t0 + h:
        raise ValueError(f"Need >= start_t+horizon points (start_t={t0}, horizon={h}), got {n}")

    rows = n - t0 - h + 1
    X = np.empty((rows, len(lag_steps)), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    t_idx = np.arange(t0, t0 + rows, dtype=int)
    for i, t in enumerate(t_idx):
        X[i, :] = np.asarray([x[t - lag] for lag in lag_steps], dtype=float)
        Y[i, :] = x[t : t + h]
    return X, Y, t_idx


def lr_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: Any,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon OLS regression on lag features.

    Fits one linear model per horizon step (shared feature matrix), then predicts
    all steps directly from the last `lags` observed values.
    """
    x = _as_1d_float_array(train)
    target_lags = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=target_lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(
        x, lags=target_lags, horizon=int(horizon), start_t=start_t
    )
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    x_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, Y, rcond=None)  # (1+lags, horizon)

    intercept = coef[0, :].astype(float, copy=False)
    w = coef[1:, :].astype(float, copy=False)  # (n_features, horizon)

    feat_base = _make_target_feat_row(x, lags=target_lags)
    feat_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    ).reshape(-1)
    yhat = intercept + feat_aug @ w
    return np.asarray(yhat, dtype=float)


def ridge_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: Any,
    alpha: float = 1.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Ridge regression on lag features (requires scikit-learn), forecast recursively.
    """
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'ridge_lag_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    target_lags = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    max_target_lag = int(max(target_lags))
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if x.size <= max_target_lag:
        raise ValueError(
            f"ridge_lag_forecast requires > lags points (lags={max_target_lag}), got {x.size}"
        )

    start_t = _compute_feature_start_t(
        lags=target_lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=target_lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(horizon):
        feat_base = _make_target_feat_row(np.asarray(history, dtype=float), lags=target_lags)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(model.predict(feat)[0])
        out.append(yhat)
        history.append(yhat)
    return np.asarray(out, dtype=float)


def rf_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 200,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon RandomForest on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'rf_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=1,
        max_features=1.0,
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def lasso_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 0.001,
    max_iter: int = 5000,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon Lasso on lag features (requires scikit-learn).
    """
    try:
        from sklearn.linear_model import Lasso  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'lasso_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=int(max_iter))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def elasticnet_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon ElasticNet on lag features (requires scikit-learn).
    """
    try:
        from sklearn.linear_model import ElasticNet  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'elasticnet_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=int(max_iter),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def knn_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_neighbors: int = 10,
    weights: str = "distance",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon KNN regression on lag features (requires scikit-learn).
    """
    try:
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
        from sklearn.neighbors import KNeighborsRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'knn_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be >= 1")

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = KNeighborsRegressor(n_neighbors=int(n_neighbors), weights=str(weights))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def gbrt_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon GradientBoosting on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'gbrt_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def ridge_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: Any,
    alpha: float = 1.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon Ridge regression on lag features (requires scikit-learn).

    Unlike `ridge_lag_forecast` (recursive), this fits a multi-target model and predicts
    all horizon steps directly from the last `lags` observed values.
    """
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'ridge_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    target_lags = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=target_lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(
        x, lags=target_lags, horizon=int(horizon), start_t=start_t
    )
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, Y)

    feat_base = _make_target_feat_row(x, lags=target_lags)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def decision_tree_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    max_depth: int | None = 5,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon DecisionTreeRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.tree import DecisionTreeRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "decision_tree_lag_direct_forecast requires scikit-learn. Install with: "
            'pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError(MAX_DEPTH_MIN_OR_NONE_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = DecisionTreeRegressor(
        max_depth=None if max_depth is None else int(max_depth),
        ccp_alpha=0.0,
        random_state=int(random_state),
    )
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def extra_trees_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon ExtraTreesRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'extra_trees_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError(MAX_DEPTH_MIN_OR_NONE_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
    )
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def adaboost_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon AdaBoostRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import AdaBoostRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'adaboost_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = AdaBoostRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def bagging_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 200,
    max_samples: float = 0.8,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon BaggingRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import BaggingRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'bagging_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if not (0.0 < float(max_samples) <= 1.0):
        raise ValueError("max_samples must be in (0,1]")

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = BaggingRegressor(
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def hgb_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int | None = 3,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon HistGradientBoostingRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'hgb_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError(MAX_DEPTH_MIN_OR_NONE_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def svr_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    C: float = 1.0,
    gamma: str | float = "scale",
    epsilon: float = 0.1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon SVR (RBF) on lag features (requires scikit-learn).
    """
    try:
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
        from sklearn.svm import SVR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'svr_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if float(C) <= 0:
        raise ValueError(SVR_C_ERROR)
    if float(epsilon) < 0:
        raise ValueError(SVR_EPSILON_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = SVR(C=float(C), gamma=gamma, epsilon=float(epsilon), kernel="rbf")
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def linear_svr_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    C: float = 1.0,
    epsilon: float = 0.0,
    max_iter: int = 5000,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon LinearSVR on lag features (requires scikit-learn).
    """
    try:
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
        from sklearn.svm import LinearSVR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'linear_svr_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if float(C) <= 0:
        raise ValueError(SVR_C_ERROR)
    if float(epsilon) < 0:
        raise ValueError(SVR_EPSILON_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = LinearSVR(
        C=float(C),
        epsilon=float(epsilon),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def kernel_ridge_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 1.0,
    kernel: str = "rbf",
    gamma: float | None = None,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon KernelRidge on lag features (requires scikit-learn).
    """
    try:
        from sklearn.kernel_ridge import KernelRidge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'kernel_ridge_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if float(alpha) < 0:
        raise ValueError(ALPHA_NON_NEGATIVE_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = KernelRidge(alpha=float(alpha), kernel=str(kernel), gamma=gamma)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def mlp_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    hidden_layer_sizes: tuple[int, ...] = (64, 64),
    alpha: float = 0.0001,
    max_iter: int = 300,
    random_state: int = 0,
    learning_rate_init: float = 0.001,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon MLPRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.neural_network import MLPRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mlp_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)
    if float(alpha) < 0:
        raise ValueError(ALPHA_NON_NEGATIVE_ERROR)
    if float(learning_rate_init) <= 0:
        raise ValueError("learning_rate_init must be > 0")

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = MLPRegressor(
        hidden_layer_sizes=tuple(int(s) for s in hidden_layer_sizes),
        alpha=float(alpha),
        max_iter=int(max_iter),
        random_state=int(random_state),
        learning_rate_init=float(learning_rate_init),
    )
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def huber_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    epsilon: float = 1.35,
    alpha: float = 0.0001,
    max_iter: int = 200,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon HuberRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.linear_model import HuberRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'huber_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if float(epsilon) <= 1.0:
        raise ValueError("epsilon must be > 1.0")
    if float(alpha) < 0:
        raise ValueError(ALPHA_NON_NEGATIVE_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = HuberRegressor(epsilon=float(epsilon), alpha=float(alpha), max_iter=int(max_iter))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def quantile_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    quantile: float = 0.5,
    alpha: float = 0.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon QuantileRegressor on lag features (requires scikit-learn).

    This returns the specified conditional quantile as a point forecast.
    """
    try:
        from sklearn.linear_model import QuantileRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'quantile_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be in (0,1)")
    if float(alpha) < 0:
        raise ValueError(ALPHA_NON_NEGATIVE_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = QuantileRegressor(quantile=q, alpha=float(alpha), fit_intercept=True)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def sgd_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 0.0001,
    penalty: str = "l2",
    max_iter: int = 2000,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon SGDRegressor on lag features (requires scikit-learn).
    """
    try:
        from sklearn.linear_model import SGDRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'sgd_lag_direct_forecast requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if float(alpha) < 0:
        raise ValueError(ALPHA_NON_NEGATIVE_ERROR)
    if max_iter <= 0:
        raise ValueError(MAX_ITER_MIN_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    base = SGDRegressor(
        alpha=float(alpha),
        penalty=str(penalty),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def _xgb_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    booster: str,
    objective: str,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    objective_params: dict[str, Any] | None = None,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if max_depth <= 0:
        raise ValueError(MAX_DEPTH_MIN_ERROR)
    if float(learning_rate) <= 0:
        raise ValueError(LEARNING_RATE_POSITIVE_ERROR)
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    if float(reg_alpha) < 0:
        raise ValueError("reg_alpha must be >= 0")
    if float(reg_lambda) < 0:
        raise ValueError(REG_LAMBDA_NON_NEGATIVE_ERROR)
    if float(min_child_weight) < 0:
        raise ValueError(MIN_CHILD_WEIGHT_NON_NEGATIVE_ERROR)
    if float(gamma) < 0:
        raise ValueError(GAMMA_NON_NEGATIVE_ERROR)
    if n_jobs == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)

    obj = str(objective).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon), start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat_base_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    fourier_pred_all: np.ndarray | None = None
    if bool(fourier_periods):
        fourier_pred_all, _ = build_fourier_features(
            np.arange(int(x.size), int(x.size) + int(horizon), dtype=int),
            periods=fourier_periods,
            orders=fourier_orders,
        )

    extra = dict(objective_params or {})

    out = np.empty((int(horizon),), dtype=float)
    for j in range(int(horizon)):
        if fourier_pred_all is None:
            x_j = x_base_aug
            feat = feat_base_aug
        else:
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_j = np.concatenate([x_base_aug, fourier_train], axis=1)
            feat = np.concatenate(
                [feat_base_aug, fourier_pred_all[int(j) : int(j) + 1, :]],
                axis=1,
            )

        model = xgb.XGBRegressor(
            booster=str(booster),
            objective=obj,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            tree_method=str(tree_method),
            verbosity=0,
            **extra,
        )
        model.fit(x_j, Y[:, j])
        out[j] = float(model.predict(feat)[0])

    return np.asarray(out, dtype=float)


def _xgb_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    booster: str,
    objective: str,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    objective_params: dict[str, Any] | None = None,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if max_depth <= 0:
        raise ValueError(MAX_DEPTH_MIN_ERROR)
    if float(learning_rate) <= 0:
        raise ValueError(LEARNING_RATE_POSITIVE_ERROR)
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    if float(reg_alpha) < 0:
        raise ValueError("reg_alpha must be >= 0")
    if float(reg_lambda) < 0:
        raise ValueError(REG_LAMBDA_NON_NEGATIVE_ERROR)
    if float(min_child_weight) < 0:
        raise ValueError(MIN_CHILD_WEIGHT_NON_NEGATIVE_ERROR)
    if float(gamma) < 0:
        raise ValueError(GAMMA_NON_NEGATIVE_ERROR)
    if n_jobs == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)

    obj = str(objective).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    extra = dict(objective_params or {})
    model = xgb.XGBRegressor(
        booster=str(booster),
        objective=obj,
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        tree_method=str(tree_method),
        verbosity=0,
        **extra,
    )
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(int(horizon)):
        feat_base = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(model.predict(feat)[0])
        out.append(float(yhat))
        history.append(float(yhat))

    return np.asarray(out, dtype=float)


def xgb_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost (XGBRegressor) on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost (XGBRegressor) on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_dart_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost DART booster on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="dart",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_dart_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost DART booster on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="dart",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgbrf_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Random Forest (XGBRFRegressor) on lag features (requires xgboost).
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if max_depth <= 0:
        raise ValueError(MAX_DEPTH_MIN_ERROR)
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    if float(reg_lambda) < 0:
        raise ValueError(REG_LAMBDA_NON_NEGATIVE_ERROR)
    if float(min_child_weight) < 0:
        raise ValueError(MIN_CHILD_WEIGHT_NON_NEGATIVE_ERROR)
    if float(gamma) < 0:
        raise ValueError(GAMMA_NON_NEGATIVE_ERROR)
    if n_jobs == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)

    h = int(horizon)
    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat_base_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    fourier_pred_all: np.ndarray | None = None
    if bool(fourier_periods):
        fourier_pred_all, _ = build_fourier_features(
            np.arange(int(x.size), int(x.size) + h, dtype=int),
            periods=fourier_periods,
            orders=fourier_orders,
        )

    out = np.empty((h,), dtype=float)
    for j in range(h):
        if fourier_pred_all is None:
            x_j = x_base_aug
            feat = feat_base_aug
        else:
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_j = np.concatenate([x_base_aug, fourier_train], axis=1)
            feat = np.concatenate([feat_base_aug, fourier_pred_all[int(j) : int(j) + 1, :]], axis=1)

        model = xgb.XGBRFRegressor(
            objective=XGB_OBJECTIVE_SQUAREDERROR,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            tree_method=str(tree_method),
            verbosity=0,
        )
        model.fit(x_j, Y[:, j])
        out[j] = float(model.predict(feat)[0])

    return np.asarray(out, dtype=float)


def xgbrf_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost Random Forest (XGBRFRegressor) on lag features (requires xgboost).
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )
    if n_estimators <= 0:
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if max_depth <= 0:
        raise ValueError(MAX_DEPTH_MIN_ERROR)
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    if float(reg_lambda) < 0:
        raise ValueError(REG_LAMBDA_NON_NEGATIVE_ERROR)
    if float(min_child_weight) < 0:
        raise ValueError(MIN_CHILD_WEIGHT_NON_NEGATIVE_ERROR)
    if float(gamma) < 0:
        raise ValueError(GAMMA_NON_NEGATIVE_ERROR)
    if n_jobs == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = xgb.XGBRFRegressor(
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        tree_method=str(tree_method),
        verbosity=0,
    )
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(int(horizon)):
        feat_base = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(model.predict(feat)[0])
        out.append(float(yhat))
        history.append(float(yhat))

    return np.asarray(out, dtype=float)


def xgb_linear_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost linear booster (gblinear) on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gblinear",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        # gblinear ignores tree-specific params like max_depth, but they are accepted by the API.
        max_depth=1,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=1.0,
        gamma=0.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_linear_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost linear booster (gblinear) on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gblinear",
        objective=XGB_OBJECTIVE_SQUAREDERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        # gblinear ignores tree-specific params like max_depth, but they are accepted by the API.
        max_depth=1,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=1.0,
        gamma=0.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_msle_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost with squared log error objective on lag features (requires xgboost, y>=0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_SQUAREDLOGERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_msle_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost with squared log error objective on lag features (requires xgboost, y>=0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_SQUAREDLOGERROR,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_logistic_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost with logistic regression objective on lag features (requires xgboost, y in [0,1]).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_LOGISTIC,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_logistic_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost with logistic regression objective on lag features (requires xgboost, y in [0,1]).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_LOGISTIC,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_mae_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost with MAE objective on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:absoluteerror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_mae_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost with MAE objective on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:absoluteerror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_huber_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    huber_slope: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost pseudo-Huber objective on lag features (requires xgboost).
    """
    if float(huber_slope) <= 0:
        raise ValueError("huber_slope must be > 0")
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:pseudohubererror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"huber_slope": float(huber_slope)},
    )


def xgb_huber_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    huber_slope: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost pseudo-Huber objective on lag features (requires xgboost).
    """
    if float(huber_slope) <= 0:
        raise ValueError("huber_slope must be > 0")
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:pseudohubererror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"huber_slope": float(huber_slope)},
    )


def xgb_quantile_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    quantile_alpha: float = 0.5,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost quantile objective on lag features (requires xgboost).
    """
    qa = float(quantile_alpha)
    if not (0.0 < qa < 1.0):
        raise ValueError("quantile_alpha must be in (0,1)")
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:quantileerror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"quantile_alpha": qa},
    )


def xgb_quantile_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    quantile_alpha: float = 0.5,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost quantile objective on lag features (requires xgboost).
    """
    qa = float(quantile_alpha)
    if not (0.0 < qa < 1.0):
        raise ValueError("quantile_alpha must be in (0,1)")
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:quantileerror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"quantile_alpha": qa},
    )


def xgb_poisson_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Poisson objective on lag features (requires xgboost; y>=0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_POISSON,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_poisson_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost Poisson objective on lag features (requires xgboost; y>=0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_POISSON,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_gamma_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Gamma objective on lag features (requires xgboost; y>0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_GAMMA,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_gamma_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost Gamma objective on lag features (requires xgboost; y>0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_GAMMA,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_tweedie_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    tweedie_variance_power: float = 1.5,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Tweedie objective on lag features (requires xgboost; y>=0).
    """
    tvp = float(tweedie_variance_power)
    if not (1.0 <= tvp < 2.0):
        raise ValueError("tweedie_variance_power must be in [1,2)")
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_TWEEDIE,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"tweedie_variance_power": tvp},
    )


def xgb_tweedie_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    tweedie_variance_power: float = 1.5,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Recursive one-step XGBoost Tweedie objective on lag features (requires xgboost; y>=0).
    """
    tvp = float(tweedie_variance_power)
    if not (1.0 <= tvp < 2.0):
        raise ValueError("tweedie_variance_power must be in [1,2)")
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective=XGB_OBJECTIVE_TWEEDIE,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        objective_params={"tweedie_variance_power": tvp},
    )


def _xgb_validate_objective_label_constraints(obj: str, x: np.ndarray) -> None:
    """
    Defensive ergonomics: a few XGBoost objectives have label-domain assumptions.

    We validate these early to surface a clearer error than the underlying trainer.
    """
    if obj in {
        XGB_OBJECTIVE_POISSON,
        XGB_OBJECTIVE_TWEEDIE,
        XGB_OBJECTIVE_SQUAREDLOGERROR,
    } and np.any(x < 0.0):
        raise ValueError(f"{obj} requires non-negative series values")
    if obj == XGB_OBJECTIVE_GAMMA and np.any(x <= 0.0):
        raise ValueError(f"{XGB_OBJECTIVE_GAMMA} requires strictly positive series values")
    if obj == XGB_OBJECTIVE_LOGISTIC and np.any((x < 0.0) | (x > 1.0)):
        raise ValueError(f"{XGB_OBJECTIVE_LOGISTIC} requires series values in [0,1]")


def _xgb_validate_common_regressor_params(params: dict[str, Any]) -> None:
    if (
        "n_estimators" in params
        and params["n_estimators"] is not None
        and int(params["n_estimators"]) <= 0
    ):
        raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if "max_depth" in params and params["max_depth"] is not None and int(params["max_depth"]) <= 0:
        raise ValueError(MAX_DEPTH_MIN_ERROR)
    if (
        "learning_rate" in params
        and params["learning_rate"] is not None
        and float(params["learning_rate"]) <= 0
    ):
        raise ValueError(LEARNING_RATE_POSITIVE_ERROR)
    if (
        "subsample" in params
        and params["subsample"] is not None
        and not (0.0 < float(params["subsample"]) <= 1.0)
    ):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if (
        "colsample_bytree" in params
        and params["colsample_bytree"] is not None
        and not (0.0 < float(params["colsample_bytree"]) <= 1.0)
    ):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    for key in ("reg_alpha", "reg_lambda", "min_child_weight", "gamma"):
        if key in params and params[key] is not None and float(params[key]) < 0:
            raise ValueError(f"{key} must be >= 0")
    if "n_jobs" in params and params["n_jobs"] is not None and int(params["n_jobs"]) == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)


def _xgb_lag_direct_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    h = int(horizon)
    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat_base_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    fourier_pred_all: np.ndarray | None = None
    if bool(fourier_periods):
        fourier_pred_all, _ = build_fourier_features(
            np.arange(int(x.size), int(x.size) + h, dtype=int),
            periods=fourier_periods,
            orders=fourier_orders,
        )

    out = np.empty((h,), dtype=float)
    for j in range(h):
        if fourier_pred_all is None:
            x_j = x_base_aug
            feat = feat_base_aug
        else:
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_j = np.concatenate([x_base_aug, fourier_train], axis=1)
            feat = np.concatenate([feat_base_aug, fourier_pred_all[int(j) : int(j) + 1, :]], axis=1)

        model = xgb.XGBRegressor(**params)
        model.fit(x_j, Y[:, j])
        out[j] = float(model.predict(feat)[0])
    return np.asarray(out, dtype=float)


def _xgb_lag_recursive_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(int(horizon)):
        feat_base = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(model.predict(feat)[0])
        out.append(float(yhat))
        history.append(float(yhat))
    return np.asarray(out, dtype=float)


def _xgb_lag_step_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Single-model multi-horizon forecasting by adding a "step index" feature.

    This trains one regressor on the expanded dataset:
      (lag_window, step) -> y_{t+step}
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    rows = int(x_base.shape[0])
    derived, _ = build_lag_derived_features(
        x_base, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal, _ = build_seasonal_lag_features(
        x, t=t_idx, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )

    if step_scale not in {"one_based", "zero_based", "unit"}:
        raise ValueError(STEP_SCALE_OPTIONS_ERROR)

    if step_scale == "zero_based":
        step = np.arange(h, dtype=float)
    elif step_scale == "unit":
        step = np.arange(1, h + 1, dtype=float) / float(h)
    else:
        step = np.arange(1, h + 1, dtype=float)

    step_idx = np.tile(step, rows).reshape(-1, 1)  # (rows*h, 1)
    x_rep = np.repeat(x_base, repeats=h, axis=0)  # (rows*h, lags)
    derived_rep = np.repeat(derived, repeats=h, axis=0)  # (rows*h, k)
    seasonal_rep = np.repeat(seasonal, repeats=h, axis=0)

    if bool(fourier_periods):
        t_flat = np.repeat(t_idx, repeats=h) + np.tile(np.arange(h, dtype=int), rows)
        fourier_long, _ = build_fourier_features(
            t_flat, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_long = np.empty((rows * h, 0), dtype=float)

    x_long = np.concatenate([x_rep, derived_rep, seasonal_rep, fourier_long, step_idx], axis=1)
    y_long = Y.reshape(-1).astype(float, copy=False)

    model = xgb.XGBRegressor(**params)
    model.fit(x_long, y_long)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    derived_pred, _ = build_lag_derived_features(
        base_feat, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal_pred, _ = build_seasonal_lag_features(
        x, t=[int(x.size)], seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    base_rep = np.repeat(base_feat, repeats=h, axis=0)  # (h, lags)
    derived_rep_pred = np.repeat(derived_pred, repeats=h, axis=0)
    seasonal_rep_pred = np.repeat(seasonal_pred, repeats=h, axis=0)
    if bool(fourier_periods):
        t_pred = np.arange(int(x.size), int(x.size) + h, dtype=int)
        fourier_pred, _ = build_fourier_features(
            t_pred, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_pred = np.empty((h, 0), dtype=float)

    feat = np.concatenate(
        [base_rep, derived_rep_pred, seasonal_rep_pred, fourier_pred, step.reshape(-1, 1)],
        axis=1,
    )
    out = model.predict(feat)
    return np.asarray(out, dtype=float)


def _xgb_lag_dirrec_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    DirRec (Direct-Recursive) strategy with per-step models.

    Each step model uses lag features plus the previous steps as additional regressors.
    During training we use the true previous-step values; during prediction we use the
    model predictions from earlier steps.
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )

    models: list[Any] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_step = np.concatenate([x_base_aug, fourier_train], axis=1)
        else:
            x_step = x_base_aug

        if j == 0:
            x_j = x_step
        else:
            x_j = np.concatenate([x_step, Y[:, :j]], axis=1)
        yj = Y[:, j]
        model = xgb.XGBRegressor(**params)
        model.fit(x_j, yj)
        models.append(model)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    base_feat_aug = _augment_lag_feat_row(
        base_feat,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    ).reshape(-1)
    out: list[float] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_pred, _ = build_fourier_features(
                [int(x.size) + int(j)],
                periods=fourier_periods,
                orders=fourier_orders,
            )
            feat_step = np.concatenate([base_feat_aug, fourier_pred.reshape(-1)], axis=0)
        else:
            feat_step = base_feat_aug

        if j == 0:
            feat = feat_step
        else:
            feat = np.concatenate([feat_step, np.asarray(out, dtype=float)], axis=0)
        yhat = float(models[j].predict(feat.reshape(1, -1))[0])
        out.append(yhat)
    return np.asarray(out, dtype=float)


def _xgb_lag_mimo_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    multi_strategy: str = "multi_output_tree",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    MIMO (multi-input multi-output) strategy with a single multi-output regressor.

    Requires XGBoost's multi-target regression support (xgboost>=2.0).
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(XGB_INSTALL_ERROR) from e

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    params.setdefault("multi_strategy", str(multi_strategy))
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError(XGB_OBJECTIVE_EMPTY_ERROR)

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    model = xgb.XGBRegressor(**params)
    model.fit(X, Y)

    pred = model.predict(feat)
    out = np.asarray(pred, dtype=float).reshape(-1)
    if out.shape[0] != h:
        raise RuntimeError(f"Unexpected MIMO predict shape {out.shape}; expected ({h},)")
    return out


def xgb_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    XGBoost multi-horizon forecast using a single model with an extra "step" feature.
    """
    return _xgb_lag_step_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        xgb_params=xgb_params,
        step_scale=step_scale,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    XGBoost DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    return _xgb_lag_dirrec_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        xgb_params=xgb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_mimo_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    multi_strategy: str = "multi_output_tree",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    XGBoost MIMO (multi-input multi-output) multi-horizon forecast on lag features.

    Trains a single multi-output regressor to predict all horizon steps at once.
    """
    return _xgb_lag_mimo_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        xgb_params=xgb_params,
        multi_strategy=multi_strategy,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_custom_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable direct multi-horizon XGBoost (XGBRegressor) on lag features.

    All XGBoost parameters are provided through `xgb_params` and passed to `XGBRegressor`.
    """
    return _xgb_lag_direct_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        xgb_params=xgb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def xgb_custom_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable recursive one-step XGBoost (XGBRegressor) on lag features.

    All XGBoost parameters are provided through `xgb_params` and passed to `XGBRegressor`.
    """
    return _xgb_lag_recursive_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        xgb_params=xgb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def _lgbm_validate_common_regressor_params(params: dict[str, Any]) -> None:
    if "n_estimators" in params and params["n_estimators"] is not None:
        if int(params["n_estimators"]) <= 0:
            raise ValueError(N_ESTIMATORS_MIN_ERROR)
    if "learning_rate" in params and params["learning_rate"] is not None:
        if float(params["learning_rate"]) <= 0:
            raise ValueError(LEARNING_RATE_POSITIVE_ERROR)
    if "max_depth" in params and params["max_depth"] is not None:
        max_depth = int(params["max_depth"])
        # LightGBM convention: -1 means "no limit".
        if max_depth == 0 or max_depth < -1:
            raise ValueError("max_depth must be -1 or >= 1")
    if "num_leaves" in params and params["num_leaves"] is not None:
        if int(params["num_leaves"]) < 2:
            raise ValueError("num_leaves must be >= 2")
    if (
        "subsample" in params
        and params["subsample"] is not None
        and not (0.0 < float(params["subsample"]) <= 1.0)
    ):
        raise ValueError(SUBSAMPLE_RANGE_ERROR)
    if (
        "colsample_bytree" in params
        and params["colsample_bytree"] is not None
        and not (0.0 < float(params["colsample_bytree"]) <= 1.0)
    ):
        raise ValueError(COLSAMPLE_BYTREE_RANGE_ERROR)
    for key in ("reg_alpha", "reg_lambda", "min_child_weight", "min_split_gain"):
        if key in params and params[key] is not None and float(params[key]) < 0:
            raise ValueError(f"{key} must be >= 0")
    if "n_jobs" in params and params["n_jobs"] is not None and int(params["n_jobs"]) == 0:
        raise ValueError(N_JOBS_NON_ZERO_ERROR)


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'lightgbm lag models require lightgbm. Install with: pip install -e ".[lgbm]"'
        ) from e
    return lgb


def _lgbm_predict(model: Any, X: Any) -> Any:
    import warnings

    # LightGBM's sklearn wrapper sets feature names even when fitting on numpy arrays
    # ("Column_0", ...), then sklearn emits noisy warnings at predict-time. These
    # models operate purely by shape, so we safely silence that specific warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but LGBMRegressor was fitted with feature names"
            ),
            category=UserWarning,
        )
        return model.predict(X)


def _lgbm_lag_direct_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    lgb = _require_lightgbm()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(lgbm_params)
    params.setdefault("verbosity", -1)
    _lgbm_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat_base_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    fourier_pred_all: np.ndarray | None = None
    if bool(fourier_periods):
        fourier_pred_all, _ = build_fourier_features(
            np.arange(int(x.size), int(x.size) + h, dtype=int),
            periods=fourier_periods,
            orders=fourier_orders,
        )

    out = np.empty((h,), dtype=float)
    for j in range(h):
        if fourier_pred_all is None:
            x_j = x_base_aug
            feat = feat_base_aug
        else:
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_j = np.concatenate([x_base_aug, fourier_train], axis=1)
            feat = np.concatenate(
                [feat_base_aug, fourier_pred_all[int(j) : int(j) + 1, :]],
                axis=1,
            )
        model = lgb.LGBMRegressor(**params)
        model.fit(x_j, Y[:, j])
        out[j] = float(_lgbm_predict(model, feat)[0])
    return np.asarray(out, dtype=float)


def _lgbm_lag_recursive_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    lgb = _require_lightgbm()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if x.size <= lags:
        raise ValueError(
            f"lightgbm recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )

    params = dict(lgbm_params)
    params.setdefault("verbosity", -1)
    _lgbm_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(h):
        feat_base = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(_lgbm_predict(model, feat)[0])
        out.append(float(yhat))
        history.append(float(yhat))
    return np.asarray(out, dtype=float)


def _lgbm_lag_step_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Single-model multi-horizon forecasting by adding a "step index" feature.

    Trains one regressor on the expanded dataset:
      (lag_window, step) -> y_{t+step}
    """
    lgb = _require_lightgbm()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(lgbm_params)
    params.setdefault("verbosity", -1)
    _lgbm_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    rows = int(x_base.shape[0])
    derived, _ = build_lag_derived_features(
        x_base, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal, _ = build_seasonal_lag_features(
        x, t=t_idx, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )

    if step_scale not in {"one_based", "zero_based", "unit"}:
        raise ValueError(STEP_SCALE_OPTIONS_ERROR)

    if step_scale == "zero_based":
        step = np.arange(h, dtype=float)
    elif step_scale == "unit":
        step = np.arange(1, h + 1, dtype=float) / float(h)
    else:
        step = np.arange(1, h + 1, dtype=float)

    step_idx = np.tile(step, rows).reshape(-1, 1)  # (rows*h, 1)
    x_rep = np.repeat(x_base, repeats=h, axis=0)  # (rows*h, lags)
    derived_rep = np.repeat(derived, repeats=h, axis=0)
    seasonal_rep = np.repeat(seasonal, repeats=h, axis=0)

    if bool(fourier_periods):
        t_flat = np.repeat(t_idx, repeats=h) + np.tile(np.arange(h, dtype=int), rows)
        fourier_long, _ = build_fourier_features(
            t_flat, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_long = np.empty((rows * h, 0), dtype=float)

    x_long = np.concatenate([x_rep, derived_rep, seasonal_rep, fourier_long, step_idx], axis=1)
    y_long = Y.reshape(-1).astype(float, copy=False)

    model = lgb.LGBMRegressor(**params)
    model.fit(x_long, y_long)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    derived_pred, _ = build_lag_derived_features(
        base_feat, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal_pred, _ = build_seasonal_lag_features(
        x, t=[int(x.size)], seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    base_rep = np.repeat(base_feat, repeats=h, axis=0)  # (h, lags)
    derived_rep_pred = np.repeat(derived_pred, repeats=h, axis=0)
    seasonal_rep_pred = np.repeat(seasonal_pred, repeats=h, axis=0)

    if bool(fourier_periods):
        t_pred = np.arange(int(x.size), int(x.size) + h, dtype=int)
        fourier_pred, _ = build_fourier_features(
            t_pred, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_pred = np.empty((h, 0), dtype=float)

    feat = np.concatenate(
        [base_rep, derived_rep_pred, seasonal_rep_pred, fourier_pred, step.reshape(-1, 1)],
        axis=1,
    )
    out = _lgbm_predict(model, feat)
    return np.asarray(out, dtype=float)


def _lgbm_lag_dirrec_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    DirRec (Direct-Recursive) strategy with per-step models.

    Each step model uses lag features plus the previous steps as additional regressors.
    During training we use the true previous-step values; during prediction we use the
    model predictions from earlier steps.
    """
    lgb = _require_lightgbm()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(lgbm_params)
    params.setdefault("verbosity", -1)
    _lgbm_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )

    models: list[Any] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_step = np.concatenate([x_base_aug, fourier_train], axis=1)
        else:
            x_step = x_base_aug

        if j == 0:
            x_j = x_step
        else:
            x_j = np.concatenate([x_step, Y[:, :j]], axis=1)
        yj = Y[:, j]
        model = lgb.LGBMRegressor(**params)
        model.fit(x_j, yj)
        models.append(model)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    base_feat_aug = _augment_lag_feat_row(
        base_feat,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    ).reshape(-1)
    out: list[float] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_pred, _ = build_fourier_features(
                [int(x.size) + int(j)],
                periods=fourier_periods,
                orders=fourier_orders,
            )
            feat_step = np.concatenate([base_feat_aug, fourier_pred.reshape(-1)], axis=0)
        else:
            feat_step = base_feat_aug

        if j == 0:
            feat = feat_step
        else:
            feat = np.concatenate([feat_step, np.asarray(out, dtype=float)], axis=0)
        yhat = float(_lgbm_predict(models[j], feat.reshape(1, -1))[0])
        out.append(yhat)
    return np.asarray(out, dtype=float)


def lgbm_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_weight: float = 0.001,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    LightGBM (LGBMRegressor) on lag features (direct multi-horizon). Requires lightgbm.
    """
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    return _lgbm_lag_direct_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_weight: float = 0.001,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    LightGBM (LGBMRegressor) on lag features (one-step trained, recursive forecast). Requires lightgbm.
    """
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    return _lgbm_lag_recursive_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_weight: float = 0.001,
    random_state: int = 0,
    n_jobs: int = 1,
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    LightGBM multi-horizon forecast using a single model with an extra "step" feature.
    """
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    return _lgbm_lag_step_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=params,
        step_scale=step_scale,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_weight: float = 0.001,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    LightGBM DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    return _lgbm_lag_dirrec_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_custom_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable direct multi-horizon LightGBM (LGBMRegressor) on lag features.

    All LightGBM parameters are provided through `lgbm_params` and passed to `LGBMRegressor`.
    """
    return _lgbm_lag_direct_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=lgbm_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_custom_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable recursive one-step LightGBM (LGBMRegressor) on lag features.

    All LightGBM parameters are provided through `lgbm_params` and passed to `LGBMRegressor`.
    """
    return _lgbm_lag_recursive_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=lgbm_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_custom_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable LightGBM multi-horizon forecast using a single model with a "step" feature.
    """
    return _lgbm_lag_step_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=lgbm_params,
        step_scale=step_scale,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def lgbm_custom_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    lgbm_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable LightGBM DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    return _lgbm_lag_dirrec_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        lgbm_params=lgbm_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def _catboost_validate_common_regressor_params(params: dict[str, Any]) -> None:
    if "iterations" in params and params["iterations"] is not None:
        if int(params["iterations"]) <= 0:
            raise ValueError("iterations must be >= 1")
    if "learning_rate" in params and params["learning_rate"] is not None:
        if float(params["learning_rate"]) <= 0:
            raise ValueError(LEARNING_RATE_POSITIVE_ERROR)
    if "depth" in params and params["depth"] is not None:
        if int(params["depth"]) <= 0:
            raise ValueError("depth must be >= 1")
    if "l2_leaf_reg" in params and params["l2_leaf_reg"] is not None:
        if float(params["l2_leaf_reg"]) < 0:
            raise ValueError("l2_leaf_reg must be >= 0")
    if (
        "thread_count" in params
        and params["thread_count"] is not None
        and int(params["thread_count"]) == 0
    ):
        raise ValueError("thread_count must be non-zero")


def _require_catboost() -> Any:
    try:
        import catboost as cb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'catboost lag models require catboost. Install with: pip install -e ".[catboost]"'
        ) from e
    return cb


def _catboost_lag_direct_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    cb = _require_catboost()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)
    _catboost_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    feat_base = x[-lags:].astype(float, copy=False).reshape(1, -1)
    feat_base_aug = _augment_lag_feat_row(
        feat_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    )

    fourier_pred_all: np.ndarray | None = None
    if bool(fourier_periods):
        fourier_pred_all, _ = build_fourier_features(
            np.arange(int(x.size), int(x.size) + h, dtype=int),
            periods=fourier_periods,
            orders=fourier_orders,
        )

    out = np.empty((h,), dtype=float)
    for j in range(h):
        model = cb.CatBoostRegressor(**params)
        if fourier_pred_all is None:
            x_j = x_base_aug
            feat = feat_base_aug
        else:
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_j = np.concatenate([x_base_aug, fourier_train], axis=1)
            feat = np.concatenate(
                [feat_base_aug, fourier_pred_all[int(j) : int(j) + 1, :]],
                axis=1,
            )

        model.fit(x_j, Y[:, j])
        out[j] = float(model.predict(feat)[0])
    return np.asarray(out, dtype=float)


def _catboost_lag_recursive_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    cb = _require_catboost()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)
    if x.size <= lags:
        raise ValueError(
            f"catboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )

    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)
    _catboost_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, y = make_lagged_xy(x, lags=lags, start_t=start_t)
    t_idx = np.arange(int(start_t), int(x.size), dtype=int)
    X = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )
    model = cb.CatBoostRegressor(**params)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(h):
        feat_base = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        feat = _augment_lag_feat_row(
            feat_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
            t_next=len(history),
            history=np.asarray(history, dtype=float),
        )
        yhat = float(model.predict(feat)[0])
        out.append(float(yhat))
        history.append(float(yhat))
    return np.asarray(out, dtype=float)


def _catboost_lag_step_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Single-model multi-horizon forecasting by adding a "step index" feature.

    Trains one regressor on the expanded dataset:
      (lag_window, step) -> y_{t+step}
    """
    cb = _require_catboost()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)
    _catboost_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    rows = int(x_base.shape[0])
    derived, _ = build_lag_derived_features(
        x_base, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal, _ = build_seasonal_lag_features(
        x, t=t_idx, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )

    if step_scale not in {"one_based", "zero_based", "unit"}:
        raise ValueError(STEP_SCALE_OPTIONS_ERROR)

    if step_scale == "zero_based":
        step = np.arange(h, dtype=float)
    elif step_scale == "unit":
        step = np.arange(1, h + 1, dtype=float) / float(h)
    else:
        step = np.arange(1, h + 1, dtype=float)

    step_idx = np.tile(step, rows).reshape(-1, 1)  # (rows*h, 1)
    x_rep = np.repeat(x_base, repeats=h, axis=0)  # (rows*h, lags)
    derived_rep = np.repeat(derived, repeats=h, axis=0)
    seasonal_rep = np.repeat(seasonal, repeats=h, axis=0)

    if bool(fourier_periods):
        t_flat = np.repeat(t_idx, repeats=h) + np.tile(np.arange(h, dtype=int), rows)
        fourier_long, _ = build_fourier_features(
            t_flat, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_long = np.empty((rows * h, 0), dtype=float)

    x_long = np.concatenate([x_rep, derived_rep, seasonal_rep, fourier_long, step_idx], axis=1)
    y_long = Y.reshape(-1).astype(float, copy=False)

    model = cb.CatBoostRegressor(**params)
    model.fit(x_long, y_long)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    derived_pred, _ = build_lag_derived_features(
        base_feat, roll_windows=roll_windows, roll_stats=roll_stats, diff_lags=diff_lags
    )
    seasonal_pred, _ = build_seasonal_lag_features(
        x, t=[int(x.size)], seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    base_rep = np.repeat(base_feat, repeats=h, axis=0)  # (h, lags)
    derived_rep_pred = np.repeat(derived_pred, repeats=h, axis=0)
    seasonal_rep_pred = np.repeat(seasonal_pred, repeats=h, axis=0)

    if bool(fourier_periods):
        t_pred = np.arange(int(x.size), int(x.size) + h, dtype=int)
        fourier_pred, _ = build_fourier_features(
            t_pred, periods=fourier_periods, orders=fourier_orders
        )
    else:
        fourier_pred = np.empty((h, 0), dtype=float)

    feat = np.concatenate(
        [base_rep, derived_rep_pred, seasonal_rep_pred, fourier_pred, step.reshape(-1, 1)],
        axis=1,
    )
    out = model.predict(feat)
    return np.asarray(out, dtype=float)


def _catboost_lag_dirrec_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    DirRec (Direct-Recursive) strategy with per-step models.

    Each step model uses lag features plus the previous steps as additional regressors.
    During training we use the true previous-step values; during prediction we use the
    model predictions from earlier steps.
    """
    cb = _require_catboost()

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError(HORIZON_MIN_ERROR)
    if lags <= 0:
        raise ValueError(LAGS_MIN_ERROR)

    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)
    _catboost_validate_common_regressor_params(params)

    start_t = _compute_feature_start_t(
        lags=lags, seasonal_lags=seasonal_lags, seasonal_diff_lags=seasonal_diff_lags
    )
    x_base, Y, t_idx = _make_lagged_xy_multi(x, lags=lags, horizon=h, start_t=start_t)
    x_base_aug = _augment_lag_matrix(
        x_base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_index=t_idx,
        series=x,
    )

    models: list[Any] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_train, _ = build_fourier_features(
                t_idx + int(j),
                periods=fourier_periods,
                orders=fourier_orders,
            )
            x_step = np.concatenate([x_base_aug, fourier_train], axis=1)
        else:
            x_step = x_base_aug

        if j == 0:
            x_j = x_step
        else:
            x_j = np.concatenate([x_step, Y[:, :j]], axis=1)
        yj = Y[:, j]
        model = cb.CatBoostRegressor(**params)
        model.fit(x_j, yj)
        models.append(model)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    base_feat_aug = _augment_lag_feat_row(
        base_feat,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=(),
        fourier_orders=fourier_orders,
        t_next=int(x.size),
        history=x,
    ).reshape(-1)
    out: list[float] = []
    for j in range(h):
        if bool(fourier_periods):
            fourier_pred, _ = build_fourier_features(
                [int(x.size) + int(j)],
                periods=fourier_periods,
                orders=fourier_orders,
            )
            feat_step = np.concatenate([base_feat_aug, fourier_pred.reshape(-1)], axis=0)
        else:
            feat_step = base_feat_aug

        if j == 0:
            feat = feat_step
        else:
            feat = np.concatenate([feat_step, np.asarray(out, dtype=float)], axis=0)
        yhat = float(models[j].predict(feat.reshape(1, -1))[0])
        out.append(yhat)
    return np.asarray(out, dtype=float)


def catboost_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 0,
    thread_count: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    CatBoost (CatBoostRegressor) on lag features (direct multi-horizon). Requires catboost.
    """
    params = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_seed),
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
    }
    return _catboost_lag_direct_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 0,
    thread_count: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    CatBoost (CatBoostRegressor) on lag features (one-step trained, recursive forecast). Requires catboost.
    """
    params = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_seed),
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
    }
    return _catboost_lag_recursive_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 0,
    thread_count: int = 1,
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    CatBoost multi-horizon forecast using a single model with an extra "step" feature.
    """
    params = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_seed),
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
    }
    return _catboost_lag_step_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=params,
        step_scale=step_scale,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 0,
    thread_count: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    CatBoost DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    params = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_seed),
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
    }
    return _catboost_lag_dirrec_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_custom_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable direct multi-horizon CatBoost (CatBoostRegressor) on lag features.

    All CatBoost parameters are provided through `cb_params` and passed to `CatBoostRegressor`.
    """
    return _catboost_lag_direct_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=cb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_custom_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable recursive one-step CatBoost (CatBoostRegressor) on lag features.

    All CatBoost parameters are provided through `cb_params` and passed to `CatBoostRegressor`.
    """
    return _catboost_lag_recursive_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=cb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_custom_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable CatBoost multi-horizon forecast using a single model with a "step" feature.
    """
    return _catboost_lag_step_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=cb_params,
        step_scale=step_scale,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )


def catboost_custom_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    cb_params: dict[str, Any],
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
) -> np.ndarray:
    """
    Customizable CatBoost DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    return _catboost_lag_dirrec_forecast_kwargs(
        train,
        horizon,
        lags=lags,
        cb_params=cb_params,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        seasonal_lags=seasonal_lags,
        seasonal_diff_lags=seasonal_diff_lags,
        fourier_periods=fourier_periods,
        fourier_orders=fourier_orders,
    )
