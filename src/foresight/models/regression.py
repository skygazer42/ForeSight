from __future__ import annotations

from typing import Any

import numpy as np

from ..features.lag import make_lagged_xy


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def lr_lag_forecast(train: Any, horizon: int, *, lags: int) -> np.ndarray:
    """
    Fit an OLS linear regression on lag features and forecast recursively.

    This is intentionally lightweight (pure numpy) and suitable as a baseline.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if x.size <= lags:
        raise ValueError(f"lr_lag_forecast requires > lags points (lags={lags}), got {x.size}")

    X, y = make_lagged_xy(x, lags=lags)
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    intercept = float(coef[0])
    w = coef[1:].astype(float, copy=False)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(horizon):
        feat = np.array(history[-lags:], dtype=float)
        yhat = intercept + float(np.dot(w, feat))
        out.append(float(yhat))
        history.append(float(yhat))
    return np.asarray(out, dtype=float)


def _make_lagged_xy_multi(
    x: np.ndarray, *, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n < lags + h:
        raise ValueError(f"Need >= lags+horizon points (lags={lags}, horizon={h}), got {n}")

    rows = n - lags - h + 1
    X = np.empty((rows, lags), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    for i in range(rows):
        t = i + lags
        X[i, :] = x[t - lags : t]
        for j in range(h):
            Y[i, j] = x[t + j]
    return X, Y


def lr_lag_direct_forecast(train: Any, horizon: int, *, lags: int) -> np.ndarray:
    """
    Direct multi-horizon OLS regression on lag features.

    Fits one linear model per horizon step (shared feature matrix), then predicts
    all steps directly from the last `lags` observed values.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    coef, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (1+lags, horizon)

    intercept = coef[0, :].astype(float, copy=False)
    w = coef[1:, :].astype(float, copy=False)  # (lags, horizon)

    feat = x[-lags:].astype(float, copy=False)
    yhat = intercept + feat @ w
    return np.asarray(yhat, dtype=float)


def ridge_lag_forecast(train: Any, horizon: int, *, lags: int, alpha: float = 1.0) -> np.ndarray:
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
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if x.size <= lags:
        raise ValueError(f"ridge_lag_forecast requires > lags points (lags={lags}), got {x.size}")

    X, y = make_lagged_xy(x, lags=lags)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(horizon):
        feat = np.array(history[-lags:], dtype=float).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def lasso_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 0.001,
    max_iter: int = 5000,
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=int(max_iter))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=int(max_iter),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def knn_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    n_neighbors: int = 10,
    weights: str = "distance",
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = KNeighborsRegressor(n_neighbors=int(n_neighbors), weights=str(weights))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)
