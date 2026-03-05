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


def ridge_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    alpha: float = 1.0,
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
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def decision_tree_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    max_depth: int | None = 5,
    random_state: int = 0,
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    model = DecisionTreeRegressor(
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
    )
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
    )
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = AdaBoostRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if not (0.0 < float(max_samples) <= 1.0):
        raise ValueError("max_samples must be in (0,1]")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = BaggingRegressor(
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if float(C) <= 0:
        raise ValueError("C must be > 0")
    if float(epsilon) < 0:
        raise ValueError("epsilon must be >= 0")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = SVR(C=float(C), gamma=gamma, epsilon=float(epsilon))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if float(C) <= 0:
        raise ValueError("C must be > 0")
    if float(epsilon) < 0:
        raise ValueError("epsilon must be >= 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = LinearSVR(
        C=float(C),
        epsilon=float(epsilon),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    model = KernelRidge(alpha=float(alpha), kernel=str(kernel), gamma=gamma)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    if float(learning_rate_init) <= 0:
        raise ValueError("learning_rate_init must be > 0")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    model = MLPRegressor(
        hidden_layer_sizes=tuple(int(s) for s in hidden_layer_sizes),
        alpha=float(alpha),
        max_iter=int(max_iter),
        random_state=int(random_state),
        learning_rate_init=float(learning_rate_init),
    )
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if float(epsilon) <= 1.0:
        raise ValueError("epsilon must be > 1.0")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = HuberRegressor(epsilon=float(epsilon), alpha=float(alpha), max_iter=int(max_iter))
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    yhat = model.predict(feat)[0]
    return np.asarray(yhat, dtype=float)


def quantile_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    quantile: float = 0.5,
    alpha: float = 0.0,
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be in (0,1)")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = QuantileRegressor(quantile=q, alpha=float(alpha), fit_intercept=True)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    base = SGDRegressor(
        alpha=float(alpha),
        penalty=str(penalty),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
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
    objective_params: dict[str, Any] | None = None,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be > 0")
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError("subsample must be in (0,1]")
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError("colsample_bytree must be in (0,1]")
    if float(reg_alpha) < 0:
        raise ValueError("reg_alpha must be >= 0")
    if float(reg_lambda) < 0:
        raise ValueError("reg_lambda must be >= 0")
    if float(min_child_weight) < 0:
        raise ValueError("min_child_weight must be >= 0")
    if float(gamma) < 0:
        raise ValueError("gamma must be >= 0")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    obj = str(objective).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    # Objective-specific label constraints (defensive ergonomics).
    if obj in {"count:poisson", "reg:tweedie"}:
        if np.any(x < 0.0):
            raise ValueError(f"{obj} requires non-negative series values")
    if obj == "reg:squaredlogerror":
        if np.any(x < 0.0):
            raise ValueError("reg:squaredlogerror requires non-negative series values")
    if obj == "reg:gamma":
        if np.any(x <= 0.0):
            raise ValueError("reg:gamma requires strictly positive series values")
    if obj == "reg:logistic":
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("reg:logistic requires series values in [0,1]")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)

    extra = dict(objective_params or {})

    out = np.empty((int(horizon),), dtype=float)
    for j in range(int(horizon)):
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
        model.fit(X, Y[:, j])
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
    objective_params: dict[str, Any] | None = None,
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be > 0")
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError("subsample must be in (0,1]")
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError("colsample_bytree must be in (0,1]")
    if float(reg_alpha) < 0:
        raise ValueError("reg_alpha must be >= 0")
    if float(reg_lambda) < 0:
        raise ValueError("reg_lambda must be >= 0")
    if float(min_child_weight) < 0:
        raise ValueError("min_child_weight must be >= 0")
    if float(gamma) < 0:
        raise ValueError("gamma must be >= 0")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    obj = str(objective).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    # Objective-specific label constraints (defensive ergonomics).
    if obj in {"count:poisson", "reg:tweedie"}:
        if np.any(x < 0.0):
            raise ValueError(f"{obj} requires non-negative series values")
    if obj == "reg:squaredlogerror":
        if np.any(x < 0.0):
            raise ValueError("reg:squaredlogerror requires non-negative series values")
    if obj == "reg:gamma":
        if np.any(x <= 0.0):
            raise ValueError("reg:gamma requires strictly positive series values")
    if obj == "reg:logistic":
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("reg:logistic requires series values in [0,1]")

    X, y = make_lagged_xy(x, lags=lags)
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
        feat = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost (XGBRegressor) on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost (XGBRegressor) on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost DART booster on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="dart",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost DART booster on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="dart",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Random Forest (XGBRFRegressor) on lag features (requires xgboost).
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError("subsample must be in (0,1]")
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError("colsample_bytree must be in (0,1]")
    if float(reg_lambda) < 0:
        raise ValueError("reg_lambda must be >= 0")
    if float(min_child_weight) < 0:
        raise ValueError("min_child_weight must be >= 0")
    if float(gamma) < 0:
        raise ValueError("gamma must be >= 0")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)

    out = np.empty((int(horizon),), dtype=float)
    for j in range(int(horizon)):
        model = xgb.XGBRFRegressor(
            objective="reg:squarederror",
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
        model.fit(X, Y[:, j])
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost Random Forest (XGBRFRegressor) on lag features (requires xgboost).
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )
    if n_estimators <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")
    if not (0.0 < float(subsample) <= 1.0):
        raise ValueError("subsample must be in (0,1]")
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise ValueError("colsample_bytree must be in (0,1]")
    if float(reg_lambda) < 0:
        raise ValueError("reg_lambda must be >= 0")
    if float(min_child_weight) < 0:
        raise ValueError("min_child_weight must be >= 0")
    if float(gamma) < 0:
        raise ValueError("gamma must be >= 0")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    X, y = make_lagged_xy(x, lags=lags)
    model = xgb.XGBRFRegressor(
        objective="reg:squarederror",
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
        feat = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost linear booster (gblinear) on lag features (requires xgboost).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gblinear",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost linear booster (gblinear) on lag features (requires xgboost).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gblinear",
        objective="reg:squarederror",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost with squared log error objective on lag features (requires xgboost, y>=0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:squaredlogerror",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost with squared log error objective on lag features (requires xgboost, y>=0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:squaredlogerror",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost with logistic regression objective on lag features (requires xgboost, y in [0,1]).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:logistic",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost with logistic regression objective on lag features (requires xgboost, y in [0,1]).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:logistic",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Poisson objective on lag features (requires xgboost; y>=0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="count:poisson",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost Poisson objective on lag features (requires xgboost; y>=0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="count:poisson",
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
) -> np.ndarray:
    """
    Direct multi-horizon XGBoost Gamma objective on lag features (requires xgboost; y>0).
    """
    return _xgb_lag_direct_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:gamma",
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
) -> np.ndarray:
    """
    Recursive one-step XGBoost Gamma objective on lag features (requires xgboost; y>0).
    """
    return _xgb_lag_recursive_forecast(
        train,
        horizon,
        lags=lags,
        booster="gbtree",
        objective="reg:gamma",
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
        objective="reg:tweedie",
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
        objective="reg:tweedie",
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
        objective_params={"tweedie_variance_power": tvp},
    )


def _xgb_validate_objective_label_constraints(obj: str, x: np.ndarray) -> None:
    """
    Defensive ergonomics: a few XGBoost objectives have label-domain assumptions.

    We validate these early to surface a clearer error than the underlying trainer.
    """
    if obj in {"count:poisson", "reg:tweedie", "reg:squaredlogerror"}:
        if np.any(x < 0.0):
            raise ValueError(f"{obj} requires non-negative series values")
    if obj == "reg:gamma":
        if np.any(x <= 0.0):
            raise ValueError("reg:gamma requires strictly positive series values")
    if obj == "reg:logistic":
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("reg:logistic requires series values in [0,1]")


def _xgb_validate_common_regressor_params(params: dict[str, Any]) -> None:
    if "n_estimators" in params and params["n_estimators"] is not None:
        if int(params["n_estimators"]) <= 0:
            raise ValueError("n_estimators must be >= 1")
    if "max_depth" in params and params["max_depth"] is not None:
        if int(params["max_depth"]) <= 0:
            raise ValueError("max_depth must be >= 1")
    if "learning_rate" in params and params["learning_rate"] is not None:
        if float(params["learning_rate"]) <= 0:
            raise ValueError("learning_rate must be > 0")
    if "subsample" in params and params["subsample"] is not None:
        subsample = float(params["subsample"])
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0,1]")
    if "colsample_bytree" in params and params["colsample_bytree"] is not None:
        colsample_bytree = float(params["colsample_bytree"])
        if not (0.0 < colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must be in (0,1]")
    for key in ("reg_alpha", "reg_lambda", "min_child_weight", "gamma"):
        if key in params and params[key] is not None and float(params[key]) < 0:
            raise ValueError(f"{key} must be >= 0")
    if "n_jobs" in params and params["n_jobs"] is not None and int(params["n_jobs"]) == 0:
        raise ValueError("n_jobs must be non-zero")


def _xgb_lag_direct_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    X, Y = _make_lagged_xy_multi(x, lags=lags, horizon=int(horizon))
    feat = x[-lags:].astype(float, copy=False).reshape(1, -1)

    out = np.empty((int(horizon),), dtype=float)
    for j in range(int(horizon)):
        model = xgb.XGBRegressor(**params)
        model.fit(X, Y[:, j])
        out[j] = float(model.predict(feat)[0])
    return np.asarray(out, dtype=float)


def _xgb_lag_recursive_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")
    if x.size <= lags:
        raise ValueError(
            f"xgboost recursive lag forecast requires > lags points (lags={lags}), got {x.size}"
        )

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    X, y = make_lagged_xy(x, lags=lags)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(int(horizon)):
        feat = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
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
) -> np.ndarray:
    """
    Single-model multi-horizon forecasting by adding a "step index" feature.

    This trains one regressor on the expanded dataset:
      (lag_window, step) -> y_{t+step}
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    X_base, Y = _make_lagged_xy_multi(x, lags=lags, horizon=h)
    rows = int(X_base.shape[0])

    if step_scale not in {"one_based", "zero_based", "unit"}:
        raise ValueError("step_scale must be one of: one_based, zero_based, unit")

    if step_scale == "zero_based":
        step = np.arange(h, dtype=float)
    elif step_scale == "unit":
        step = np.arange(1, h + 1, dtype=float) / float(h)
    else:
        step = np.arange(1, h + 1, dtype=float)

    step_idx = np.tile(step, rows).reshape(-1, 1)  # (rows*h, 1)
    X_rep = np.repeat(X_base, repeats=h, axis=0)  # (rows*h, lags)
    X_long = np.concatenate([X_rep, step_idx], axis=1)
    y_long = Y.reshape(-1).astype(float, copy=False)

    model = xgb.XGBRegressor(**params)
    model.fit(X_long, y_long)

    base_feat = x[-lags:].astype(float, copy=False).reshape(1, -1)
    base_rep = np.repeat(base_feat, repeats=h, axis=0)  # (h, lags)
    feat = np.concatenate([base_rep, step.reshape(-1, 1)], axis=1)
    out = model.predict(feat)
    return np.asarray(out, dtype=float)


def _xgb_lag_dirrec_forecast_kwargs(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
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
        raise ImportError(
            'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lags <= 0:
        raise ValueError("lags must be >= 1")

    params = dict(xgb_params)
    params.setdefault("verbosity", 0)
    obj = str(params.get("objective", "")).strip()
    if not obj:
        raise ValueError("objective must be non-empty")

    _xgb_validate_objective_label_constraints(obj, x)
    _xgb_validate_common_regressor_params(params)

    X_base, Y = _make_lagged_xy_multi(x, lags=lags, horizon=h)

    models: list[Any] = []
    for j in range(h):
        if j == 0:
            Xj = X_base
        else:
            Xj = np.concatenate([X_base, Y[:, :j]], axis=1)
        yj = Y[:, j]
        model = xgb.XGBRegressor(**params)
        model.fit(Xj, yj)
        models.append(model)

    base_feat = x[-lags:].astype(float, copy=False)
    out: list[float] = []
    for j in range(h):
        if j == 0:
            feat = base_feat
        else:
            feat = np.concatenate([base_feat, np.asarray(out, dtype=float)], axis=0)
        yhat = float(models[j].predict(feat.reshape(1, -1))[0])
        out.append(yhat)
    return np.asarray(out, dtype=float)


def xgb_step_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
    step_scale: str = "one_based",
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
    )


def xgb_dirrec_lag_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    """
    XGBoost DirRec (direct-recursive) multi-horizon forecast on lag features.
    """
    return _xgb_lag_dirrec_forecast_kwargs(train, horizon, lags=lags, xgb_params=xgb_params)


def xgb_custom_lag_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    """
    Customizable direct multi-horizon XGBoost (XGBRegressor) on lag features.

    All XGBoost parameters are provided through `xgb_params` and passed to `XGBRegressor`.
    """
    return _xgb_lag_direct_forecast_kwargs(train, horizon, lags=lags, xgb_params=xgb_params)


def xgb_custom_lag_recursive_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    """
    Customizable recursive one-step XGBoost (XGBRegressor) on lag features.

    All XGBoost parameters are provided through `xgb_params` and passed to `XGBRegressor`.
    """
    return _xgb_lag_recursive_forecast_kwargs(train, horizon, lags=lags, xgb_params=xgb_params)
