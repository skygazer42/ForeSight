from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _pop_legacy_keyword(kwargs: dict[str, Any], *, legacy_name: str, value: Any) -> Any:
    if legacy_name in kwargs:
        return kwargs.pop(legacy_name)
    return value


def _raise_unexpected_kwargs(function_name: str, kwargs: dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(f"{function_name}() got unexpected keyword arguments: {sorted(kwargs)}")


def _validated_ar_ols_input(
    train: Any,
    *,
    horizon: int,
    p: int,
    subject: str,
) -> tuple[np.ndarray, int, int]:
    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    order = int(p)
    if order <= 0:
        raise ValueError("p must be >= 1")
    if x.size <= order:
        raise ValueError(f"{subject} requires > p points (p={order}), got {x.size}")
    return x, h, order


def _ar_ols_design_matrix(x: np.ndarray, *, p: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    order = int(p)
    y = x[order:]
    rows = n - order
    X = np.ones((rows, order + 1), dtype=float)
    for i in range(1, order + 1):
        X[:, i] = x[order - i : n - i]
    return X, np.asarray(y, dtype=float)


def ar_ols_forecast(train: Any, horizon: int, *, p: int) -> np.ndarray:
    """
    Autoregression AR(p) fit by OLS, forecast recursively.
    """
    x, h, order = _validated_ar_ols_input(
        train,
        horizon=horizon,
        p=p,
        subject="ar_ols_forecast",
    )

    X, y = _ar_ols_design_matrix(x, p=order)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(coef[0])
    ar = coef[1:].astype(float, copy=False)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _step in range(h):
        lags = np.array(history[-order:][::-1], dtype=float)  # most recent first
        yhat = intercept + float(np.dot(ar, lags))
        out.append(float(yhat))
        history.append(float(yhat))

    return np.asarray(out, dtype=float)


def _validate_lags(lags: Any) -> tuple[int, ...]:
    if isinstance(lags, int):
        lags_tup = (int(lags),)
    elif isinstance(lags, str):
        s = lags.strip()
        if not s:
            raise ValueError("lags must be non-empty")
        lags_tup = tuple(int(p.strip()) for p in s.split(",") if p.strip())
    elif isinstance(lags, list | tuple):
        lags_tup = tuple(int(v) for v in lags)
    else:
        lags_tup = (int(lags),)

    if not lags_tup:
        raise ValueError("lags must be non-empty")
    if any(lag <= 0 for lag in lags_tup):
        raise ValueError("lags must be positive integers")
    # keep order but drop duplicates
    out: list[int] = []
    seen: set[int] = set()
    for lag in lags_tup:
        if lag not in seen:
            seen.add(lag)
            out.append(int(lag))
    return tuple(out)


def ar_ols_lags_forecast(train: Any, horizon: int, *, lags: Any) -> np.ndarray:
    """
    Autoregression with an arbitrary set of lag indices, fit by OLS, forecast recursively.

    Example:
      lags=(1, 2, 12) uses y_{t-1}, y_{t-2}, y_{t-12}.
    """
    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    lags_tup = _validate_lags(lags)
    max_lag = int(max(lags_tup))
    if x.size <= max_lag:
        raise ValueError(
            f"ar_ols_lags_forecast requires > max(lags) points (max_lag={max_lag}), got {x.size}"
        )

    n = int(x.size)
    rows = n - max_lag
    y = x[max_lag:]

    X = np.ones((rows, len(lags_tup) + 1), dtype=float)
    for j, lag in enumerate(lags_tup, start=1):
        X[:, j] = x[max_lag - lag : n - lag]

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(coef[0])
    w = coef[1:].astype(float, copy=False)

    history = x.astype(float, copy=True).tolist()
    out: list[float] = []
    for _h in range(int(horizon)):
        feats = np.array([history[-lag] for lag in lags_tup], dtype=float)
        yhat = intercept + float(np.dot(w, feats))
        out.append(float(yhat))
        history.append(float(yhat))

    return np.asarray(out, dtype=float)


def sar_ols_forecast(
    train: Any,
    horizon: int,
    *,
    p: int = 1,
    seasonal_p: int = 1,
    season_length: int = 12,
    **kwargs: Any,
) -> np.ndarray:
    """
    Seasonal autoregression using OLS with both short and seasonal lags.

    Uses lags:
      1..p  and  season_length, 2*season_length, .., P*season_length
    """
    seasonal_p = int(_pop_legacy_keyword(kwargs, legacy_name="P", value=seasonal_p))
    _raise_unexpected_kwargs("sar_ols_forecast", kwargs)

    if p < 0:
        raise ValueError("p must be >= 0")
    if seasonal_p < 0:
        raise ValueError("P must be >= 0")
    if season_length <= 0:
        raise ValueError("season_length must be >= 1")

    lags: list[int] = []
    lags.extend(range(1, int(p) + 1))
    lags.extend([int(season_length) * k for k in range(1, int(seasonal_p) + 1)])
    if not lags:
        raise ValueError("At least one of p or P must be > 0")

    return ar_ols_lags_forecast(train, horizon, lags=tuple(lags))


def select_ar_order_aic(train: Any, *, max_p: int = 10) -> int:
    """
    Select AR(p) order by AIC on the provided training series (OLS fit).
    """
    x = _as_1d_float_array(train)
    if max_p <= 0:
        raise ValueError("max_p must be >= 1")
    if x.size < 4:
        raise ValueError("select_ar_order_aic requires at least 4 points")

    max_p_eff = min(int(max_p), int(x.size) - 1)
    best_p: int | None = None
    best_aic = float("inf")

    for p in range(1, max_p_eff + 1):
        if x.size <= p + 1:
            continue

        X, y = _ar_ols_design_matrix(x, p=p)
        rows = int(X.shape[0])

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sse = float(np.sum(resid**2))
        sigma2 = sse / float(max(rows, 1))
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            sigma2 = 1e-12

        k = int(p + 1)  # intercept + p coefs
        aic = float(rows * np.log(sigma2) + 2.0 * k)
        if aic < best_aic:
            best_aic = aic
            best_p = int(p)

    if best_p is None:
        raise ValueError("Could not select an AR order; check max_p and series length.")
    return int(best_p)


def ar_ols_auto_forecast(train: Any, horizon: int, *, max_p: int = 10) -> np.ndarray:
    """
    Auto AR(p) by AIC (OLS fit), forecast recursively.
    """
    p = select_ar_order_aic(train, max_p=int(max_p))
    return ar_ols_forecast(train, horizon, p=int(p))
