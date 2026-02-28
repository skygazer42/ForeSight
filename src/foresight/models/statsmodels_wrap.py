from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def arima_forecast(train: Any, horizon: int, *, order: tuple[int, int, int]) -> np.ndarray:
    """
    ARIMA forecast via statsmodels (optional dependency).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("arima_forecast requires at least 3 training points")

    p, d, q = map(int, order)
    model = ARIMA(x, order=(p, d, q))
    res = model.fit()
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def sarimax_forecast(
    train: Any,
    horizon: int,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> np.ndarray:
    """
    SARIMAX / seasonal ARIMA via statsmodels (optional dependency).
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'sarimax_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("sarimax_forecast requires at least 3 training points")

    model = SARIMAX(
        x,
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 12,
    trend: str = "c",
    seasonal: bool = False,
    period: int | None = None,
) -> np.ndarray:
    """
    AutoReg (AR with optional deterministic terms) via statsmodels (optional dependency).
    """
    try:
        from statsmodels.tsa.ar_model import AutoReg  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'autoreg_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("autoreg_forecast requires at least 3 training points")
    if int(lags) <= 0:
        raise ValueError("lags must be >= 1")

    model = AutoReg(
        x,
        lags=int(lags),
        trend=str(trend),
        seasonal=bool(seasonal),
        period=(None if period is None else int(period)),
        old_names=False,
    )
    res = model.fit()
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def unobserved_components_forecast(
    train: Any,
    horizon: int,
    *,
    level: str = "local level",
) -> np.ndarray:
    """
    Structural / state-space model via statsmodels UnobservedComponents (optional dependency).

    Common `level` strings:
      - "local level"
      - "local linear trend"
      - "random walk"
    """
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'unobserved_components_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("unobserved_components_forecast requires at least 3 training points")

    model = UnobservedComponents(x, level=str(level))
    res = model.fit(disp=False)
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def stl_arima_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal: int = 7,
    robust: bool = False,
) -> np.ndarray:
    """
    STL + ARIMA remainder forecasting via statsmodels STLForecast (optional dependency).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        from statsmodels.tsa.forecasting.stl import STLForecast  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("stl_arima_forecast requires at least 3 training points")
    if int(period) <= 1:
        raise ValueError("period must be >= 2")

    stlf = STLForecast(
        x,
        ARIMA,
        model_kwargs={"order": tuple(int(v) for v in order)},
        period=int(period),
        seasonal=int(seasonal),
        robust=bool(robust),
    )
    res = stlf.fit()
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    information_criterion: str = "aic",
) -> np.ndarray:
    """
    Lightweight AutoARIMA-style grid search via statsmodels ARIMA (optional dependency).

    This tries orders in a small (p,d,q) grid and selects the best by AIC/BIC.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'auto_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 5:
        raise ValueError("auto_arima_forecast requires at least 5 training points")
    if int(max_p) < 0 or int(max_d) < 0 or int(max_q) < 0:
        raise ValueError("max_p/max_d/max_q must be >= 0")

    ic_key = str(information_criterion).strip().lower()
    if ic_key not in {"aic", "bic"}:
        raise ValueError("information_criterion must be 'aic' or 'bic'")

    best_ic = float("inf")
    best_res = None

    for p in range(0, int(max_p) + 1):
        for d in range(0, int(max_d) + 1):
            for q in range(0, int(max_q) + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    res = ARIMA(x, order=(int(p), int(d), int(q))).fit()
                except Exception:  # noqa: BLE001
                    continue
                ic = float(getattr(res, ic_key))
                if ic < best_ic:
                    best_ic = ic
                    best_res = res

    if best_res is None:
        raise ValueError("auto_arima_forecast failed to fit any ARIMA model in the grid")

    fc = best_res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def _normalize_periods(periods: Any) -> tuple[int, ...]:
    if periods is None:
        raise ValueError("periods must be provided")
    if isinstance(periods, int | float):
        return (int(periods),)
    if isinstance(periods, str):
        s = periods.strip()
        if not s:
            raise ValueError("periods must be non-empty")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    if isinstance(periods, list | tuple):
        return tuple(int(p) for p in periods)
    return (int(periods),)


def _repeat_last_season(seasonal: np.ndarray, *, period: int, horizon: int) -> np.ndarray:
    if int(period) <= 0:
        raise ValueError("period must be >= 1")
    p = int(period)
    if seasonal.size == 0:
        return np.zeros((int(horizon),), dtype=float)
    pattern = seasonal[-p:] if seasonal.size >= p else seasonal
    return np.resize(np.asarray(pattern, dtype=float), (int(horizon),))


def mstl_arima_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    order: tuple[int, int, int] = (1, 0, 0),
    iterate: int = 2,
    lmbda: float | str | None = None,
) -> np.ndarray:
    """
    MSTL (multi-seasonal STL) decomposition + ARIMA on the seasonally-adjusted series.

    Seasonal forecast is a simple repetition of the last seasonal cycle(s).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 10:
        raise ValueError("mstl_arima_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError("periods must contain integers >= 2")
    max_p = int(max(periods_tup))
    if x.size < 2 * max_p:
        raise ValueError(f"Need at least 2*max(periods)={2 * max_p} points, got {x.size}")

    decomp = MSTL(
        x,
        periods=periods_tup,
        iterate=int(iterate),
        lmbda=lmbda,
    ).fit()

    seasonal = np.asarray(decomp.seasonal, dtype=float)
    if seasonal.ndim == 1:
        seasonal = seasonal.reshape(-1, 1)
    if seasonal.shape[0] != x.size:
        raise ValueError("Unexpected MSTL seasonal shape")
    if seasonal.shape[1] != len(periods_tup):
        raise ValueError("MSTL returned unexpected number of seasonal components")

    seasonal_sum = np.sum(seasonal, axis=1)
    y_adj = x - seasonal_sum

    model = ARIMA(y_adj, order=tuple(int(v) for v in order))
    res = model.fit()
    fc_adj = np.asarray(res.forecast(steps=int(horizon)), dtype=float)

    seasonal_fc = np.zeros((int(horizon),), dtype=float)
    for j, p in enumerate(periods_tup):
        seasonal_fc += _repeat_last_season(seasonal[:, j], period=int(p), horizon=int(horizon))

    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def mstl_auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    iterate: int = 2,
    lmbda: float | str | None = None,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    information_criterion: str = "aic",
) -> np.ndarray:
    """
    MSTL decomposition + AutoARIMA-style grid search on the seasonally-adjusted series.
    """
    try:
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_auto_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 10:
        raise ValueError("mstl_auto_arima_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError("periods must contain integers >= 2")
    max_period = int(max(periods_tup))
    if x.size < 2 * max_period:
        raise ValueError(f"Need at least 2*max(periods)={2 * max_period} points, got {x.size}")

    decomp = MSTL(
        x,
        periods=periods_tup,
        iterate=int(iterate),
        lmbda=lmbda,
    ).fit()

    seasonal = np.asarray(decomp.seasonal, dtype=float)
    if seasonal.ndim == 1:
        seasonal = seasonal.reshape(-1, 1)
    if seasonal.shape[1] != len(periods_tup):
        raise ValueError("MSTL returned unexpected number of seasonal components")

    seasonal_sum = np.sum(seasonal, axis=1)
    y_adj = x - seasonal_sum

    fc_adj = auto_arima_forecast(
        y_adj,
        horizon,
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
        information_criterion=str(information_criterion),
    )

    seasonal_fc = np.zeros((int(horizon),), dtype=float)
    for j, p in enumerate(periods_tup):
        seasonal_fc += _repeat_last_season(seasonal[:, j], period=int(p), horizon=int(horizon))

    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def _boxcox(x: np.ndarray, *, lmbda: float) -> np.ndarray:
    lam = float(lmbda)
    if np.any(x <= 0.0):
        raise ValueError("Box-Cox transform requires strictly positive values")
    if abs(lam) < 1e-12:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _inv_boxcox(y: np.ndarray, *, lmbda: float) -> np.ndarray:
    lam = float(lmbda)
    if abs(lam) < 1e-12:
        return np.exp(y)
    return np.power(lam * y + 1.0, 1.0 / lam)


def tbats_lite_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    arima_order: tuple[int, int, int] = (1, 0, 0),
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + ARIMA errors (optional Box-Cox).

    This is not a full TBATS implementation (no damping, no full ARMA error structure tuning),
    but captures the common "Fourier seasonality + ARIMA residuals" idea.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'tbats_lite_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 10:
        raise ValueError("tbats_lite_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError("periods must contain integers >= 2")

    # Normalize orders to match number of periods.
    if isinstance(orders, int | float):
        orders_tup = tuple([int(orders)] * len(periods_tup))
    elif isinstance(orders, str):
        parts = [p.strip() for p in str(orders).split(",") if p.strip()]
        if len(parts) == 1 and len(periods_tup) > 1:
            orders_tup = tuple([int(parts[0])] * len(periods_tup))
        else:
            if len(parts) != len(periods_tup):
                raise ValueError("orders must be an int or have the same length as periods")
            orders_tup = tuple(int(p) for p in parts)
    elif isinstance(orders, list | tuple):
        if len(orders) == 1 and len(periods_tup) > 1:
            orders_tup = tuple([int(orders[0])] * len(periods_tup))
        else:
            if len(orders) != len(periods_tup):
                raise ValueError("orders must be an int or have the same length as periods")
            orders_tup = tuple(int(o) for o in orders)
    else:
        orders_tup = tuple([int(orders)] * len(periods_tup))

    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("All orders must be >= 0")

    n = int(x_work.size)
    t = np.arange(n, dtype=float)

    cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        cols.append(t)

    for period, order in zip(periods_tup, orders_tup, strict=True):
        w = 2.0 * np.pi / float(period)
        for k in range(1, int(order) + 1):
            cols.append(np.sin(w * float(k) * t))
            cols.append(np.cos(w * float(k) * t))

    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    resid_model = ARIMA(resid, order=tuple(int(v) for v in arima_order))
    resid_res = resid_model.fit()
    resid_fc = np.asarray(resid_res.forecast(steps=int(horizon)), dtype=float)

    tf = np.arange(n, n + int(horizon), dtype=float)
    cols_f: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        cols_f.append(tf)
    for period, order in zip(periods_tup, orders_tup, strict=True):
        w = 2.0 * np.pi / float(period)
        for k in range(1, int(order) + 1):
            cols_f.append(np.sin(w * float(k) * tf))
            cols_f.append(np.cos(w * float(k) * tf))

    Xf = np.stack(cols_f, axis=1)
    base_fc = Xf @ coef
    yhat_work = np.asarray(base_fc + resid_fc, dtype=float)

    if boxcox_lambda is not None:
        yhat = _inv_boxcox(yhat_work, lmbda=float(boxcox_lambda))
        return np.asarray(yhat, dtype=float)

    return yhat_work


def ets_forecast(
    train: Any,
    horizon: int,
    *,
    trend: str | None = "add",
    seasonal: str | None = "add",
    seasonal_periods: int | None = 12,
    damped_trend: bool = False,
) -> np.ndarray:
    """
    ETS / Holt-Winters-style exponential smoothing via statsmodels (optional dependency).

    This uses statsmodels' ExponentialSmoothing implementation and lets statsmodels
    optimize smoothing parameters.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'ets_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("ets_forecast requires at least 3 training points")

    trend_final = (
        None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    )
    seasonal_final = (
        None
        if (seasonal is None or str(seasonal).lower() in {"none", "null", ""})
        else str(seasonal)
    )
    seasonal_periods_final = (
        None
        if seasonal_final is None
        else (None if seasonal_periods is None else int(seasonal_periods))
    )

    model = ExponentialSmoothing(
        x,
        trend=trend_final,
        damped_trend=bool(damped_trend),
        seasonal=seasonal_final,
        seasonal_periods=seasonal_periods_final,
    )
    res = model.fit(optimized=True)
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)
