from __future__ import annotations

from typing import Any

import numpy as np

HORIZON_MUST_BE_AT_LEAST_ONE = "horizon must be >= 1"
TRAIN_EXOG_ROWS_MUST_MATCH_TRAIN = "train_exog must have the same number of rows as train"
FUTURE_EXOG_ROWS_MUST_MATCH_HORIZON = "future_exog must have horizon rows"
EXOG_INPUTS_MUST_BE_PAIRED = (
    "train_exog and future_exog must either both be provided or both be omitted"
)
LAGS_MUST_BE_NON_NEGATIVE = "lags must be >= 0"
LOCAL_LEVEL = "local level"
PERIOD_MUST_BE_AT_LEAST_TWO = "period must be >= 2"
FOURIER_ORDERS_MUST_MATCH_PERIODS = "orders must be an int or have the same length as periods"
FOURIER_PERIODS_MUST_BE_VALID = "periods must contain integers >= 2"


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")
    return arr


def _validate_positive_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError(HORIZON_MUST_BE_AT_LEAST_ONE)


def _validate_non_negative_lags(lags: int) -> None:
    if int(lags) < 0:
        raise ValueError(LAGS_MUST_BE_NON_NEGATIVE)


def _validate_period_at_least_two(period: int) -> None:
    if int(period) <= 1:
        raise ValueError(PERIOD_MUST_BE_AT_LEAST_TWO)


def _validate_exog_pair(
    *,
    train_size: int,
    horizon: int,
    train_exog: Any | None,
    future_exog: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    exog_train = None
    exog_future = None
    if train_exog is not None:
        exog_train = _as_2d_float_array(train_exog, name="train_exog")
        if int(exog_train.shape[0]) != int(train_size):
            raise ValueError(TRAIN_EXOG_ROWS_MUST_MATCH_TRAIN)
    if future_exog is not None:
        exog_future = _as_2d_float_array(future_exog, name="future_exog")
        if int(exog_future.shape[0]) != int(horizon):
            raise ValueError(FUTURE_EXOG_ROWS_MUST_MATCH_HORIZON)
    if (exog_train is None) != (exog_future is None):
        raise ValueError(EXOG_INPUTS_MUST_BE_PAIRED)
    return exog_train, exog_future


def _normalize_interval_levels(interval_levels: Any) -> tuple[float, ...]:
    if isinstance(interval_levels, list | tuple):
        items = list(interval_levels)
    else:
        items = [interval_levels]

    out: list[float] = []
    for item in items:
        level = float(item)
        if level >= 1.0:
            level = level / 100.0
        if not (0.0 < level < 1.0):
            raise ValueError("interval_levels must be in (0,1) or percentages like 80,90")
        out.append(level)
    return tuple(sorted(set(out)))


def _forecast_with_interval_levels(
    result: Any,
    *,
    horizon: int,
    interval_levels: Any,
    future_exog: np.ndarray | None = None,
) -> dict[str, Any]:
    levels = _normalize_interval_levels(interval_levels)
    if not levels:
        raise ValueError("interval_levels must be non-empty")

    if future_exog is None:
        pred_res = result.get_forecast(steps=int(horizon))
    else:
        pred_res = result.get_forecast(steps=int(horizon), exog=future_exog)

    mean = np.asarray(pred_res.predicted_mean, dtype=float)
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for level in levels:
        ci = np.asarray(pred_res.conf_int(alpha=1.0 - float(level)), dtype=float)
        if ci.ndim != 2 or int(ci.shape[0]) != int(horizon) or int(ci.shape[1]) < 2:
            raise ValueError(
                f"forecast conf_int must have shape ({int(horizon)}, 2), got {tuple(ci.shape)}"
            )
        intervals[float(level)] = (ci[:, 0].astype(float), ci[:, 1].astype(float))
    return {"mean": mean.astype(float), "intervals": intervals}


def _fit_sarimax_result(
    train: Any,
    horizon: int,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
) -> tuple[Any, np.ndarray | None]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'sarimax_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("sarimax_forecast requires at least 3 training points")

    exog_train, exog_future = _validate_exog_pair(
        train_size=int(x.size),
        horizon=int(horizon),
        train_exog=train_exog,
        future_exog=future_exog,
    )

    model = SARIMAX(
        x,
        exog=exog_train,
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )
    return model.fit(disp=False), exog_future


def _fit_auto_arima_best_result(
    train: Any,
    horizon: int,
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 0,
    max_D: int = 0,
    max_Q: int = 0,
    seasonal_period: int | None = None,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
    information_criterion: str = "aic",
) -> tuple[Any, np.ndarray | None]:
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'auto_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 5:
        raise ValueError("auto_arima_forecast requires at least 5 training points")
    if int(max_p) < 0 or int(max_d) < 0 or int(max_q) < 0:
        raise ValueError("max_p/max_d/max_q must be >= 0")
    if int(max_P) < 0 or int(max_D) < 0 or int(max_Q) < 0:
        raise ValueError("max_P/max_D/max_Q must be >= 0")

    exog_train, exog_future = _validate_exog_pair(
        train_size=int(x.size),
        horizon=int(horizon),
        train_exog=train_exog,
        future_exog=future_exog,
    )

    seasonal_period_int = (
        None
        if seasonal_period is None or str(seasonal_period).strip().lower() in {"none", "null", ""}
        else int(seasonal_period)
    )
    if seasonal_period_int is None and any(int(v) > 0 for v in (max_P, max_D, max_Q)):
        raise ValueError("seasonal_period must be provided when max_P/max_D/max_Q are non-zero")
    if seasonal_period_int is not None and seasonal_period_int <= 1:
        raise ValueError("seasonal_period must be at least 2")

    ic_key = str(information_criterion).strip().lower()
    if ic_key not in {"aic", "bic"}:
        raise ValueError("information_criterion must be 'aic' or 'bic'")

    trend_final = None if trend is None or str(trend).lower() in {"none", "null", ""} else str(trend)
    allow_zero_order = exog_train is not None or trend_final not in {None, "n"}

    best_ic = float("inf")
    best_res = None

    for p in range(0, int(max_p) + 1):
        for d in range(0, int(max_d) + 1):
            for q in range(0, int(max_q) + 1):
                seasonal_grid = (
                    [(0, 0, 0, 0)]
                    if seasonal_period_int is None
                    else [
                        (P, D, Q, seasonal_period_int)
                        for P in range(0, int(max_P) + 1)
                        for D in range(0, int(max_D) + 1)
                        for Q in range(0, int(max_Q) + 1)
                    ]
                )
                for P, D, Q, s in seasonal_grid:
                    if (
                        p == 0
                        and d == 0
                        and q == 0
                        and P == 0
                        and D == 0
                        and Q == 0
                        and not allow_zero_order
                    ):
                        continue
                    try:
                        res = ARIMA(
                            x,
                            exog=exog_train,
                            order=(int(p), int(d), int(q)),
                            seasonal_order=(int(P), int(D), int(Q), int(s)),
                            trend=trend_final,
                            enforce_stationarity=bool(enforce_stationarity),
                            enforce_invertibility=bool(enforce_invertibility),
                        ).fit()
                    except Exception:  # noqa: BLE001
                        continue
                    ic = float(getattr(res, ic_key))
                    if ic < best_ic:
                        best_ic = ic
                        best_res = res

    if best_res is None:
        raise ValueError("auto_arima_forecast failed to fit any ARIMA model in the grid")
    return best_res, exog_future


def arima_forecast(
    train: Any,
    horizon: int,
    *,
    order: tuple[int, int, int],
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
) -> np.ndarray:
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
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("arima_forecast requires at least 3 training points")

    exog_train, exog_future = _validate_exog_pair(
        train_size=int(x.size),
        horizon=int(horizon),
        train_exog=train_exog,
        future_exog=future_exog,
    )

    p, d, q = map(int, order)
    model = ARIMA(
        x,
        exog=exog_train,
        order=(p, d, q),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )
    res = model.fit()
    if exog_future is None:
        fc = res.forecast(steps=int(horizon))
    else:
        fc = res.forecast(steps=int(horizon), exog=exog_future)
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
    train_exog: Any | None = None,
    future_exog: Any | None = None,
) -> np.ndarray:
    """
    SARIMAX / seasonal ARIMA via statsmodels (optional dependency).
    """
    res, exog_future = _fit_sarimax_result(
        train,
        horizon,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        train_exog=train_exog,
        future_exog=future_exog,
    )
    if exog_future is None:
        fc = res.forecast(steps=int(horizon))
    else:
        fc = res.forecast(steps=int(horizon), exog=exog_future)
    return np.asarray(fc, dtype=float)


def sarimax_forecast_with_intervals(
    train: Any,
    horizon: int,
    *,
    interval_levels: Any,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
) -> dict[str, Any]:
    res, exog_future = _fit_sarimax_result(
        train,
        horizon,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        train_exog=train_exog,
        future_exog=future_exog,
    )
    return _forecast_with_interval_levels(
        res,
        horizon=int(horizon),
        interval_levels=interval_levels,
        future_exog=exog_future,
    )


def autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 12,
    trend: str = "c",
    seasonal: bool = False,
    period: int | None = None,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
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
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("autoreg_forecast requires at least 3 training points")
    _validate_non_negative_lags(lags)

    exog_train, exog_future = _validate_exog_pair(
        train_size=int(x.size),
        horizon=int(horizon),
        train_exog=train_exog,
        future_exog=future_exog,
    )

    model = AutoReg(
        x,
        lags=int(lags),
        trend=str(trend),
        seasonal=bool(seasonal),
        exog=exog_train,
        period=(None if period is None else int(period)),
        old_names=False,
    )
    res = model.fit()
    if exog_future is None:
        fc = res.forecast(steps=int(horizon))
    else:
        fc = res.forecast(steps=int(horizon), exog=exog_future)
    return np.asarray(fc, dtype=float)


def unobserved_components_forecast(
    train: Any,
    horizon: int,
    *,
    level: str = LOCAL_LEVEL,
    seasonal: int | None = None,
) -> np.ndarray:
    """
    Structural / state-space model via statsmodels UnobservedComponents (optional dependency).

    Common `level` strings:
      - the default structural level setting
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
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("unobserved_components_forecast requires at least 3 training points")
    seasonal_int = None if seasonal is None else int(seasonal)
    if seasonal_int is not None and seasonal_int <= 1:
        raise ValueError("seasonal must be >= 2")

    model_kwargs: dict[str, Any] = {"level": str(level)}
    if seasonal_int is not None:
        model_kwargs["seasonal"] = seasonal_int

    model = UnobservedComponents(x, **model_kwargs)
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
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("stl_arima_forecast requires at least 3 training points")
    _validate_period_at_least_two(period)

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


def stl_ets_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    trend: str | None = "add",
    damped_trend: bool = False,
    robust: bool = False,
) -> np.ndarray:
    """
    STL + ETS remainder forecasting via statsmodels STLForecast (optional dependency).
    """
    try:
        from statsmodels.tsa.forecasting.stl import STLForecast  # type: ignore
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_ets_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("stl_ets_forecast requires at least 3 training points")
    _validate_period_at_least_two(period)

    trend_final = None if trend is None or str(trend).lower() in {"none", "null", ""} else str(trend)
    stlf = STLForecast(
        x,
        ExponentialSmoothing,
        model_kwargs={
            "trend": trend_final,
            "damped_trend": bool(damped_trend),
            "seasonal": None,
        },
        period=int(period),
        robust=bool(robust),
    )
    res = stlf.fit()
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def stl_autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    lags: int = 1,
    trend: str = "c",
    seasonal: int = 7,
    robust: bool = False,
) -> np.ndarray:
    """
    STL + AutoReg remainder forecasting via statsmodels STLForecast (optional dependency).
    """
    try:
        from statsmodels.tsa.ar_model import AutoReg  # type: ignore
        from statsmodels.tsa.forecasting.stl import STLForecast  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_autoreg_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("stl_autoreg_forecast requires at least 3 training points")
    _validate_period_at_least_two(period)
    _validate_non_negative_lags(lags)

    stlf = STLForecast(
        x,
        AutoReg,
        model_kwargs={
            "lags": int(lags),
            "trend": str(trend),
            "old_names": False,
        },
        period=int(period),
        seasonal=int(seasonal),
        robust=bool(robust),
    )
    res = stlf.fit()
    fc = res.forecast(steps=int(horizon))
    return np.asarray(fc, dtype=float)


def stl_uc_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    level: str = LOCAL_LEVEL,
    seasonal: int = 7,
    robust: bool = False,
) -> np.ndarray:
    """
    STL decomposition + UnobservedComponents on the seasonally-adjusted series.

    Seasonal forecast is produced by repeating the last estimated seasonal cycle.
    """
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_uc_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("stl_uc_forecast requires at least 3 training points")
    _validate_period_at_least_two(period)

    decomp = STL(
        x,
        period=int(period),
        seasonal=int(seasonal),
        robust=bool(robust),
    ).fit()
    seasonal_component = np.asarray(decomp.seasonal, dtype=float)
    y_adj = x - seasonal_component

    fc_adj = unobserved_components_forecast(
        y_adj,
        int(horizon),
        level=str(level),
        seasonal=None,
    )
    seasonal_fc = _repeat_last_season(
        seasonal_component,
        period=int(period),
        horizon=int(horizon),
    )
    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def stl_sarimax_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    seasonal: int = 7,
    robust: bool = False,
) -> np.ndarray:
    """
    STL decomposition + SARIMAX on the seasonally-adjusted series.

    Seasonal forecast is produced by repeating the last estimated seasonal cycle.
    """
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_sarimax_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("stl_sarimax_forecast requires at least 3 training points")
    _validate_period_at_least_two(period)

    decomp = STL(
        x,
        period=int(period),
        seasonal=int(seasonal),
        robust=bool(robust),
    ).fit()
    seasonal_component = np.asarray(decomp.seasonal, dtype=float)
    y_adj = x - seasonal_component

    fc_adj = sarimax_forecast(
        y_adj,
        int(horizon),
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )
    seasonal_fc = _repeat_last_season(
        seasonal_component,
        period=int(period),
        horizon=int(horizon),
    )
    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def stl_auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    period: int,
    seasonal: int = 7,
    robust: bool = False,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
) -> np.ndarray:
    """
    STL decomposition + AutoARIMA-style grid search on the seasonally-adjusted series.
    """
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'stl_auto_arima_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 5:
        raise ValueError("stl_auto_arima_forecast requires at least 5 training points")
    _validate_period_at_least_two(period)

    decomp = STL(
        x,
        period=int(period),
        seasonal=int(seasonal),
        robust=bool(robust),
    ).fit()
    seasonal_component = np.asarray(decomp.seasonal, dtype=float)
    y_adj = x - seasonal_component

    fc_adj = auto_arima_forecast(
        y_adj,
        int(horizon),
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
        information_criterion=str(information_criterion),
    )
    seasonal_fc = _repeat_last_season(
        seasonal_component,
        period=int(period),
        horizon=int(horizon),
    )
    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 0,
    max_D: int = 0,
    max_Q: int = 0,
    seasonal_period: int | None = None,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
    information_criterion: str = "aic",
) -> np.ndarray:
    """
    Lightweight AutoARIMA-style grid search via statsmodels ARIMA (optional dependency).

    This tries orders in a small (p,d,q) grid, optionally with seasonal
    (P,D,Q,s), and selects the best by AIC/BIC.
    """
    best_res, exog_future = _fit_auto_arima_best_result(
        train,
        horizon,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        max_P=max_P,
        max_D=max_D,
        max_Q=max_Q,
        seasonal_period=seasonal_period,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        train_exog=train_exog,
        future_exog=future_exog,
        information_criterion=information_criterion,
    )
    if exog_future is None:
        fc = best_res.forecast(steps=int(horizon))
    else:
        fc = best_res.forecast(steps=int(horizon), exog=exog_future)
    return np.asarray(fc, dtype=float)


def auto_arima_forecast_with_intervals(
    train: Any,
    horizon: int,
    *,
    interval_levels: Any,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 0,
    max_D: int = 0,
    max_Q: int = 0,
    seasonal_period: int | None = None,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    train_exog: Any | None = None,
    future_exog: Any | None = None,
    information_criterion: str = "aic",
) -> dict[str, Any]:
    best_res, exog_future = _fit_auto_arima_best_result(
        train,
        horizon,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        max_P=max_P,
        max_D=max_D,
        max_Q=max_Q,
        seasonal_period=seasonal_period,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        train_exog=train_exog,
        future_exog=future_exog,
        information_criterion=information_criterion,
    )
    return _forecast_with_interval_levels(
        best_res,
        horizon=int(horizon),
        interval_levels=interval_levels,
        future_exog=exog_future,
    )


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


def _normalize_valid_periods(periods: Any) -> tuple[int, ...]:
    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
    return tuple(int(p) for p in periods_tup)


def _normalize_fourier_orders(orders: Any, *, n_periods: int) -> tuple[int, ...]:
    if orders is None:
        return tuple([2] * int(n_periods))
    if isinstance(orders, int | float):
        return tuple([int(orders)] * int(n_periods))
    if isinstance(orders, str):
        s = orders.strip()
        if not s:
            return tuple([2] * int(n_periods))
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 1 and int(n_periods) > 1:
            return tuple([int(parts[0])] * int(n_periods))
        if len(parts) != int(n_periods):
            raise ValueError(FOURIER_ORDERS_MUST_MATCH_PERIODS)
        return tuple(int(p) for p in parts)
    if isinstance(orders, list | tuple):
        if len(orders) == 1 and int(n_periods) > 1:
            return tuple([int(orders[0])] * int(n_periods))
        if len(orders) != int(n_periods):
            raise ValueError(FOURIER_ORDERS_MUST_MATCH_PERIODS)
        return tuple(int(o) for o in orders)
    return tuple([int(orders)] * int(n_periods))


def _build_fourier_exog(
    *,
    start: int,
    steps: int,
    periods: tuple[int, ...],
    orders: tuple[int, ...],
) -> np.ndarray | None:
    t = np.arange(int(start), int(start) + int(steps), dtype=float)
    cols: list[np.ndarray] = []
    for period, order in zip(periods, orders, strict=True):
        w = 2.0 * np.pi / float(period)
        for k in range(1, int(order) + 1):
            cols.append(np.sin(w * float(k) * t))
            cols.append(np.cos(w * float(k) * t))
    if not cols:
        return None
    return np.stack(cols, axis=1)


def fourier_auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + AutoARIMA residual search.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 5:
        raise ValueError("fourier_auto_arima_forecast requires at least 5 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    return auto_arima_forecast(
        x,
        int(horizon),
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
        train_exog=train_exog,
        future_exog=future_exog,
        information_criterion=str(information_criterion),
    )


def fourier_arima_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    order: tuple[int, int, int] = (1, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + fixed-order ARIMA errors.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 5:
        raise ValueError("fourier_arima_forecast requires at least 5 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    return arima_forecast(
        x,
        int(horizon),
        order=tuple(int(v) for v in order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
        train_exog=train_exog,
        future_exog=future_exog,
    )


def fourier_sarimax_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + fixed-order SARIMAX errors.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 5:
        raise ValueError("fourier_sarimax_forecast requires at least 5 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    return sarimax_forecast(
        x,
        int(horizon),
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
        train_exog=train_exog,
        future_exog=future_exog,
    )


def fourier_ets_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    trend: str | None = None,
    damped_trend: bool = False,
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + ETS residuals.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'fourier_ets_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("fourier_ets_forecast requires at least 3 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    X_cols: list[np.ndarray] = [np.ones((int(x.size),), dtype=float)]
    if train_exog is not None:
        X_cols.extend([train_exog[:, j] for j in range(train_exog.shape[1])])
    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    fitted = X @ coef
    resid = x - fitted

    trend_final = None if trend is None or str(trend).lower() in {"none", "null", ""} else str(trend)
    resid_model = ExponentialSmoothing(
        resid,
        trend=trend_final,
        damped_trend=bool(damped_trend),
        seasonal=None,
    )
    resid_res = resid_model.fit()
    resid_fc = np.asarray(resid_res.forecast(steps=int(horizon)), dtype=float)

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if future_exog is not None:
        Xf_cols.extend([future_exog[:, j] for j in range(future_exog.shape[1])])
    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    return np.asarray(base_fc + resid_fc, dtype=float)


def fourier_uc_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    level: str = LOCAL_LEVEL,
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + UnobservedComponents residuals.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("fourier_uc_forecast requires at least 3 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    X_cols: list[np.ndarray] = [np.ones((int(x.size),), dtype=float)]
    if train_exog is not None:
        X_cols.extend([train_exog[:, j] for j in range(train_exog.shape[1])])
    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    fitted = X @ coef
    resid = x - fitted

    resid_fc = unobserved_components_forecast(
        resid,
        int(horizon),
        level=str(level),
        seasonal=None,
    )

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if future_exog is not None:
        Xf_cols.extend([future_exog[:, j] for j in range(future_exog.shape[1])])
    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    return np.asarray(base_fc + resid_fc, dtype=float)


def fourier_autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    lags: int = 0,
    trend: str = "c",
) -> np.ndarray:
    """
    Dynamic harmonic regression: Fourier seasonal terms + AutoReg / AR-X errors.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 3:
        raise ValueError("fourier_autoreg_forecast requires at least 3 training points")

    periods_tup = _normalize_valid_periods(periods)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")
    orders_tup = tuple(int(o) for o in orders_tup)

    train_exog = _build_fourier_exog(
        start=0,
        steps=int(x.size),
        periods=periods_tup,
        orders=orders_tup,
    )
    future_exog = _build_fourier_exog(
        start=int(x.size),
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )

    return autoreg_forecast(
        x,
        int(horizon),
        lags=int(lags),
        trend=str(trend),
        seasonal=False,
        period=None,
        train_exog=train_exog,
        future_exog=future_exog,
    )


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
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_arima_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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


def mstl_autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    lags: int = 1,
    trend: str = "c",
    iterate: int = 2,
    lmbda: float | str | None = None,
) -> np.ndarray:
    """
    MSTL (multi-seasonal STL) decomposition + AutoReg on the seasonally-adjusted series.

    Seasonal forecast is a simple repetition of the last seasonal cycle(s).
    """
    try:
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_autoreg_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_autoreg_forecast requires at least 10 training points")
    _validate_non_negative_lags(lags)

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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
    fc_adj = autoreg_forecast(
        y_adj,
        int(horizon),
        lags=int(lags),
        trend=str(trend),
        seasonal=False,
        period=None,
    )

    seasonal_fc = np.zeros((int(horizon),), dtype=float)
    for j, p in enumerate(periods_tup):
        seasonal_fc += _repeat_last_season(seasonal[:, j], period=int(p), horizon=int(horizon))

    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def mstl_ets_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    trend: str | None = "add",
    damped_trend: bool = False,
    iterate: int = 2,
    lmbda: float | str | None = None,
) -> np.ndarray:
    """
    MSTL (multi-seasonal STL) decomposition + ETS on the seasonally-adjusted series.

    Seasonal forecast is a simple repetition of the last seasonal cycle(s).
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_ets_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_ets_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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

    trend_final = None if trend is None or str(trend).lower() in {"none", "null", ""} else str(trend)
    model = ExponentialSmoothing(
        y_adj,
        trend=trend_final,
        damped_trend=bool(damped_trend),
        seasonal=None,
    )
    res = model.fit()
    fc_adj = np.asarray(res.forecast(steps=int(horizon)), dtype=float)

    seasonal_fc = np.zeros((int(horizon),), dtype=float)
    for j, p in enumerate(periods_tup):
        seasonal_fc += _repeat_last_season(seasonal[:, j], period=int(p), horizon=int(horizon))

    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def mstl_uc_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    level: str = LOCAL_LEVEL,
    iterate: int = 2,
    lmbda: float | str | None = None,
) -> np.ndarray:
    """
    MSTL (multi-seasonal STL) decomposition + UnobservedComponents on the adjusted series.

    Seasonal forecast is a simple repetition of the last seasonal cycle(s).
    """
    try:
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_uc_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_uc_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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
    if seasonal.shape[0] != x.size:
        raise ValueError("Unexpected MSTL seasonal shape")
    if seasonal.shape[1] != len(periods_tup):
        raise ValueError("MSTL returned unexpected number of seasonal components")

    seasonal_sum = np.sum(seasonal, axis=1)
    y_adj = x - seasonal_sum

    fc_adj = unobserved_components_forecast(
        y_adj,
        int(horizon),
        level=str(level),
        seasonal=None,
    )

    seasonal_fc = np.zeros((int(horizon),), dtype=float)
    for j, p in enumerate(periods_tup):
        seasonal_fc += _repeat_last_season(seasonal[:, j], period=int(p), horizon=int(horizon))

    return np.asarray(fc_adj + seasonal_fc, dtype=float)


def mstl_sarimax_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    iterate: int = 2,
    lmbda: float | str | None = None,
) -> np.ndarray:
    """
    MSTL (multi-seasonal STL) decomposition + SARIMAX on the seasonally-adjusted series.

    Seasonal forecast is a simple repetition of the last seasonal cycle(s).
    """
    try:
        from statsmodels.tsa.seasonal import MSTL  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mstl_sarimax_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_sarimax_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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

    fc_adj = sarimax_forecast(
        y_adj,
        int(horizon),
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )

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
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
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
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("mstl_auto_arima_forecast requires at least 10 training points")

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)
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
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
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
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))

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


def tbats_lite_autoreg_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    lags: int = 1,
    trend: str = "n",
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + AutoReg errors (optional Box-Cox).
    """
    try:
        from statsmodels.tsa.ar_model import AutoReg  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'tbats_lite_autoreg_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_autoreg_forecast requires at least 10 training points")
    _validate_non_negative_lags(lags)

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")

    n = int(x_work.size)
    X_cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        X_cols.append(np.arange(n, dtype=float))
    fourier_train = _build_fourier_exog(
        start=0,
        steps=n,
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_train is not None:
        X_cols.extend([fourier_train[:, j] for j in range(fourier_train.shape[1])])

    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    resid_model = AutoReg(
        resid,
        lags=int(lags),
        trend=str(trend),
        seasonal=False,
        old_names=False,
    )
    resid_res = resid_model.fit()
    resid_fc = np.asarray(resid_res.forecast(steps=int(horizon)), dtype=float)

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        Xf_cols.append(np.arange(n, n + int(horizon), dtype=float))
    fourier_future = _build_fourier_exog(
        start=n,
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_future is not None:
        Xf_cols.extend([fourier_future[:, j] for j in range(fourier_future.shape[1])])

    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    yhat_work = np.asarray(base_fc + resid_fc, dtype=float)

    if boxcox_lambda is not None:
        yhat = _inv_boxcox(yhat_work, lmbda=float(boxcox_lambda))
        return np.asarray(yhat, dtype=float)

    return yhat_work


def tbats_lite_ets_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    trend: str | None = None,
    damped_trend: bool = False,
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + ETS residuals (optional Box-Cox).
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'tbats_lite_ets_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_ets_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")

    n = int(x_work.size)
    X_cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        X_cols.append(np.arange(n, dtype=float))
    fourier_train = _build_fourier_exog(
        start=0,
        steps=n,
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_train is not None:
        X_cols.extend([fourier_train[:, j] for j in range(fourier_train.shape[1])])

    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    trend_final = None if trend is None or str(trend).lower() in {"none", "null", ""} else str(trend)
    resid_model = ExponentialSmoothing(
        resid,
        trend=trend_final,
        damped_trend=bool(damped_trend),
        seasonal=None,
    )
    resid_res = resid_model.fit()
    resid_fc = np.asarray(resid_res.forecast(steps=int(horizon)), dtype=float)

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        Xf_cols.append(np.arange(n, n + int(horizon), dtype=float))
    fourier_future = _build_fourier_exog(
        start=n,
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_future is not None:
        Xf_cols.extend([fourier_future[:, j] for j in range(fourier_future.shape[1])])

    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    yhat_work = np.asarray(base_fc + resid_fc, dtype=float)

    if boxcox_lambda is not None:
        yhat = _inv_boxcox(yhat_work, lmbda=float(boxcox_lambda))
        return np.asarray(yhat, dtype=float)

    return yhat_work


def tbats_lite_sarimax_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + SARIMAX residuals (optional Box-Cox).
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_sarimax_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")

    n = int(x_work.size)
    X_cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        X_cols.append(np.arange(n, dtype=float))
    fourier_train = _build_fourier_exog(
        start=0,
        steps=n,
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_train is not None:
        X_cols.extend([fourier_train[:, j] for j in range(fourier_train.shape[1])])

    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    resid_fc = sarimax_forecast(
        resid,
        int(horizon),
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
    )

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        Xf_cols.append(np.arange(n, n + int(horizon), dtype=float))
    fourier_future = _build_fourier_exog(
        start=n,
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_future is not None:
        Xf_cols.extend([fourier_future[:, j] for j in range(fourier_future.shape[1])])

    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    yhat_work = np.asarray(base_fc + resid_fc, dtype=float)

    if boxcox_lambda is not None:
        yhat = _inv_boxcox(yhat_work, lmbda=float(boxcox_lambda))
        return np.asarray(yhat, dtype=float)

    return yhat_work


def tbats_lite_auto_arima_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = "c",
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + AutoARIMA residual search (optional Box-Cox).
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_auto_arima_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")

    n = int(x_work.size)
    X_cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        X_cols.append(np.arange(n, dtype=float))
    fourier_train = _build_fourier_exog(
        start=0,
        steps=n,
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_train is not None:
        X_cols.extend([fourier_train[:, j] for j in range(fourier_train.shape[1])])

    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    resid_fc = auto_arima_forecast(
        resid,
        int(horizon),
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
        trend=trend,
        enforce_stationarity=bool(enforce_stationarity),
        enforce_invertibility=bool(enforce_invertibility),
        information_criterion=str(information_criterion),
    )

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        Xf_cols.append(np.arange(n, n + int(horizon), dtype=float))
    fourier_future = _build_fourier_exog(
        start=n,
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_future is not None:
        Xf_cols.extend([fourier_future[:, j] for j in range(fourier_future.shape[1])])

    Xf = np.stack(Xf_cols, axis=1)
    base_fc = Xf @ coef
    yhat_work = np.asarray(base_fc + resid_fc, dtype=float)

    if boxcox_lambda is not None:
        yhat = _inv_boxcox(yhat_work, lmbda=float(boxcox_lambda))
        return np.asarray(yhat, dtype=float)

    return yhat_work


def tbats_lite_uc_forecast(
    train: Any,
    horizon: int,
    *,
    periods: Any,
    orders: Any = 2,
    include_trend: bool = True,
    level: str = LOCAL_LEVEL,
    boxcox_lambda: float | None = None,
) -> np.ndarray:
    """
    TBATS-like baseline: multi-season Fourier terms + UnobservedComponents residuals.
    """
    x = _as_1d_float_array(train)
    _validate_positive_horizon(horizon)
    if x.size < 10:
        raise ValueError("tbats_lite_uc_forecast requires at least 10 training points")

    x_work = x
    if boxcox_lambda is not None:
        x_work = _boxcox(x_work, lmbda=float(boxcox_lambda))

    periods_tup = _normalize_periods(periods)
    if not periods_tup or any(int(p) <= 1 for p in periods_tup):
        raise ValueError(FOURIER_PERIODS_MUST_BE_VALID)

    orders_tup = _normalize_fourier_orders(orders, n_periods=len(periods_tup))
    if any(int(o) < 0 for o in orders_tup):
        raise ValueError("orders must contain integers >= 0")

    n = int(x_work.size)
    X_cols: list[np.ndarray] = [np.ones((n,), dtype=float)]
    if bool(include_trend):
        X_cols.append(np.arange(n, dtype=float))
    fourier_train = _build_fourier_exog(
        start=0,
        steps=n,
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_train is not None:
        X_cols.extend([fourier_train[:, j] for j in range(fourier_train.shape[1])])

    X = np.stack(X_cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, x_work, rcond=None)
    fitted = X @ coef
    resid = x_work - fitted

    resid_fc = unobserved_components_forecast(
        resid,
        int(horizon),
        level=str(level),
        seasonal=None,
    )

    Xf_cols: list[np.ndarray] = [np.ones((int(horizon),), dtype=float)]
    if bool(include_trend):
        Xf_cols.append(np.arange(n, n + int(horizon), dtype=float))
    fourier_future = _build_fourier_exog(
        start=n,
        steps=int(horizon),
        periods=periods_tup,
        orders=orders_tup,
    )
    if fourier_future is not None:
        Xf_cols.extend([fourier_future[:, j] for j in range(fourier_future.shape[1])])

    Xf = np.stack(Xf_cols, axis=1)
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
    _validate_positive_horizon(horizon)
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
    seasonal_periods_final = None
    if seasonal_final is not None and seasonal_periods is not None:
        seasonal_periods_final = int(seasonal_periods)

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
