from __future__ import annotations

from typing import Any


def build_stats_catalog(context: Any) -> dict[str, Any]:
    model_spec = context.ModelSpec
    _factory_arima = context._factory_arima
    _factory_auto_arima = context._factory_auto_arima
    _factory_autoreg = context._factory_autoreg
    _factory_ets = context._factory_ets
    _factory_fourier_arima = context._factory_fourier_arima
    _factory_fourier_auto_arima = context._factory_fourier_auto_arima
    _factory_fourier_autoreg = context._factory_fourier_autoreg
    _factory_fourier_ets = context._factory_fourier_ets
    _factory_fourier_sarimax = context._factory_fourier_sarimax
    _factory_fourier_uc = context._factory_fourier_uc
    _factory_mstl_arima = context._factory_mstl_arima
    _factory_mstl_auto_arima = context._factory_mstl_auto_arima
    _factory_mstl_autoreg = context._factory_mstl_autoreg
    _factory_mstl_ets = context._factory_mstl_ets
    _factory_mstl_sarimax = context._factory_mstl_sarimax
    _factory_mstl_uc = context._factory_mstl_uc
    _factory_sarimax = context._factory_sarimax
    _factory_stl_arima = context._factory_stl_arima
    _factory_stl_auto_arima = context._factory_stl_auto_arima
    _factory_stl_autoreg = context._factory_stl_autoreg
    _factory_stl_ets = context._factory_stl_ets
    _factory_stl_sarimax = context._factory_stl_sarimax
    _factory_stl_uc = context._factory_stl_uc
    _factory_tbats_lite = context._factory_tbats_lite
    _factory_tbats_lite_auto_arima = context._factory_tbats_lite_auto_arima
    _factory_tbats_lite_autoreg = context._factory_tbats_lite_autoreg
    _factory_tbats_lite_ets = context._factory_tbats_lite_ets
    _factory_tbats_lite_sarimax = context._factory_tbats_lite_sarimax
    _factory_tbats_lite_uc = context._factory_tbats_lite_uc
    _factory_unobserved_components = context._factory_unobserved_components
    CANDIDATE_STATIONARITY_PARAM_HELP = (
        "Enforce stationarity for candidate models (true/false)"
    )
    CANDIDATE_INVERTIBILITY_PARAM_HELP = (
        "Enforce invertibility for candidate models (true/false)"
    )
    MODEL_SELECTION_CRITERION_PARAM_HELP = "Model selection criterion: aic or bic"
    FOURIER_PERIODS_PARAM_HELP = (
        "Comma-separated seasonal periods for Fourier terms (e.g. 7,365)"
    )
    FOURIER_ORDERS_PARAM_HELP = (
        "Fourier order per period (int or comma-separated list)"
    )
    LOCAL_LEVEL = "local level"
    STL_PERIOD_PARAM_HELP = "Seasonal period for STL"
    STL_SEASONAL_PARAM_HELP = (
        "STL seasonal smoother length (odd integer >= 3; default 7)"
    )
    ROBUST_STL_PARAM_HELP = "Robust STL (true/false)"
    MULTI_SEASONAL_PERIODS_PARAM_HELP = "Comma-separated seasonal periods (e.g. 7,365)"
    MSTL_ITERATIONS_PARAM_HELP = "MSTL iterations (default: 2)"
    MSTL_BOXCOX_PARAM_HELP = "Box-Cox lambda for MSTL (float, 'auto', or none)"
    INCLUDE_LINEAR_TREND_PARAM_HELP = "Include linear trend term (true/false)"
    OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP = (
        "Optional Box-Cox lambda (float); requires y > 0"
    )
    return {
    "ets": model_spec(
        key="ets",
        description="ETS (ExponentialSmoothing) via statsmodels. Optional dependency.",
        factory=_factory_ets,
        default_params={
            "season_length": 12,
            "trend": "add",
            "seasonal": "add",
            "damped_trend": False,
        },
        param_help={
            "season_length": "Season length (used as seasonal_periods when seasonal is not None)",
            "trend": "Trend component: add, mul, or none",
            "seasonal": "Seasonal component: add, mul, or none",
            "damped_trend": "Whether to use a damped trend (true/false)",
        },
        requires=("stats",),
        capability_overrides={"supports_interval_forecast_with_x_cols": True},
    ),
    "arima": model_spec(
        key="arima",
        description="ARIMA(p,d,q) via statsmodels. Optional dependency.",
        factory=_factory_arima,
        default_params={
            "order": (1, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        param_help={
            "order": "ARIMA order tuple (p,d,q)",
            "trend": "Trend term (e.g. n, c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity (true/false)",
            "enforce_invertibility": "Enforce invertibility (true/false)",
        },
        requires=("stats",),
        capability_overrides={"supports_interval_forecast_with_x_cols": True},
    ),
    "auto-arima": model_spec(
        key="auto-arima",
        description="AutoARIMA-style grid search via statsmodels. Optional dependency.",
        factory=_factory_auto_arima,
        default_params={
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "seasonal_period": None,
            "trend": None,
            "x_cols": (),
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "information_criterion": "aic",
        },
        param_help={
            "max_p": "Max AR order p to consider",
            "max_d": "Max differencing order d to consider",
            "max_q": "Max MA order q to consider",
            "max_P": "Max seasonal AR order P to consider",
            "max_D": "Max seasonal differencing order D to consider",
            "max_Q": "Max seasonal MA order Q to consider",
            "seasonal_period": "Optional seasonal period s for SARIMA-style search",
            "trend": "Trend term (e.g. n, c, t, ct) or none",
            "x_cols": "Optional future covariate columns for forecast_model_long_df / forecast csv",
            "enforce_stationarity": CANDIDATE_STATIONARITY_PARAM_HELP,
            "enforce_invertibility": CANDIDATE_INVERTIBILITY_PARAM_HELP,
            "information_criterion": MODEL_SELECTION_CRITERION_PARAM_HELP,
        },
        requires=("stats",),
        capability_overrides={"supports_interval_forecast_with_x_cols": True},
    ),
    "fourier-auto-arima": model_spec(
        key="fourier-auto-arima",
        description="Dynamic harmonic regression: Fourier seasonal terms + AutoARIMA errors. Optional dependency.",
        factory=_factory_fourier_auto_arima,
        default_params={
            "periods": (12,),
            "orders": 2,
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "information_criterion": "aic",
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "max_p": "Max AR order p to consider for the residual model",
            "max_d": "Max differencing order d to consider for the residual model",
            "max_q": "Max MA order q to consider for the residual model",
            "trend": "Trend term for the residual ARIMA search (e.g. n, c, t, ct) or none",
            "enforce_stationarity": CANDIDATE_STATIONARITY_PARAM_HELP,
            "enforce_invertibility": CANDIDATE_INVERTIBILITY_PARAM_HELP,
            "information_criterion": MODEL_SELECTION_CRITERION_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "fourier-arima": model_spec(
        key="fourier-arima",
        description="Dynamic harmonic regression: Fourier seasonal terms + fixed-order ARIMA errors. Optional dependency.",
        factory=_factory_fourier_arima,
        default_params={
            "periods": (12,),
            "orders": 2,
            "order": (1, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "order": "ARIMA order for the residual model (p,d,q)",
            "trend": "Trend term for the residual ARIMA model (e.g. n, c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity for the residual ARIMA model (true/false)",
            "enforce_invertibility": "Enforce invertibility for the residual ARIMA model (true/false)",
        },
        requires=("stats",),
    ),
    "fourier-sarimax": model_spec(
        key="fourier-sarimax",
        description="Dynamic harmonic regression: Fourier seasonal terms + fixed-order SARIMAX errors. Optional dependency.",
        factory=_factory_fourier_sarimax,
        default_params={
            "periods": (12,),
            "orders": 2,
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "order": "SARIMAX non-seasonal order for the residual model (p,d,q)",
            "seasonal_order": "SARIMAX seasonal order for the residual model (P,D,Q,s)",
            "trend": "Trend term for the residual SARIMAX model (e.g. n, c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity for the residual SARIMAX model (true/false)",
            "enforce_invertibility": "Enforce invertibility for the residual SARIMAX model (true/false)",
        },
        requires=("stats",),
    ),
    "fourier-autoreg": model_spec(
        key="fourier-autoreg",
        description="Dynamic harmonic regression: Fourier seasonal terms + AutoReg / AR-X errors. Optional dependency.",
        factory=_factory_fourier_autoreg,
        default_params={
            "periods": (12,),
            "orders": 2,
            "lags": 0,
            "trend": "c",
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "lags": "AutoReg lag order; 0 disables AR lags and uses deterministic regression only",
            "trend": "Deterministic trend: n, c, t, ct",
        },
        requires=("stats",),
    ),
    "fourier-ets": model_spec(
        key="fourier-ets",
        description="Dynamic harmonic regression: Fourier seasonal terms + ETS residuals. Optional dependency.",
        factory=_factory_fourier_ets,
        default_params={
            "periods": (12,),
            "orders": 2,
            "trend": None,
            "damped_trend": False,
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "trend": "Residual ETS trend: add, mul, or none",
            "damped_trend": "Use damped trend in the residual ETS model (true/false)",
        },
        requires=("stats",),
    ),
    "fourier-uc": model_spec(
        key="fourier-uc",
        description="Dynamic harmonic regression: Fourier seasonal terms + UnobservedComponents residuals. Optional dependency.",
        factory=_factory_fourier_uc,
        default_params={
            "periods": (12,),
            "orders": 2,
            "level": LOCAL_LEVEL,
        },
        param_help={
            "periods": FOURIER_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "level": "Residual structural level model (e.g. local level, local linear trend, random walk)",
        },
        requires=("stats",),
    ),
    "sarimax": model_spec(
        key="sarimax",
        description="SARIMAX / seasonal ARIMA via statsmodels. Optional dependency.",
        factory=_factory_sarimax,
        default_params={
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "x_cols": (),
        },
        param_help={
            "order": "Non-seasonal order (p,d,q)",
            "seasonal_order": "Seasonal order (P,D,Q,s)",
            "trend": "Trend term (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity (true/false)",
            "enforce_invertibility": "Enforce invertibility (true/false)",
            "x_cols": "Optional future covariate columns for forecast_model_long_df / forecast csv",
        },
        requires=("stats",),
        capability_overrides={"supports_interval_forecast_with_x_cols": True},
    ),
    "autoreg": model_spec(
        key="autoreg",
        description="AutoReg (AR) via statsmodels. Optional dependency.",
        factory=_factory_autoreg,
        default_params={"lags": 12, "trend": "c", "seasonal": False, "period": None},
        param_help={
            "lags": "AR lags",
            "trend": "Deterministic trend: n, c, t, ct",
            "seasonal": "Include seasonal dummies (true/false)",
            "period": "Season length when seasonal=true",
        },
        requires=("stats",),
    ),
    "uc-local-level": model_spec(
        key="uc-local-level",
        description="UnobservedComponents local level via statsmodels. Optional dependency.",
        factory=_factory_unobserved_components,
        default_params={"level": LOCAL_LEVEL},
        param_help={"level": "Level specification string (default: 'local level')"},
        requires=("stats",),
    ),
    "uc-local-linear-trend": model_spec(
        key="uc-local-linear-trend",
        description="UnobservedComponents local linear trend via statsmodels. Optional dependency.",
        factory=_factory_unobserved_components,
        default_params={"level": "local linear trend"},
        param_help={"level": "Level specification string (default: 'local linear trend')"},
        requires=("stats",),
    ),
    "uc-seasonal": model_spec(
        key="uc-seasonal",
        description="UnobservedComponents local level + seasonal component via statsmodels. Optional dependency.",
        factory=_factory_unobserved_components,
        default_params={"level": LOCAL_LEVEL, "seasonal": 12},
        param_help={
            "level": "Level specification string (default: 'local level')",
            "seasonal": "Seasonal cycle length for the structural seasonal component",
        },
        requires=("stats",),
    ),
    "stl-arima": model_spec(
        key="stl-arima",
        description="STL + ARIMA remainder forecasting via statsmodels. Optional dependency.",
        factory=_factory_stl_arima,
        default_params={"period": 12, "order": (1, 0, 0), "seasonal": 7, "robust": False},
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "order": "ARIMA order for remainder model (p,d,q)",
            "seasonal": STL_SEASONAL_PARAM_HELP,
            "robust": ROBUST_STL_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "stl-ets": model_spec(
        key="stl-ets",
        description="STL + ETS remainder forecasting via statsmodels. Optional dependency.",
        factory=_factory_stl_ets,
        default_params={"period": 12, "trend": "add", "damped_trend": False, "robust": False},
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "trend": "Remainder ETS trend: add, mul, or none",
            "damped_trend": "Use damped trend in the ETS remainder model (true/false)",
            "robust": ROBUST_STL_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "stl-autoreg": model_spec(
        key="stl-autoreg",
        description="STL + AutoReg remainder forecasting via statsmodels. Optional dependency.",
        factory=_factory_stl_autoreg,
        default_params={"period": 12, "lags": 1, "trend": "c", "seasonal": 7, "robust": False},
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "lags": "AutoReg lag order for the remainder model",
            "trend": "Deterministic trend for the remainder AutoReg model: n, c, t, ct",
            "seasonal": STL_SEASONAL_PARAM_HELP,
            "robust": ROBUST_STL_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "stl-uc": model_spec(
        key="stl-uc",
        description="STL decomposition + UnobservedComponents on the seasonally-adjusted series. Optional dependency.",
        factory=_factory_stl_uc,
        default_params={"period": 12, "level": LOCAL_LEVEL, "seasonal": 7, "robust": False},
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "level": "Adjusted-series structural level model (e.g. local level, local linear trend, random walk)",
            "seasonal": STL_SEASONAL_PARAM_HELP,
            "robust": ROBUST_STL_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "stl-auto-arima": model_spec(
        key="stl-auto-arima",
        description="STL decomposition + AutoARIMA-style grid search on the seasonally-adjusted series. Optional dependency.",
        factory=_factory_stl_auto_arima,
        default_params={
            "period": 12,
            "seasonal": 7,
            "robust": False,
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "information_criterion": "aic",
        },
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "seasonal": STL_SEASONAL_PARAM_HELP,
            "robust": ROBUST_STL_PARAM_HELP,
            "max_p": "Maximum AR order p for adjusted-series AutoARIMA search",
            "max_d": "Maximum differencing order d for adjusted-series AutoARIMA search",
            "max_q": "Maximum MA order q for adjusted-series AutoARIMA search",
            "trend": "Trend term for the adjusted-series AutoARIMA model (e.g. n, c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity in the adjusted-series AutoARIMA model (true/false)",
            "enforce_invertibility": "Enforce invertibility in the adjusted-series AutoARIMA model (true/false)",
            "information_criterion": "Model selection criterion for adjusted-series AutoARIMA search: aic or bic",
        },
        requires=("stats",),
    ),
    "stl-sarimax": model_spec(
        key="stl-sarimax",
        description="STL decomposition + SARIMAX on the seasonally-adjusted series. Optional dependency.",
        factory=_factory_stl_sarimax,
        default_params={
            "period": 12,
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "seasonal": 7,
            "robust": False,
        },
        param_help={
            "period": STL_PERIOD_PARAM_HELP,
            "order": "SARIMAX order for the adjusted series (p,d,q)",
            "seasonal_order": "Seasonal SARIMAX order for the adjusted series (P,D,Q,s)",
            "trend": "Trend term for the adjusted-series SARIMAX model (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity in the adjusted-series SARIMAX model (true/false)",
            "enforce_invertibility": "Enforce invertibility in the adjusted-series SARIMAX model (true/false)",
            "seasonal": STL_SEASONAL_PARAM_HELP,
            "robust": ROBUST_STL_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-arima": model_spec(
        key="mstl-arima",
        description="MSTL (multi-seasonal STL) + ARIMA on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_arima,
        default_params={"periods": (12,), "order": (1, 0, 0), "iterate": 2, "lmbda": None},
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "order": "ARIMA order for adjusted series (p,d,q)",
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-autoreg": model_spec(
        key="mstl-autoreg",
        description="MSTL (multi-seasonal STL) + AutoReg on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_autoreg,
        default_params={"periods": (12,), "lags": 1, "trend": "c", "iterate": 2, "lmbda": None},
        param_help={
            "periods": "Comma-separated seasonal periods (e.g. 7,24)",
            "lags": "AutoReg lag order for the adjusted series",
            "trend": "Deterministic trend for the adjusted-series AutoReg model: n, c, t, ct",
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-ets": model_spec(
        key="mstl-ets",
        description="MSTL (multi-seasonal STL) + ETS on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_ets,
        default_params={
            "periods": (12,),
            "trend": "add",
            "damped_trend": False,
            "iterate": 2,
            "lmbda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "trend": "Adjusted-series ETS trend: add, mul, or none",
            "damped_trend": "Use damped trend in the adjusted-series ETS model (true/false)",
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-uc": model_spec(
        key="mstl-uc",
        description="MSTL (multi-seasonal STL) + UnobservedComponents on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_uc,
        default_params={
            "periods": (12,),
            "level": LOCAL_LEVEL,
            "iterate": 2,
            "lmbda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "level": "Adjusted-series structural level model (e.g. local level, local linear trend, random walk)",
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-sarimax": model_spec(
        key="mstl-sarimax",
        description="MSTL (multi-seasonal STL) + SARIMAX on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_sarimax,
        default_params={
            "periods": (12,),
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "iterate": 2,
            "lmbda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "order": "SARIMAX order for adjusted series (p,d,q)",
            "seasonal_order": "Seasonal SARIMAX order for adjusted series (P,D,Q,s)",
            "trend": "Trend term for the adjusted-series SARIMAX model (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity in the adjusted-series SARIMAX model (true/false)",
            "enforce_invertibility": "Enforce invertibility in the adjusted-series SARIMAX model (true/false)",
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "mstl-auto-arima": model_spec(
        key="mstl-auto-arima",
        description="MSTL + AutoARIMA-style grid search on adjusted series. Optional dependency.",
        factory=_factory_mstl_auto_arima,
        default_params={
            "periods": (12,),
            "iterate": 2,
            "lmbda": None,
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "information_criterion": "aic",
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "iterate": MSTL_ITERATIONS_PARAM_HELP,
            "lmbda": MSTL_BOXCOX_PARAM_HELP,
            "max_p": "Max AR order p to consider",
            "max_d": "Max differencing order d to consider",
            "max_q": "Max MA order q to consider",
            "trend": "Trend term for the adjusted-series ARIMA search (e.g. n, c, t, ct) or none",
            "enforce_stationarity": CANDIDATE_STATIONARITY_PARAM_HELP,
            "enforce_invertibility": CANDIDATE_INVERTIBILITY_PARAM_HELP,
            "information_criterion": MODEL_SELECTION_CRITERION_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite": model_spec(
        key="tbats-lite",
        description="TBATS-like: multi-season Fourier + ARIMA residuals (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "arima_order": (1, 0, 0),
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "arima_order": "ARIMA order for residual errors (p,d,q)",
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite-autoreg": model_spec(
        key="tbats-lite-autoreg",
        description="TBATS-like: multi-season Fourier + AutoReg residuals (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite_autoreg,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "lags": 1,
            "trend": "n",
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "lags": "AutoReg lag order for residual errors",
            "trend": "Residual AutoReg deterministic trend: n, c, t, ct",
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite-ets": model_spec(
        key="tbats-lite-ets",
        description="TBATS-like: multi-season Fourier + ETS residuals (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite_ets,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "trend": None,
            "damped_trend": False,
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "trend": "Residual ETS trend: add, mul, or none",
            "damped_trend": "Use damped trend in the residual ETS model (true/false)",
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite-sarimax": model_spec(
        key="tbats-lite-sarimax",
        description="TBATS-like: multi-season Fourier + SARIMAX residuals (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite_sarimax,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "order": "SARIMAX order for residual errors (p,d,q)",
            "seasonal_order": "Seasonal SARIMAX order for residual errors (P,D,Q,s)",
            "trend": "Residual SARIMAX trend term (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity for residual SARIMAX model (true/false)",
            "enforce_invertibility": "Enforce invertibility for residual SARIMAX model (true/false)",
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite-auto-arima": model_spec(
        key="tbats-lite-auto-arima",
        description="TBATS-like: multi-season Fourier + AutoARIMA residual search (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite_auto_arima,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "trend": "c",
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "information_criterion": "aic",
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "max_p": "Max AR order p to consider for residual AutoARIMA",
            "max_d": "Max differencing order d to consider for residual AutoARIMA",
            "max_q": "Max MA order q to consider for residual AutoARIMA",
            "trend": "Residual AutoARIMA trend term (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity for residual AutoARIMA candidates (true/false)",
            "enforce_invertibility": "Enforce invertibility for residual AutoARIMA candidates (true/false)",
            "information_criterion": MODEL_SELECTION_CRITERION_PARAM_HELP,
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    "tbats-lite-uc": model_spec(
        key="tbats-lite-uc",
        description="TBATS-like: multi-season Fourier + UnobservedComponents residuals (optional Box-Cox). Optional dependency.",
        factory=_factory_tbats_lite_uc,
        default_params={
            "periods": (12,),
            "orders": 2,
            "include_trend": True,
            "level": LOCAL_LEVEL,
            "boxcox_lambda": None,
        },
        param_help={
            "periods": MULTI_SEASONAL_PERIODS_PARAM_HELP,
            "orders": FOURIER_ORDERS_PARAM_HELP,
            "include_trend": INCLUDE_LINEAR_TREND_PARAM_HELP,
            "level": "Residual structural level model (e.g. local level, local linear trend, random walk)",
            "boxcox_lambda": OPTIONAL_BOXCOX_LAMBDA_PARAM_HELP,
        },
        requires=("stats",),
    ),
    }

