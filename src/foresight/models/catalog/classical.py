from __future__ import annotations

from typing import Any

_LEVEL_SMOOTHING_HELP = "Level smoothing in [0, 1]"
_TREND_SMOOTHING_HELP = "Trend smoothing in [0, 1]"
_SEASON_LENGTH_HELP = "Season length"
_ALPHA_GRID_SIZE_HELP = "Number of alpha values to try (default: 19)"
_SMOOTHING_PARAM_HELP = "Smoothing parameter in [0,1]"


def build_classical_catalog(context: Any) -> dict[str, Any]:
    ModelSpec = context.ModelSpec
    _factory_adida = context._factory_adida
    _factory_analog_knn = context._factory_analog_knn
    _factory_ar_ols = context._factory_ar_ols
    _factory_ar_ols_auto = context._factory_ar_ols_auto
    _factory_ar_ols_lags = context._factory_ar_ols_lags
    _factory_croston = context._factory_croston
    _factory_croston_opt = context._factory_croston_opt
    _factory_croston_sba = context._factory_croston_sba
    _factory_croston_sbj = context._factory_croston_sbj
    _factory_drift = context._factory_drift
    _factory_ensemble_mean = context._factory_ensemble_mean
    _factory_ensemble_median = context._factory_ensemble_median
    _factory_fft = context._factory_fft
    _factory_fourier = context._factory_fourier
    _factory_fourier_multi = context._factory_fourier_multi
    _factory_holt = context._factory_holt
    _factory_holt_auto = context._factory_holt_auto
    _factory_holt_damped = context._factory_holt_damped
    _factory_holt_winters_add = context._factory_holt_winters_add
    _factory_holt_winters_add_auto = context._factory_holt_winters_add_auto
    _factory_holt_winters_mul = context._factory_holt_winters_mul
    _factory_holt_winters_mul_auto = context._factory_holt_winters_mul_auto
    _factory_kalman_level = context._factory_kalman_level
    _factory_kalman_trend = context._factory_kalman_trend
    _factory_les = context._factory_les
    _factory_mean = context._factory_mean
    _factory_median = context._factory_median
    _factory_moving_average = context._factory_moving_average
    _factory_moving_median = context._factory_moving_median
    _factory_naive_last = context._factory_naive_last
    _factory_pipeline = context._factory_pipeline
    _factory_poly_trend = context._factory_poly_trend
    _factory_sar_ols = context._factory_sar_ols
    _factory_seasonal_drift = context._factory_seasonal_drift
    _factory_seasonal_mean = context._factory_seasonal_mean
    _factory_seasonal_naive = context._factory_seasonal_naive
    _factory_seasonal_naive_auto = context._factory_seasonal_naive_auto
    _factory_ses = context._factory_ses
    _factory_ses_auto = context._factory_ses_auto
    _factory_ssa = context._factory_ssa
    _factory_theta = context._factory_theta
    _factory_theta_auto = context._factory_theta_auto
    _factory_tsb = context._factory_tsb
    _factory_weighted_moving_average = context._factory_weighted_moving_average
    return {
        "naive-last": ModelSpec(
            key="naive-last",
            description="Repeat the last observed value.",
            factory=_factory_naive_last,
        ),
        "seasonal-naive": ModelSpec(
            key="seasonal-naive",
            description="Repeat the last season of length `season_length`.",
            factory=_factory_seasonal_naive,
            default_params={"season_length": 12},
            param_help={"season_length": "Season length for repeating the last season"},
        ),
        "seasonal-naive-auto": ModelSpec(
            key="seasonal-naive-auto",
            description="Auto seasonal-naive baseline (infer season length via ACF scan).",
            factory=_factory_seasonal_naive_auto,
            default_params={
                "min_season_length": 2,
                "max_season_length": 24,
                "detrend": True,
                "min_corr": 0.2,
            },
            param_help={
                "min_season_length": "Minimum season length to consider (default: 2)",
                "max_season_length": "Maximum season length to consider (default: 24)",
                "detrend": "Use first differences when scanning ACF (true/false)",
                "min_corr": "Minimum correlation threshold to accept seasonality (default: 0.2)",
            },
        ),
        "mean": ModelSpec(
            key="mean",
            description="Repeat the mean of the training window.",
            factory=_factory_mean,
        ),
        "median": ModelSpec(
            key="median",
            description="Repeat the median of the training window.",
            factory=_factory_median,
        ),
        "drift": ModelSpec(
            key="drift",
            description="Random walk with drift (linear extrapolation from first to last).",
            factory=_factory_drift,
        ),
        "moving-average": ModelSpec(
            key="moving-average",
            description="Repeat the mean of the last `window` values.",
            factory=_factory_moving_average,
            default_params={"window": 3},
            param_help={"window": "Trailing window size for the moving average"},
        ),
        "weighted-moving-average": ModelSpec(
            key="weighted-moving-average",
            description="Repeat the linearly weighted mean of the last `window` values.",
            factory=_factory_weighted_moving_average,
            default_params={"window": 3},
            param_help={"window": "Trailing window size for the weighted moving average"},
        ),
        "moving-median": ModelSpec(
            key="moving-median",
            description="Repeat the median of the last `window` values.",
            factory=_factory_moving_median,
            default_params={"window": 3},
            param_help={"window": "Trailing window size for the moving median"},
        ),
        "seasonal-mean": ModelSpec(
            key="seasonal-mean",
            description="Repeat the seasonal means for each position in a season.",
            factory=_factory_seasonal_mean,
            default_params={"season_length": 12},
            param_help={"season_length": "Season length for seasonal means"},
        ),
        "seasonal-drift": ModelSpec(
            key="seasonal-drift",
            description="Repeat the last season with per-position drift estimated from the previous season.",
            factory=_factory_seasonal_drift,
            default_params={"season_length": 12},
            param_help={"season_length": "Season length for per-position seasonal drift"},
        ),
        "ses": ModelSpec(
            key="ses",
            description="Simple Exponential Smoothing (SES).",
            factory=_factory_ses,
            default_params={"alpha": 0.2},
            param_help={"alpha": "Smoothing level in [0, 1]"},
        ),
        "ses-auto": ModelSpec(
            key="ses-auto",
            description="Auto-tuned SES (grid search over alpha).",
            factory=_factory_ses_auto,
            default_params={"grid_size": 19},
            param_help={"grid_size": "Number of alpha values to try (default: 19)"},
        ),
        "holt": ModelSpec(
            key="holt",
            description="Holt linear trend exponential smoothing.",
            factory=_factory_holt,
            default_params={"alpha": 0.2, "beta": 0.1},
            param_help={
                "alpha": _LEVEL_SMOOTHING_HELP,
                "beta": _TREND_SMOOTHING_HELP,
            },
        ),
        "holt-auto": ModelSpec(
            key="holt-auto",
            description="Auto-tuned Holt (grid search over alpha,beta).",
            factory=_factory_holt_auto,
            default_params={"grid_size": 10},
            param_help={"grid_size": "Grid size per parameter (default: 10)"},
        ),
        "holt-damped": ModelSpec(
            key="holt-damped",
            description="Holt damped trend exponential smoothing.",
            factory=_factory_holt_damped,
            default_params={"alpha": 0.2, "beta": 0.1, "phi": 0.9},
            param_help={
                "alpha": _LEVEL_SMOOTHING_HELP,
                "beta": _TREND_SMOOTHING_HELP,
                "phi": "Damping parameter in [0, 1] (phi=1 reduces to Holt)",
            },
        ),
        "holt-winters-add": ModelSpec(
            key="holt-winters-add",
            description="Holt-Winters additive seasonality + additive trend.",
            factory=_factory_holt_winters_add,
            default_params={"season_length": 12, "alpha": 0.2, "beta": 0.1, "gamma": 0.1},
            param_help={
                "season_length": _SEASON_LENGTH_HELP,
                "alpha": _LEVEL_SMOOTHING_HELP,
                "beta": _TREND_SMOOTHING_HELP,
                "gamma": "Seasonal smoothing in [0, 1]",
            },
        ),
        "holt-winters-mul": ModelSpec(
            key="holt-winters-mul",
            description="Holt-Winters multiplicative seasonality + additive trend (positive series only).",
            factory=_factory_holt_winters_mul,
            default_params={"season_length": 12, "alpha": 0.2, "beta": 0.1, "gamma": 0.1},
            param_help={
                "season_length": _SEASON_LENGTH_HELP,
                "alpha": _LEVEL_SMOOTHING_HELP,
                "beta": _TREND_SMOOTHING_HELP,
                "gamma": "Seasonal smoothing in [0, 1]",
            },
        ),
        "holt-winters-add-auto": ModelSpec(
            key="holt-winters-add-auto",
            description="Auto-tuned Holt-Winters additive (small grid search).",
            factory=_factory_holt_winters_add_auto,
            default_params={"season_length": 12, "grid_size": 7},
            param_help={
                "season_length": _SEASON_LENGTH_HELP,
                "grid_size": "Grid size per parameter (default: 7)",
            },
        ),
        "holt-winters-mul-auto": ModelSpec(
            key="holt-winters-mul-auto",
            description="Auto-tuned Holt-Winters multiplicative (small grid search).",
            factory=_factory_holt_winters_mul_auto,
            default_params={"season_length": 12, "grid_size": 7},
            param_help={
                "season_length": _SEASON_LENGTH_HELP,
                "grid_size": "Grid size per parameter (default: 7)",
            },
        ),
        "theta": ModelSpec(
            key="theta",
            description="Theta-style baseline (SES level + half-slope drift).",
            factory=_factory_theta,
            default_params={"alpha": 0.2},
            param_help={"alpha": "SES smoothing level in [0, 1]"},
        ),
        "theta-auto": ModelSpec(
            key="theta-auto",
            description="Auto-tuned Theta-style baseline (grid search over alpha).",
            factory=_factory_theta_auto,
            default_params={"grid_size": 19},
            param_help={"grid_size": _ALPHA_GRID_SIZE_HELP},
        ),
        "ar-ols": ModelSpec(
            key="ar-ols",
            description="Autoregression AR(p) fitted by OLS (recursive forecast).",
            factory=_factory_ar_ols,
            default_params={"p": 5},
            param_help={"p": "AR order"},
        ),
        "ar-ols-lags": ModelSpec(
            key="ar-ols-lags",
            description="Autoregression with custom lag set, fitted by OLS (recursive forecast).",
            factory=_factory_ar_ols_lags,
            default_params={"lags": (1, 2, 3, 4, 5)},
            param_help={"lags": "Lag indices (e.g. 1,2,12)"},
        ),
        "sar-ols": ModelSpec(
            key="sar-ols",
            description="Seasonal AR using OLS with short and seasonal lags (recursive forecast).",
            factory=_factory_sar_ols,
            default_params={"p": 1, "P": 1, "season_length": 12},
            param_help={
                "p": "Non-seasonal AR order",
                "P": "Seasonal AR order",
                "season_length": "Season length (e.g. 12 for monthly)",
            },
        ),
        "ar-ols-auto": ModelSpec(
            key="ar-ols-auto",
            description="Auto AR(p) by AIC (OLS), recursive forecast.",
            factory=_factory_ar_ols_auto,
            default_params={"max_p": 10},
            param_help={"max_p": "Maximum AR order to consider"},
        ),
        "croston": ModelSpec(
            key="croston",
            description="Croston classic intermittent-demand method.",
            factory=_factory_croston,
            default_params={"alpha": 0.1},
            param_help={"alpha": _SMOOTHING_PARAM_HELP},
        ),
        "croston-sba": ModelSpec(
            key="croston-sba",
            description="Croston-SBA intermittent-demand method (bias-corrected).",
            factory=_factory_croston_sba,
            default_params={"alpha": 0.1},
            param_help={"alpha": _SMOOTHING_PARAM_HELP},
        ),
        "croston-sbj": ModelSpec(
            key="croston-sbj",
            description="Croston-SBJ intermittent-demand method (bias-corrected).",
            factory=_factory_croston_sbj,
            default_params={"alpha": 0.1},
            param_help={"alpha": _SMOOTHING_PARAM_HELP},
        ),
        "croston-opt": ModelSpec(
            key="croston-opt",
            description="Croston classic with alpha tuned by in-sample SSE grid search.",
            factory=_factory_croston_opt,
            default_params={"grid_size": 19},
            param_help={"grid_size": _ALPHA_GRID_SIZE_HELP},
        ),
        "les": ModelSpec(
            key="les",
            description="LES intermittent-demand method (linear decay under no demand).",
            factory=_factory_les,
            default_params={"alpha": 0.1, "beta": 0.1},
            param_help={
                "alpha": "Demand-size smoothing in [0,1]",
                "beta": "Interval smoothing in [0,1] (also controls decay rate)",
            },
        ),
        "tsb": ModelSpec(
            key="tsb",
            description="TSB intermittent-demand method (probability + size smoothing).",
            factory=_factory_tsb,
            default_params={"alpha": 0.1, "beta": 0.1},
            param_help={
                "alpha": "Size smoothing in [0,1]",
                "beta": "Probability smoothing in [0,1]",
            },
        ),
        "adida": ModelSpec(
            key="adida",
            description="ADIDA aggregation/disaggregation intermittent-demand baseline.",
            factory=_factory_adida,
            default_params={"agg_period": 4, "base": "ses", "alpha": 0.2},
            param_help={
                "agg_period": "Aggregation period (block size)",
                "base": "Base method on aggregated series: naive-last, mean, ses",
                "alpha": "SES alpha when base='ses'",
            },
        ),
        "fourier": ModelSpec(
            key="fourier",
            description="Fourier regression (seasonality + optional trend).",
            factory=_factory_fourier,
            default_params={"period": 12, "order": 2, "include_trend": True},
            param_help={
                "period": "Seasonal period (e.g. 12 for monthly)",
                "order": "Number of Fourier harmonics",
                "include_trend": "Include linear trend term (true/false)",
            },
        ),
        "fourier-multi": ModelSpec(
            key="fourier-multi",
            description="Fourier regression with multiple seasonalities.",
            factory=_factory_fourier_multi,
            default_params={"periods": (7, 365), "orders": 2, "include_trend": True},
            param_help={
                "periods": "Comma-separated seasonal periods (e.g. 7,365)",
                "orders": "Fourier order per period (int or comma-separated list)",
                "include_trend": "Include linear trend term (true/false)",
            },
        ),
        "poly-trend": ModelSpec(
            key="poly-trend",
            description="Polynomial trend regression on time index.",
            factory=_factory_poly_trend,
            default_params={"degree": 1},
            param_help={"degree": "Polynomial degree (0=mean, 1=linear, 2=quadratic, ...)"},
        ),
        "fft": ModelSpec(
            key="fft",
            description="FFT-based extrapolation (top-K frequencies + optional trend).",
            factory=_factory_fft,
            default_params={"top_k": 3, "include_trend": True},
            param_help={
                "top_k": "Number of dominant frequencies to keep",
                "include_trend": "Detrend with linear regression before FFT (true/false)",
            },
        ),
        "ssa": ModelSpec(
            key="ssa",
            description="Singular Spectrum Analysis (SSA) recurrent forecast (rank-truncated SVD).",
            factory=_factory_ssa,
            default_params={"window_length": 24, "rank": 5},
            param_help={
                "window_length": "SSA embedding window length L (2 <= L <= n-1)",
                "rank": "Truncated SVD rank r (>=1)",
            },
        ),
        "analog-knn": ModelSpec(
            key="analog-knn",
            description="Analog kNN forecasting on lag windows (non-parametric).",
            factory=_factory_analog_knn,
            default_params={"lags": 12, "k": 5, "normalize": True, "weights": "uniform"},
            param_help={
                "lags": "Window length",
                "k": "Number of nearest neighbors",
                "normalize": "Z-score windows before distance (true/false)",
                "weights": "Neighbor weights: uniform or distance",
            },
        ),
        "kalman-level": ModelSpec(
            key="kalman-level",
            description="Kalman filter local-level model (random walk level).",
            factory=_factory_kalman_level,
            default_params={"process_variance": None, "obs_variance": None},
            param_help={
                "process_variance": "State noise variance (q); None for heuristic default",
                "obs_variance": "Observation noise variance (r); None for heuristic default",
            },
        ),
        "kalman-trend": ModelSpec(
            key="kalman-trend",
            description="Kalman filter local linear trend model (level + trend).",
            factory=_factory_kalman_trend,
            default_params={"level_variance": None, "trend_variance": None, "obs_variance": None},
            param_help={
                "level_variance": "Level noise variance; None for heuristic default",
                "trend_variance": "Trend noise variance; None for heuristic default",
                "obs_variance": "Observation noise variance; None for heuristic default",
            },
        ),
        "pipeline": ModelSpec(
            key="pipeline",
            description="Meta-model: apply transforms then run a base model (params forwarded).",
            factory=_factory_pipeline,
            default_params={"base": "naive-last", "transforms": ()},
            param_help={
                "base": "Base model key (e.g. theta, holt-winters-add, lr-lag-direct)",
                "transforms": "Transform list (e.g. log1p,diff1,standardize)",
            },
        ),
        "ensemble-mean": ModelSpec(
            key="ensemble-mean",
            description="Meta-model: average predictions from several member models.",
            factory=_factory_ensemble_mean,
            default_params={"members": ("naive-last", "seasonal-naive", "theta")},
            param_help={"members": "Comma-separated model keys to average"},
        ),
        "ensemble-median": ModelSpec(
            key="ensemble-median",
            description="Meta-model: median of predictions from several member models.",
            factory=_factory_ensemble_median,
            default_params={"members": ("naive-last", "seasonal-naive", "theta")},
            param_help={"members": "Comma-separated model keys to take the median of"},
        ),
    }
