# ruff: noqa: F401
from __future__ import annotations

import functools
import inspect
import sys
from contextlib import nullcontext
from typing import Any

import numpy as np

from .. import base as _base
from ..base import (
    BaseForecaster,
    BaseGlobalForecaster,
)
from ..transforms import fit_transform, inverse_forecast, normalize_transform_list
from . import specs as _specs
from .analog import analog_knn_forecast
from .ar import ar_ols_auto_forecast, ar_ols_forecast, ar_ols_lags_forecast, sar_ols_forecast
from .baselines import (
    drift_forecast,
    mean_forecast,
    median_forecast,
    moving_average_forecast,
    moving_median_forecast,
    seasonal_drift_forecast,
    seasonal_mean_forecast,
    weighted_moving_average_forecast,
)
from .catalog import build_catalog
from .factories import (
    build_global_forecaster,
    build_global_forecaster_object,
    build_local_forecaster,
    build_local_forecaster_object,
    build_multivariate_forecaster,
)
from .fourier import fourier_multi_regression_forecast, fourier_regression_forecast
from .global_regression import (
    adaboost_step_lag_global_forecaster,
    ard_step_lag_global_forecaster,
    bagging_step_lag_global_forecaster,
    bayesian_ridge_step_lag_global_forecaster,
    catboost_step_lag_global_forecaster,
    decision_tree_step_lag_global_forecaster,
    elasticnet_step_lag_global_forecaster,
    extra_trees_step_lag_global_forecaster,
    gamma_step_lag_global_forecaster,
    gbrt_step_lag_global_forecaster,
    hgb_step_lag_global_forecaster,
    huber_step_lag_global_forecaster,
    kernel_ridge_step_lag_global_forecaster,
    knn_step_lag_global_forecaster,
    lasso_step_lag_global_forecaster,
    lgbm_step_lag_global_forecaster,
    linear_svr_step_lag_global_forecaster,
    mlp_step_lag_global_forecaster,
    omp_step_lag_global_forecaster,
    passive_aggressive_step_lag_global_forecaster,
    poisson_step_lag_global_forecaster,
    quantile_step_lag_global_forecaster,
    rf_step_lag_global_forecaster,
    ridge_step_lag_global_forecaster,
    sgd_step_lag_global_forecaster,
    svr_step_lag_global_forecaster,
    tweedie_step_lag_global_forecaster,
    xgb_dart_step_lag_global_forecaster,
    xgb_gamma_step_lag_global_forecaster,
    xgb_huber_step_lag_global_forecaster,
    xgb_linear_step_lag_global_forecaster,
    xgb_logistic_step_lag_global_forecaster,
    xgb_mae_step_lag_global_forecaster,
    xgb_msle_step_lag_global_forecaster,
    xgb_poisson_step_lag_global_forecaster,
    xgb_step_lag_global_forecaster,
    xgb_tweedie_step_lag_global_forecaster,
    xgbrf_step_lag_global_forecaster,
)
from .hf_time_series import hf_timeseries_transformer_direct_forecast
from .intermittent import (
    adida_forecast,
    croston_classic_forecast,
    croston_optimized_forecast,
    croston_sba_forecast,
    croston_sbj_forecast,
    les_forecast,
    tsb_forecast,
)
from .kalman import kalman_local_level_forecast, kalman_local_linear_trend_forecast
from .multivariate import (
    torch_graphwavenet_forecast,
    torch_stgcn_forecast,
    torch_stid_forecast,
    var_forecast,
)
from .naive import naive_last, seasonal_naive, seasonal_naive_auto
from .regression import (
    adaboost_lag_direct_forecast,
    ard_lag_direct_forecast,
    bagging_lag_direct_forecast,
    bayesian_ridge_lag_direct_forecast,
    catboost_custom_dirrec_lag_forecast,
    catboost_custom_lag_direct_forecast,
    catboost_custom_lag_recursive_forecast,
    catboost_custom_step_lag_direct_forecast,
    catboost_dirrec_lag_forecast,
    catboost_lag_direct_forecast,
    catboost_lag_recursive_forecast,
    catboost_step_lag_direct_forecast,
    decision_tree_lag_direct_forecast,
    elasticnet_lag_direct_forecast,
    extra_trees_lag_direct_forecast,
    gamma_lag_direct_forecast,
    gbrt_lag_direct_forecast,
    hgb_lag_direct_forecast,
    huber_lag_direct_forecast,
    kernel_ridge_lag_direct_forecast,
    knn_lag_direct_forecast,
    lasso_lag_direct_forecast,
    lgbm_custom_dirrec_lag_forecast,
    lgbm_custom_lag_direct_forecast,
    lgbm_custom_lag_recursive_forecast,
    lgbm_custom_step_lag_direct_forecast,
    lgbm_dirrec_lag_forecast,
    lgbm_lag_direct_forecast,
    lgbm_lag_recursive_forecast,
    lgbm_step_lag_direct_forecast,
    linear_svr_lag_direct_forecast,
    lr_lag_direct_forecast,
    lr_lag_forecast,
    mlp_lag_direct_forecast,
    omp_lag_direct_forecast,
    passive_aggressive_lag_direct_forecast,
    poisson_lag_direct_forecast,
    quantile_lag_direct_forecast,
    rf_lag_direct_forecast,
    ridge_lag_direct_forecast,
    ridge_lag_forecast,
    sgd_lag_direct_forecast,
    svr_lag_direct_forecast,
    tweedie_lag_direct_forecast,
    xgb_custom_lag_direct_forecast,
    xgb_custom_lag_recursive_forecast,
    xgb_dart_lag_direct_forecast,
    xgb_dart_lag_recursive_forecast,
    xgb_dirrec_lag_forecast,
    xgb_gamma_lag_direct_forecast,
    xgb_gamma_lag_recursive_forecast,
    xgb_huber_lag_direct_forecast,
    xgb_huber_lag_recursive_forecast,
    xgb_lag_direct_forecast,
    xgb_lag_recursive_forecast,
    xgb_linear_lag_direct_forecast,
    xgb_linear_lag_recursive_forecast,
    xgb_logistic_lag_direct_forecast,
    xgb_logistic_lag_recursive_forecast,
    xgb_mae_lag_direct_forecast,
    xgb_mae_lag_recursive_forecast,
    xgb_mimo_lag_direct_forecast,
    xgb_msle_lag_direct_forecast,
    xgb_msle_lag_recursive_forecast,
    xgb_poisson_lag_direct_forecast,
    xgb_poisson_lag_recursive_forecast,
    xgb_quantile_lag_direct_forecast,
    xgb_quantile_lag_recursive_forecast,
    xgb_step_lag_direct_forecast,
    xgb_tweedie_lag_direct_forecast,
    xgb_tweedie_lag_recursive_forecast,
    xgbrf_lag_direct_forecast,
    xgbrf_lag_recursive_forecast,
)
from .smoothing import (
    holt_auto_forecast,
    holt_damped_forecast,
    holt_forecast,
    holt_winters_additive_auto_forecast,
    holt_winters_additive_forecast,
    holt_winters_multiplicative_auto_forecast,
    holt_winters_multiplicative_forecast,
    ses_auto_forecast,
    ses_forecast,
)
from .specs import ForecasterFn, GlobalForecasterFn, ModelSpec, MultivariateForecasterFn
from .spectral import fft_topk_forecast
from .ssa import ssa_forecast
from .statsmodels_wrap import (
    arima_forecast as _arima_forecast_impl,
    auto_arima_forecast as _auto_arima_forecast_impl,
    autoreg_forecast as _autoreg_forecast_impl,
    ets_forecast as _ets_forecast_impl,
    fourier_arima_forecast as _fourier_arima_forecast_impl,
    fourier_auto_arima_forecast as _fourier_auto_arima_forecast_impl,
    fourier_autoreg_forecast as _fourier_autoreg_forecast_impl,
    fourier_ets_forecast as _fourier_ets_forecast_impl,
    fourier_sarimax_forecast as _fourier_sarimax_forecast_impl,
    fourier_uc_forecast as _fourier_uc_forecast_impl,
    mstl_arima_forecast as _mstl_arima_forecast_impl,
    mstl_auto_arima_forecast as _mstl_auto_arima_forecast_impl,
    mstl_autoreg_forecast as _mstl_autoreg_forecast_impl,
    mstl_ets_forecast as _mstl_ets_forecast_impl,
    mstl_sarimax_forecast as _mstl_sarimax_forecast_impl,
    mstl_uc_forecast as _mstl_uc_forecast_impl,
    sarimax_forecast as _sarimax_forecast_impl,
    stl_arima_forecast as _stl_arima_forecast_impl,
    stl_auto_arima_forecast as _stl_auto_arima_forecast_impl,
    stl_autoreg_forecast as _stl_autoreg_forecast_impl,
    stl_ets_forecast as _stl_ets_forecast_impl,
    stl_sarimax_forecast as _stl_sarimax_forecast_impl,
    stl_uc_forecast as _stl_uc_forecast_impl,
    tbats_lite_auto_arima_forecast as _tbats_lite_auto_arima_forecast_impl,
    tbats_lite_autoreg_forecast as _tbats_lite_autoreg_forecast_impl,
    tbats_lite_ets_forecast as _tbats_lite_ets_forecast_impl,
    tbats_lite_forecast as _tbats_lite_forecast_impl,
    tbats_lite_sarimax_forecast as _tbats_lite_sarimax_forecast_impl,
    tbats_lite_uc_forecast as _tbats_lite_uc_forecast_impl,
    unobserved_components_forecast as _unobserved_components_forecast_impl,
)
from .theta import theta_auto_forecast, theta_forecast
from .torch_ct_rnn import (
    torch_cfc_direct_forecast,
    torch_griffin_direct_forecast,
    torch_hawk_direct_forecast,
    torch_lmu_direct_forecast,
    torch_ltc_direct_forecast,
    torch_xlstm_direct_forecast,
)
from .torch_global import (
    torch_autoformer_global_forecaster,
    torch_crossformer_global_forecaster,
    torch_deepar_global_forecaster,
    torch_dilated_rnn_global_forecaster,
    torch_dlinear_global_forecaster,
    torch_esrnn_global_forecaster,
    torch_etsformer_global_forecaster,
    torch_fedformer_global_forecaster,
    torch_fnet_global_forecaster,
    torch_gmlp_global_forecaster,
    torch_hyena_global_forecaster,
    torch_inception_global_forecaster,
    torch_informer_global_forecaster,
    torch_itransformer_global_forecaster,
    torch_kan_global_forecaster,
    torch_lstnet_global_forecaster,
    torch_mamba_global_forecaster,
    torch_nbeats_global_forecaster,
    torch_nhits_global_forecaster,
    torch_nlinear_global_forecaster,
    torch_nonstationary_transformer_global_forecaster,
    torch_patchtst_global_forecaster,
    torch_pyraformer_global_forecaster,
    torch_resnet1d_global_forecaster,
    torch_retnet_global_forecaster,
    torch_rnn_global_forecaster,
    torch_rwkv_global_forecaster,
    torch_scinet_global_forecaster,
    torch_seq2seq_global_forecaster,
    torch_ssm_global_forecaster,
    torch_tcn_global_forecaster,
    torch_tft_global_forecaster,
    torch_tide_global_forecaster,
    torch_timesnet_global_forecaster,
    torch_timexer_global_forecaster,
    torch_transformer_encdec_global_forecaster,
    torch_tsmixer_global_forecaster,
    torch_wavenet_global_forecaster,
    torch_xformer_global_forecaster,
)
from .torch_nn import (
    torch_attn_gru_direct_forecast,
    torch_autoformer_direct_forecast,
    torch_bigru_direct_forecast,
    torch_basisformer_direct_forecast,
    torch_bilstm_direct_forecast,
    torch_cnn_direct_forecast,
    torch_crossgnn_direct_forecast,
    torch_crossformer_direct_forecast,
    torch_deepar_recursive_forecast,
    torch_dilated_rnn_direct_forecast,
    torch_dlinear_direct_forecast,
    torch_esrnn_direct_forecast,
    torch_etsformer_direct_forecast,
    torch_fedformer_direct_forecast,
    torch_fits_direct_forecast,
    torch_film_direct_forecast,
    torch_fnet_direct_forecast,
    torch_frets_direct_forecast,
    torch_gmlp_direct_forecast,
    torch_gru_direct_forecast,
    torch_hyena_direct_forecast,
    torch_inception_direct_forecast,
    torch_informer_direct_forecast,
    torch_itransformer_direct_forecast,
    torch_kan_direct_forecast,
    torch_koopa_direct_forecast,
    torch_lightts_direct_forecast,
    torch_linear_attention_direct_forecast,
    torch_lstm_direct_forecast,
    torch_mamba_direct_forecast,
    torch_micn_direct_forecast,
    torch_moderntcn_direct_forecast,
    torch_mlp_lag_direct_forecast,
    torch_nbeats_direct_forecast,
    torch_nhits_direct_forecast,
    torch_nlinear_direct_forecast,
    torch_nonstationary_transformer_direct_forecast,
    torch_patchtst_direct_forecast,
    torch_pathformer_direct_forecast,
    torch_perceiver_direct_forecast,
    torch_pyraformer_direct_forecast,
    torch_qrnn_recursive_forecast,
    torch_resnet1d_direct_forecast,
    torch_retnet_direct_forecast,
    torch_retnet_recursive_forecast,
    torch_rwkv_direct_forecast,
    torch_samformer_direct_forecast,
    torch_scinet_direct_forecast,
    torch_segrnn_direct_forecast,
    torch_sparsetsf_direct_forecast,
    torch_tcn_direct_forecast,
    torch_tft_direct_forecast,
    torch_tide_direct_forecast,
    torch_timemixer_direct_forecast,
    torch_tinytimemixer_direct_forecast,
    torch_timesmamba_direct_forecast,
    torch_timesnet_direct_forecast,
    torch_timexer_direct_forecast,
    torch_transformer_direct_forecast,
    torch_train_config_override,
    torch_tsmixer_direct_forecast,
    torch_witran_direct_forecast,
    torch_wavenet_direct_forecast,
)


_XGB_REG_SQUAREDERROR_OBJECTIVE = "reg:squarederror"
_MEMBERS_NON_EMPTY_ERROR = "members must be non-empty"
_ORDER_TUPLE_ERROR = "order must be a 3-tuple like (p, d, q)"
_SEASONAL_ORDER_TUPLE_ERROR = "seasonal_order must be a 4-tuple like (P, D, Q, s)"
_LOCAL_LEVEL_LITERAL = "local level"
_DEFERRED_TORCH_RUNTIME_PARAM_KEYS = frozenset(
    {
        "tensorboard_log_dir",
        "tensorboard_run_name",
        "tensorboard_flush_secs",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "mlflow_run_name",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_dir",
        "wandb_mode",
    }
)


def _registry_statsmodels_symbol(name: str, fallback: Any) -> Any:
    def _call(*args: Any, **kwargs: Any) -> Any:
        from . import registry as _registry

        target = getattr(_registry, name, fallback)
        if target is _call:
            target = fallback
        return target(*args, **kwargs)

    _call.__name__ = name
    return _call


arima_forecast = _registry_statsmodels_symbol("arima_forecast", _arima_forecast_impl)
auto_arima_forecast = _registry_statsmodels_symbol(
    "auto_arima_forecast",
    _auto_arima_forecast_impl,
)
autoreg_forecast = _registry_statsmodels_symbol("autoreg_forecast", _autoreg_forecast_impl)
ets_forecast = _registry_statsmodels_symbol("ets_forecast", _ets_forecast_impl)
fourier_arima_forecast = _registry_statsmodels_symbol(
    "fourier_arima_forecast",
    _fourier_arima_forecast_impl,
)
fourier_auto_arima_forecast = _registry_statsmodels_symbol(
    "fourier_auto_arima_forecast",
    _fourier_auto_arima_forecast_impl,
)
fourier_autoreg_forecast = _registry_statsmodels_symbol(
    "fourier_autoreg_forecast",
    _fourier_autoreg_forecast_impl,
)


def _wrap_runtime_torch_callable(func: Any) -> Any:
    if not callable(func):
        return func

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        deferred = {
            str(key): value
            for key, value in dict(kwargs).items()
            if str(key) in _DEFERRED_TORCH_RUNTIME_PARAM_KEYS
        }
        direct_kwargs = {
            str(key): value
            for key, value in dict(kwargs).items()
            if str(key) not in _DEFERRED_TORCH_RUNTIME_PARAM_KEYS
        }
        ctx = (
            torch_train_config_override(deferred)
            if deferred
            else nullcontext()
        )
        with ctx:
            return func(*args, **direct_kwargs)

    return _wrapped


def _wrap_runtime_torch_callables() -> None:
    for name, value in list(globals().items()):
        if not str(name).startswith("torch_"):
            continue
        if "forecast" not in str(name) and "forecaster" not in str(name):
            continue
        if inspect.isclass(value):
            continue
        if not callable(value):
            continue
        globals()[str(name)] = _wrap_runtime_torch_callable(value)


_wrap_runtime_torch_callables()
fourier_ets_forecast = _registry_statsmodels_symbol(
    "fourier_ets_forecast",
    _fourier_ets_forecast_impl,
)
fourier_sarimax_forecast = _registry_statsmodels_symbol(
    "fourier_sarimax_forecast",
    _fourier_sarimax_forecast_impl,
)
fourier_uc_forecast = _registry_statsmodels_symbol(
    "fourier_uc_forecast",
    _fourier_uc_forecast_impl,
)
mstl_arima_forecast = _registry_statsmodels_symbol(
    "mstl_arima_forecast",
    _mstl_arima_forecast_impl,
)
mstl_auto_arima_forecast = _registry_statsmodels_symbol(
    "mstl_auto_arima_forecast",
    _mstl_auto_arima_forecast_impl,
)
mstl_autoreg_forecast = _registry_statsmodels_symbol(
    "mstl_autoreg_forecast",
    _mstl_autoreg_forecast_impl,
)
mstl_ets_forecast = _registry_statsmodels_symbol(
    "mstl_ets_forecast",
    _mstl_ets_forecast_impl,
)
mstl_sarimax_forecast = _registry_statsmodels_symbol(
    "mstl_sarimax_forecast",
    _mstl_sarimax_forecast_impl,
)
mstl_uc_forecast = _registry_statsmodels_symbol(
    "mstl_uc_forecast",
    _mstl_uc_forecast_impl,
)
sarimax_forecast = _registry_statsmodels_symbol("sarimax_forecast", _sarimax_forecast_impl)
stl_arima_forecast = _registry_statsmodels_symbol(
    "stl_arima_forecast",
    _stl_arima_forecast_impl,
)
stl_auto_arima_forecast = _registry_statsmodels_symbol(
    "stl_auto_arima_forecast",
    _stl_auto_arima_forecast_impl,
)
stl_autoreg_forecast = _registry_statsmodels_symbol(
    "stl_autoreg_forecast",
    _stl_autoreg_forecast_impl,
)
stl_ets_forecast = _registry_statsmodels_symbol("stl_ets_forecast", _stl_ets_forecast_impl)
stl_sarimax_forecast = _registry_statsmodels_symbol(
    "stl_sarimax_forecast",
    _stl_sarimax_forecast_impl,
)
stl_uc_forecast = _registry_statsmodels_symbol("stl_uc_forecast", _stl_uc_forecast_impl)
tbats_lite_auto_arima_forecast = _registry_statsmodels_symbol(
    "tbats_lite_auto_arima_forecast",
    _tbats_lite_auto_arima_forecast_impl,
)
tbats_lite_autoreg_forecast = _registry_statsmodels_symbol(
    "tbats_lite_autoreg_forecast",
    _tbats_lite_autoreg_forecast_impl,
)
tbats_lite_ets_forecast = _registry_statsmodels_symbol(
    "tbats_lite_ets_forecast",
    _tbats_lite_ets_forecast_impl,
)
tbats_lite_forecast = _registry_statsmodels_symbol(
    "tbats_lite_forecast",
    _tbats_lite_forecast_impl,
)
tbats_lite_sarimax_forecast = _registry_statsmodels_symbol(
    "tbats_lite_sarimax_forecast",
    _tbats_lite_sarimax_forecast_impl,
)
tbats_lite_uc_forecast = _registry_statsmodels_symbol(
    "tbats_lite_uc_forecast",
    _tbats_lite_uc_forecast_impl,
)
unobserved_components_forecast = _registry_statsmodels_symbol(
    "unobserved_components_forecast",
    _unobserved_components_forecast_impl,
)
from .torch_rnn_paper_zoo import list_rnnpaper_specs, torch_rnnpaper_direct_forecast
from .torch_rnn_zoo import list_rnnzoo_specs, torch_rnnzoo_direct_forecast
from .torch_seq2seq import torch_lstnet_direct_forecast, torch_seq2seq_direct_forecast
from .torch_ssm import (
    torch_mamba2_direct_forecast,
    torch_s4_direct_forecast,
    torch_s4d_direct_forecast,
    torch_s5_direct_forecast,
)
from .torch_xformer import torch_xformer_direct_forecast
from .trend import poly_trend_forecast
from .neural_runtime import coerce_torch_train_config_params as _coerce_torch_train_config_params

# Compatibility re-exports for downstream imports that historically used this module.
RegistryForecaster = _base.RegistryForecaster
RegistryGlobalForecaster = _base.RegistryGlobalForecaster
LocalForecasterFn = _specs.LocalForecasterFn
ModelFactory = _specs.ModelFactory


def _normalize_bool_like(value: Any) -> bool:
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "y", "on"}:
            return True
        if lower in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_torch_extra_train_params(params: dict[str, Any]) -> dict[str, Any]:
    return _coerce_torch_train_config_params(params)


_TORCH_COMMON_DEFAULTS: dict[str, Any] = {
    "epochs": 50,
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "seed": 0,
    "normalize": True,
    "device": "cpu",
    "patience": 10,
    "loss": "mse",
    "val_split": 0.0,
    "grad_clip_norm": 0.0,
    "optimizer": "adam",
    "momentum": 0.9,
    "scheduler": "none",
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.1,
    "restore_best": True,
    "min_epochs": 1,
    "amp": False,
    "amp_dtype": "auto",
    "warmup_epochs": 0,
    "min_lr": 0.0,
    "scheduler_restart_period": 10,
    "scheduler_restart_mult": 1,
    "scheduler_pct_start": 0.3,
    "grad_accum_steps": 1,
    "monitor": "auto",
    "monitor_mode": "min",
    "min_delta": 0.0,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "scheduler_patience": 5,
    "grad_clip_mode": "norm",
    "grad_clip_value": 0.0,
    "scheduler_plateau_factor": 0.1,
    "scheduler_plateau_threshold": 1e-4,
    "ema_decay": 0.0,
    "ema_warmup_epochs": 0,
    "swa_start_epoch": -1,
    "lookahead_steps": 0,
    "lookahead_alpha": 0.5,
    "sam_rho": 0.0,
    "sam_adaptive": False,
    "horizon_loss_decay": 1.0,
    "input_dropout": 0.0,
    "temporal_dropout": 0.0,
    "grad_noise_std": 0.0,
    "gc_mode": "off",
    "agc_clip_factor": 0.0,
    "agc_eps": 1e-3,
    "checkpoint_dir": "",
    "save_best_checkpoint": False,
    "save_last_checkpoint": False,
    "resume_checkpoint_path": "",
    "resume_checkpoint_strict": True,
    "tensorboard_log_dir": "",
    "tensorboard_run_name": "",
    "tensorboard_flush_secs": 10,
    "mlflow_tracking_uri": "",
    "mlflow_experiment_name": "",
    "mlflow_run_name": "",
    "wandb_project": "",
    "wandb_entity": "",
    "wandb_run_name": "",
    "wandb_dir": "",
    "wandb_mode": "online",
}

_TORCH_COMMON_PARAM_HELP: dict[str, str] = {
    "epochs": "Max training epochs (per window)",
    "lr": "Learning rate",
    "weight_decay": "L2 weight decay",
    "batch_size": "Mini-batch size",
    "seed": "Random seed",
    "normalize": "Z-score normalize the series before fitting (true/false)",
    "device": "Torch device (cpu or cuda)",
    "patience": "Early-stop patience (epochs w/o improvement)",
    "loss": "Loss: mse, mae, huber",
    "val_split": "Validation fraction in [0,0.5) for early stopping",
    "grad_clip_norm": "Gradient clipping max norm (0 disables)",
    "optimizer": "Optimizer: adam, adamw, sgd",
    "momentum": "Momentum (only for sgd)",
    "scheduler": "LR scheduler: none, cosine, step, plateau, onecycle, cosine_restarts",
    "scheduler_step_size": "StepLR step_size (only for scheduler=step)",
    "scheduler_gamma": "StepLR gamma (only for scheduler=step)",
    "restore_best": "Restore best checkpoint at end (true/false)",
    "min_epochs": "Minimum epochs before early stopping can trigger",
    "amp": "Enable CUDA automatic mixed precision (true/false)",
    "amp_dtype": "AMP compute dtype: auto, float16, bfloat16",
    "warmup_epochs": "Linear LR warmup epochs before the main scheduler",
    "min_lr": "Lower bound for learning rate during scheduler updates",
    "scheduler_restart_period": "Initial restart period in epochs for scheduler=cosine_restarts",
    "scheduler_restart_mult": "Cycle-length multiplier for scheduler=cosine_restarts",
    "scheduler_pct_start": "Warmup fraction for scheduler=onecycle, must be in (0,1)",
    "grad_accum_steps": "Gradient accumulation steps (>=1)",
    "monitor": "Early-stop metric: auto, train_loss, val_loss",
    "monitor_mode": "Whether the monitor should be minimized or maximized: min, max",
    "min_delta": "Minimum improvement required to reset patience",
    "num_workers": "DataLoader worker count (0 uses main process)",
    "pin_memory": "Pin DataLoader memory before host-to-device transfer",
    "persistent_workers": "Keep DataLoader workers alive across epochs (requires num_workers>0)",
    "scheduler_patience": "ReduceLROnPlateau patience in epochs (only for scheduler=plateau)",
    "grad_clip_mode": "Gradient clipping strategy: norm, value",
    "grad_clip_value": "Gradient clipping absolute value threshold (only for grad_clip_mode=value)",
    "scheduler_plateau_factor": "ReduceLROnPlateau decay factor in (0,1) (only for scheduler=plateau)",
    "scheduler_plateau_threshold": "ReduceLROnPlateau minimum monitored change before a decay step",
    "ema_decay": "EMA decay in [0,1); 0 disables exponential moving average weights",
    "ema_warmup_epochs": "Warmup epochs before EMA updates start",
    "swa_start_epoch": "Epoch index where stochastic weight averaging starts; -1 disables SWA",
    "lookahead_steps": "Lookahead sync interval in optimizer steps; 0 disables Lookahead",
    "lookahead_alpha": "Lookahead slow-weight interpolation factor in (0,1]",
    "sam_rho": "SAM neighborhood size rho; 0 disables Sharpness-Aware Minimization",
    "sam_adaptive": "Use adaptive SAM scaling based on parameter magnitudes",
    "horizon_loss_decay": "Per-step exponential horizon loss decay (>0); 1 disables weighting",
    "input_dropout": "Feature dropout applied to training inputs only; 0 disables",
    "temporal_dropout": "Drop whole training timesteps across all features; 0 disables",
    "grad_noise_std": "Gradient noise stddev before AGC/clipping; 0 disables",
    "gc_mode": "Gradient centralization mode: off, all, conv_only",
    "agc_clip_factor": "Adaptive Gradient Clipping factor; 0 disables AGC",
    "agc_eps": "Adaptive Gradient Clipping epsilon for parameter-norm stabilization",
    "checkpoint_dir": "Directory for trainer checkpoints (writes best.pt/last.pt when enabled)",
    "save_best_checkpoint": "Persist the best training checkpoint to checkpoint_dir/best.pt",
    "save_last_checkpoint": "Persist the last training checkpoint to checkpoint_dir/last.pt",
    "resume_checkpoint_path": "Load initial model weights from this checkpoint path before training",
    "resume_checkpoint_strict": "Use strict state_dict loading when resume_checkpoint_path is set",
    "tensorboard_log_dir": "Optional TensorBoard log root directory; enables SummaryWriter export when set",
    "tensorboard_run_name": "Optional TensorBoard run subdirectory name (default: timestamped run-* name)",
    "tensorboard_flush_secs": "TensorBoard flush interval in seconds (>=1)",
    "mlflow_tracking_uri": "Optional MLflow tracking URI; uses the MLflow default backend when unset",
    "mlflow_experiment_name": "Optional MLflow experiment name; enables MLflow tracking when set",
    "mlflow_run_name": "Optional MLflow run name (default: timestamped run-* name)",
    "wandb_project": "Optional Weights & Biases project; enables W&B tracking when set",
    "wandb_entity": "Optional Weights & Biases entity / team for wandb_project",
    "wandb_run_name": "Optional Weights & Biases run name",
    "wandb_dir": "Optional Weights & Biases local run directory",
    "wandb_mode": "Weights & Biases mode: online, offline, disabled",
}

_LAG_DERIVED_DEFAULTS: dict[str, Any] = {"roll_windows": (), "roll_stats": (), "diff_lags": ()}

_LAG_DERIVED_PARAM_HELP: dict[str, str] = {
    "roll_windows": "Optional rolling windows (<=lags) for derived lag stats, e.g. 3,7,14",
    "roll_stats": "Derived stats per window: mean,std,min,max,median,slope,iqr,mad,skew,kurt",
    "diff_lags": "Optional diffs: diff_k = last - lag(k+1); each k must be < lags",
}

_SEASONAL_FOURIER_DEFAULTS: dict[str, Any] = {
    "seasonal_lags": (),
    "seasonal_diff_lags": (),
    "fourier_periods": (),
    "fourier_orders": 2,
}

_SEASONAL_FOURIER_PARAM_HELP: dict[str, str] = {
    "seasonal_lags": "Optional seasonal lags (in steps): y[t-p], e.g. 7,14",
    "seasonal_diff_lags": "Optional seasonal diffs: y[t-1]-y[t-1-p], e.g. 7",
    "fourier_periods": "Optional Fourier periods (in steps), e.g. 7,365",
    "fourier_orders": "Fourier harmonic order(s): int or list matching periods (default 2)",
}


def _resolve_target_lags_param(*, lags: Any, target_lags: Any = ()) -> Any:
    spec = target_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return lags
    return spec


def _factory_naive_last(**_params: Any) -> ForecasterFn:
    return naive_last


def _factory_seasonal_naive(*, season_length: int = 12, **_params: Any) -> ForecasterFn:
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_naive(train, horizon, season_length=season_length_int)

    return _f


def _factory_seasonal_naive_auto(
    *,
    min_season_length: int = 2,
    max_season_length: int = 24,
    detrend: bool = True,
    min_corr: float = 0.2,
    **_params: Any,
) -> ForecasterFn:
    min_season_length_int = int(min_season_length)
    max_season_length_int = int(max_season_length)
    detrend_bool = bool(detrend)
    min_corr_f = float(min_corr)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_naive_auto(
            train,
            horizon,
            min_season_length=min_season_length_int,
            max_season_length=max_season_length_int,
            detrend=detrend_bool,
            min_corr=min_corr_f,
        )

    return _f


def _factory_mean(**_params: Any) -> ForecasterFn:
    return mean_forecast


def _factory_median(**_params: Any) -> ForecasterFn:
    return median_forecast


def _factory_drift(**_params: Any) -> ForecasterFn:
    return drift_forecast


def _factory_moving_average(*, window: int = 3, **_params: Any) -> ForecasterFn:
    window_int = int(window)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return moving_average_forecast(train, horizon, window=window_int)

    return _f


def _factory_weighted_moving_average(*, window: int = 3, **_params: Any) -> ForecasterFn:
    window_int = int(window)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return weighted_moving_average_forecast(train, horizon, window=window_int)

    return _f


def _factory_moving_median(*, window: int = 3, **_params: Any) -> ForecasterFn:
    window_int = int(window)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return moving_median_forecast(train, horizon, window=window_int)

    return _f


def _factory_seasonal_mean(*, season_length: int = 12, **_params: Any) -> ForecasterFn:
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_mean_forecast(train, horizon, season_length=season_length_int)

    return _f


def _factory_seasonal_drift(*, season_length: int = 12, **_params: Any) -> ForecasterFn:
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_drift_forecast(train, horizon, season_length=season_length_int)

    return _f


def _factory_ses(*, alpha: float = 0.2, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ses_forecast(train, horizon, alpha=alpha_f)

    return _f


def _factory_holt(*, alpha: float = 0.2, beta: float = 0.1, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)
    beta_f = float(beta)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_forecast(train, horizon, alpha=alpha_f, beta=beta_f)

    return _f


def _factory_holt_damped(
    *,
    alpha: float = 0.2,
    beta: float = 0.1,
    phi: float = 0.9,
    **_params: Any,
) -> ForecasterFn:
    alpha_f = float(alpha)
    beta_f = float(beta)
    phi_f = float(phi)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_damped_forecast(train, horizon, alpha=alpha_f, beta=beta_f, phi=phi_f)

    return _f


def _factory_holt_winters_add(
    *,
    season_length: int = 12,
    alpha: float = 0.2,
    beta: float = 0.1,
    gamma: float = 0.1,
    **_params: Any,
) -> ForecasterFn:
    season_length_int = int(season_length)
    alpha_f = float(alpha)
    beta_f = float(beta)
    gamma_f = float(gamma)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_winters_additive_forecast(
            train,
            horizon,
            season_length=season_length_int,
            alpha=alpha_f,
            beta=beta_f,
            gamma=gamma_f,
        )

    return _f


def _factory_holt_winters_mul(
    *,
    season_length: int = 12,
    alpha: float = 0.2,
    beta: float = 0.1,
    gamma: float = 0.1,
    **_params: Any,
) -> ForecasterFn:
    season_length_int = int(season_length)
    alpha_f = float(alpha)
    beta_f = float(beta)
    gamma_f = float(gamma)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_winters_multiplicative_forecast(
            train,
            horizon,
            season_length=season_length_int,
            alpha=alpha_f,
            beta=beta_f,
            gamma=gamma_f,
        )

    return _f


def _factory_theta(*, alpha: float = 0.2, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return theta_forecast(train, horizon, alpha=alpha_f)

    return _f


def _factory_theta_auto(*, grid_size: int = 19, **_params: Any) -> ForecasterFn:
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return theta_auto_forecast(train, horizon, grid_size=grid_size_int)

    return _f


def _factory_ar_ols(*, p: int = 5, **_params: Any) -> ForecasterFn:
    p_int = int(p)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ar_ols_forecast(train, horizon, p=p_int)

    return _f


def _factory_ar_ols_lags(*, lags: Any = (1, 2, 3, 4, 5), **_params: Any) -> ForecasterFn:
    def _f(train: Any, horizon: int) -> np.ndarray:
        return ar_ols_lags_forecast(train, horizon, lags=lags)

    return _f


def _factory_sar_ols(
    *,
    p: int = 1,
    seasonal_ar_order: int = 1,
    season_length: int = 12,
    **_params: Any,
) -> ForecasterFn:
    legacy_seasonal_ar_order = _params.pop("P", seasonal_ar_order)
    p_int = int(p)
    seasonal_ar_order_int = int(legacy_seasonal_ar_order)
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return sar_ols_forecast(
            train,
            horizon,
            p=p_int,
            P=seasonal_ar_order_int,
            season_length=season_length_int,
        )

    return _f


def _factory_ar_ols_auto(*, max_p: int = 10, **_params: Any) -> ForecasterFn:
    max_p_int = int(max_p)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ar_ols_auto_forecast(train, horizon, max_p=max_p_int)

    return _f


def _factory_lr_lag(
    *,
    lags: Any = 5,
    target_lags: Any = (),
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lag_spec = _resolve_target_lags_param(lags=lags, target_lags=target_lags)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lr_lag_forecast(
            train,
            horizon,
            lags=lag_spec,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lr_lag_direct(
    *,
    lags: Any = 5,
    target_lags: Any = (),
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lag_spec = _resolve_target_lags_param(lags=lags, target_lags=target_lags)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lr_lag_direct_forecast(
            train,
            horizon,
            lags=lag_spec,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_ridge_lag(
    *,
    lags: Any = 5,
    target_lags: Any = (),
    alpha: float = 1.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lag_spec = _resolve_target_lags_param(lags=lags, target_lags=target_lags)
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ridge_lag_forecast(
            train,
            horizon,
            lags=lag_spec,
            alpha=alpha_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_rf_lag(
    *,
    lags: int = 5,
    n_estimators: int = 200,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return rf_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lasso_lag(
    *,
    lags: int = 10,
    alpha: float = 0.001,
    max_iter: int = 5000,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lasso_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_elasticnet_lag(
    *,
    lags: int = 10,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    l1_ratio_f = float(l1_ratio)
    max_iter_int = int(max_iter)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return elasticnet_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            l1_ratio=l1_ratio_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_knn_lag(
    *,
    lags: int = 12,
    n_neighbors: int = 10,
    weights: str = "distance",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_neighbors_int = int(n_neighbors)
    weights_s = str(weights)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return knn_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_neighbors=n_neighbors_int,
            weights=weights_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_gbrt_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return gbrt_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_ridge_lag_direct(
    *,
    lags: Any = 12,
    target_lags: Any = (),
    alpha: float = 1.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lag_spec = _resolve_target_lags_param(lags=lags, target_lags=target_lags)
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ridge_lag_direct_forecast(
            train,
            horizon,
            lags=lag_spec,
            alpha=alpha_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_bayesian_ridge_lag(
    *,
    lags: int = 12,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_1_f = float(alpha_1)
    alpha_2_f = float(alpha_2)
    lambda_1_f = float(lambda_1)
    lambda_2_f = float(lambda_2)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return bayesian_ridge_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha_1=alpha_1_f,
            alpha_2=alpha_2_f,
            lambda_1=lambda_1_f,
            lambda_2=lambda_2_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_ard_lag(
    *,
    lags: int = 12,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    max_iter: int = 300,
    tol: float = 1e-3,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_1_f = float(alpha_1)
    alpha_2_f = float(alpha_2)
    lambda_1_f = float(lambda_1)
    lambda_2_f = float(lambda_2)
    max_iter_int = int(max_iter)
    tol_f = float(tol)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ard_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha_1=alpha_1_f,
            alpha_2=alpha_2_f,
            lambda_1=lambda_1_f,
            lambda_2=lambda_2_f,
            max_iter=max_iter_int,
            tol=tol_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_omp_lag(
    *,
    lags: int = 12,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_nonzero_opt = None if n_nonzero_coefs is None else int(n_nonzero_coefs)
    tol_opt = None if tol is None else float(tol)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return omp_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_nonzero_coefs=n_nonzero_opt,
            tol=tol_opt,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_passive_aggressive_lag(
    *,
    lags: int = 12,
    c: float = 1.0,
    epsilon: float = 0.1,
    max_iter: int = 1000,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    legacy_c = _params.pop("C", c)
    lags_int = int(lags)
    c_value = float(legacy_c)
    epsilon_f = float(epsilon)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return passive_aggressive_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            C=c_value,
            epsilon=epsilon_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_decision_tree_lag(
    *,
    lags: int = 12,
    max_depth: int | None = 5,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    max_depth_opt = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return decision_tree_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            max_depth=max_depth_opt,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_extra_trees_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_depth_opt = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return extra_trees_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            max_depth=max_depth_opt,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_adaboost_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return adaboost_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_bagging_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_samples_f = float(max_samples)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return bagging_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            max_samples=max_samples_f,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_hgb_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    max_iter_int = int(max_iter)
    learning_rate_f = float(learning_rate)
    max_depth_opt = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return hgb_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            max_iter=max_iter_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_opt,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_svr_lag(
    *,
    lags: int = 12,
    c: float = 1.0,
    gamma: Any = "scale",
    epsilon: float = 0.1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    legacy_c = _params.pop("C", c)
    lags_int = int(lags)
    c_value = float(legacy_c)
    gamma_v: Any = gamma
    epsilon_f = float(epsilon)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return svr_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            C=c_value,
            gamma=gamma_v,
            epsilon=epsilon_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_linear_svr_lag(
    *,
    lags: int = 12,
    c: float = 1.0,
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
    **_params: Any,
) -> ForecasterFn:
    legacy_c = _params.pop("C", c)
    lags_int = int(lags)
    c_value = float(legacy_c)
    epsilon_f = float(epsilon)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return linear_svr_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            C=c_value,
            epsilon=epsilon_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_kernel_ridge_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    kernel_s = str(kernel)
    gamma_opt = None if gamma is None else float(gamma)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return kernel_ridge_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            kernel=kernel_s,
            gamma=gamma_opt,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_mlp_lag(
    *,
    lags: int = 12,
    hidden_layer_sizes: Any = (64, 64),
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)
    learning_rate_init_f = float(learning_rate_init)

    sizes_raw = hidden_layer_sizes
    if isinstance(sizes_raw, tuple | list):
        sizes = tuple(int(s) for s in sizes_raw)
    else:
        sizes = (int(sizes_raw),)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mlp_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_layer_sizes=sizes,
            alpha=alpha_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
            learning_rate_init=learning_rate_init_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_huber_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    epsilon_f = float(epsilon)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return huber_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            epsilon=epsilon_f,
            alpha=alpha_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_poisson_lag(
    *,
    lags: int = 12,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    fourier_orders_int = int(fourier_orders)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return poisson_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders_int,
        )

    return _f


def _factory_gamma_lag(
    *,
    lags: int = 12,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    fourier_orders_int = int(fourier_orders)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return gamma_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders_int,
        )

    return _f


def _factory_tweedie_lag(
    *,
    lags: int = 12,
    power: float = 1.5,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    power_f = float(power)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    fourier_orders_int = int(fourier_orders)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tweedie_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            power=power_f,
            alpha=alpha_f,
            max_iter=max_iter_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders_int,
        )

    return _f


def _factory_quantile_lag(
    *,
    lags: int = 12,
    quantile: float = 0.5,
    alpha: float = 0.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    quantile_f = float(quantile)
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return quantile_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            quantile=quantile_f,
            alpha=alpha_f,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_sgd_lag(
    *,
    lags: int = 12,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)
    penalty_s = str(penalty)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return sgd_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            alpha=alpha_f,
            penalty=penalty_s,
            max_iter=max_iter_int,
            random_state=random_state_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_custom_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **lgbm_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(lgbm_params)
    params.setdefault("boosting_type", "gbdt")
    params.setdefault("objective", "regression")
    params.setdefault("verbosity", -1)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_custom_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            lgbm_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_custom_lag_recursive(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **lgbm_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(lgbm_params)
    params.setdefault("boosting_type", "gbdt")
    params.setdefault("objective", "regression")
    params.setdefault("verbosity", -1)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_custom_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            lgbm_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_custom_step_lag(
    *,
    lags: int = 24,
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **lgbm_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    step_scale_s = str(step_scale)
    params = dict(lgbm_params)
    params.setdefault("boosting_type", "gbdt")
    params.setdefault("objective", "regression")
    params.setdefault("verbosity", -1)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_custom_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            lgbm_params=dict(params),
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_custom_dirrec_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **lgbm_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(lgbm_params)
    params.setdefault("boosting_type", "gbdt")
    params.setdefault("objective", "regression")
    params.setdefault("verbosity", -1)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_custom_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            lgbm_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    num_leaves_int = int(num_leaves)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            num_leaves=num_leaves_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    num_leaves_int = int(num_leaves)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            num_leaves=num_leaves_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_step_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    num_leaves_int = int(num_leaves)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    step_scale_s = str(step_scale)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            num_leaves=num_leaves_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_lgbm_dirrec_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    num_leaves_int = int(num_leaves)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lgbm_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            num_leaves=num_leaves_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_custom_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **cb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_custom_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_custom_lag_recursive(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **cb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_custom_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            cb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_custom_step_lag(
    *,
    lags: int = 24,
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **cb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    step_scale_s = str(step_scale)
    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_custom_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cb_params=dict(params),
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_custom_dirrec_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **cb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(cb_params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_custom_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            cb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    iterations_int = int(iterations)
    learning_rate_f = float(learning_rate)
    depth_int = int(depth)
    l2_leaf_reg_f = float(l2_leaf_reg)
    random_seed_int = int(random_seed)
    thread_count_int = int(thread_count)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            iterations=iterations_int,
            learning_rate=learning_rate_f,
            depth=depth_int,
            l2_leaf_reg=l2_leaf_reg_f,
            random_seed=random_seed_int,
            thread_count=thread_count_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    iterations_int = int(iterations)
    learning_rate_f = float(learning_rate)
    depth_int = int(depth)
    l2_leaf_reg_f = float(l2_leaf_reg)
    random_seed_int = int(random_seed)
    thread_count_int = int(thread_count)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            iterations=iterations_int,
            learning_rate=learning_rate_f,
            depth=depth_int,
            l2_leaf_reg=l2_leaf_reg_f,
            random_seed=random_seed_int,
            thread_count=thread_count_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_step_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    iterations_int = int(iterations)
    learning_rate_f = float(learning_rate)
    depth_int = int(depth)
    l2_leaf_reg_f = float(l2_leaf_reg)
    random_seed_int = int(random_seed)
    thread_count_int = int(thread_count)
    step_scale_s = str(step_scale)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            iterations=iterations_int,
            learning_rate=learning_rate_f,
            depth=depth_int,
            l2_leaf_reg=l2_leaf_reg_f,
            random_seed=random_seed_int,
            thread_count=thread_count_int,
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_catboost_dirrec_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    iterations_int = int(iterations)
    learning_rate_f = float(learning_rate)
    depth_int = int(depth)
    l2_leaf_reg_f = float(l2_leaf_reg)
    random_seed_int = int(random_seed)
    thread_count_int = int(thread_count)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return catboost_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            iterations=iterations_int,
            learning_rate=learning_rate_f,
            depth=depth_int,
            l2_leaf_reg=l2_leaf_reg_f,
            random_seed=random_seed_int,
            thread_count=thread_count_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_custom_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **xgb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(xgb_params)
    params.setdefault("booster", "gbtree")
    params.setdefault("objective", _XGB_REG_SQUAREDERROR_OBJECTIVE)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_custom_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_custom_lag_recursive(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **xgb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    params = dict(xgb_params)
    params.setdefault("booster", "gbtree")
    params.setdefault("objective", _XGB_REG_SQUAREDERROR_OBJECTIVE)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_custom_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_custom_step_lag(
    *,
    lags: int = 24,
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **xgb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    step_scale_s = str(step_scale)

    params = dict(xgb_params)
    params.setdefault("booster", "gbtree")
    params.setdefault("objective", _XGB_REG_SQUAREDERROR_OBJECTIVE)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(params),
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_custom_dirrec_lag(
    *,
    lags: int = 24,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **xgb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)

    params = dict(xgb_params)
    params.setdefault("booster", "gbtree")
    params.setdefault("objective", _XGB_REG_SQUAREDERROR_OBJECTIVE)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_custom_mimo_lag(
    *,
    lags: int = 24,
    multi_strategy: str = "multi_output_tree",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **xgb_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    multi_strategy_s = str(multi_strategy)

    params = dict(xgb_params)
    params.setdefault("booster", "gbtree")
    params.setdefault("objective", _XGB_REG_SQUAREDERROR_OBJECTIVE)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_mimo_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(params),
            multi_strategy=multi_strategy_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_step_lag(
    *,
    lags: int = 24,
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
    step_scale: str = "one_based",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)
    step_scale_s = str(step_scale)

    xgb_params = {
        "booster": "gbtree",
        "objective": _XGB_REG_SQUAREDERROR_OBJECTIVE,
        "n_estimators": n_estimators_int,
        "learning_rate": learning_rate_f,
        "max_depth": max_depth_int,
        "subsample": subsample_f,
        "colsample_bytree": colsample_bytree_f,
        "reg_alpha": reg_alpha_f,
        "reg_lambda": reg_lambda_f,
        "min_child_weight": min_child_weight_f,
        "gamma": gamma_f,
        "random_state": random_state_int,
        "n_jobs": n_jobs_int,
        "tree_method": tree_method_s,
    }

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_step_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(xgb_params),
            step_scale=step_scale_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_dirrec_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    xgb_params = {
        "booster": "gbtree",
        "objective": _XGB_REG_SQUAREDERROR_OBJECTIVE,
        "n_estimators": n_estimators_int,
        "learning_rate": learning_rate_f,
        "max_depth": max_depth_int,
        "subsample": subsample_f,
        "colsample_bytree": colsample_bytree_f,
        "reg_alpha": reg_alpha_f,
        "reg_lambda": reg_lambda_f,
        "min_child_weight": min_child_weight_f,
        "gamma": gamma_f,
        "random_state": random_state_int,
        "n_jobs": n_jobs_int,
        "tree_method": tree_method_s,
    }

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_dirrec_lag_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(xgb_params),
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_mimo_lag(
    *,
    lags: int = 24,
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
    multi_strategy: str = "multi_output_tree",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    seasonal_lags: Any = (),
    seasonal_diff_lags: Any = (),
    fourier_periods: Any = (),
    fourier_orders: Any = 2,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)
    multi_strategy_s = str(multi_strategy)

    xgb_params = {
        "booster": "gbtree",
        "objective": _XGB_REG_SQUAREDERROR_OBJECTIVE,
        "n_estimators": n_estimators_int,
        "learning_rate": learning_rate_f,
        "max_depth": max_depth_int,
        "subsample": subsample_f,
        "colsample_bytree": colsample_bytree_f,
        "reg_alpha": reg_alpha_f,
        "reg_lambda": reg_lambda_f,
        "min_child_weight": min_child_weight_f,
        "gamma": gamma_f,
        "random_state": random_state_int,
        "n_jobs": n_jobs_int,
        "tree_method": tree_method_s,
    }

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_mimo_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            xgb_params=dict(xgb_params),
            multi_strategy=multi_strategy_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_msle_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_msle_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_msle_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_msle_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_logistic_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_logistic_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_logistic_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_logistic_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_dart_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_dart_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_dart_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_dart_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgbrf_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgbrf_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgbrf_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgbrf_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_linear_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_linear_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_linear_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_linear_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_mae_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_mae_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_mae_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_mae_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            seasonal_lags=seasonal_lags,
            seasonal_diff_lags=seasonal_diff_lags,
            fourier_periods=fourier_periods,
            fourier_orders=fourier_orders,
        )

    return _f


def _factory_xgb_huber_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    huber_slope_f = float(huber_slope)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_huber_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            huber_slope=huber_slope_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_huber_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    huber_slope_f = float(huber_slope)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_huber_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            huber_slope=huber_slope_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_quantile_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    quantile_alpha_f = float(quantile_alpha)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_quantile_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            quantile_alpha=quantile_alpha_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_quantile_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    quantile_alpha_f = float(quantile_alpha)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_quantile_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            quantile_alpha=quantile_alpha_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_poisson_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_poisson_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_poisson_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_poisson_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_gamma_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_gamma_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_gamma_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_gamma_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_tweedie_lag(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    tweedie_variance_power_f = float(tweedie_variance_power)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_tweedie_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            tweedie_variance_power=tweedie_variance_power_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_xgb_tweedie_lag_recursive(
    *,
    lags: int = 24,
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
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    subsample_f = float(subsample)
    colsample_bytree_f = float(colsample_bytree)
    reg_alpha_f = float(reg_alpha)
    reg_lambda_f = float(reg_lambda)
    min_child_weight_f = float(min_child_weight)
    gamma_f = float(gamma)
    tweedie_variance_power_f = float(tweedie_variance_power)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    tree_method_s = str(tree_method)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return xgb_tweedie_lag_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            subsample=subsample_f,
            colsample_bytree=colsample_bytree_f,
            reg_alpha=reg_alpha_f,
            reg_lambda=reg_lambda_f,
            min_child_weight=min_child_weight_f,
            gamma=gamma_f,
            tweedie_variance_power=tweedie_variance_power_f,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
            tree_method=tree_method_s,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )

    return _f


def _factory_torch_mlp_direct(
    *,
    lags: int = 24,
    hidden_sizes: Any = (64, 64),
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_mlp_lag_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_sizes=hidden_sizes,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_lstm_direct(
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_lstm_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_gru_direct(
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_gru_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_tcn_direct(
    *,
    lags: int = 24,
    channels: Any = (16, 16, 16),
    kernel_size: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_tcn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            channels=channels,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_nbeats_direct(
    *,
    lags: int = 48,
    num_blocks: int = 3,
    num_layers: int = 2,
    layer_width: int = 64,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    num_blocks_int = int(num_blocks)
    num_layers_int = int(num_layers)
    layer_width_int = int(layer_width)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_nbeats_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            num_blocks=num_blocks_int,
            num_layers=num_layers_int,
            layer_width=layer_width_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_nlinear_direct(
    *,
    lags: int = 48,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_nlinear_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_dlinear_direct(
    *,
    lags: int = 48,
    ma_window: int = 25,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    ma_window_int = int(ma_window)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_dlinear_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            ma_window=ma_window_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_transformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_transformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_informer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_informer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_autoformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    ma_window: int = 7,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    ma_window_int = int(ma_window)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_autoformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            ma_window=ma_window_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_nonstationary_transformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_nonstationary_transformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_fedformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 256,
    modes: int = 16,
    ma_window: int = 7,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    modes_int = int(modes)
    ma_window_int = int(ma_window)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_fedformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            modes=modes_int,
            ma_window=ma_window_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_itransformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_itransformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_timesnet_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    top_k: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    top_k_int = int(top_k)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_timesnet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            top_k=top_k_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_tft_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    lstm_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    lstm_layers_int = int(lstm_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_tft_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            lstm_layers=lstm_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_timemixer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_blocks: int = 4,
    multiscale_factors: Any = (1, 2, 4),
    token_mixing_hidden: int = 128,
    channel_mixing_hidden: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    token_mixing_hidden_int = int(token_mixing_hidden)
    channel_mixing_hidden_int = int(channel_mixing_hidden)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_timemixer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            multiscale_factors=multiscale_factors,
            token_mixing_hidden=token_mixing_hidden_int,
            channel_mixing_hidden=channel_mixing_hidden_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_tinytimemixer_direct(
    *,
    lags: int = 96,
    patch_len: int = 8,
    d_model: int = 64,
    num_blocks: int = 4,
    token_mixing_hidden: int = 128,
    channel_mixing_hidden: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    patch_len_int = int(patch_len)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    token_mixing_hidden_int = int(token_mixing_hidden)
    channel_mixing_hidden_int = int(channel_mixing_hidden)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_tinytimemixer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            patch_len=patch_len_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            token_mixing_hidden=token_mixing_hidden_int,
            channel_mixing_hidden=channel_mixing_hidden_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,
            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_sparsetsf_direct(
    *,
    lags: int = 192,
    period_len: int = 24,
    d_model: int = 64,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    period_len_int = int(period_len)
    d_model_int = int(d_model)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_sparsetsf_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            period_len=period_len_int,
            d_model=d_model_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_lightts_direct(
    *,
    lags: int = 96,
    chunk_len: int = 12,
    d_model: int = 64,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    chunk_len_int = int(chunk_len)
    d_model_int = int(d_model)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_lightts_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            chunk_len=chunk_len_int,
            d_model=d_model_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_frets_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    top_k_freqs: int = 8,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    top_k_freqs_int = int(top_k_freqs)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_frets_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            top_k_freqs=top_k_freqs_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_film_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ma_window: int = 7,
    kernel_size: int = 7,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ma_window_int = int(ma_window)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_film_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            ma_window=ma_window_int,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_micn_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    kernel_sizes: Any = (3, 5, 7),
    ma_window: int = 7,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ma_window_int = int(ma_window)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_micn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            kernel_sizes=kernel_sizes,
            ma_window=ma_window_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_koopa_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    latent_dim: int = 32,
    num_blocks: int = 2,
    ma_window: int = 7,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    latent_dim_int = int(latent_dim)
    num_blocks_int = int(num_blocks)
    ma_window_int = int(ma_window)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_koopa_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            latent_dim=latent_dim_int,
            num_blocks=num_blocks_int,
            ma_window=ma_window_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_samformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_samformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_retnet_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_retnet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_retnet_recursive(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_retnet_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_timexer_direct(
    *,
    x_cols: Any = (),
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)
    x_cols_v = x_cols

    def _f(
        train: Any,
        horizon: int,
        *,
        train_exog: Any | None = None,
        future_exog: Any | None = None,
    ) -> np.ndarray:
        return torch_timexer_direct_forecast(
            train,
            horizon,
            x_cols=x_cols_v,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
            train_exog=train_exog,
            future_exog=future_exog,
        )

    return _f


def _factory_torch_fits_direct(
    *,
    lags: int = 96,
    low_freq_bins: int = 12,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    low_freq_bins_int = int(low_freq_bins)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_fits_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            low_freq_bins=low_freq_bins_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,
            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_lmu_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    memory_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    memory_dim_int = int(memory_dim)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_lmu_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            memory_dim=memory_dim_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_ltc_direct(
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_ltc_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_cfc_direct(
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    backbone_hidden: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    backbone_hidden_int = int(backbone_hidden)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_cfc_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            backbone_hidden=backbone_hidden_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_xlstm_direct(
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    proj_factor: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    proj_factor_int = int(proj_factor)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_xlstm_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            proj_factor=proj_factor_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_griffin_direct(
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    conv_kernel: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    conv_kernel_int = int(conv_kernel)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_griffin_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            conv_kernel=conv_kernel_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_hawk_direct(
    *,
    lags: int = 96,
    hidden_size: int = 64,
    num_layers: int = 1,
    expansion_factor: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    expansion_factor_int = int(expansion_factor)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_hawk_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            expansion_factor=expansion_factor_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_s4d_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_s4d_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_mamba2_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    conv_kernel: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    conv_kernel_int = int(conv_kernel)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_mamba2_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            conv_kernel=conv_kernel_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_s4_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    state_dim: int = 32,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    state_dim_int = int(state_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_s4_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            state_dim=state_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_s5_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    state_dim: int = 32,
    heads: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    state_dim_int = int(state_dim)
    heads_int = int(heads)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_s5_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            state_dim=state_dim_int,
            heads=heads_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_timexer_global(
    *,
    context_length: int = 96,
    x_cols: Any = (),
    static_cols: Any = (),
    add_time_features: bool = True,
    normalize: bool = True,
    max_train_size: int | None = None,
    sample_step: int = 1,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    seed: int = 0,
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    device: str = "cpu",
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    id_emb_dim: int = 8,
    dropout: float = 0.1,
    **_params: Any,
) -> GlobalForecasterFn:
    return torch_timexer_global_forecaster(
        context_length=int(context_length),
        x_cols=x_cols,
        static_cols=static_cols,
        add_time_features=bool(add_time_features),
        normalize=bool(normalize),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        patience=int(patience),
        loss=str(loss),
        val_split=float(val_split),
        grad_clip_norm=float(grad_clip_norm),
        optimizer=str(optimizer),
        momentum=float(momentum),
        scheduler=str(scheduler),
        scheduler_step_size=int(scheduler_step_size),
        scheduler_gamma=float(scheduler_gamma),
        scheduler_restart_period=int(scheduler_restart_period),
        scheduler_restart_mult=int(scheduler_restart_mult),
        scheduler_pct_start=float(scheduler_pct_start),
        restore_best=bool(restore_best),

        **_coerce_torch_extra_train_params(_params),
        device=str(device),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        id_emb_dim=int(id_emb_dim),
        dropout=float(dropout),
    )


def _factory_torch_mamba_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    conv_kernel: int = 3,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    conv_kernel_int = int(conv_kernel)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_mamba_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            conv_kernel=conv_kernel_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_rwkv_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_rwkv_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_hyena_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    ffn_dim: int = 128,
    kernel_size: int = 64,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_hyena_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_dilated_rnn_direct(
    *,
    lags: int = 96,
    cell: str = "gru",
    hidden_size: int = 64,
    num_layers: int = 3,
    dilation_base: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    cell_s = str(cell)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dilation_base_int = int(dilation_base)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_dilated_rnn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cell=cell_s,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dilation_base=dilation_base_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_kan_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    grid_size: int = 16,
    grid_range: float = 2.0,
    dropout: float = 0.1,
    linear_skip: bool = True,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    grid_size_int = int(grid_size)
    grid_range_f = float(grid_range)
    dropout_f = float(dropout)
    linear_skip_bool = bool(linear_skip)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_kan_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            grid_size=grid_size_int,
            grid_range=grid_range_f,
            dropout=dropout_f,
            linear_skip=linear_skip_bool,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_scinet_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_stages: int = 3,
    conv_kernel: int = 5,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_stages_int = int(num_stages)
    conv_kernel_int = int(conv_kernel)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_scinet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_stages=num_stages_int,
            conv_kernel=conv_kernel_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_etsformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    alpha_init_f = float(alpha_init)
    beta_init_f = float(beta_init)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_etsformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            alpha_init=alpha_init_f,
            beta_init=beta_init_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_esrnn_direct(
    *,
    lags: int = 96,
    cell: str = "gru",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    cell_s = str(cell)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    alpha_init_f = float(alpha_init)
    beta_init_f = float(beta_init)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_esrnn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cell=cell_s,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            alpha_init=alpha_init_f,
            beta_init=beta_init_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_patchtst_direct(
    *,
    lags: int = 192,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    patch_len_int = int(patch_len)
    stride_int = int(stride)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_patchtst_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            patch_len=patch_len_int,
            stride=stride_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_crossformer_direct(
    *,
    lags: int = 192,
    segment_len: int = 16,
    stride: int = 16,
    num_scales: int = 3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    segment_len_int = int(segment_len)
    stride_int = int(stride)
    num_scales_int = int(num_scales)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_crossformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            segment_len=segment_len_int,
            stride=stride_int,
            num_scales=num_scales_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_pyraformer_direct(
    *,
    lags: int = 192,
    segment_len: int = 16,
    stride: int = 16,
    num_levels: int = 3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    segment_len_int = int(segment_len)
    stride_int = int(stride)
    num_levels_int = int(num_levels)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_pyraformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            segment_len=segment_len_int,
            stride=stride_int,
            num_levels=num_levels_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_perceiver_direct(
    *,
    lags: int = 192,
    d_model: int = 64,
    latent_len: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    latent_len_int = int(latent_len)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_perceiver_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            latent_len=latent_len_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_tsmixer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_blocks: int = 4,
    token_mixing_hidden: int = 128,
    channel_mixing_hidden: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    token_mixing_hidden_int = int(token_mixing_hidden)
    channel_mixing_hidden_int = int(channel_mixing_hidden)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_tsmixer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            token_mixing_hidden=token_mixing_hidden_int,
            channel_mixing_hidden=channel_mixing_hidden_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_cnn_direct(
    *,
    lags: int = 48,
    channels: Any = (32, 32, 32),
    kernel_size: int = 3,
    dropout: float = 0.1,
    pool: str = "last",
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    pool_s = str(pool)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_cnn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            channels=channels,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            pool=pool_s,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_resnet1d_direct(
    *,
    lags: int = 96,
    channels: int = 32,
    num_blocks: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    pool: str = "last",
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    channels_int = int(channels)
    num_blocks_int = int(num_blocks)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    pool_s = str(pool)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_resnet1d_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            channels=channels_int,
            num_blocks=num_blocks_int,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            pool=pool_s,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_wavenet_direct(
    *,
    lags: int = 96,
    channels: int = 32,
    num_layers: int = 6,
    kernel_size: int = 2,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    channels_int = int(channels)
    num_layers_int = int(num_layers)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_wavenet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            channels=channels_int,
            num_layers=num_layers_int,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_bilstm_direct(
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_bilstm_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_bigru_direct(
    *,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_bigru_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_attn_gru_direct(
    *,
    lags: int = 48,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_attn_gru_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_segrnn_direct(
    *,
    lags: int = 96,
    segment_len: int = 12,
    d_model: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    segment_len_int = int(segment_len)
    d_model_int = int(d_model)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_segrnn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            segment_len=segment_len_int,
            d_model=d_model_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_moderntcn_direct(
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    num_blocks: int = 3,
    expansion_factor: float = 2.0,
    kernel_size: int = 9,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    patch_len_int = int(patch_len)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    expansion_factor_f = float(expansion_factor)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_moderntcn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            patch_len=patch_len_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            expansion_factor=expansion_factor_f,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_basisformer_direct(
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    num_bases: int = 16,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    patch_len_int = int(patch_len)
    d_model_int = int(d_model)
    num_bases_int = int(num_bases)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_basisformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            patch_len=patch_len_int,
            d_model=d_model_int,
            num_bases=num_bases_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_witran_direct(
    *,
    lags: int = 192,
    grid_cols: int = 12,
    d_model: int = 64,
    hidden_size: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    grid_cols_int = int(grid_cols)
    d_model_int = int(d_model)
    hidden_size_int = int(hidden_size)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_witran_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            grid_cols=grid_cols_int,
            d_model=d_model_int,
            hidden_size=hidden_size_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_crossgnn_direct(
    *,
    lags: int = 192,
    d_model: int = 64,
    num_blocks: int = 3,
    top_k: int = 8,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    top_k_int = int(top_k)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_crossgnn_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            top_k=top_k_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_pathformer_direct(
    *,
    lags: int = 192,
    d_model: int = 64,
    expert_patch_lens: Any = (4, 8, 16),
    num_blocks: int = 3,
    top_k: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    top_k_int = int(top_k)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_pathformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            expert_patch_lens=expert_patch_lens,
            num_blocks=num_blocks_int,
            top_k=top_k_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_timesmamba_direct(
    *,
    lags: int = 192,
    patch_len: int = 8,
    d_model: int = 64,
    state_size: int = 64,
    num_blocks: int = 3,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    patch_len_int = int(patch_len)
    d_model_int = int(d_model)
    state_size_int = int(state_size)
    num_blocks_int = int(num_blocks)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_timesmamba_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            patch_len=patch_len_int,
            d_model=d_model_int,
            state_size=state_size_int,
            num_blocks=num_blocks_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_fnet_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_fnet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_gmlp_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_gmlp_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_nhits_direct(
    *,
    lags: int = 192,
    pool_sizes: Any = (1, 2, 4),
    num_blocks: int = 6,
    num_layers: int = 2,
    layer_width: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    num_blocks_int = int(num_blocks)
    num_layers_int = int(num_layers)
    layer_width_int = int(layer_width)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_nhits_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            pool_sizes=pool_sizes,
            num_blocks=num_blocks_int,
            num_layers=num_layers_int,
            layer_width=layer_width_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_tide_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    hidden_size: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    hidden_size_int = int(hidden_size)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_tide_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            hidden_size=hidden_size_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_deepar_recursive(
    *,
    lags: int = 48,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    # `loss` is accepted for API consistency but DeepAR uses a Gaussian NLL objective.
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_deepar_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_qrnn_recursive(
    *,
    lags: int = 48,
    q: float = 0.5,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    q_f = float(q)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    # `loss` is accepted for API consistency but QRNN uses pinball loss at quantile q.
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_qrnn_recursive_forecast(
            train,
            horizon,
            lags=lags_int,
            q=q_f,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_xformer_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    attn: str = "full",
    pos_emb: str = "learned",
    norm: str = "layer",
    ffn: str = "gelu",
    local_window: int = 16,
    bigbird_random_k: int = 8,
    performer_features: int = 64,
    linformer_k: int = 32,
    nystrom_landmarks: int = 16,
    reformer_bucket_size: int = 8,
    reformer_n_hashes: int = 1,
    probsparse_top_u: int = 32,
    autocorr_top_k: int = 4,
    horizon_tokens: str = "zeros",
    revin: bool = False,
    residual_gating: bool = False,
    drop_path: float = 0.0,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    attn_s = str(attn)
    pos_emb_s = str(pos_emb)
    norm_s = str(norm)
    ffn_s = str(ffn)
    local_window_int = int(local_window)
    bigbird_random_k_int = int(bigbird_random_k)
    performer_features_int = int(performer_features)
    linformer_k_int = int(linformer_k)
    nystrom_landmarks_int = int(nystrom_landmarks)
    reformer_bucket_size_int = int(reformer_bucket_size)
    reformer_n_hashes_int = int(reformer_n_hashes)
    probsparse_top_u_int = int(probsparse_top_u)
    autocorr_top_k_int = int(autocorr_top_k)
    horizon_tokens_s = str(horizon_tokens)
    revin_bool = bool(revin)
    residual_gating_bool = bool(residual_gating)
    drop_path_f = float(drop_path)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_xformer_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            nhead=nhead_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            attn=attn_s,
            pos_emb=pos_emb_s,
            norm=norm_s,
            ffn=ffn_s,
            local_window=local_window_int,
            bigbird_random_k=bigbird_random_k_int,
            performer_features=performer_features_int,
            linformer_k=linformer_k_int,
            nystrom_landmarks=nystrom_landmarks_int,
            reformer_bucket_size=reformer_bucket_size_int,
            reformer_n_hashes=reformer_n_hashes_int,
            probsparse_top_u=probsparse_top_u_int,
            autocorr_top_k=autocorr_top_k_int,
            horizon_tokens=horizon_tokens_s,
            revin=revin_bool,
            residual_gating=residual_gating_bool,
            drop_path=drop_path_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_seq2seq_direct(
    *,
    lags: int = 48,
    cell: str = "lstm",
    attention: str = "none",
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    teacher_forcing: float = 0.5,
    teacher_forcing_final: float | None = None,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.1,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    cell_s = str(cell)
    attention_s = str(attention)
    hidden_size_int = int(hidden_size)
    num_layers_int = int(num_layers)
    dropout_f = float(dropout)
    teacher_forcing_f = float(teacher_forcing)
    teacher_forcing_final_v = (
        None if teacher_forcing_final is None else float(teacher_forcing_final)
    )
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_seq2seq_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cell=cell_s,
            attention=attention_s,
            hidden_size=hidden_size_int,
            num_layers=num_layers_int,
            dropout=dropout_f,
            teacher_forcing=teacher_forcing_f,
            teacher_forcing_final=teacher_forcing_final_v,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_lstnet_direct(
    *,
    lags: int = 96,
    cnn_channels: int = 16,
    kernel_size: int = 6,
    rnn_hidden: int = 32,
    skip: int = 24,
    highway_window: int = 24,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    cnn_channels_int = int(cnn_channels)
    kernel_size_int = int(kernel_size)
    rnn_hidden_int = int(rnn_hidden)
    skip_int = int(skip)
    highway_window_int = int(highway_window)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_lstnet_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            cnn_channels=cnn_channels_int,
            kernel_size=kernel_size_int,
            rnn_hidden=rnn_hidden_int,
            skip=skip_int,
            highway_window=highway_window_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_linear_attn_direct(
    *,
    lags: int = 96,
    d_model: int = 64,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_layers_int = int(num_layers)
    dim_feedforward_int = int(dim_feedforward)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_linear_attention_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_layers=num_layers_int,
            dim_feedforward=dim_feedforward_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_inception_direct(
    *,
    lags: int = 96,
    channels: int = 32,
    num_blocks: int = 3,
    kernel_sizes: Any = (3, 5, 7),
    bottleneck_channels: int = 16,
    dropout: float = 0.1,
    pool: str = "last",
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    channels_int = int(channels)
    num_blocks_int = int(num_blocks)
    bottleneck_channels_int = int(bottleneck_channels)
    dropout_f = float(dropout)
    pool_s = str(pool)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_inception_direct_forecast(
            train,
            horizon,
            lags=lags_int,
            channels=channels_int,
            num_blocks=num_blocks_int,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels_int,
            dropout=dropout_f,
            pool=pool_s,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_ses_auto(*, grid_size: int = 19, **_params: Any) -> ForecasterFn:
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ses_auto_forecast(train, horizon, grid_size=grid_size_int)

    return _f


def _factory_holt_auto(*, grid_size: int = 10, **_params: Any) -> ForecasterFn:
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_auto_forecast(train, horizon, grid_size=grid_size_int)

    return _f


def _factory_holt_winters_add_auto(
    *, season_length: int = 12, grid_size: int = 7, **_params: Any
) -> ForecasterFn:
    season_length_int = int(season_length)
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_winters_additive_auto_forecast(
            train,
            horizon,
            season_length=season_length_int,
            grid_size=grid_size_int,
        )

    return _f


def _factory_holt_winters_mul_auto(
    *, season_length: int = 12, grid_size: int = 7, **_params: Any
) -> ForecasterFn:
    season_length_int = int(season_length)
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return holt_winters_multiplicative_auto_forecast(
            train,
            horizon,
            season_length=season_length_int,
            grid_size=grid_size_int,
        )

    return _f


def _factory_croston(*, alpha: float = 0.1, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return croston_classic_forecast(train, horizon, alpha=alpha_f)

    return _f


def _factory_croston_sba(*, alpha: float = 0.1, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return croston_sba_forecast(train, horizon, alpha=alpha_f)

    return _f


def _factory_croston_sbj(*, alpha: float = 0.1, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return croston_sbj_forecast(train, horizon, alpha=alpha_f)

    return _f


def _factory_croston_opt(*, grid_size: int = 19, **_params: Any) -> ForecasterFn:
    grid_size_int = int(grid_size)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return croston_optimized_forecast(train, horizon, grid_size=grid_size_int)

    return _f


def _factory_les(
    *,
    alpha: float = 0.1,
    beta: float = 0.1,
    **_params: Any,
) -> ForecasterFn:
    alpha_f = float(alpha)
    beta_f = float(beta)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return les_forecast(train, horizon, alpha=alpha_f, beta=beta_f)

    return _f


def _factory_tsb(*, alpha: float = 0.1, beta: float = 0.1, **_params: Any) -> ForecasterFn:
    alpha_f = float(alpha)
    beta_f = float(beta)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tsb_forecast(train, horizon, alpha=alpha_f, beta=beta_f)

    return _f


def _factory_adida(
    *, agg_period: int = 4, base: str = "ses", alpha: float = 0.2, **_params: Any
) -> ForecasterFn:
    agg_period_int = int(agg_period)
    base_s = str(base)
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return adida_forecast(
            train,
            horizon,
            agg_period=agg_period_int,
            base=base_s,
            alpha=alpha_f,
        )

    return _f


def _factory_fourier(
    *, period: int = 12, order: int = 2, include_trend: bool = True, **_params: Any
) -> ForecasterFn:
    period_int = int(period)
    order_int = int(order)
    include_trend_bool = bool(include_trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_regression_forecast(
            train,
            horizon,
            period=period_int,
            order=order_int,
            include_trend=include_trend_bool,
        )

    return _f


def _factory_fourier_multi(
    *,
    periods: Any = (7, 365),
    orders: Any = 2,
    include_trend: bool = True,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = bool(include_trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_multi_regression_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
        )

    return _f


def _factory_poly_trend(*, degree: int = 1, **_params: Any) -> ForecasterFn:
    degree_int = int(degree)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return poly_trend_forecast(train, horizon, degree=degree_int)

    return _f


def _factory_analog_knn(
    *,
    lags: int = 12,
    k: int = 5,
    normalize: bool = True,
    weights: str = "uniform",
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    k_int = int(k)
    normalize_bool = bool(normalize)
    weights_s = str(weights)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return analog_knn_forecast(
            train,
            horizon,
            lags=lags_int,
            k=k_int,
            normalize=normalize_bool,
            weights=weights_s,
        )

    return _f


def _factory_fft(*, top_k: int = 3, include_trend: bool = True, **_params: Any) -> ForecasterFn:
    top_k_int = int(top_k)
    include_trend_bool = bool(include_trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fft_topk_forecast(train, horizon, top_k=top_k_int, include_trend=include_trend_bool)

    return _f


def _factory_ssa(
    *,
    window_length: int = 24,
    rank: int = 5,
    **_params: Any,
) -> ForecasterFn:
    window_length_int = int(window_length)
    rank_int = int(rank)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ssa_forecast(
            train,
            horizon,
            window_length=window_length_int,
            rank=rank_int,
        )

    return _f


def _factory_kalman_level(
    *,
    process_variance: float | None = None,
    obs_variance: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    q = None if process_variance is None else float(process_variance)
    r = None if obs_variance is None else float(obs_variance)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return kalman_local_level_forecast(train, horizon, process_variance=q, obs_variance=r)

    return _f


def _factory_kalman_trend(
    *,
    level_variance: float | None = None,
    trend_variance: float | None = None,
    obs_variance: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    q_level = None if level_variance is None else float(level_variance)
    q_trend = None if trend_variance is None else float(trend_variance)
    r = None if obs_variance is None else float(obs_variance)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return kalman_local_linear_trend_forecast(
            train,
            horizon,
            level_variance=q_level,
            trend_variance=q_trend,
            obs_variance=r,
        )

    return _f


def _factory_pipeline(
    *,
    base: str = "naive-last",
    transforms: Any = (),
    **base_params: Any,
) -> ForecasterFn:
    base_key = str(base).strip()
    if base_key == "pipeline":
        raise ValueError("pipeline base model cannot be 'pipeline'")

    transforms_list = normalize_transform_list(transforms)
    base_forecaster = make_forecaster(base_key, **base_params)

    def _f(train: Any, horizon: int) -> np.ndarray:
        y = np.asarray(train, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"Expected 1D series, got shape {y.shape}")

        states = []
        yt = y
        for name in transforms_list:
            yt, st = fit_transform(str(name), yt)
            states.append(st)

        yhat_t = np.asarray(base_forecaster(yt, int(horizon)), dtype=float)
        if yhat_t.shape != (int(horizon),):
            raise ValueError(
                f"base forecaster must return shape ({int(horizon)},), got {yhat_t.shape}"
            )

        yhat = yhat_t
        for st in reversed(states):
            yhat = inverse_forecast(st, yhat)
        return np.asarray(yhat, dtype=float)

    return _f


def _normalize_members(members: Any) -> tuple[str, ...]:
    if members is None:
        raise ValueError("members must be provided")

    if isinstance(members, str):
        s = members.strip()
        if not s:
            raise ValueError(_MEMBERS_NON_EMPTY_ERROR)
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(parts)

    if isinstance(members, list | tuple):
        parts = [str(m).strip() for m in members if str(m).strip()]
        return tuple(parts)

    s = str(members).strip()
    if not s:
        raise ValueError(_MEMBERS_NON_EMPTY_ERROR)
    return (s,)


def _factory_ensemble_mean(
    *, members: Any = ("naive-last", "seasonal-naive", "theta"), **_p: Any
) -> ForecasterFn:
    member_keys = _normalize_members(members)
    if not member_keys:
        raise ValueError(_MEMBERS_NON_EMPTY_ERROR)
    if any(k == "ensemble-mean" for k in member_keys):
        raise ValueError("ensemble-mean cannot include itself")

    member_forecasters = [make_forecaster(k) for k in member_keys]

    def _f(train: Any, horizon: int) -> np.ndarray:
        preds = [np.asarray(m(train, horizon), dtype=float) for m in member_forecasters]
        arr = np.stack(preds, axis=0)
        return np.mean(arr, axis=0)

    return _f


def _factory_ensemble_median(
    *, members: Any = ("naive-last", "seasonal-naive", "theta"), **_p: Any
) -> ForecasterFn:
    member_keys = _normalize_members(members)
    if not member_keys:
        raise ValueError(_MEMBERS_NON_EMPTY_ERROR)
    if any(k == "ensemble-median" for k in member_keys):
        raise ValueError("ensemble-median cannot include itself")

    member_forecasters = [make_forecaster(k) for k in member_keys]

    def _f(train: Any, horizon: int) -> np.ndarray:
        preds = [np.asarray(m(train, horizon), dtype=float) for m in member_forecasters]
        arr = np.stack(preds, axis=0)
        return np.median(arr, axis=0)

    return _f


def _factory_arima(
    *,
    order: Any = (1, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return arima_forecast(
            train,
            horizon,
            order=order_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
        )

    return _f


def _factory_auto_arima(
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_seasonal_p: int = 0,
    max_seasonal_d: int = 0,
    max_seasonal_q: int = 0,
    seasonal_period: int | None = None,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    legacy_max_seasonal_p = _params.pop("max_P", max_seasonal_p)
    legacy_max_seasonal_d = _params.pop("max_D", max_seasonal_d)
    legacy_max_seasonal_q = _params.pop("max_Q", max_seasonal_q)
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    max_seasonal_p_int = int(legacy_max_seasonal_p)
    max_seasonal_d_int = int(legacy_max_seasonal_d)
    max_seasonal_q_int = int(legacy_max_seasonal_q)
    seasonal_period_int = (
        None
        if seasonal_period is None or str(seasonal_period).strip().lower() in {"none", "null", ""}
        else int(seasonal_period)
    )
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    ic_s = str(information_criterion)

    def _f(
        train: Any,
        horizon: int,
        *,
        train_exog: Any | None = None,
        future_exog: Any | None = None,
    ) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "max_p": max_p_int,
            "max_d": max_d_int,
            "max_q": max_q_int,
            "max_P": max_seasonal_p_int,
            "max_D": max_seasonal_d_int,
            "max_Q": max_seasonal_q_int,
            "seasonal_period": seasonal_period_int,
            "trend": trend_s,
            "enforce_stationarity": enforce_stationarity_bool,
            "enforce_invertibility": enforce_invertibility_bool,
            "information_criterion": ic_s,
        }
        if train_exog is not None or future_exog is not None:
            kwargs["train_exog"] = train_exog
            kwargs["future_exog"] = future_exog

        return auto_arima_forecast(train, horizon, **kwargs)

    return _f


def _factory_fourier_auto_arima(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    ic_s = str(information_criterion)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_auto_arima_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            max_p=max_p_int,
            max_d=max_d_int,
            max_q=max_q_int,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            information_criterion=ic_s,
        )

    return _f


def _factory_fourier_arima(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    order: Any = (1, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_arima_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            order=order_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
        )

    return _f


def _factory_fourier_sarimax(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    order: Any = (1, 0, 0),
    seasonal_order: Any = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_SEASONAL_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_sarimax_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
        )

    return _f


def _factory_fourier_autoreg(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    lags: int = 0,
    trend: str = "c",
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    trend_s = str(trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_autoreg_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            lags=lags_int,
            trend=trend_s,
        )

    return _f


def _factory_fourier_ets(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    trend: str | None = None,
    damped_trend: bool = False,
    **_params: Any,
) -> ForecasterFn:
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    damped_trend_bool = _normalize_bool_like(damped_trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_ets_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            trend=trend_s,
            damped_trend=damped_trend_bool,
        )

    return _f


def _factory_fourier_uc(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    level: str = _LOCAL_LEVEL_LITERAL,
    **_params: Any,
) -> ForecasterFn:
    level_s = str(level)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return fourier_uc_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            level=level_s,
        )

    return _f


def _factory_sarimax(
    *,
    order: Any = (1, 0, 0),
    seasonal_order: Any = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_SEASONAL_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_final = (
        None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    )

    enforce_stationarity_bool = bool(enforce_stationarity)
    enforce_invertibility_bool = bool(enforce_invertibility)

    def _f(
        train: Any,
        horizon: int,
        *,
        train_exog: Any | None = None,
        future_exog: Any | None = None,
    ) -> np.ndarray:
        return sarimax_forecast(
            train,
            horizon,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_final,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            train_exog=train_exog,
            future_exog=future_exog,
        )

    return _f


def _factory_autoreg(
    *,
    lags: int = 12,
    trend: str = "c",
    seasonal: bool = False,
    period: int | None = None,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    trend_s = str(trend)
    seasonal_bool = bool(seasonal)
    period_int = None if period is None else int(period)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return autoreg_forecast(
            train,
            horizon,
            lags=lags_int,
            trend=trend_s,
            seasonal=seasonal_bool,
            period=period_int,
        )

    return _f


def _factory_unobserved_components(
    *,
    level: str = _LOCAL_LEVEL_LITERAL,
    seasonal: int | None = None,
    **_params: Any,
) -> ForecasterFn:
    level_s = str(level)
    seasonal_int = None if seasonal is None else int(seasonal)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return unobserved_components_forecast(
            train,
            horizon,
            level=level_s,
            seasonal=seasonal_int,
        )

    return _f


def _factory_stl_arima(
    *,
    period: int = 12,
    order: Any = (1, 0, 0),
    seasonal: int = 7,
    robust: bool = False,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    period_int = int(period)
    order_tup = (int(p), int(d), int(q))
    seasonal_int = int(seasonal)
    robust_bool = bool(robust)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_arima_forecast(
            train,
            horizon,
            period=period_int,
            order=order_tup,
            seasonal=seasonal_int,
            robust=robust_bool,
        )

    return _f


def _factory_stl_ets(
    *,
    period: int = 12,
    trend: str | None = "add",
    damped_trend: bool = False,
    robust: bool = False,
    **_params: Any,
) -> ForecasterFn:
    period_int = int(period)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    damped_trend_bool = _normalize_bool_like(damped_trend)
    robust_bool = _normalize_bool_like(robust)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_ets_forecast(
            train,
            horizon,
            period=period_int,
            trend=trend_s,
            damped_trend=damped_trend_bool,
            robust=robust_bool,
        )

    return _f


def _factory_stl_autoreg(
    *,
    period: int = 12,
    lags: int = 1,
    trend: str = "c",
    seasonal: int = 7,
    robust: bool = False,
    **_params: Any,
) -> ForecasterFn:
    period_int = int(period)
    lags_int = int(lags)
    trend_s = str(trend)
    seasonal_int = int(seasonal)
    robust_bool = _normalize_bool_like(robust)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_autoreg_forecast(
            train,
            horizon,
            period=period_int,
            lags=lags_int,
            trend=trend_s,
            seasonal=seasonal_int,
            robust=robust_bool,
        )

    return _f


def _factory_stl_uc(
    *,
    period: int = 12,
    level: str = _LOCAL_LEVEL_LITERAL,
    seasonal: int = 7,
    robust: bool = False,
    **_params: Any,
) -> ForecasterFn:
    period_int = int(period)
    level_s = str(level)
    seasonal_int = int(seasonal)
    robust_bool = _normalize_bool_like(robust)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_uc_forecast(
            train,
            horizon,
            period=period_int,
            level=level_s,
            seasonal=seasonal_int,
            robust=robust_bool,
        )

    return _f


def _factory_stl_sarimax(
    *,
    period: int = 12,
    order: Any = (1, 0, 0),
    seasonal_order: Any = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    seasonal: int = 7,
    robust: bool = False,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_SEASONAL_ORDER_TUPLE_ERROR) from e

    period_int = int(period)
    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    seasonal_int = int(seasonal)
    robust_bool = _normalize_bool_like(robust)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_sarimax_forecast(
            train,
            horizon,
            period=period_int,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            seasonal=seasonal_int,
            robust=robust_bool,
        )

    return _f


def _factory_stl_auto_arima(
    *,
    period: int = 12,
    seasonal: int = 7,
    robust: bool = False,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    period_int = int(period)
    seasonal_int = int(seasonal)
    robust_bool = _normalize_bool_like(robust)
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    ic_s = str(information_criterion)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return stl_auto_arima_forecast(
            train,
            horizon,
            period=period_int,
            seasonal=seasonal_int,
            robust=robust_bool,
            max_p=max_p_int,
            max_d=max_d_int,
            max_q=max_q_int,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            information_criterion=ic_s,
        )

    return _f


def _factory_mstl_arima(
    *,
    periods: Any = (12,),
    order: Any = (1, 0, 0),
    iterate: int = 2,
    lmbda: float | str | None = None,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    iterate_int = int(iterate)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_arima_forecast(
            train,
            horizon,
            periods=periods,
            order=order_tup,
            iterate=iterate_int,
            lmbda=lmbda,
        )

    return _f


def _factory_mstl_autoreg(
    *,
    periods: Any = (12,),
    lags: int = 1,
    trend: str = "c",
    iterate: int = 2,
    lmbda: float | str | None = None,
    **_params: Any,
) -> ForecasterFn:
    lags_int = int(lags)
    trend_s = str(trend)
    iterate_int = int(iterate)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_autoreg_forecast(
            train,
            horizon,
            periods=periods,
            lags=lags_int,
            trend=trend_s,
            iterate=iterate_int,
            lmbda=lmbda,
        )

    return _f


def _factory_mstl_uc(
    *,
    periods: Any = (12,),
    level: str = _LOCAL_LEVEL_LITERAL,
    iterate: int = 2,
    lmbda: float | str | None = None,
    **_params: Any,
) -> ForecasterFn:
    level_s = str(level)
    iterate_int = int(iterate)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_uc_forecast(
            train,
            horizon,
            periods=periods,
            level=level_s,
            iterate=iterate_int,
            lmbda=lmbda,
        )

    return _f


def _factory_mstl_sarimax(
    *,
    periods: Any = (12,),
    order: Any = (1, 0, 0),
    seasonal_order: Any = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    iterate: int = 2,
    lmbda: float | str | None = None,
    **_params: Any,
) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e

    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_SEASONAL_ORDER_TUPLE_ERROR) from e

    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    iterate_int = int(iterate)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_sarimax_forecast(
            train,
            horizon,
            periods=periods,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            iterate=iterate_int,
            lmbda=lmbda,
        )

    return _f


def _factory_mstl_auto_arima(
    *,
    periods: Any = (12,),
    iterate: int = 2,
    lmbda: float | str | None = None,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    iterate_int = int(iterate)
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    ic_s = str(information_criterion)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_auto_arima_forecast(
            train,
            horizon,
            periods=periods,
            iterate=iterate_int,
            lmbda=lmbda,
            max_p=max_p_int,
            max_d=max_d_int,
            max_q=max_q_int,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            information_criterion=ic_s,
        )

    return _f


def _factory_mstl_ets(
    *,
    periods: Any = (12,),
    trend: str | None = "add",
    damped_trend: bool = False,
    iterate: int = 2,
    lmbda: float | str | None = None,
    **_params: Any,
) -> ForecasterFn:
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    damped_trend_bool = _normalize_bool_like(damped_trend)
    iterate_int = int(iterate)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return mstl_ets_forecast(
            train,
            horizon,
            periods=periods,
            trend=trend_s,
            damped_trend=damped_trend_bool,
            iterate=iterate_int,
            lmbda=lmbda,
        )

    return _f


def _factory_tbats_lite(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    include_trend: bool = True,
    arima_order: Any = (1, 0, 0),
    boxcox_lambda: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = bool(include_trend)

    try:
        p, d, q = arima_order
    except Exception as e:  # noqa: BLE001
        raise TypeError("arima_order must be a 3-tuple like (p, d, q)") from e
    arima_order_tup = (int(p), int(d), int(q))

    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            arima_order=arima_order_tup,
            boxcox_lambda=lam,
        )

    return _f


def _factory_tbats_lite_autoreg(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    include_trend: bool = True,
    lags: int = 1,
    trend: str = "n",
    boxcox_lambda: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = _normalize_bool_like(include_trend)
    lags_int = int(lags)
    trend_s = str(trend)
    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_autoreg_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            lags=lags_int,
            trend=trend_s,
            boxcox_lambda=lam,
        )

    return _f


def _factory_tbats_lite_ets(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    include_trend: bool = True,
    trend: str | None = None,
    damped_trend: bool = False,
    boxcox_lambda: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = _normalize_bool_like(include_trend)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    damped_trend_bool = _normalize_bool_like(damped_trend)
    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_ets_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            trend=trend_s,
            damped_trend=damped_trend_bool,
            boxcox_lambda=lam,
        )

    return _f


def _factory_tbats_lite_sarimax(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    include_trend: bool = True,
    order: Any = (1, 0, 0),
    seasonal_order: Any = (0, 0, 0, 0),
    trend: str | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    boxcox_lambda: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = _normalize_bool_like(include_trend)
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_ORDER_TUPLE_ERROR) from e
    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError(_SEASONAL_ORDER_TUPLE_ERROR) from e
    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_sarimax_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            boxcox_lambda=lam,
        )

    return _f


def _factory_tbats_lite_auto_arima(
    *,
    periods: Any = (12,),
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
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = _normalize_bool_like(include_trend)
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    trend_s = None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    enforce_stationarity_bool = _normalize_bool_like(enforce_stationarity)
    enforce_invertibility_bool = _normalize_bool_like(enforce_invertibility)
    ic_s = str(information_criterion)
    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_auto_arima_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            max_p=max_p_int,
            max_d=max_d_int,
            max_q=max_q_int,
            trend=trend_s,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
            information_criterion=ic_s,
            boxcox_lambda=lam,
        )

    return _f


def _factory_tbats_lite_uc(
    *,
    periods: Any = (12,),
    orders: Any = 2,
    include_trend: bool = True,
    level: str = _LOCAL_LEVEL_LITERAL,
    boxcox_lambda: float | None = None,
    **_params: Any,
) -> ForecasterFn:
    include_trend_bool = _normalize_bool_like(include_trend)
    level_s = str(level)
    lam = None if boxcox_lambda is None else float(boxcox_lambda)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return tbats_lite_uc_forecast(
            train,
            horizon,
            periods=periods,
            orders=orders,
            include_trend=include_trend_bool,
            level=level_s,
            boxcox_lambda=lam,
        )

    return _f


def _factory_ets(
    *,
    season_length: int = 12,
    trend: str | None = "add",
    seasonal: str | None = "add",
    damped_trend: bool = False,
    **_params: Any,
) -> ForecasterFn:
    season_length_int = int(season_length)
    trend_final = (
        None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    )
    seasonal_final = (
        None
        if (seasonal is None or str(seasonal).lower() in {"none", "null", ""})
        else str(seasonal)
    )
    damped_trend_bool = bool(damped_trend)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ets_forecast(
            train,
            horizon,
            trend=trend_final,
            seasonal=seasonal_final,
            seasonal_periods=(season_length_int if seasonal_final is not None else None),
            damped_trend=damped_trend_bool,
        )

    return _f


def _factory_var(
    *,
    maxlags: int = 1,
    trend: str = "c",
    ic: str | None = None,
    **_params: Any,
) -> MultivariateForecasterFn:
    maxlags_int = int(maxlags)
    trend_s = str(trend)
    ic_final = None if ic is None else str(ic)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return var_forecast(train, horizon, maxlags=maxlags_int, trend=trend_s, ic=ic_final)

    return _f


def _factory_torch_stid_multivariate(
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> MultivariateForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    dropout_f = float(dropout)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_stid_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            dropout=dropout_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_stgcn_multivariate(
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.1,
    adj: Any = "corr",
    adj_path: str = "",
    adj_top_k: int = 8,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> MultivariateForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    kernel_size_int = int(kernel_size)
    dropout_f = float(dropout)
    adj_value = adj
    adj_path_s = str(adj_path)
    adj_top_k_int = int(adj_top_k)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_stgcn_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            kernel_size=kernel_size_int,
            dropout=dropout_f,
            adj=adj_value,
            adj_path=adj_path_s,
            adj_top_k=adj_top_k_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_graphwavenet_multivariate(
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 4,
    kernel_size: int = 2,
    dilation_base: int = 2,
    dropout: float = 0.1,
    adj: Any = "corr",
    adj_path: str = "",
    adj_top_k: int = 8,
    adaptive_adj: bool = True,
    adj_emb_dim: int = 8,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> MultivariateForecasterFn:
    lags_int = int(lags)
    d_model_int = int(d_model)
    num_blocks_int = int(num_blocks)
    kernel_size_int = int(kernel_size)
    dilation_base_int = int(dilation_base)
    dropout_f = float(dropout)
    adj_value = adj
    adj_path_s = str(adj_path)
    adj_top_k_int = int(adj_top_k)
    adaptive_adj_bool = bool(adaptive_adj)
    adj_emb_dim_int = int(adj_emb_dim)
    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_graphwavenet_forecast(
            train,
            horizon,
            lags=lags_int,
            d_model=d_model_int,
            num_blocks=num_blocks_int,
            kernel_size=kernel_size_int,
            dilation_base=dilation_base_int,
            dropout=dropout_f,
            adj=adj_value,
            adj_path=adj_path_s,
            adj_top_k=adj_top_k_int,
            adaptive_adj=adaptive_adj_bool,
            adj_emb_dim=adj_emb_dim_int,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_hf_timeseries_transformer_direct(
    *,
    context_length: int = 48,
    lags_sequence: Any = (1, 2, 3, 4, 5, 6, 7),
    d_model: int = 64,
    nhead: int = 2,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    num_time_features: int = 0,
    num_samples: int = 100,
    pretrained_model: str = "",
    local_files_only: bool = True,
    normalize: bool = True,
    device: str = "cpu",
    seed: int = 0,
    epochs: int = 0,
    **_params: Any,
) -> ForecasterFn:
    context_length_int = int(context_length)
    lags_sequence_value = lags_sequence
    d_model_int = int(d_model)
    nhead_int = int(nhead)
    encoder_layers_int = int(encoder_layers)
    decoder_layers_int = int(decoder_layers)
    ffn_dim_int = int(ffn_dim)
    dropout_f = float(dropout)
    num_time_features_int = int(num_time_features)
    num_samples_int = int(num_samples)
    pretrained_model_s = str(pretrained_model)
    local_files_only_bool = bool(local_files_only)
    normalize_bool = bool(normalize)
    device_s = str(device)
    seed_int = int(seed)
    epochs_int = int(epochs)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return hf_timeseries_transformer_direct_forecast(
            train,
            horizon,
            context_length=context_length_int,
            lags_sequence=lags_sequence_value,
            d_model=d_model_int,
            nhead=nhead_int,
            encoder_layers=encoder_layers_int,
            decoder_layers=decoder_layers_int,
            ffn_dim=ffn_dim_int,
            dropout=dropout_f,
            num_time_features=num_time_features_int,
            num_samples=num_samples_int,
            pretrained_model=pretrained_model_s,
            local_files_only=local_files_only_bool,
            normalize=normalize_bool,
            device=device_s,
            seed=seed_int,
            epochs=epochs_int,
        )

    return _f


def _factory_torch_rnnzoo_direct(
    *,
    base: str,
    variant: str = "direct",
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    proj_size: int = 16,
    attn_hidden: int = 32,
    clock_periods: Any = (1, 2, 4, 8),
    qrnn_kernel_size: int = 3,
    rhn_depth: int = 2,
    phased_tau: float = 32.0,
    phased_r_on: float = 0.05,
    phased_leak: float = 0.001,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    base_s = str(base)
    variant_s = str(variant)
    lags_int = int(lags)
    hidden_int = int(hidden_size)
    layers_int = int(num_layers)
    dropout_f = float(dropout)
    proj_int = int(proj_size)
    attn_int = int(attn_hidden)
    qrnn_k_int = int(qrnn_kernel_size)
    rhn_depth_int = int(rhn_depth)
    phased_tau_f = float(phased_tau)
    phased_r_on_f = float(phased_r_on)
    phased_leak_f = float(phased_leak)

    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    clock_periods_val: Any = clock_periods

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_rnnzoo_direct_forecast(
            train,
            horizon,
            base=base_s,
            variant=variant_s,  # type: ignore[arg-type]
            lags=lags_int,
            hidden_size=hidden_int,
            num_layers=layers_int,
            dropout=dropout_f,
            proj_size=proj_int,
            attn_hidden=attn_int,
            clock_periods=clock_periods_val,
            qrnn_kernel_size=qrnn_k_int,
            rhn_depth=rhn_depth_int,
            phased_tau=phased_tau_f,
            phased_r_on=phased_r_on_f,
            phased_leak=phased_leak_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def _factory_torch_rnnpaper_direct(
    *,
    paper: str,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    # generic knobs used by some architectures
    attn_hidden: int = 32,
    kernel_size: int = 3,
    hops: int = 2,
    memory_slots: int = 16,
    memory_dim: int = 32,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    # training
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    scheduler_restart_period: int = 10,
    scheduler_restart_mult: int = 1,
    scheduler_pct_start: float = 0.3,
    restore_best: bool = True,
    **_params: Any,
) -> ForecasterFn:
    paper_s = str(paper)
    lags_int = int(lags)
    hidden_int = int(hidden_size)
    layers_int = int(num_layers)
    dropout_f = float(dropout)

    attn_int = int(attn_hidden)
    kernel_int = int(kernel_size)
    hops_int = int(hops)
    mem_slots_int = int(memory_slots)
    mem_dim_int = int(memory_dim)
    spectral_f = float(spectral_radius)
    leak_f = float(leak)

    epochs_int = int(epochs)
    lr_f = float(lr)
    weight_decay_f = float(weight_decay)
    batch_size_int = int(batch_size)
    seed_int = int(seed)
    normalize_bool = bool(normalize)
    device_s = str(device)
    patience_int = int(patience)
    loss_s = str(loss)
    val_split_f = float(val_split)
    grad_clip_norm_f = float(grad_clip_norm)
    optimizer_s = str(optimizer)
    momentum_f = float(momentum)
    scheduler_s = str(scheduler)
    scheduler_step_size_int = int(scheduler_step_size)
    scheduler_gamma_f = float(scheduler_gamma)
    restore_best_bool = bool(restore_best)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return torch_rnnpaper_direct_forecast(
            train,
            horizon,
            paper=paper_s,
            lags=lags_int,
            hidden_size=hidden_int,
            num_layers=layers_int,
            dropout=dropout_f,
            attn_hidden=attn_int,
            kernel_size=kernel_int,
            hops=hops_int,
            memory_slots=mem_slots_int,
            memory_dim=mem_dim_int,
            spectral_radius=spectral_f,
            leak=leak_f,
            epochs=epochs_int,
            lr=lr_f,
            weight_decay=weight_decay_f,
            batch_size=batch_size_int,
            seed=seed_int,
            normalize=normalize_bool,
            device=device_s,
            patience=patience_int,
            loss=loss_s,
            val_split=val_split_f,
            grad_clip_norm=grad_clip_norm_f,
            optimizer=optimizer_s,
            momentum=momentum_f,
            scheduler=scheduler_s,
            scheduler_step_size=scheduler_step_size_int,
            scheduler_gamma=scheduler_gamma_f,
            scheduler_restart_period=int(scheduler_restart_period),
            scheduler_restart_mult=int(scheduler_restart_mult),
            scheduler_pct_start=float(scheduler_pct_start),
            restore_best=restore_best_bool,

            **_coerce_torch_extra_train_params(_params),
        )

    return _f


def make_forecaster(key: str, **params: Any) -> ForecasterFn:
    from .resolution import get_model_spec

    spec = get_model_spec(key)
    return build_local_forecaster(key=key, spec=spec, params=params)


def make_global_forecaster(key: str, **params: Any) -> GlobalForecasterFn:
    from .resolution import get_model_spec

    spec = get_model_spec(key)
    return build_global_forecaster(key=key, spec=spec, params=params)


def make_multivariate_forecaster(key: str, **params: Any) -> MultivariateForecasterFn:
    from .resolution import get_model_spec

    spec = get_model_spec(key)
    return build_multivariate_forecaster(key=key, spec=spec, params=params)


def make_forecaster_object(key: str, **params: Any) -> BaseForecaster:
    from .resolution import get_model_spec

    spec = get_model_spec(key)
    return build_local_forecaster_object(key=key, spec=spec, params=params)


def make_global_forecaster_object(key: str, **params: Any) -> BaseGlobalForecaster:
    from .resolution import get_model_spec

    spec = get_model_spec(key)
    return build_global_forecaster_object(key=key, spec=spec, params=params)
