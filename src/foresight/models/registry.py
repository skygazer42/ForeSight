from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..transforms import fit_transform, inverse_forecast, normalize_transform_list
from .analog import analog_knn_forecast
from .ar import ar_ols_auto_forecast, ar_ols_forecast, ar_ols_lags_forecast, sar_ols_forecast
from .baselines import (
    drift_forecast,
    mean_forecast,
    median_forecast,
    moving_average_forecast,
    seasonal_mean_forecast,
)
from .fourier import fourier_multi_regression_forecast, fourier_regression_forecast
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
from .naive import naive_last, seasonal_naive
from .regression import (
    elasticnet_lag_direct_forecast,
    gbrt_lag_direct_forecast,
    knn_lag_direct_forecast,
    lasso_lag_direct_forecast,
    lr_lag_direct_forecast,
    lr_lag_forecast,
    rf_lag_direct_forecast,
    ridge_lag_forecast,
)
from .smoothing import (
    holt_auto_forecast,
    holt_damped_forecast,
    holt_forecast,
    holt_winters_additive_auto_forecast,
    holt_winters_additive_forecast,
    ses_auto_forecast,
    ses_forecast,
)
from .spectral import fft_topk_forecast
from .statsmodels_wrap import (
    arima_forecast,
    auto_arima_forecast,
    autoreg_forecast,
    ets_forecast,
    mstl_arima_forecast,
    mstl_auto_arima_forecast,
    sarimax_forecast,
    stl_arima_forecast,
    tbats_lite_forecast,
    unobserved_components_forecast,
)
from .theta import theta_auto_forecast, theta_forecast
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
    torch_rnn_global_forecaster,
    torch_rwkv_global_forecaster,
    torch_scinet_global_forecaster,
    torch_seq2seq_global_forecaster,
    torch_ssm_global_forecaster,
    torch_tcn_global_forecaster,
    torch_tft_global_forecaster,
    torch_tide_global_forecaster,
    torch_timesnet_global_forecaster,
    torch_transformer_encdec_global_forecaster,
    torch_tsmixer_global_forecaster,
    torch_wavenet_global_forecaster,
    torch_xformer_global_forecaster,
)
from .torch_nn import (
    torch_attn_gru_direct_forecast,
    torch_bigru_direct_forecast,
    torch_bilstm_direct_forecast,
    torch_cnn_direct_forecast,
    torch_crossformer_direct_forecast,
    torch_deepar_recursive_forecast,
    torch_dilated_rnn_direct_forecast,
    torch_dlinear_direct_forecast,
    torch_esrnn_direct_forecast,
    torch_etsformer_direct_forecast,
    torch_fnet_direct_forecast,
    torch_gmlp_direct_forecast,
    torch_gru_direct_forecast,
    torch_hyena_direct_forecast,
    torch_inception_direct_forecast,
    torch_kan_direct_forecast,
    torch_linear_attention_direct_forecast,
    torch_lstm_direct_forecast,
    torch_mamba_direct_forecast,
    torch_mlp_lag_direct_forecast,
    torch_nbeats_direct_forecast,
    torch_nhits_direct_forecast,
    torch_nlinear_direct_forecast,
    torch_patchtst_direct_forecast,
    torch_pyraformer_direct_forecast,
    torch_qrnn_recursive_forecast,
    torch_resnet1d_direct_forecast,
    torch_rwkv_direct_forecast,
    torch_scinet_direct_forecast,
    torch_tcn_direct_forecast,
    torch_tide_direct_forecast,
    torch_transformer_direct_forecast,
    torch_tsmixer_direct_forecast,
    torch_wavenet_direct_forecast,
)
from .torch_rnn_paper_zoo import list_rnnpaper_specs, torch_rnnpaper_direct_forecast
from .torch_rnn_zoo import list_rnnzoo_specs, torch_rnnzoo_direct_forecast
from .torch_seq2seq import torch_lstnet_direct_forecast, torch_seq2seq_direct_forecast
from .torch_xformer import torch_xformer_direct_forecast
from .trend import poly_trend_forecast

LocalForecasterFn = Callable[[Any, int], np.ndarray]
GlobalForecasterFn = Callable[[pd.DataFrame, Any, int], pd.DataFrame]
ModelFactory = Callable[..., Any]
ForecasterFn = LocalForecasterFn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str
    factory: ModelFactory
    default_params: dict[str, Any] = field(default_factory=dict)
    param_help: dict[str, str] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    interface: str = (
        "local"  # local: (train_1d, horizon)->yhat ; global: (long_df, cutoff, horizon)->pred_df
    )


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
    "scheduler": "LR scheduler: none, cosine, step",
    "scheduler_step_size": "StepLR step_size (only for scheduler=step)",
    "scheduler_gamma": "StepLR gamma (only for scheduler=step)",
    "restore_best": "Restore best checkpoint at end (true/false)",
}


def _factory_naive_last(**_params: Any) -> ForecasterFn:
    return naive_last


def _factory_seasonal_naive(*, season_length: int = 12, **_params: Any) -> ForecasterFn:
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_naive(train, horizon, season_length=season_length_int)

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


def _factory_seasonal_mean(*, season_length: int = 12, **_params: Any) -> ForecasterFn:
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return seasonal_mean_forecast(train, horizon, season_length=season_length_int)

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
    P: int = 1,
    season_length: int = 12,
    **_params: Any,
) -> ForecasterFn:
    p_int = int(p)
    P_int = int(P)
    season_length_int = int(season_length)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return sar_ols_forecast(
            train,
            horizon,
            p=p_int,
            P=P_int,
            season_length=season_length_int,
        )

    return _f


def _factory_ar_ols_auto(*, max_p: int = 10, **_params: Any) -> ForecasterFn:
    max_p_int = int(max_p)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ar_ols_auto_forecast(train, horizon, max_p=max_p_int)

    return _f


def _factory_lr_lag(*, lags: int = 5, **_params: Any) -> ForecasterFn:
    lags_int = int(lags)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lr_lag_forecast(train, horizon, lags=lags_int)

    return _f


def _factory_lr_lag_direct(*, lags: int = 5, **_params: Any) -> ForecasterFn:
    lags_int = int(lags)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return lr_lag_direct_forecast(train, horizon, lags=lags_int)

    return _f


def _factory_ridge_lag(*, lags: int = 5, alpha: float = 1.0, **_params: Any) -> ForecasterFn:
    lags_int = int(lags)
    alpha_f = float(alpha)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return ridge_lag_forecast(train, horizon, lags=lags_int, alpha=alpha_f)

    return _f


def _factory_rf_lag(
    *, lags: int = 5, n_estimators: int = 200, random_state: int = 0, **_params: Any
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
        )

    return _f


def _factory_lasso_lag(
    *, lags: int = 10, alpha: float = 0.001, max_iter: int = 5000, **_params: Any
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
        )

    return _f


def _factory_elasticnet_lag(
    *,
    lags: int = 10,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
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
        )

    return _f


def _factory_knn_lag(
    *, lags: int = 12, n_neighbors: int = 10, weights: str = "distance", **_params: Any
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
        )

    return _f


def _factory_gbrt_lag(
    *,
    lags: int = 12,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 0,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
        )

    return _f


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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
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
            raise ValueError("members must be non-empty")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(parts)

    if isinstance(members, list | tuple):
        parts = [str(m).strip() for m in members if str(m).strip()]
        return tuple(parts)

    s = str(members).strip()
    if not s:
        raise ValueError("members must be non-empty")
    return (s,)


def _factory_ensemble_mean(
    *, members: Any = ("naive-last", "seasonal-naive", "theta"), **_p: Any
) -> ForecasterFn:
    member_keys = _normalize_members(members)
    if not member_keys:
        raise ValueError("members must be non-empty")
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
        raise ValueError("members must be non-empty")
    if any(k == "ensemble-median" for k in member_keys):
        raise ValueError("ensemble-median cannot include itself")

    member_forecasters = [make_forecaster(k) for k in member_keys]

    def _f(train: Any, horizon: int) -> np.ndarray:
        preds = [np.asarray(m(train, horizon), dtype=float) for m in member_forecasters]
        arr = np.stack(preds, axis=0)
        return np.median(arr, axis=0)

    return _f


def _factory_arima(*, order: Any = (1, 0, 0), **_params: Any) -> ForecasterFn:
    try:
        p, d, q = order
    except Exception as e:  # noqa: BLE001
        raise TypeError("order must be a 3-tuple like (p, d, q)") from e

    order_tup = (int(p), int(d), int(q))

    def _f(train: Any, horizon: int) -> np.ndarray:
        return arima_forecast(train, horizon, order=order_tup)

    return _f


def _factory_auto_arima(
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
    ic_s = str(information_criterion)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return auto_arima_forecast(
            train,
            horizon,
            max_p=max_p_int,
            max_d=max_d_int,
            max_q=max_q_int,
            information_criterion=ic_s,
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
        raise TypeError("order must be a 3-tuple like (p, d, q)") from e

    try:
        P, D, Q, s = seasonal_order
    except Exception as e:  # noqa: BLE001
        raise TypeError("seasonal_order must be a 4-tuple like (P, D, Q, s)") from e

    order_tup = (int(p), int(d), int(q))
    seasonal_tup = (int(P), int(D), int(Q), int(s))
    trend_final = (
        None if (trend is None or str(trend).lower() in {"none", "null", ""}) else str(trend)
    )

    enforce_stationarity_bool = bool(enforce_stationarity)
    enforce_invertibility_bool = bool(enforce_invertibility)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return sarimax_forecast(
            train,
            horizon,
            order=order_tup,
            seasonal_order=seasonal_tup,
            trend=trend_final,
            enforce_stationarity=enforce_stationarity_bool,
            enforce_invertibility=enforce_invertibility_bool,
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


def _factory_unobserved_components(*, level: str = "local level", **_params: Any) -> ForecasterFn:
    level_s = str(level)

    def _f(train: Any, horizon: int) -> np.ndarray:
        return unobserved_components_forecast(train, horizon, level=level_s)

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
        raise TypeError("order must be a 3-tuple like (p, d, q)") from e

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
        raise TypeError("order must be a 3-tuple like (p, d, q)") from e

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


def _factory_mstl_auto_arima(
    *,
    periods: Any = (12,),
    iterate: int = 2,
    lmbda: float | str | None = None,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    information_criterion: str = "aic",
    **_params: Any,
) -> ForecasterFn:
    iterate_int = int(iterate)
    max_p_int = int(max_p)
    max_d_int = int(max_d)
    max_q_int = int(max_q)
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
            information_criterion=ic_s,
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


_REGISTRY: dict[str, ModelSpec] = {
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
    "seasonal-mean": ModelSpec(
        key="seasonal-mean",
        description="Repeat the seasonal means for each position in a season.",
        factory=_factory_seasonal_mean,
        default_params={"season_length": 12},
        param_help={"season_length": "Season length for seasonal means"},
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
            "alpha": "Level smoothing in [0, 1]",
            "beta": "Trend smoothing in [0, 1]",
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
            "alpha": "Level smoothing in [0, 1]",
            "beta": "Trend smoothing in [0, 1]",
            "phi": "Damping parameter in [0, 1] (phi=1 reduces to Holt)",
        },
    ),
    "holt-winters-add": ModelSpec(
        key="holt-winters-add",
        description="Holt-Winters additive seasonality + additive trend.",
        factory=_factory_holt_winters_add,
        default_params={"season_length": 12, "alpha": 0.2, "beta": 0.1, "gamma": 0.1},
        param_help={
            "season_length": "Season length",
            "alpha": "Level smoothing in [0, 1]",
            "beta": "Trend smoothing in [0, 1]",
            "gamma": "Seasonal smoothing in [0, 1]",
        },
    ),
    "holt-winters-add-auto": ModelSpec(
        key="holt-winters-add-auto",
        description="Auto-tuned Holt-Winters additive (small grid search).",
        factory=_factory_holt_winters_add_auto,
        default_params={"season_length": 12, "grid_size": 7},
        param_help={
            "season_length": "Season length",
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
        param_help={"grid_size": "Number of alpha values to try (default: 19)"},
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
        param_help={"alpha": "Smoothing parameter in [0,1]"},
    ),
    "croston-sba": ModelSpec(
        key="croston-sba",
        description="Croston-SBA intermittent-demand method (bias-corrected).",
        factory=_factory_croston_sba,
        default_params={"alpha": 0.1},
        param_help={"alpha": "Smoothing parameter in [0,1]"},
    ),
    "croston-sbj": ModelSpec(
        key="croston-sbj",
        description="Croston-SBJ intermittent-demand method (bias-corrected).",
        factory=_factory_croston_sbj,
        default_params={"alpha": 0.1},
        param_help={"alpha": "Smoothing parameter in [0,1]"},
    ),
    "croston-opt": ModelSpec(
        key="croston-opt",
        description="Croston classic with alpha tuned by in-sample SSE grid search.",
        factory=_factory_croston_opt,
        default_params={"grid_size": 19},
        param_help={"grid_size": "Number of alpha values to try (default: 19)"},
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
    "lr-lag": ModelSpec(
        key="lr-lag",
        description="Linear regression on lag features (OLS, recursive forecast).",
        factory=_factory_lr_lag,
        default_params={"lags": 5},
        param_help={"lags": "Number of lag features"},
    ),
    "lr-lag-direct": ModelSpec(
        key="lr-lag-direct",
        description="Linear regression on lag features (OLS, direct multi-horizon).",
        factory=_factory_lr_lag_direct,
        default_params={"lags": 5},
        param_help={"lags": "Number of lag features"},
    ),
    "ridge-lag": ModelSpec(
        key="ridge-lag",
        description="Ridge regression on lag features (recursive forecast). Requires scikit-learn.",
        factory=_factory_ridge_lag,
        default_params={"lags": 5, "alpha": 1.0},
        param_help={"lags": "Number of lag features", "alpha": "Ridge regularization strength"},
        requires=("ml",),
    ),
    "rf-lag": ModelSpec(
        key="rf-lag",
        description="RandomForest on lag features (direct multi-horizon). Requires scikit-learn.",
        factory=_factory_rf_lag,
        default_params={"lags": 10, "n_estimators": 200, "random_state": 0},
        param_help={
            "lags": "Number of lag features",
            "n_estimators": "RandomForest n_estimators",
            "random_state": "RandomForest random_state",
        },
        requires=("ml",),
    ),
    "lasso-lag": ModelSpec(
        key="lasso-lag",
        description="Lasso on lag features (direct multi-horizon). Requires scikit-learn.",
        factory=_factory_lasso_lag,
        default_params={"lags": 12, "alpha": 0.001, "max_iter": 5000},
        param_help={
            "lags": "Number of lag features",
            "alpha": "L1 regularization strength",
            "max_iter": "Max solver iterations",
        },
        requires=("ml",),
    ),
    "elasticnet-lag": ModelSpec(
        key="elasticnet-lag",
        description="ElasticNet on lag features (direct multi-horizon). Requires scikit-learn.",
        factory=_factory_elasticnet_lag,
        default_params={"lags": 12, "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000},
        param_help={
            "lags": "Number of lag features",
            "alpha": "Regularization strength",
            "l1_ratio": "ElasticNet l1_ratio in [0,1]",
            "max_iter": "Max solver iterations",
        },
        requires=("ml",),
    ),
    "knn-lag": ModelSpec(
        key="knn-lag",
        description="KNN regression on lag features (direct multi-horizon). Requires scikit-learn.",
        factory=_factory_knn_lag,
        default_params={"lags": 12, "n_neighbors": 10, "weights": "distance"},
        param_help={
            "lags": "Number of lag features",
            "n_neighbors": "KNN n_neighbors",
            "weights": "KNN weights: uniform or distance",
        },
        requires=("ml",),
    ),
    "gbrt-lag": ModelSpec(
        key="gbrt-lag",
        description="GradientBoosting on lag features (direct multi-horizon). Requires scikit-learn.",
        factory=_factory_gbrt_lag,
        default_params={
            "lags": 12,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 0,
        },
        param_help={
            "lags": "Number of lag features",
            "n_estimators": "Number of boosting stages",
            "learning_rate": "Boosting learning rate",
            "max_depth": "Tree max_depth",
            "random_state": "Random seed",
        },
        requires=("ml",),
    ),
    "torch-mlp-direct": ModelSpec(
        key="torch-mlp-direct",
        description="Torch MLP on lag features (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_mlp_direct,
        default_params={
            "lags": 24,
            "hidden_sizes": (64, 64),
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_sizes": "Hidden layer sizes (e.g. 64,64)",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-lstm-direct": ModelSpec(
        key="torch-lstm-direct",
        description="Torch LSTM on lag windows (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_lstm_direct,
        default_params={
            "lags": 24,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "LSTM hidden size",
            "num_layers": "Number of LSTM layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-gru-direct": ModelSpec(
        key="torch-gru-direct",
        description="Torch GRU on lag windows (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_gru_direct,
        default_params={
            "lags": 24,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "GRU hidden size",
            "num_layers": "Number of GRU layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-tcn-direct": ModelSpec(
        key="torch-tcn-direct",
        description="Torch TCN on lag windows (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_tcn_direct,
        default_params={
            "lags": 24,
            "channels": (16, 16, 16),
            "kernel_size": 3,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "channels": "Conv channel sizes (e.g. 16,16,16)",
            "kernel_size": "Conv kernel size",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-nbeats-direct": ModelSpec(
        key="torch-nbeats-direct",
        description="Torch N-BEATS-style model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_nbeats_direct,
        default_params={
            "lags": 48,
            "num_blocks": 3,
            "num_layers": 2,
            "layer_width": 64,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "num_blocks": "Number of residual blocks",
            "num_layers": "Hidden layers per block",
            "layer_width": "Hidden width per layer",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-nlinear-direct": ModelSpec(
        key="torch-nlinear-direct",
        description="Torch NLinear-style baseline (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_nlinear_direct,
        default_params={
            "lags": 48,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-dlinear-direct": ModelSpec(
        key="torch-dlinear-direct",
        description="Torch DLinear-style baseline (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_dlinear_direct,
        default_params={
            "lags": 48,
            "ma_window": 25,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "ma_window": "Moving average window size for decomposition",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-transformer-direct": ModelSpec(
        key="torch-transformer-direct",
        description="Torch Transformer encoder (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_transformer_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Transformer embedding dimension",
            "nhead": "Number of attention heads",
            "num_layers": "Number of Transformer encoder layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-mamba-direct": ModelSpec(
        key="torch-mamba-direct",
        description="Torch Mamba-style selective SSM (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_mamba_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "conv_kernel": 3,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Model dimension",
            "num_layers": "Number of stacked Mamba blocks",
            "dropout": "Dropout probability in [0,1)",
            "conv_kernel": "Causal depthwise conv kernel size (>=1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-rwkv-direct": ModelSpec(
        key="torch-rwkv-direct",
        description="Torch RWKV-style time-mix + channel-mix (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_rwkv_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 2,
            "ffn_dim": 128,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Model dimension",
            "num_layers": "Number of stacked RWKV blocks",
            "ffn_dim": "Channel-mix hidden size",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-hyena-direct": ModelSpec(
        key="torch-hyena-direct",
        description="Torch Hyena-style long convolution model (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_hyena_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 2,
            "ffn_dim": 128,
            "kernel_size": 64,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Model dimension",
            "num_layers": "Number of Hyena blocks",
            "ffn_dim": "Channel-mixing FFN hidden size",
            "kernel_size": "Depthwise causal conv kernel size (>=1)",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-dilated-rnn-direct": ModelSpec(
        key="torch-dilated-rnn-direct",
        description="Torch Dilated RNN (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_dilated_rnn_direct,
        default_params={
            "lags": 96,
            "cell": "gru",
            "hidden_size": 64,
            "num_layers": 3,
            "dilation_base": 2,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "cell": "Recurrent cell: gru or lstm",
            "hidden_size": "Hidden size / model dimension",
            "num_layers": "Number of dilated recurrent layers",
            "dilation_base": "Dilation base (>=2); dilations are base^layer_index",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-kan-direct": ModelSpec(
        key="torch-kan-direct",
        description="Torch KAN-style spline MLP (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_kan_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 2,
            "grid_size": 16,
            "grid_range": 2.0,
            "dropout": 0.1,
            "linear_skip": True,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Hidden width of the KAN network",
            "num_layers": "Number of KAN spline layers",
            "grid_size": "Number of spline knots (>=4)",
            "grid_range": "Spline grid range (+/- range) in normalized y units",
            "dropout": "Dropout probability in [0,1)",
            "linear_skip": "Add a linear skip connection per layer (true/false)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-scinet-direct": ModelSpec(
        key="torch-scinet-direct",
        description="Torch SCINet-style sample-convolution interaction network (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_scinet_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_stages": 3,
            "conv_kernel": 5,
            "ffn_dim": 128,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Model dimension",
            "num_stages": "Number of SCINet interaction stages",
            "conv_kernel": "Conv1D kernel size (>=1) inside interaction blocks",
            "ffn_dim": "FFN hidden size inside blocks",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-etsformer-direct": ModelSpec(
        key="torch-etsformer-direct",
        description="Torch ETSformer-style exponential smoothing + Transformer residual model (lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_etsformer_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "alpha_init": 0.3,
            "beta_init": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Transformer embedding dimension",
            "nhead": "Number of attention heads",
            "num_layers": "Number of Transformer encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "dropout": "Dropout probability in [0,1)",
            "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
            "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-esrnn-direct": ModelSpec(
        key="torch-esrnn-direct",
        description="Torch ESRNN-style hybrid (Holt smoothing + RNN residual, lite) (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_esrnn_direct,
        default_params={
            "lags": 96,
            "cell": "gru",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "alpha_init": 0.3,
            "beta_init": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "cell": "RNN cell: gru or lstm",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
            "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-patchtst-direct": ModelSpec(
        key="torch-patchtst-direct",
        description="Torch PatchTST-style model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_patchtst_direct,
        default_params={
            "lags": 192,
            "patch_len": 16,
            "stride": 8,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "patch_len": "Patch length",
            "stride": "Patch stride",
            "d_model": "Transformer embedding dimension",
            "nhead": "Number of attention heads",
            "num_layers": "Number of Transformer encoder layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-crossformer-direct": ModelSpec(
        key="torch-crossformer-direct",
        description="Torch Crossformer-style (lite) multi-scale segmented Transformer encoder (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_crossformer_direct,
        default_params={
            "lags": 192,
            "segment_len": 16,
            "stride": 16,
            "num_scales": 3,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "segment_len": "Base segment length (scale i uses segment_len * 2^i)",
            "stride": "Base segment stride (scale i uses stride * 2^i)",
            "num_scales": "Number of scales (>=1)",
            "d_model": "Transformer embedding dimension",
            "nhead": "Number of attention heads",
            "num_layers": "Number of Transformer encoder layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-pyraformer-direct": ModelSpec(
        key="torch-pyraformer-direct",
        description="Torch Pyraformer-style (lite) pyramid-pooled segmented Transformer encoder (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_pyraformer_direct,
        default_params={
            "lags": 192,
            "segment_len": 16,
            "stride": 16,
            "num_levels": 3,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "segment_len": "Segment length at level-0",
            "stride": "Segment stride at level-0 (>=1)",
            "num_levels": "Number of pyramid levels (>=1)",
            "d_model": "Transformer embedding dimension",
            "nhead": "Number of attention heads",
            "num_layers": "Number of Transformer encoder layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-tsmixer-direct": ModelSpec(
        key="torch-tsmixer-direct",
        description="Torch TSMixer-style model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_tsmixer_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_blocks": 4,
            "token_mixing_hidden": 128,
            "channel_mixing_hidden": 128,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Mixer embedding dimension",
            "num_blocks": "Number of mixer blocks",
            "token_mixing_hidden": "Token-mixing MLP hidden size",
            "channel_mixing_hidden": "Channel-mixing MLP hidden size",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-cnn-direct": ModelSpec(
        key="torch-cnn-direct",
        description="Torch Conv1D stack (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_cnn_direct,
        default_params={
            "lags": 48,
            "channels": (32, 32, 32),
            "kernel_size": 3,
            "dropout": 0.1,
            "pool": "last",
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "channels": "Conv channel sizes (e.g. 32,32,32)",
            "kernel_size": "Conv kernel size",
            "dropout": "Dropout probability in [0,1)",
            "pool": "Pooling: last, mean, max",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-resnet1d-direct": ModelSpec(
        key="torch-resnet1d-direct",
        description="Torch ResNet-1D (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_resnet1d_direct,
        default_params={
            "lags": 96,
            "channels": 32,
            "num_blocks": 4,
            "kernel_size": 3,
            "dropout": 0.1,
            "pool": "last",
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "channels": "ResNet hidden channels",
            "num_blocks": "Number of residual blocks",
            "kernel_size": "Conv kernel size",
            "dropout": "Dropout probability in [0,1)",
            "pool": "Pooling: last, mean, max",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-wavenet-direct": ModelSpec(
        key="torch-wavenet-direct",
        description="Torch WaveNet-style gated dilated CNN (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_wavenet_direct,
        default_params={
            "lags": 96,
            "channels": 32,
            "num_layers": 6,
            "kernel_size": 2,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "channels": "Hidden channels",
            "num_layers": "Number of dilated gated layers",
            "kernel_size": "Conv kernel size",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-bilstm-direct": ModelSpec(
        key="torch-bilstm-direct",
        description="Torch bidirectional LSTM on lag windows (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_bilstm_direct,
        default_params={
            "lags": 24,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "BiLSTM hidden size (per direction)",
            "num_layers": "Number of stacked BiLSTM layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-bigru-direct": ModelSpec(
        key="torch-bigru-direct",
        description="Torch bidirectional GRU on lag windows (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_bigru_direct,
        default_params={
            "lags": 24,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "BiGRU hidden size (per direction)",
            "num_layers": "Number of stacked BiGRU layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-attn-gru-direct": ModelSpec(
        key="torch-attn-gru-direct",
        description="Torch GRU + attention pooling (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_attn_gru_direct,
        default_params={
            "lags": 48,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "GRU hidden size",
            "num_layers": "Number of GRU layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-fnet-direct": ModelSpec(
        key="torch-fnet-direct",
        description="Torch FNet-style (Fourier mixing) model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_fnet_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Embedding dimension",
            "num_layers": "Number of FNet layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-linear-attn-direct": ModelSpec(
        key="torch-linear-attn-direct",
        description="Torch linear-attention encoder (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_linear_attn_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Embedding dimension",
            "num_layers": "Number of encoder layers",
            "dim_feedforward": "Feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-inception-direct": ModelSpec(
        key="torch-inception-direct",
        description="Torch InceptionTime-style Conv1D model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_inception_direct,
        default_params={
            "lags": 96,
            "channels": 32,
            "num_blocks": 3,
            "kernel_sizes": (3, 5, 7),
            "bottleneck_channels": 16,
            "dropout": 0.1,
            "pool": "last",
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "channels": "Hidden channels",
            "num_blocks": "Number of inception blocks",
            "kernel_sizes": "Comma-separated kernel sizes (e.g. 3,5,7)",
            "bottleneck_channels": "Bottleneck (1x1) conv channels",
            "dropout": "Dropout probability in [0,1)",
            "pool": "Pooling: last, mean, max",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-gmlp-direct": ModelSpec(
        key="torch-gmlp-direct",
        description="Torch gMLP-style model (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_gmlp_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "num_layers": 4,
            "ffn_dim": 128,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Embedding dimension",
            "num_layers": "Number of gMLP layers",
            "ffn_dim": "gMLP feed-forward dimension",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-nhits-direct": ModelSpec(
        key="torch-nhits-direct",
        description="Torch N-HiTS-style multi-rate residual MLP (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_nhits_direct,
        default_params={
            "lags": 192,
            "pool_sizes": (1, 2, 4),
            "num_blocks": 6,
            "num_layers": 2,
            "layer_width": 128,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "pool_sizes": "Comma-separated pooling sizes (e.g. 1,2,4)",
            "num_blocks": "Number of residual blocks",
            "num_layers": "Hidden layers per block",
            "layer_width": "Hidden width per layer",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-tide-direct": ModelSpec(
        key="torch-tide-direct",
        description="Torch TiDE-style encoder/decoder MLP (direct multi-horizon). Requires PyTorch.",
        factory=_factory_torch_tide_direct,
        default_params={
            "lags": 96,
            "d_model": 64,
            "hidden_size": 128,
            "dropout": 0.1,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "d_model": "Context embedding dimension",
            "hidden_size": "Hidden size for encoder/decoder MLP",
            "dropout": "Dropout probability in [0,1)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
    ),
    "torch-deepar-recursive": ModelSpec(
        key="torch-deepar-recursive",
        description="Torch DeepAR-style Gaussian RNN (one-step trained, recursive forecast). Requires PyTorch.",
        factory=_factory_torch_deepar_recursive,
        default_params={
            "lags": 48,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of GRU layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
            "loss": "Ignored (DeepAR uses Gaussian NLL)",
        },
        requires=("torch",),
    ),
    "torch-qrnn-recursive": ModelSpec(
        key="torch-qrnn-recursive",
        description="Torch quantile-regression RNN (one-step trained, recursive forecast). Requires PyTorch.",
        factory=_factory_torch_qrnn_recursive,
        default_params={
            "lags": 48,
            "q": 0.5,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help={
            "lags": "Lag window length",
            "q": "Quantile in (0,1) for pinball loss",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of GRU layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            **_TORCH_COMMON_PARAM_HELP,
            "loss": "Ignored (QRNN uses pinball loss at quantile q)",
        },
        requires=("torch",),
    ),
    "torch-tft-global": ModelSpec(
        key="torch-tft-global",
        description="Torch Temporal Fusion Transformer (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_tft_global_forecaster,
        default_params={
            "context_length": 48,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "lstm_layers": 1,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Embedding dimension",
            "nhead": "Attention heads",
            "lstm_layers": "Number of LSTM layers",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-informer-global": ModelSpec(
        key="torch-informer-global",
        description="Torch Informer (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_informer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-autoformer-global": ModelSpec(
        key="torch-autoformer-global",
        description="Torch Autoformer (lite) with series decomposition, trained globally across panel series. Requires PyTorch.",
        factory=torch_autoformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "ma_window": 7,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "ma_window": "Moving-average window for trend/seasonal decomposition",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-fedformer-global": ModelSpec(
        key="torch-fedformer-global",
        description="Torch FEDformer-style (lite) decomposition + frequency mixing, trained globally across panel series. Requires PyTorch.",
        factory=torch_fedformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 2,
            "ffn_dim": 256,
            "modes": 16,
            "ma_window": 7,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of frequency-mixing blocks",
            "ffn_dim": "FFN hidden dimension",
            "modes": "Number of low-frequency FFT modes to keep (>=1)",
            "ma_window": "Moving-average window for trend extraction (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-nonstationary-transformer-global": ModelSpec(
        key="torch-nonstationary-transformer-global",
        description="Torch Non-stationary Transformer (lite) with RevIN + de-stationary attention factors, trained globally across panel series. Requires PyTorch.",
        factory=torch_nonstationary_transformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-patchtst-global": ModelSpec(
        key="torch-patchtst-global",
        description="Torch PatchTST-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_patchtst_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "patch_len": 16,
            "stride": 8,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "patch_len": "Patch length over (context+horizon) tokens",
            "stride": "Patch stride (>=1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-crossformer-global": ModelSpec(
        key="torch-crossformer-global",
        description="Torch Crossformer-style (lite) multi-scale segmented Transformer encoder trained globally across panel series. Requires PyTorch.",
        factory=torch_crossformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "segment_len": 16,
            "stride": 16,
            "num_scales": 3,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "segment_len": "Base segment length over (context+horizon) tokens",
            "stride": "Base segment stride over (context+horizon) tokens",
            "num_scales": "Number of scales (>=1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-pyraformer-global": ModelSpec(
        key="torch-pyraformer-global",
        description="Torch Pyraformer-style (lite) pyramid-pooled segmented Transformer encoder trained globally across panel series. Requires PyTorch.",
        factory=torch_pyraformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "segment_len": 16,
            "stride": 16,
            "num_levels": 3,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "segment_len": "Segment length at level-0 over (context+horizon) tokens",
            "stride": "Segment stride at level-0 over (context+horizon) tokens",
            "num_levels": "Number of pyramid levels (>=1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-tsmixer-global": ModelSpec(
        key="torch-tsmixer-global",
        description="Torch TSMixer-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_tsmixer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_blocks": 4,
            "token_mixing_hidden": 128,
            "channel_mixing_hidden": 128,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Mixer model dimension",
            "num_blocks": "Number of mixer blocks",
            "token_mixing_hidden": "Hidden size for token-mixing MLP",
            "channel_mixing_hidden": "Hidden size for channel-mixing MLP",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-itransformer-global": ModelSpec(
        key="torch-itransformer-global",
        description="Torch iTransformer-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_itransformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-timesnet-global": ModelSpec(
        key="torch-timesnet-global",
        description="Torch TimesNet-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_timesnet_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 2,
            "top_k": 3,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Embedding dimension",
            "num_layers": "Number of TimesBlocks",
            "top_k": "Number of dominant periods to use (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-tcn-global": ModelSpec(
        key="torch-tcn-global",
        description="Torch TCN (causal dilated Conv1D) trained globally across panel series. Requires PyTorch.",
        factory=torch_tcn_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "channels": (64, 64, 64),
            "kernel_size": 3,
            "dilation_base": 2,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "channels": "Channel sizes for TCN layers (comma-separated)",
            "kernel_size": "Conv1D kernel size (>=1)",
            "dilation_base": "Dilation base (>=1), layer i uses dilation_base**i",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-nbeats-global": ModelSpec(
        key="torch-nbeats-global",
        description="Torch N-BEATS-style (generic) trained globally across panel series. Requires PyTorch.",
        factory=torch_nbeats_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "num_blocks": 3,
            "num_layers": 2,
            "layer_width": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "num_blocks": "Number of residual blocks",
            "num_layers": "MLP layers per block",
            "layer_width": "Hidden width of MLP layers",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-nhits-global": ModelSpec(
        key="torch-nhits-global",
        description="Torch N-HiTS-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_nhits_global_forecaster,
        default_params={
            "context_length": 192,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "pool_sizes": (1, 2, 4),
            "num_blocks": 6,
            "num_layers": 2,
            "layer_width": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "pool_sizes": "Avg-pool downsample factors (comma-separated, <=context_length)",
            "num_blocks": "Number of residual blocks",
            "num_layers": "MLP layers per block",
            "layer_width": "Hidden width of MLP layers",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-tide-global": ModelSpec(
        key="torch-tide-global",
        description="Torch TiDE-style (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_tide_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "hidden_size": 128,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Latent model dimension",
            "hidden_size": "Hidden size for encoder/decoder MLPs",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-wavenet-global": ModelSpec(
        key="torch-wavenet-global",
        description="Torch WaveNet-style gated dilated CNN trained globally across panel series. Requires PyTorch.",
        factory=torch_wavenet_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "channels": 32,
            "num_layers": 6,
            "kernel_size": 2,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "channels": "Hidden channels in WaveNet stack",
            "num_layers": "Number of dilated convolution layers",
            "kernel_size": "Conv1D kernel size (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-resnet1d-global": ModelSpec(
        key="torch-resnet1d-global",
        description="Torch ResNet-1D Conv1D model trained globally across panel series. Requires PyTorch.",
        factory=torch_resnet1d_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "channels": 32,
            "num_blocks": 4,
            "kernel_size": 3,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "channels": "Conv1D channels",
            "num_blocks": "Number of residual blocks",
            "kernel_size": "Conv1D kernel size (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-inception-global": ModelSpec(
        key="torch-inception-global",
        description="Torch InceptionTime-style Conv1D model trained globally across panel series. Requires PyTorch.",
        factory=torch_inception_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "channels": 32,
            "num_blocks": 3,
            "kernel_sizes": (3, 5, 7),
            "bottleneck_channels": 16,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "channels": "Conv1D channels",
            "num_blocks": "Number of inception blocks",
            "kernel_sizes": "Kernel sizes (comma-separated)",
            "bottleneck_channels": "Bottleneck channels inside inception blocks",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-lstnet-global": ModelSpec(
        key="torch-lstnet-global",
        description="Torch LSTNet-style (CNN+GRU+skip+highway, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_lstnet_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cnn_channels": 16,
            "kernel_size": 6,
            "rnn_hidden": 32,
            "skip": 24,
            "highway_window": 24,
            "id_emb_dim": 8,
            "dropout": 0.2,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cnn_channels": "Conv1D output channels",
            "kernel_size": "Conv1D kernel size (>=1)",
            "rnn_hidden": "GRU hidden size",
            "skip": "Skip length for skip-GRU (0 disables)",
            "highway_window": "Highway window length over the context y channel (0 disables)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-fnet-global": ModelSpec(
        key="torch-fnet-global",
        description="Torch FNet-style (FFT token mixing, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_fnet_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of FNet layers",
            "dim_feedforward": "FFN hidden dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-gmlp-global": ModelSpec(
        key="torch-gmlp-global",
        description="Torch gMLP-style (spatial gating, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_gmlp_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "ffn_dim": 128,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of gMLP layers",
            "ffn_dim": "FFN dimension inside gMLP",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-ssm-global": ModelSpec(
        key="torch-ssm-global",
        description="Torch diagonal state-space model (SSM, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_ssm_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of SSM blocks",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-mamba-global": ModelSpec(
        key="torch-mamba-global",
        description="Torch Mamba-style selective SSM (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_mamba_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "conv_kernel": 3,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of stacked Mamba blocks",
            "conv_kernel": "Causal depthwise conv kernel size (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-rwkv-global": ModelSpec(
        key="torch-rwkv-global",
        description="Torch RWKV-style time-mix + channel-mix (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_rwkv_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "ffn_dim": 128,
            "id_emb_dim": 8,
            "dropout": 0.0,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of stacked RWKV blocks",
            "ffn_dim": "Channel-mix hidden size",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-hyena-global": ModelSpec(
        key="torch-hyena-global",
        description="Torch Hyena-style long convolution (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_hyena_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 4,
            "ffn_dim": 128,
            "kernel_size": 64,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_layers": "Number of Hyena blocks",
            "ffn_dim": "Channel-mixing FFN hidden size",
            "kernel_size": "Depthwise causal conv kernel size (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-dilated-rnn-global": ModelSpec(
        key="torch-dilated-rnn-global",
        description="Torch Dilated RNN (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_dilated_rnn_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "gru",
            "d_model": 64,
            "num_layers": 3,
            "dilation_base": 2,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "Recurrent cell: gru or lstm",
            "d_model": "Hidden size / model dimension",
            "num_layers": "Number of dilated recurrent layers",
            "dilation_base": "Dilation base (>=2); dilations are base^layer_index",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-kan-global": ModelSpec(
        key="torch-kan-global",
        description="Torch KAN-style spline MLP (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_kan_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_layers": 2,
            "grid_size": 16,
            "grid_range": 2.0,
            "linear_skip": True,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Hidden width of the KAN network",
            "num_layers": "Number of KAN spline layers",
            "grid_size": "Number of spline knots (>=4)",
            "grid_range": "Spline grid range (+/- range) in normalized y units",
            "linear_skip": "Add a linear skip connection per layer (true/false)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-scinet-global": ModelSpec(
        key="torch-scinet-global",
        description="Torch SCINet-style sample-convolution interaction network (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_scinet_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "num_stages": 3,
            "conv_kernel": 5,
            "ffn_dim": 128,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Model dimension",
            "num_stages": "Number of SCINet interaction stages",
            "conv_kernel": "Conv1D kernel size (>=1) inside interaction blocks",
            "ffn_dim": "FFN hidden size inside blocks",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-etsformer-global": ModelSpec(
        key="torch-etsformer-global",
        description="Torch ETSformer-style exponential smoothing + Transformer residual model (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_etsformer_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "alpha_init": 0.3,
            "beta_init": 0.1,
            "id_emb_dim": 8,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Transformer encoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "dropout": "Dropout probability in [0,1)",
            "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
            "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-esrnn-global": ModelSpec(
        key="torch-esrnn-global",
        description="Torch ESRNN-style hybrid (Holt smoothing + RNN residual, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_esrnn_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "gru",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "alpha_init": 0.3,
            "beta_init": 0.1,
            "id_emb_dim": 8,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "RNN cell: gru or lstm",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "alpha_init": "Initial smoothing alpha in (0,1) (learned during training)",
            "beta_init": "Initial smoothing beta in (0,1) (learned during training)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-transformer-encdec-global": ModelSpec(
        key="torch-transformer-encdec-global",
        description="Torch encoder-decoder Transformer (lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_transformer_encdec_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "id_emb_dim": 8,
            "dropout": 0.1,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "d_model": "Transformer model dimension",
            "nhead": "Attention heads",
            "num_layers": "Encoder/decoder layers",
            "dim_feedforward": "Transformer FFN dimension",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-nlinear-global": ModelSpec(
        key="torch-nlinear-global",
        description="Torch NLinear-style (last-value centering + linear head, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_nlinear_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "id_emb_dim": 8,
            "dropout": 0.0,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-dlinear-global": ModelSpec(
        key="torch-dlinear-global",
        description="Torch DLinear-style (moving-average decomposition + linear heads, lite) trained globally across panel series. Requires PyTorch.",
        factory=torch_dlinear_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "ma_window": 7,
            "id_emb_dim": 8,
            "dropout": 0.0,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "ma_window": "Moving-average window size for trend extraction (>=1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-deepar-global": ModelSpec(
        key="torch-deepar-global",
        description="Torch DeepAR-style global Gaussian RNN (direct multi-horizon, lite). Requires PyTorch.",
        factory=torch_deepar_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "hidden_size": 64,
            "num_layers": 1,
            "id_emb_dim": 8,
            "dropout": 0.0,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "hidden_size": "GRU hidden size",
            "num_layers": "Number of GRU layers",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "quantiles": "Optional quantiles (computed from Normal params; adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
            "loss": "Ignored (DeepAR uses Gaussian NLL)",
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-seq2seq-lstm-global": ModelSpec(
        key="torch-seq2seq-lstm-global",
        description="Torch Seq2Seq (encoder-decoder) global model (LSTM, no attention). Requires PyTorch.",
        factory=torch_seq2seq_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "lstm",
            "attention": "none",
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
            "id_emb_dim": 8,
            "teacher_forcing": 0.5,
            "teacher_forcing_final": None,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "RNN cell: lstm, gru",
            "attention": "Attention: none, bahdanau",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "teacher_forcing": "Teacher forcing ratio at the start of training",
            "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-seq2seq-gru-global": ModelSpec(
        key="torch-seq2seq-gru-global",
        description="Torch Seq2Seq (encoder-decoder) global model (GRU, no attention). Requires PyTorch.",
        factory=torch_seq2seq_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "gru",
            "attention": "none",
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
            "id_emb_dim": 8,
            "teacher_forcing": 0.5,
            "teacher_forcing_final": None,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "RNN cell: lstm, gru",
            "attention": "Attention: none, bahdanau",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "teacher_forcing": "Teacher forcing ratio at the start of training",
            "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-seq2seq-attn-lstm-global": ModelSpec(
        key="torch-seq2seq-attn-lstm-global",
        description="Torch Seq2Seq (encoder-decoder) global model (LSTM + Bahdanau attention). Requires PyTorch.",
        factory=torch_seq2seq_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "lstm",
            "attention": "bahdanau",
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
            "id_emb_dim": 8,
            "teacher_forcing": 0.5,
            "teacher_forcing_final": None,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "RNN cell: lstm, gru",
            "attention": "Attention: none, bahdanau",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "teacher_forcing": "Teacher forcing ratio at the start of training",
            "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
    ),
    "torch-seq2seq-attn-gru-global": ModelSpec(
        key="torch-seq2seq-attn-gru-global",
        description="Torch Seq2Seq (encoder-decoder) global model (GRU + Bahdanau attention). Requires PyTorch.",
        factory=torch_seq2seq_global_forecaster,
        default_params={
            "context_length": 96,
            "x_cols": (),
            "add_time_features": True,
            "sample_step": 1,
            "cell": "gru",
            "attention": "bahdanau",
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
            "id_emb_dim": 8,
            "teacher_forcing": 0.5,
            "teacher_forcing_final": None,
            "quantiles": (),
            "max_train_size": None,
            **_TORCH_COMMON_DEFAULTS,
            "epochs": 30,
            "batch_size": 64,
            "val_split": 0.1,
        },
        param_help={
            "context_length": "Context window length (encoder length)",
            "x_cols": "Optional covariate columns from long_df (comma-separated)",
            "add_time_features": "Add built-in time features from ds (true/false)",
            "sample_step": "Stride when generating training windows (>=1)",
            "cell": "RNN cell: lstm, gru",
            "attention": "Attention: none, bahdanau",
            "hidden_size": "RNN hidden size",
            "num_layers": "Number of stacked RNN layers",
            "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
            "id_emb_dim": "Series-id embedding dim (panel/global models)",
            "teacher_forcing": "Teacher forcing ratio at the start of training",
            "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
            "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
            "max_train_size": "Optional per-series rolling training window length (None for expanding)",
            **_TORCH_COMMON_PARAM_HELP,
        },
        requires=("torch",),
        interface="global",
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
    "ets": ModelSpec(
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
    ),
    "arima": ModelSpec(
        key="arima",
        description="ARIMA(p,d,q) via statsmodels. Optional dependency.",
        factory=_factory_arima,
        default_params={"order": (1, 0, 0)},
        param_help={"order": "ARIMA order tuple (p,d,q)"},
        requires=("stats",),
    ),
    "auto-arima": ModelSpec(
        key="auto-arima",
        description="AutoARIMA-style grid search via statsmodels. Optional dependency.",
        factory=_factory_auto_arima,
        default_params={"max_p": 3, "max_d": 2, "max_q": 3, "information_criterion": "aic"},
        param_help={
            "max_p": "Max AR order p to consider",
            "max_d": "Max differencing order d to consider",
            "max_q": "Max MA order q to consider",
            "information_criterion": "Model selection criterion: aic or bic",
        },
        requires=("stats",),
    ),
    "sarimax": ModelSpec(
        key="sarimax",
        description="SARIMAX / seasonal ARIMA via statsmodels. Optional dependency.",
        factory=_factory_sarimax,
        default_params={
            "order": (1, 0, 0),
            "seasonal_order": (0, 0, 0, 0),
            "trend": None,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        param_help={
            "order": "Non-seasonal order (p,d,q)",
            "seasonal_order": "Seasonal order (P,D,Q,s)",
            "trend": "Trend term (e.g. c, t, ct) or none",
            "enforce_stationarity": "Enforce stationarity (true/false)",
            "enforce_invertibility": "Enforce invertibility (true/false)",
        },
        requires=("stats",),
    ),
    "autoreg": ModelSpec(
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
    "uc-local-level": ModelSpec(
        key="uc-local-level",
        description="UnobservedComponents local level via statsmodels. Optional dependency.",
        factory=_factory_unobserved_components,
        default_params={"level": "local level"},
        param_help={"level": "Level specification string (default: 'local level')"},
        requires=("stats",),
    ),
    "uc-local-linear-trend": ModelSpec(
        key="uc-local-linear-trend",
        description="UnobservedComponents local linear trend via statsmodels. Optional dependency.",
        factory=_factory_unobserved_components,
        default_params={"level": "local linear trend"},
        param_help={"level": "Level specification string (default: 'local linear trend')"},
        requires=("stats",),
    ),
    "stl-arima": ModelSpec(
        key="stl-arima",
        description="STL + ARIMA remainder forecasting via statsmodels. Optional dependency.",
        factory=_factory_stl_arima,
        default_params={"period": 12, "order": (1, 0, 0), "seasonal": 7, "robust": False},
        param_help={
            "period": "Seasonal period for STL",
            "order": "ARIMA order for remainder model (p,d,q)",
            "seasonal": "STL seasonal smoother length (odd integer >= 3; default 7)",
            "robust": "Robust STL (true/false)",
        },
        requires=("stats",),
    ),
    "mstl-arima": ModelSpec(
        key="mstl-arima",
        description="MSTL (multi-seasonal STL) + ARIMA on seasonally-adjusted series. Optional dependency.",
        factory=_factory_mstl_arima,
        default_params={"periods": (12,), "order": (1, 0, 0), "iterate": 2, "lmbda": None},
        param_help={
            "periods": "Comma-separated seasonal periods (e.g. 7,365)",
            "order": "ARIMA order for adjusted series (p,d,q)",
            "iterate": "MSTL iterations (default: 2)",
            "lmbda": "Box-Cox lambda for MSTL (float, 'auto', or none)",
        },
        requires=("stats",),
    ),
    "mstl-auto-arima": ModelSpec(
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
            "information_criterion": "aic",
        },
        param_help={
            "periods": "Comma-separated seasonal periods (e.g. 7,365)",
            "iterate": "MSTL iterations (default: 2)",
            "lmbda": "Box-Cox lambda for MSTL (float, 'auto', or none)",
            "max_p": "Max AR order p to consider",
            "max_d": "Max differencing order d to consider",
            "max_q": "Max MA order q to consider",
            "information_criterion": "Model selection criterion: aic or bic",
        },
        requires=("stats",),
    ),
    "tbats-lite": ModelSpec(
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
            "periods": "Comma-separated seasonal periods (e.g. 7,365)",
            "orders": "Fourier order per period (int or comma-separated list)",
            "include_trend": "Include linear trend term (true/false)",
            "arima_order": "ARIMA order for residual errors (p,d,q)",
            "boxcox_lambda": "Optional Box-Cox lambda (float); requires y > 0",
        },
        requires=("stats",),
    ),
}


def _make_torch_dl_variant_specs() -> dict[str, ModelSpec]:
    extra: dict[str, ModelSpec] = {}

    xformer_help = {
        "lags": "Lag window length",
        "d_model": "Transformer model dimension",
        "nhead": "Attention heads",
        "num_layers": "Number of encoder layers",
        "dim_feedforward": "FFN hidden dimension",
        "dropout": "Dropout probability in [0,1)",
        "attn": "Attention type: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer",
        "pos_emb": "Positional embedding: learned, sincos, rope, time2vec, none",
        "norm": "Normalization: layer, rms",
        "ffn": "FFN: gelu, swiglu",
        "local_window": "Local attention window radius (attn=local/logsparse/longformer/bigbird)",
        "bigbird_random_k": "Random key connections per token (attn=bigbird)",
        "performer_features": "Performer random feature count (attn=performer)",
        "linformer_k": "Linformer projection length (attn=linformer)",
        "nystrom_landmarks": "Nyström landmarks (attn=nystrom)",
        "reformer_bucket_size": "Reformer LSH bucket size (attn=reformer)",
        "reformer_n_hashes": "Reformer LSH hash rounds (attn=reformer)",
        "probsparse_top_u": "Top-u queries for ProbSparse attention (attn=probsparse)",
        "autocorr_top_k": "Top-k delays for AutoCorrelation attention (attn=autocorr)",
        "horizon_tokens": "Future token placeholders: zeros, learned",
        "revin": "RevIN per-window normalization (true/false)",
        "residual_gating": "Residual gating (true/false)",
        "drop_path": "Stochastic depth drop probability in [0,1)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    xformer_base_defaults = {
        "lags": 96,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "attn": "full",
        "pos_emb": "learned",
        "norm": "layer",
        "ffn": "gelu",
        "local_window": 16,
        "bigbird_random_k": 8,
        "performer_features": 64,
        "linformer_k": 32,
        "nystrom_landmarks": 16,
        "reformer_bucket_size": 8,
        "reformer_n_hashes": 1,
        "probsparse_top_u": 32,
        "autocorr_top_k": 4,
        "horizon_tokens": "zeros",
        "revin": False,
        "residual_gating": False,
        "drop_path": 0.0,
        **_TORCH_COMMON_DEFAULTS,
    }

    def _add_local_xformer(
        key: str,
        description: str,
        **overrides: Any,
    ) -> None:
        extra[key] = ModelSpec(
            key=key,
            description=description,
            factory=_factory_torch_xformer_direct,
            default_params={**xformer_base_defaults, **overrides},
            param_help=dict(xformer_help),
            requires=("torch",),
        )

    # Local xFormer variants: (attn) x (norm) x (ffn)
    for attn_s, attn_label in [
        ("full", "full"),
        ("local", "local-window"),
        ("logsparse", "log-sparse"),
        ("longformer", "longformer-windowed+global"),
        ("bigbird", "bigbird-random+local+global"),
        ("performer", "performer"),
        ("linformer", "linformer"),
        ("nystrom", "nystrom"),
        ("probsparse", "prob-sparse"),
        ("autocorr", "auto-correlation"),
        ("reformer", "reformer-lsh"),
    ]:
        for norm_s, norm_label in [("layer", "LayerNorm"), ("rms", "RMSNorm")]:
            for ffn_s, ffn_label in [("gelu", "GELU"), ("swiglu", "SwiGLU")]:
                norm_short = "ln" if norm_s == "layer" else "rms"
                key = f"torch-xformer-{attn_s}-{norm_short}-{ffn_s}-direct"
                _add_local_xformer(
                    key,
                    f"Torch xFormer ({attn_label} attention) with {norm_label}+{ffn_label} (direct multi-horizon). Requires PyTorch.",
                    attn=attn_s,
                    norm=norm_s,
                    ffn=ffn_s,
                )

    # 41–44: RoPE positional variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        _add_local_xformer(
            f"torch-xformer-{attn_s}-rope-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with RoPE positional encoding (LN+GELU). Requires PyTorch.",
            attn=attn_s,
            pos_emb="rope",
            norm="layer",
            ffn="gelu",
        )

    # 45–48: sincos pos variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        _add_local_xformer(
            f"torch-xformer-{attn_s}-sincos-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with sinusoidal positional encoding (LN+GELU). Requires PyTorch.",
            attn=attn_s,
            pos_emb="sincos",
            norm="layer",
            ffn="gelu",
        )

    # 49–52: Time2Vec pos variants (LN+GELU)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        _add_local_xformer(
            f"torch-xformer-{attn_s}-time2vec-ln-gelu-direct",
            f"Torch xFormer ({attn_s} attention) with Time2Vec positional features (LN+GELU). Requires PyTorch.",
            attn=attn_s,
            pos_emb="time2vec",
            norm="layer",
            ffn="gelu",
        )

    # 53–56: RevIN variants
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        _add_local_xformer(
            f"torch-xformer-{attn_s}-revin-direct",
            f"Torch xFormer ({attn_s} attention) with RevIN (direct multi-horizon). Requires PyTorch.",
            attn=attn_s,
            revin=True,
        )

    # 57–60: deeper/wider configs
    _add_local_xformer(
        "torch-xformer-full-deep-direct",
        "Torch xFormer (full attention) deeper config (4 layers). Requires PyTorch.",
        attn="full",
        num_layers=4,
    )
    _add_local_xformer(
        "torch-xformer-performer-deep-direct",
        "Torch xFormer (performer attention) deeper config (4 layers). Requires PyTorch.",
        attn="performer",
        num_layers=4,
    )
    _add_local_xformer(
        "torch-xformer-full-wide-direct",
        "Torch xFormer (full attention) wider config (d_model=128). Requires PyTorch.",
        attn="full",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )
    _add_local_xformer(
        "torch-xformer-performer-wide-direct",
        "Torch xFormer (performer attention) wider config (d_model=128). Requires PyTorch.",
        attn="performer",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )

    # ---- Local RNN family (Seq2Seq + LSTNet) ----
    seq2seq_help = {
        "lags": "Lag window length (encoder length)",
        "cell": "RNN cell: lstm, gru",
        "attention": "Attention: none, bahdanau",
        "hidden_size": "RNN hidden size",
        "num_layers": "Number of stacked RNN layers",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
        "teacher_forcing": "Teacher forcing ratio at the start of training",
        "teacher_forcing_final": "Teacher forcing ratio at the end of training (None keeps it constant)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    seq2seq_base_defaults = {
        "lags": 48,
        "cell": "lstm",
        "attention": "none",
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "teacher_forcing": 0.5,
        "teacher_forcing_final": None,
        **_TORCH_COMMON_DEFAULTS,
        "val_split": 0.1,
    }

    def _add_local_seq2seq(key: str, description: str, **overrides: Any) -> None:
        extra[key] = ModelSpec(
            key=key,
            description=description,
            factory=_factory_torch_seq2seq_direct,
            default_params={**seq2seq_base_defaults, **overrides},
            param_help=dict(seq2seq_help),
            requires=("torch",),
        )

    _add_local_seq2seq(
        "torch-seq2seq-lstm-direct",
        "Torch Seq2Seq LSTM (encoder-decoder) direct multi-horizon. Requires PyTorch.",
        cell="lstm",
        attention="none",
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-direct",
        "Torch Seq2Seq GRU (encoder-decoder) direct multi-horizon. Requires PyTorch.",
        cell="gru",
        attention="none",
    )
    _add_local_seq2seq(
        "torch-seq2seq-attn-lstm-direct",
        "Torch Seq2Seq LSTM with Bahdanau attention (direct multi-horizon). Requires PyTorch.",
        cell="lstm",
        attention="bahdanau",
    )
    _add_local_seq2seq(
        "torch-seq2seq-attn-gru-direct",
        "Torch Seq2Seq GRU with Bahdanau attention (direct multi-horizon). Requires PyTorch.",
        cell="gru",
        attention="bahdanau",
    )
    _add_local_seq2seq(
        "torch-seq2seq-lstm-deep-direct",
        "Torch Seq2Seq LSTM deeper config (2 layers). Requires PyTorch.",
        cell="lstm",
        attention="none",
        num_layers=2,
        dropout=0.1,
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-deep-direct",
        "Torch Seq2Seq GRU deeper config (2 layers). Requires PyTorch.",
        cell="gru",
        attention="none",
        num_layers=2,
        dropout=0.1,
    )
    _add_local_seq2seq(
        "torch-seq2seq-lstm-wide-direct",
        "Torch Seq2Seq LSTM wider config (hidden_size=128). Requires PyTorch.",
        cell="lstm",
        attention="none",
        hidden_size=128,
    )
    _add_local_seq2seq(
        "torch-seq2seq-gru-wide-direct",
        "Torch Seq2Seq GRU wider config (hidden_size=128). Requires PyTorch.",
        cell="gru",
        attention="none",
        hidden_size=128,
    )

    lstnet_help = {
        "lags": "Lag window length",
        "cnn_channels": "CNN output channels",
        "kernel_size": "CNN kernel size",
        "rnn_hidden": "GRU hidden size",
        "skip": "Skip period (0 disables)",
        "highway_window": "Highway window length (0 disables)",
        "dropout": "Dropout probability in [0,1)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    extra["torch-lstnet-direct"] = ModelSpec(
        key="torch-lstnet-direct",
        description="Torch LSTNet-style CNN+GRU(+skip)+highway (lite) direct multi-horizon. Requires PyTorch.",
        factory=_factory_torch_lstnet_direct,
        default_params={
            "lags": 96,
            "cnn_channels": 16,
            "kernel_size": 6,
            "rnn_hidden": 32,
            "skip": 24,
            "highway_window": 24,
            "dropout": 0.2,
            **_TORCH_COMMON_DEFAULTS,
        },
        param_help=dict(lstnet_help),
        requires=("torch",),
    )

    # ---- Global Transformer-family variants ----
    xformer_global_help = {
        "context_length": "Context window length (encoder length)",
        "x_cols": "Optional covariate columns from long_df (comma-separated)",
        "add_time_features": "Add built-in time features from ds (true/false)",
        "normalize": "Z-score normalize per-series inside each cutoff window (true/false)",
        "max_train_size": "Optional per-series rolling training window length (None for expanding)",
        "sample_step": "Stride when generating training windows (>=1)",
        "d_model": "Transformer model dimension",
        "nhead": "Attention heads",
        "num_layers": "Number of encoder layers",
        "dim_feedforward": "FFN hidden dimension",
        "id_emb_dim": "Series-id embedding dim (panel/global models)",
        "dropout": "Dropout probability in [0,1)",
        "attn": "Attention type: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer",
        "pos_emb": "Positional embedding: learned, sincos, rope, time2vec, none",
        "norm": "Normalization: layer, rms",
        "ffn": "FFN: gelu, swiglu",
        "local_window": "Local attention window radius (attn=local/logsparse/longformer/bigbird)",
        "bigbird_random_k": "Random key connections per token (attn=bigbird)",
        "performer_features": "Performer random feature count (attn=performer)",
        "linformer_k": "Linformer projection length (attn=linformer)",
        "nystrom_landmarks": "Nyström landmarks (attn=nystrom)",
        "reformer_bucket_size": "Reformer LSH bucket size (attn=reformer)",
        "reformer_n_hashes": "Reformer LSH hash rounds (attn=reformer)",
        "probsparse_top_u": "Top-u queries for ProbSparse attention (attn=probsparse)",
        "autocorr_top_k": "Top-k delays for AutoCorrelation attention (attn=autocorr)",
        "residual_gating": "Residual gating (true/false)",
        "drop_path": "Stochastic depth drop probability in [0,1)",
        "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    xformer_global_base_defaults = {
        "context_length": 96,
        "x_cols": (),
        "add_time_features": True,
        "normalize": True,
        "max_train_size": None,
        "sample_step": 1,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "id_emb_dim": 8,
        "dropout": 0.1,
        "attn": "full",
        "pos_emb": "learned",
        "norm": "layer",
        "ffn": "gelu",
        "local_window": 16,
        "bigbird_random_k": 8,
        "performer_features": 64,
        "linformer_k": 32,
        "nystrom_landmarks": 16,
        "reformer_bucket_size": 8,
        "reformer_n_hashes": 1,
        "probsparse_top_u": 32,
        "autocorr_top_k": 4,
        "residual_gating": False,
        "drop_path": 0.0,
        "quantiles": (),
        **_TORCH_COMMON_DEFAULTS,
        "epochs": 30,
        "batch_size": 64,
        "val_split": 0.1,
    }

    def _add_global_xformer(key: str, description: str, **overrides: Any) -> None:
        extra[key] = ModelSpec(
            key=key,
            description=description,
            factory=torch_xformer_global_forecaster,
            default_params={**xformer_global_base_defaults, **overrides},
            param_help=dict(xformer_global_help),
            requires=("torch",),
            interface="global",
        )

    # 61–65: baseline attention variants
    for attn_s in [
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    ]:
        _add_global_xformer(
            f"torch-xformer-{attn_s}-global",
            f"Torch global xFormer ({attn_s} attention) baseline. Requires PyTorch.",
            attn=attn_s,
        )

    # 66–70: RMSNorm variants
    for attn_s in [
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    ]:
        _add_global_xformer(
            f"torch-xformer-{attn_s}-rms-global",
            f"Torch global xFormer ({attn_s} attention) with RMSNorm. Requires PyTorch.",
            attn=attn_s,
            norm="rms",
        )

    # 71–74: SwiGLU variants (subset)
    for attn_s in ["full", "performer", "linformer", "nystrom"]:
        _add_global_xformer(
            f"torch-xformer-{attn_s}-swiglu-global",
            f"Torch global xFormer ({attn_s} attention) with SwiGLU FFN. Requires PyTorch.",
            attn=attn_s,
            ffn="swiglu",
        )

    # 75–78: positional variants
    _add_global_xformer(
        "torch-xformer-full-rope-global",
        "Torch global xFormer (full attention) with RoPE positional encoding. Requires PyTorch.",
        attn="full",
        pos_emb="rope",
    )
    _add_global_xformer(
        "torch-xformer-performer-rope-global",
        "Torch global xFormer (performer attention) with RoPE positional encoding. Requires PyTorch.",
        attn="performer",
        pos_emb="rope",
    )
    _add_global_xformer(
        "torch-xformer-full-sincos-global",
        "Torch global xFormer (full attention) with sinusoidal positional encoding. Requires PyTorch.",
        attn="full",
        pos_emb="sincos",
    )
    _add_global_xformer(
        "torch-xformer-full-time2vec-global",
        "Torch global xFormer (full attention) with Time2Vec positional features. Requires PyTorch.",
        attn="full",
        pos_emb="time2vec",
    )

    # 79–80: deeper/wider configs
    _add_global_xformer(
        "torch-xformer-full-deep-global",
        "Torch global xFormer (full attention) deeper config (4 layers). Requires PyTorch.",
        attn="full",
        num_layers=4,
    )
    _add_global_xformer(
        "torch-xformer-full-wide-global",
        "Torch global xFormer (full attention) wider config (d_model=128). Requires PyTorch.",
        attn="full",
        d_model=128,
        nhead=8,
        dim_feedforward=512,
    )

    # ---- Global RNN variants ----
    rnn_global_help = {
        "context_length": "Context window length (encoder length)",
        "x_cols": "Optional covariate columns from long_df (comma-separated)",
        "add_time_features": "Add built-in time features from ds (true/false)",
        "normalize": "Z-score normalize per-series inside each cutoff window (true/false)",
        "max_train_size": "Optional per-series rolling training window length (None for expanding)",
        "sample_step": "Stride when generating training windows (>=1)",
        "cell": "RNN cell: lstm, gru",
        "hidden_size": "RNN hidden size",
        "num_layers": "Number of stacked RNN layers",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1)",
        "id_emb_dim": "Series-id embedding dim (panel/global models)",
        "quantiles": "Optional quantiles for pinball loss, e.g. 0.1,0.5,0.9 (adds yhat_pXX columns)",
        **_TORCH_COMMON_PARAM_HELP,
    }
    rnn_global_base_defaults = {
        "context_length": 96,
        "x_cols": (),
        "add_time_features": True,
        "normalize": True,
        "max_train_size": None,
        "sample_step": 1,
        "cell": "lstm",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "id_emb_dim": 8,
        "quantiles": (),
        **_TORCH_COMMON_DEFAULTS,
        "epochs": 30,
        "batch_size": 64,
        "val_split": 0.1,
    }

    def _add_global_rnn(key: str, description: str, **overrides: Any) -> None:
        extra[key] = ModelSpec(
            key=key,
            description=description,
            factory=torch_rnn_global_forecaster,
            default_params={**rnn_global_base_defaults, **overrides},
            param_help=dict(rnn_global_help),
            requires=("torch",),
            interface="global",
        )

    _add_global_rnn(
        "torch-rnn-lstm-global",
        "Torch global RNN backbone (LSTM) with token-wise horizon head. Requires PyTorch.",
        cell="lstm",
    )
    _add_global_rnn(
        "torch-rnn-gru-global",
        "Torch global RNN backbone (GRU) with token-wise horizon head. Requires PyTorch.",
        cell="gru",
    )
    _add_global_rnn(
        "torch-rnn-encoder-global",
        "Torch global encoder-only RNN horizon head (seq2seq-lite). Requires PyTorch.",
        cell="lstm",
        hidden_size=32,
    )

    return extra


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
            restore_best=restore_best_bool,
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
            restore_best=restore_best_bool,
        )

    return _f


def _make_torch_rnnpaper_specs() -> dict[str, ModelSpec]:
    extra: dict[str, ModelSpec] = {}

    help_map = {
        "paper": "Paper-named RNN architecture (fixed per key)",
        "lags": "Lag window length",
        "hidden_size": "Hidden size",
        "num_layers": "Stacked layers (only for some built-in torch RNN bases)",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1 for torch bases)",
        "attn_hidden": "Attention MLP hidden size (for attention/memory variants)",
        "kernel_size": "Conv1d kernel size (for QRNN / Conv* variants)",
        "hops": "Attention hops (memory networks)",
        "memory_slots": "External memory slots (NTM/DNC variants)",
        "memory_dim": "External memory slot size (NTM/DNC variants)",
        "spectral_radius": "Reservoir spectral radius (ESN/LSM/Conceptor variants)",
        "leak": "Reservoir leaking rate in (0,1] (ESN/LSM/Conceptor variants)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "attn_hidden": 32,
        "kernel_size": 3,
        "hops": 2,
        "memory_slots": 16,
        "memory_dim": 32,
        "spectral_radius": 0.9,
        "leak": 1.0,
        **_TORCH_COMMON_DEFAULTS,
    }

    for spec in list_rnnpaper_specs():
        extra[spec.key] = ModelSpec(
            key=spec.key,
            description=spec.description + ". Requires PyTorch.",
            factory=_factory_torch_rnnpaper_direct,
            default_params={**base_defaults, "paper": spec.paper_id},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


def _make_torch_rnnzoo_specs() -> dict[str, ModelSpec]:
    extra: dict[str, ModelSpec] = {}

    help_map = {
        "base": "Base RNN architecture (paper-named; fixed per key)",
        "variant": "Architecture wrapper: direct, bidir, ln, attn, proj (fixed per key)",
        "lags": "Lag window length",
        "hidden_size": "Hidden size",
        "num_layers": "Number of stacked layers (only for torch RNN/LSTM/GRU bases)",
        "dropout": "Dropout probability in [0,1) (only if num_layers>1 for torch bases)",
        "proj_size": "Projection size for variant=proj",
        "attn_hidden": "Attention MLP hidden size for variant=attn",
        "clock_periods": "Clockwork periods as tuple or comma-separated string (clockwork base)",
        "qrnn_kernel_size": "QRNN Conv1d kernel size (qrnn base)",
        "rhn_depth": "RHN transition depth (rhn base)",
        "phased_tau": "Phased LSTM time-gate period (phased-lstm base)",
        "phased_r_on": "Phased LSTM open ratio in (0,1) (phased-lstm base)",
        "phased_leak": "Phased LSTM closed-phase leak in [0,1) (phased-lstm base)",
        **_TORCH_COMMON_PARAM_HELP,
    }

    base_defaults = {
        "lags": 24,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "proj_size": 16,
        "attn_hidden": 32,
        "clock_periods": (1, 2, 4, 8),
        "qrnn_kernel_size": 3,
        "rhn_depth": 2,
        "phased_tau": 32.0,
        "phased_r_on": 0.05,
        "phased_leak": 0.001,
        **_TORCH_COMMON_DEFAULTS,
    }

    for spec in list_rnnzoo_specs():
        extra[spec.key] = ModelSpec(
            key=spec.key,
            description=spec.description + ". Requires PyTorch.",
            factory=_factory_torch_rnnzoo_direct,
            default_params={**base_defaults, "base": spec.base, "variant": str(spec.variant)},
            param_help=dict(help_map),
            requires=("torch",),
            interface="local",
        )

    return extra


_EXTRA_RNNPAPER = _make_torch_rnnpaper_specs()
_RNNPAPER_CLASH = set(_EXTRA_RNNPAPER).intersection(_REGISTRY)
if _RNNPAPER_CLASH:
    raise RuntimeError(f"Internal error: model key collision(s): {sorted(_RNNPAPER_CLASH)}")
_REGISTRY.update(_EXTRA_RNNPAPER)


_EXTRA_RNNZOO = _make_torch_rnnzoo_specs()
_RNNZOO_CLASH = set(_EXTRA_RNNZOO).intersection(_REGISTRY)
if _RNNZOO_CLASH:
    raise RuntimeError(f"Internal error: model key collision(s): {sorted(_RNNZOO_CLASH)}")
_REGISTRY.update(_EXTRA_RNNZOO)


_EXTRA_TORCH_VARIANTS = _make_torch_dl_variant_specs()
_CLASH = set(_EXTRA_TORCH_VARIANTS).intersection(_REGISTRY)
if _CLASH:
    raise RuntimeError(f"Internal error: model key collision(s): {sorted(_CLASH)}")
_REGISTRY.update(_EXTRA_TORCH_VARIANTS)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown model key: {key!r}. Try one of: {', '.join(list_models())}") from e


def make_forecaster(key: str, **params: Any) -> ForecasterFn:
    """
    Build a (train, horizon) -> y_pred callable based on a registered model.

    The returned callable is suitable for `foresight.backtesting.walk_forward`.
    """
    spec = get_model_spec(key)
    if str(spec.interface).lower().strip() != "local":
        raise ValueError(
            f"Model {key!r} uses interface={spec.interface!r} (not 'local'). "
            "Use `make_global_forecaster()` instead."
        )
    merged = dict(spec.default_params)
    merged.update(params)
    return spec.factory(**merged)


def make_global_forecaster(key: str, **params: Any) -> GlobalForecasterFn:
    """
    Build a (long_df, cutoff, horizon) -> predictions DataFrame callable.

    Global models are trained across all series in a long-format (panel) DataFrame.
    """
    spec = get_model_spec(key)
    if str(spec.interface).lower().strip() != "global":
        raise ValueError(
            f"Model {key!r} uses interface={spec.interface!r} (not 'global'). "
            "Use `make_forecaster()` instead."
        )
    merged = dict(spec.default_params)
    merged.update(params)
    out = spec.factory(**merged)
    if not callable(out):
        raise TypeError(f"Global model factory must return a callable, got: {type(out).__name__}")
    return out
