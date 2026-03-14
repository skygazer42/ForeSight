import importlib.util

import numpy as np
import pytest

from foresight.models import baselines, multivariate, naive, regression, smoothing, torch_ct_rnn, torch_ssm

SERIES = np.arange(20.0, dtype=float)
POSITIVE_SERIES = np.arange(1.0, 21.0, dtype=float)
MULTIVARIATE_SERIES = np.arange(40.0, dtype=float).reshape(20, 2)
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def _assert_value_error(fn, /, *args, message: str, **kwargs) -> None:
    with pytest.raises(ValueError) as exc_info:
        fn(*args, **kwargs)
    assert str(exc_info.value) == message


PURE_VALIDATION_CASES = [
    ("mean horizon", baselines.mean_forecast, ([1.0],), {"horizon": 0}, baselines.HORIZON_MIN_ERROR),
    ("median horizon", baselines.median_forecast, ([1.0],), {"horizon": 0}, baselines.HORIZON_MIN_ERROR),
    ("drift horizon", baselines.drift_forecast, ([1.0, 2.0],), {"horizon": 0}, baselines.HORIZON_MIN_ERROR),
    (
        "moving average horizon",
        baselines.moving_average_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "window": 1},
        baselines.HORIZON_MIN_ERROR,
    ),
    (
        "moving average window",
        baselines.moving_average_forecast,
        ([1.0, 2.0],),
        {"horizon": 1, "window": 0},
        baselines.WINDOW_MIN_ERROR,
    ),
    (
        "weighted moving average horizon",
        baselines.weighted_moving_average_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "window": 1},
        baselines.HORIZON_MIN_ERROR,
    ),
    (
        "weighted moving average window",
        baselines.weighted_moving_average_forecast,
        ([1.0, 2.0],),
        {"horizon": 1, "window": 0},
        baselines.WINDOW_MIN_ERROR,
    ),
    (
        "moving median horizon",
        baselines.moving_median_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "window": 1},
        baselines.HORIZON_MIN_ERROR,
    ),
    (
        "moving median window",
        baselines.moving_median_forecast,
        ([1.0, 2.0],),
        {"horizon": 1, "window": 0},
        baselines.WINDOW_MIN_ERROR,
    ),
    (
        "seasonal mean horizon",
        baselines.seasonal_mean_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "season_length": 1},
        baselines.HORIZON_MIN_ERROR,
    ),
    (
        "seasonal mean season length",
        baselines.seasonal_mean_forecast,
        ([1.0, 2.0],),
        {"horizon": 1, "season_length": 0},
        baselines.SEASON_LENGTH_MIN_ERROR,
    ),
    (
        "seasonal drift horizon",
        baselines.seasonal_drift_forecast,
        ([1.0, 2.0, 3.0, 4.0],),
        {"horizon": 0, "season_length": 2},
        baselines.HORIZON_MIN_ERROR,
    ),
    (
        "seasonal drift season length",
        baselines.seasonal_drift_forecast,
        ([1.0, 2.0, 3.0, 4.0],),
        {"horizon": 1, "season_length": 0},
        baselines.SEASON_LENGTH_MIN_ERROR,
    ),
    ("naive last horizon", naive.naive_last, ([1.0],), {"horizon": 0}, naive.HORIZON_MIN_ERROR),
    (
        "seasonal naive horizon",
        naive.seasonal_naive,
        ([1.0, 2.0],),
        {"horizon": 0, "season_length": 1},
        naive.HORIZON_MIN_ERROR,
    ),
    (
        "seasonal naive auto horizon",
        naive.seasonal_naive_auto,
        ([1.0],),
        {"horizon": 0},
        naive.HORIZON_MIN_ERROR,
    ),
    (
        "ses horizon",
        smoothing.ses_forecast,
        ([1.0],),
        {"horizon": 0, "alpha": 0.5},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "ses auto horizon",
        smoothing.ses_auto_forecast,
        ([1.0, 2.0],),
        {"horizon": 0},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "ses auto grid size",
        smoothing.ses_auto_forecast,
        ([1.0, 2.0],),
        {"horizon": 1, "grid_size": 1},
        smoothing.GRID_SIZE_MIN_ERROR,
    ),
    (
        "holt horizon",
        smoothing.holt_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "alpha": 0.5, "beta": 0.5},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "holt damped horizon",
        smoothing.holt_damped_forecast,
        ([1.0, 2.0],),
        {"horizon": 0, "alpha": 0.5, "beta": 0.5},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "holt auto horizon",
        smoothing.holt_auto_forecast,
        ([1.0, 2.0, 3.0],),
        {"horizon": 0},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "holt auto grid size",
        smoothing.holt_auto_forecast,
        ([1.0, 2.0, 3.0],),
        {"horizon": 1, "grid_size": 1},
        smoothing.GRID_SIZE_MIN_ERROR,
    ),
    (
        "hw additive horizon",
        smoothing.holt_winters_additive_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 0, "season_length": 2, "alpha": 0.5, "beta": 0.5, "gamma": 0.5},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "hw additive season length",
        smoothing.holt_winters_additive_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 0, "alpha": 0.5, "beta": 0.5, "gamma": 0.5},
        smoothing.SEASON_LENGTH_MIN_ERROR,
    ),
    (
        "hw additive auto horizon",
        smoothing.holt_winters_additive_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 0, "season_length": 2},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "hw additive auto season length",
        smoothing.holt_winters_additive_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 0},
        smoothing.SEASON_LENGTH_MIN_ERROR,
    ),
    (
        "hw additive auto grid size",
        smoothing.holt_winters_additive_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 2, "grid_size": 1},
        smoothing.GRID_SIZE_MIN_ERROR,
    ),
    (
        "hw multiplicative horizon",
        smoothing.holt_winters_multiplicative_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 0, "season_length": 2, "alpha": 0.5, "beta": 0.5, "gamma": 0.5},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "hw multiplicative season length",
        smoothing.holt_winters_multiplicative_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 0, "alpha": 0.5, "beta": 0.5, "gamma": 0.5},
        smoothing.SEASON_LENGTH_MIN_ERROR,
    ),
    (
        "hw multiplicative auto horizon",
        smoothing.holt_winters_multiplicative_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 0, "season_length": 2},
        smoothing.HORIZON_MIN_ERROR,
    ),
    (
        "hw multiplicative auto season length",
        smoothing.holt_winters_multiplicative_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 0},
        smoothing.SEASON_LENGTH_MIN_ERROR,
    ),
    (
        "hw multiplicative auto grid size",
        smoothing.holt_winters_multiplicative_auto_forecast,
        (POSITIVE_SERIES[:4],),
        {"horizon": 1, "season_length": 2, "grid_size": 1},
        smoothing.GRID_SIZE_MIN_ERROR,
    ),
    (
        "compute feature start target lags",
        regression._compute_feature_start_t,
        (),
        {"lags": (), "seasonal_lags": (), "seasonal_diff_lags": ()},
        regression.TARGET_LAGS_MIN_ERROR,
    ),
    (
        "make target feat row target lags",
        regression._make_target_feat_row,
        ([1.0, 2.0],),
        {"lags": ()},
        regression.TARGET_LAGS_MIN_ERROR,
    ),
    (
        "make lagged xy multi target lags",
        regression._make_lagged_xy_multi,
        (SERIES,),
        {"lags": (), "horizon": 1},
        regression.TARGET_LAGS_MIN_ERROR,
    ),
    (
        "multivariate lagged xy horizon",
        multivariate._make_lagged_xy_multivariate,
        (MULTIVARIATE_SERIES,),
        {"lags": 2, "horizon": 0},
        multivariate.HORIZON_MIN_ERROR,
    ),
    (
        "multivariate lagged xy lags",
        multivariate._make_lagged_xy_multivariate,
        (MULTIVARIATE_SERIES,),
        {"lags": 0, "horizon": 1},
        multivariate.LAGS_MIN_ERROR,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "fn", "args", "kwargs", "message"),
    PURE_VALIDATION_CASES,
    ids=[case[0] for case in PURE_VALIDATION_CASES],
)
def test_pure_model_validation_messages(case_id: str, fn, args, kwargs, message: str) -> None:
    _assert_value_error(fn, *args, message=message, **kwargs)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="requires statsmodels")
def test_var_forecast_uses_shared_horizon_validation_message() -> None:
    _assert_value_error(
        multivariate.var_forecast,
        MULTIVARIATE_SERIES,
        message=multivariate.HORIZON_MIN_ERROR,
        horizon=0,
        maxlags=1,
    )


TORCH_SMALL_VALIDATION_CASES = [
    (
        "stid d_model",
        multivariate.torch_stid_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        multivariate.D_MODEL_MIN_ERROR,
    ),
    (
        "stid num_blocks",
        multivariate.torch_stid_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "num_blocks": 0},
        multivariate.NUM_BLOCKS_MIN_ERROR,
    ),
    (
        "stid dropout",
        multivariate.torch_stid_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        multivariate.DROPOUT_RANGE_ERROR,
    ),
    (
        "stgcn horizon",
        multivariate.torch_stgcn_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 0, "lags": 2},
        multivariate.HORIZON_MIN_ERROR,
    ),
    (
        "stgcn lags",
        multivariate.torch_stgcn_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 0},
        multivariate.LAGS_MIN_ERROR,
    ),
    (
        "stgcn d_model",
        multivariate.torch_stgcn_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        multivariate.D_MODEL_MIN_ERROR,
    ),
    (
        "stgcn num_blocks",
        multivariate.torch_stgcn_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "num_blocks": 0},
        multivariate.NUM_BLOCKS_MIN_ERROR,
    ),
    (
        "stgcn dropout",
        multivariate.torch_stgcn_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        multivariate.DROPOUT_RANGE_ERROR,
    ),
    (
        "graphwavenet horizon",
        multivariate.torch_graphwavenet_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 0, "lags": 2},
        multivariate.HORIZON_MIN_ERROR,
    ),
    (
        "graphwavenet lags",
        multivariate.torch_graphwavenet_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 0},
        multivariate.LAGS_MIN_ERROR,
    ),
    (
        "graphwavenet d_model",
        multivariate.torch_graphwavenet_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        multivariate.D_MODEL_MIN_ERROR,
    ),
    (
        "graphwavenet num_blocks",
        multivariate.torch_graphwavenet_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "num_blocks": 0},
        multivariate.NUM_BLOCKS_MIN_ERROR,
    ),
    (
        "graphwavenet dropout",
        multivariate.torch_graphwavenet_forecast,
        {"train": MULTIVARIATE_SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        multivariate.DROPOUT_RANGE_ERROR,
    ),
    (
        "lmu horizon",
        torch_ct_rnn.torch_lmu_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "lmu lags",
        torch_ct_rnn.torch_lmu_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "lmu num_layers",
        torch_ct_rnn.torch_lmu_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "lmu dropout",
        torch_ct_rnn.torch_lmu_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "ltc horizon",
        torch_ct_rnn.torch_ltc_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "ltc lags",
        torch_ct_rnn.torch_ltc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "ltc hidden_size",
        torch_ct_rnn.torch_ltc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "hidden_size": 0},
        torch_ct_rnn.HIDDEN_SIZE_MIN_ERROR,
    ),
    (
        "ltc num_layers",
        torch_ct_rnn.torch_ltc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "ltc dropout",
        torch_ct_rnn.torch_ltc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "cfc horizon",
        torch_ct_rnn.torch_cfc_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "cfc lags",
        torch_ct_rnn.torch_cfc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "cfc hidden_size",
        torch_ct_rnn.torch_cfc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "hidden_size": 0},
        torch_ct_rnn.HIDDEN_SIZE_MIN_ERROR,
    ),
    (
        "cfc num_layers",
        torch_ct_rnn.torch_cfc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "cfc dropout",
        torch_ct_rnn.torch_cfc_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "xlstm horizon",
        torch_ct_rnn.torch_xlstm_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "xlstm lags",
        torch_ct_rnn.torch_xlstm_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "xlstm hidden_size",
        torch_ct_rnn.torch_xlstm_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "hidden_size": 0},
        torch_ct_rnn.HIDDEN_SIZE_MIN_ERROR,
    ),
    (
        "xlstm num_layers",
        torch_ct_rnn.torch_xlstm_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "xlstm dropout",
        torch_ct_rnn.torch_xlstm_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "griffin horizon",
        torch_ct_rnn.torch_griffin_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "griffin lags",
        torch_ct_rnn.torch_griffin_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "griffin hidden_size",
        torch_ct_rnn.torch_griffin_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "hidden_size": 0},
        torch_ct_rnn.HIDDEN_SIZE_MIN_ERROR,
    ),
    (
        "griffin num_layers",
        torch_ct_rnn.torch_griffin_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "griffin dropout",
        torch_ct_rnn.torch_griffin_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "hawk horizon",
        torch_ct_rnn.torch_hawk_direct_forecast,
        {"train": SERIES, "horizon": 0, "lags": 2},
        torch_ct_rnn.HORIZON_MIN_ERROR,
    ),
    (
        "hawk lags",
        torch_ct_rnn.torch_hawk_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 0},
        torch_ct_rnn.LAGS_MIN_ERROR,
    ),
    (
        "hawk hidden_size",
        torch_ct_rnn.torch_hawk_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "hidden_size": 0},
        torch_ct_rnn.HIDDEN_SIZE_MIN_ERROR,
    ),
    (
        "hawk num_layers",
        torch_ct_rnn.torch_hawk_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ct_rnn.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "hawk dropout",
        torch_ct_rnn.torch_hawk_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ct_rnn.DROPOUT_RANGE_ERROR,
    ),
    (
        "s4d d_model",
        torch_ssm.torch_s4d_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        torch_ssm.DMODEL_MIN_ERROR,
    ),
    (
        "s4d num_layers",
        torch_ssm.torch_s4d_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ssm.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "s4d dropout",
        torch_ssm.torch_s4d_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ssm.DROPOUT_RANGE_ERROR,
    ),
    (
        "mamba2 d_model",
        torch_ssm.torch_mamba2_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        torch_ssm.DMODEL_MIN_ERROR,
    ),
    (
        "mamba2 num_layers",
        torch_ssm.torch_mamba2_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ssm.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "mamba2 dropout",
        torch_ssm.torch_mamba2_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ssm.DROPOUT_RANGE_ERROR,
    ),
    (
        "s4 d_model",
        torch_ssm.torch_s4_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        torch_ssm.DMODEL_MIN_ERROR,
    ),
    (
        "s4 num_layers",
        torch_ssm.torch_s4_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ssm.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "s4 dropout",
        torch_ssm.torch_s4_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ssm.DROPOUT_RANGE_ERROR,
    ),
    (
        "s5 d_model",
        torch_ssm.torch_s5_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "d_model": 0},
        torch_ssm.DMODEL_MIN_ERROR,
    ),
    (
        "s5 num_layers",
        torch_ssm.torch_s5_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "num_layers": 0},
        torch_ssm.NUM_LAYERS_MIN_ERROR,
    ),
    (
        "s5 dropout",
        torch_ssm.torch_s5_direct_forecast,
        {"train": SERIES, "horizon": 1, "lags": 2, "dropout": 1.0},
        torch_ssm.DROPOUT_RANGE_ERROR,
    ),
]


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize(
    ("case_id", "fn", "kwargs", "message"),
    TORCH_SMALL_VALIDATION_CASES,
    ids=[case[0] for case in TORCH_SMALL_VALIDATION_CASES],
)
def test_torch_model_validation_messages(case_id: str, fn, kwargs, message: str) -> None:
    _assert_value_error(fn, message=message, **kwargs)
