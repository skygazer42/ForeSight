import importlib.util

import numpy as np
import pytest

import foresight.models.registry as registry_mod
from foresight.models.registry import get_model_spec, make_forecaster


def test_arima_model_is_registered_as_stats_optional():
    spec = get_model_spec("arima")
    assert "stats" in spec.requires


def test_ets_model_is_registered_as_stats_optional():
    spec = get_model_spec("ets")
    assert "stats" in spec.requires


def test_arima_raises_importerror_when_statsmodels_missing():
    if importlib.util.find_spec("statsmodels") is not None:
        pytest.skip("statsmodels installed; this test targets the missing-dep path")

    f = make_forecaster("arima", order=(1, 0, 0))
    with pytest.raises(ImportError):
        f([1.0, 2.0, 3.0, 4.0, 5.0], 2)


def test_ets_raises_importerror_when_statsmodels_missing():
    if importlib.util.find_spec("statsmodels") is not None:
        pytest.skip("statsmodels installed; this test targets the missing-dep path")

    f = make_forecaster("ets", season_length=12, trend="add", seasonal="add")
    with pytest.raises(ImportError):
        f([1.0, 2.0, 3.0, 4.0, 5.0], 2)


def test_arima_spec_exposes_trend_and_enforcement_params():
    spec = get_model_spec("arima")

    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_arima_supports_trend_when_statsmodels_installed():
    if importlib.util.find_spec("statsmodels") is None:
        pytest.skip("statsmodels not installed; smoke test requires it")

    y = np.arange(1.0, 31.0, dtype=float)

    f = make_forecaster("arima", order=(0, 1, 0), trend="t")
    yhat = f(y, 3)

    assert yhat.shape == (3,)
    assert np.allclose(yhat, np.array([31.0, 32.0, 33.0]), atol=1e-3)


def test_arima_factory_normalizes_none_and_bool_like_strings(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_arima_forecast(
        train,
        horizon,
        *,
        order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["order"] = order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "arima_forecast", _fake_arima_forecast)

    yhat = make_forecaster(
        "arima",
        order=(0, 1, 0),
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
    )([1.0, 2.0, 3.0, 4.0], 2)

    assert yhat.shape == (2,)
    assert captured["order"] == (0, 1, 0)
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False


def test_auto_arima_spec_exposes_trend_and_seasonal_search_params():
    spec = get_model_spec("auto-arima")

    assert spec.default_params["trend"] is None
    assert spec.default_params["max_P"] == 0
    assert spec.default_params["max_D"] == 0
    assert spec.default_params["max_Q"] == 0
    assert spec.default_params["seasonal_period"] is None
    assert spec.default_params["x_cols"] == ()
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "trend" in spec.param_help
    assert "max_P" in spec.param_help
    assert "max_D" in spec.param_help
    assert "max_Q" in spec.param_help
    assert "seasonal_period" in spec.param_help
    assert "x_cols" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_auto_arima_factory_normalizes_trend_and_seasonal_search_params(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def _fake_auto_arima_forecast(
        train,
        horizon,
        *,
        max_p,
        max_d,
        max_q,
        seasonal_period,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        information_criterion,
        **kwargs,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["max_p"] = max_p
        captured["max_d"] = max_d
        captured["max_q"] = max_q
        captured["max_P"] = kwargs["max_P"]
        captured["max_D"] = kwargs["max_D"]
        captured["max_Q"] = kwargs["max_Q"]
        captured["seasonal_period"] = seasonal_period
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["information_criterion"] = information_criterion
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "auto_arima_forecast", _fake_auto_arima_forecast)

    yhat = make_forecaster(
        "auto-arima",
        max_p="1",
        max_d="1",
        max_q="1",
        max_P="2",
        max_D="1",
        max_Q="2",
        seasonal_period="12",
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        information_criterion="bic",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["max_p"] == 1
    assert captured["max_d"] == 1
    assert captured["max_q"] == 1
    assert captured["max_P"] == 2
    assert captured["max_D"] == 1
    assert captured["max_Q"] == 2
    assert captured["seasonal_period"] == 12
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["information_criterion"] == "bic"


def test_mstl_auto_arima_spec_exposes_trend_and_enforcement_params():
    spec = get_model_spec("mstl-auto-arima")

    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_mstl_auto_arima_factory_normalizes_trend_and_bool_like_strings(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def _fake_mstl_auto_arima_forecast(
        train,
        horizon,
        *,
        periods,
        iterate,
        lmbda,
        max_p,
        max_d,
        max_q,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        information_criterion,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["iterate"] = iterate
        captured["lmbda"] = lmbda
        captured["max_p"] = max_p
        captured["max_d"] = max_d
        captured["max_q"] = max_q
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["information_criterion"] = information_criterion
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "mstl_auto_arima_forecast", _fake_mstl_auto_arima_forecast)

    yhat = make_forecaster(
        "mstl-auto-arima",
        periods=(12,),
        iterate="3",
        lmbda="auto",
        max_p="1",
        max_d="1",
        max_q="1",
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        information_criterion="bic",
    )([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (12,)
    assert captured["iterate"] == 3
    assert captured["lmbda"] == "auto"
    assert captured["max_p"] == 1
    assert captured["max_d"] == 1
    assert captured["max_q"] == 1
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["information_criterion"] == "bic"


def test_mstl_autoreg_spec_exposes_periods_and_autoreg_params():
    spec = get_model_spec("mstl-autoreg")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["lags"] == 1
    assert spec.default_params["trend"] == "c"
    assert spec.default_params["iterate"] == 2
    assert spec.default_params["lmbda"] is None
    assert "periods" in spec.param_help
    assert "lags" in spec.param_help
    assert "trend" in spec.param_help
    assert "iterate" in spec.param_help
    assert "lmbda" in spec.param_help


def test_mstl_autoreg_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_mstl_autoreg_forecast(
        train,
        horizon,
        *,
        periods,
        lags,
        trend,
        iterate,
        lmbda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["lags"] = lags
        captured["trend"] = trend
        captured["iterate"] = iterate
        captured["lmbda"] = lmbda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "mstl_autoreg_forecast", _fake_mstl_autoreg_forecast)

    yhat = make_forecaster(
        "mstl-autoreg",
        periods=(12, 30),
        lags="0",
        trend="ct",
        iterate="3",
        lmbda="auto",
    )([1.0] * 60, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (12, 30)
    assert captured["lags"] == 0
    assert captured["trend"] == "ct"
    assert captured["iterate"] == 3
    assert captured["lmbda"] == "auto"


def test_mstl_ets_spec_exposes_periods_and_ets_params():
    spec = get_model_spec("mstl-ets")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["trend"] == "add"
    assert spec.default_params["damped_trend"] is False
    assert spec.default_params["iterate"] == 2
    assert spec.default_params["lmbda"] is None
    assert "periods" in spec.param_help
    assert "trend" in spec.param_help
    assert "damped_trend" in spec.param_help
    assert "iterate" in spec.param_help
    assert "lmbda" in spec.param_help


def test_mstl_ets_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_mstl_ets_forecast(
        train,
        horizon,
        *,
        periods,
        trend,
        damped_trend,
        iterate,
        lmbda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["trend"] = trend
        captured["damped_trend"] = damped_trend
        captured["iterate"] = iterate
        captured["lmbda"] = lmbda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "mstl_ets_forecast", _fake_mstl_ets_forecast)

    yhat = make_forecaster(
        "mstl-ets",
        periods=(12, 30),
        trend="add",
        damped_trend="true",
        iterate="3",
        lmbda="auto",
    )([1.0] * 60, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (12, 30)
    assert captured["trend"] == "add"
    assert captured["damped_trend"] is True
    assert captured["iterate"] == 3
    assert captured["lmbda"] == "auto"


def test_mstl_uc_spec_exposes_mstl_and_uc_params():
    spec = get_model_spec("mstl-uc")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["level"] == "local level"
    assert spec.default_params["iterate"] == 2
    assert spec.default_params["lmbda"] is None
    assert "periods" in spec.param_help
    assert "level" in spec.param_help
    assert "iterate" in spec.param_help
    assert "lmbda" in spec.param_help


def test_mstl_uc_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_mstl_uc_forecast(
        train,
        horizon,
        *,
        periods,
        level,
        iterate,
        lmbda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["level"] = level
        captured["iterate"] = iterate
        captured["lmbda"] = lmbda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "mstl_uc_forecast", _fake_mstl_uc_forecast)

    yhat = make_forecaster(
        "mstl-uc",
        periods=(12, 30),
        level="local linear trend",
        iterate="3",
        lmbda="auto",
    )([1.0] * 60, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (12, 30)
    assert captured["level"] == "local linear trend"
    assert captured["iterate"] == 3
    assert captured["lmbda"] == "auto"


def test_mstl_sarimax_spec_exposes_mstl_and_sarimax_params():
    spec = get_model_spec("mstl-sarimax")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["order"] == (1, 0, 0)
    assert spec.default_params["seasonal_order"] == (0, 0, 0, 0)
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert spec.default_params["iterate"] == 2
    assert spec.default_params["lmbda"] is None
    assert "periods" in spec.param_help
    assert "order" in spec.param_help
    assert "seasonal_order" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help
    assert "iterate" in spec.param_help
    assert "lmbda" in spec.param_help


def test_mstl_sarimax_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_mstl_sarimax_forecast(
        train,
        horizon,
        *,
        periods,
        order,
        seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        iterate,
        lmbda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["order"] = order
        captured["seasonal_order"] = seasonal_order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["iterate"] = iterate
        captured["lmbda"] = lmbda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "mstl_sarimax_forecast", _fake_mstl_sarimax_forecast)

    yhat = make_forecaster(
        "mstl-sarimax",
        periods=(12, 30),
        order=("1", "0", "0"),
        seasonal_order=("0", "0", "0", "0"),
        trend="c",
        enforce_stationarity="false",
        enforce_invertibility="false",
        iterate="3",
        lmbda="auto",
    )([1.0] * 60, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (12, 30)
    assert captured["order"] == (1, 0, 0)
    assert captured["seasonal_order"] == (0, 0, 0, 0)
    assert captured["trend"] == "c"
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["iterate"] == 3
    assert captured["lmbda"] == "auto"


def test_tbats_lite_autoreg_spec_exposes_fourier_and_autoreg_params():
    spec = get_model_spec("tbats-lite-autoreg")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["include_trend"] is True
    assert spec.default_params["lags"] == 1
    assert spec.default_params["trend"] == "n"
    assert spec.default_params["boxcox_lambda"] is None
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "include_trend" in spec.param_help
    assert "lags" in spec.param_help
    assert "trend" in spec.param_help
    assert "boxcox_lambda" in spec.param_help


def test_tbats_lite_autoreg_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_tbats_lite_autoreg_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        include_trend,
        lags,
        trend,
        boxcox_lambda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["include_trend"] = include_trend
        captured["lags"] = lags
        captured["trend"] = trend
        captured["boxcox_lambda"] = boxcox_lambda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod,
        "tbats_lite_autoreg_forecast",
        _fake_tbats_lite_autoreg_forecast,
    )

    yhat = make_forecaster(
        "tbats-lite-autoreg",
        periods=(7, 30),
        orders=(2, 1),
        include_trend="true",
        lags="0",
        trend="n",
        boxcox_lambda="0.5",
    )([1.0] * 40, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["include_trend"] is True
    assert captured["lags"] == 0
    assert captured["trend"] == "n"
    assert captured["boxcox_lambda"] == pytest.approx(0.5)


def test_tbats_lite_auto_arima_spec_exposes_fourier_and_auto_arima_params():
    spec = get_model_spec("tbats-lite-auto-arima")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["include_trend"] is True
    assert spec.default_params["max_p"] == 3
    assert spec.default_params["max_d"] == 2
    assert spec.default_params["max_q"] == 3
    assert spec.default_params["trend"] == "c"
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert spec.default_params["information_criterion"] == "aic"
    assert spec.default_params["boxcox_lambda"] is None
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "include_trend" in spec.param_help
    assert "max_p" in spec.param_help
    assert "max_d" in spec.param_help
    assert "max_q" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help
    assert "information_criterion" in spec.param_help
    assert "boxcox_lambda" in spec.param_help


def test_tbats_lite_auto_arima_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_tbats_lite_auto_arima_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        include_trend,
        max_p,
        max_d,
        max_q,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        information_criterion,
        boxcox_lambda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["include_trend"] = include_trend
        captured["max_p"] = max_p
        captured["max_d"] = max_d
        captured["max_q"] = max_q
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["information_criterion"] = information_criterion
        captured["boxcox_lambda"] = boxcox_lambda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod,
        "tbats_lite_auto_arima_forecast",
        _fake_tbats_lite_auto_arima_forecast,
    )

    yhat = make_forecaster(
        "tbats-lite-auto-arima",
        periods=(7, 30),
        orders=(2, 1),
        include_trend="true",
        max_p="0",
        max_d="0",
        max_q="0",
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        information_criterion="bic",
        boxcox_lambda="0.5",
    )([1.0] * 40, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["include_trend"] is True
    assert captured["max_p"] == 0
    assert captured["max_d"] == 0
    assert captured["max_q"] == 0
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["information_criterion"] == "bic"
    assert captured["boxcox_lambda"] == pytest.approx(0.5)


def test_tbats_lite_uc_spec_exposes_fourier_and_uc_params():
    spec = get_model_spec("tbats-lite-uc")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["include_trend"] is True
    assert spec.default_params["level"] == "local level"
    assert spec.default_params["boxcox_lambda"] is None
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "include_trend" in spec.param_help
    assert "level" in spec.param_help
    assert "boxcox_lambda" in spec.param_help


def test_tbats_lite_uc_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_tbats_lite_uc_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        include_trend,
        level,
        boxcox_lambda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["include_trend"] = include_trend
        captured["level"] = level
        captured["boxcox_lambda"] = boxcox_lambda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod,
        "tbats_lite_uc_forecast",
        _fake_tbats_lite_uc_forecast,
    )

    yhat = make_forecaster(
        "tbats-lite-uc",
        periods=(7, 30),
        orders=(2, 1),
        include_trend="true",
        level="local linear trend",
        boxcox_lambda="0.5",
    )([1.0] * 40, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["include_trend"] is True
    assert captured["level"] == "local linear trend"
    assert captured["boxcox_lambda"] == pytest.approx(0.5)


def test_tbats_lite_ets_spec_exposes_fourier_and_ets_params():
    spec = get_model_spec("tbats-lite-ets")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["include_trend"] is True
    assert spec.default_params["trend"] is None
    assert spec.default_params["damped_trend"] is False
    assert spec.default_params["boxcox_lambda"] is None
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "include_trend" in spec.param_help
    assert "trend" in spec.param_help
    assert "damped_trend" in spec.param_help
    assert "boxcox_lambda" in spec.param_help


def test_tbats_lite_ets_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_tbats_lite_ets_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        include_trend,
        trend,
        damped_trend,
        boxcox_lambda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["include_trend"] = include_trend
        captured["trend"] = trend
        captured["damped_trend"] = damped_trend
        captured["boxcox_lambda"] = boxcox_lambda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod,
        "tbats_lite_ets_forecast",
        _fake_tbats_lite_ets_forecast,
    )

    yhat = make_forecaster(
        "tbats-lite-ets",
        periods=(7, 30),
        orders=(2, 1),
        include_trend="true",
        trend="none",
        damped_trend="true",
        boxcox_lambda="0.5",
    )([1.0] * 40, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["include_trend"] is True
    assert captured["trend"] is None
    assert captured["damped_trend"] is True
    assert captured["boxcox_lambda"] == pytest.approx(0.5)


def test_tbats_lite_sarimax_spec_exposes_fourier_and_sarimax_params():
    spec = get_model_spec("tbats-lite-sarimax")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["include_trend"] is True
    assert spec.default_params["order"] == (1, 0, 0)
    assert spec.default_params["seasonal_order"] == (0, 0, 0, 0)
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert spec.default_params["boxcox_lambda"] is None
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "include_trend" in spec.param_help
    assert "order" in spec.param_help
    assert "seasonal_order" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help
    assert "boxcox_lambda" in spec.param_help


def test_tbats_lite_sarimax_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_tbats_lite_sarimax_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        include_trend,
        order,
        seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        boxcox_lambda,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["include_trend"] = include_trend
        captured["order"] = order
        captured["seasonal_order"] = seasonal_order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["boxcox_lambda"] = boxcox_lambda
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod,
        "tbats_lite_sarimax_forecast",
        _fake_tbats_lite_sarimax_forecast,
    )

    yhat = make_forecaster(
        "tbats-lite-sarimax",
        periods=(7, 30),
        orders=(2, 1),
        include_trend="true",
        order=("1", "0", "0"),
        seasonal_order=("0", "0", "0", "0"),
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        boxcox_lambda="0.5",
    )([1.0] * 40, 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["include_trend"] is True
    assert captured["order"] == (1, 0, 0)
    assert captured["seasonal_order"] == (0, 0, 0, 0)
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["boxcox_lambda"] == pytest.approx(0.5)


def test_fourier_auto_arima_spec_exposes_periods_orders_and_arima_params():
    spec = get_model_spec("fourier-auto-arima")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["max_p"] == 3
    assert spec.default_params["max_d"] == 2
    assert spec.default_params["max_q"] == 3
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "max_p" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_fourier_auto_arima_factory_normalizes_trend_and_bool_like_strings(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def _fake_fourier_auto_arima_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        max_p,
        max_d,
        max_q,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        information_criterion,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["max_p"] = max_p
        captured["max_d"] = max_d
        captured["max_q"] = max_q
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["information_criterion"] = information_criterion
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod, "fourier_auto_arima_forecast", _fake_fourier_auto_arima_forecast
    )

    yhat = make_forecaster(
        "fourier-auto-arima",
        periods=(7, 30),
        orders=(2, 1),
        max_p="1",
        max_d="0",
        max_q="1",
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        information_criterion="bic",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["max_p"] == 1
    assert captured["max_d"] == 0
    assert captured["max_q"] == 1
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["information_criterion"] == "bic"


def test_fourier_arima_spec_exposes_periods_orders_and_arima_params():
    spec = get_model_spec("fourier-arima")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["order"] == (1, 0, 0)
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "order" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_fourier_arima_factory_normalizes_trend_and_bool_like_strings(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def _fake_fourier_arima_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["order"] = order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "fourier_arima_forecast", _fake_fourier_arima_forecast)

    yhat = make_forecaster(
        "fourier-arima",
        periods=(7, 30),
        orders=(2, 1),
        order=(0, 0, 0),
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["order"] == (0, 0, 0)
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False


def test_fourier_sarimax_spec_exposes_periods_orders_and_sarimax_params():
    spec = get_model_spec("fourier-sarimax")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["order"] == (1, 0, 0)
    assert spec.default_params["seasonal_order"] == (0, 0, 0, 0)
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "order" in spec.param_help
    assert "seasonal_order" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help


def test_fourier_sarimax_factory_normalizes_trend_and_bool_like_strings(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def _fake_fourier_sarimax_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        order,
        seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["order"] = order
        captured["seasonal_order"] = seasonal_order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "fourier_sarimax_forecast", _fake_fourier_sarimax_forecast)

    yhat = make_forecaster(
        "fourier-sarimax",
        periods=(7, 30),
        orders=(2, 1),
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["order"] == (0, 0, 0)
    assert captured["seasonal_order"] == (0, 0, 0, 0)
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False


def test_fourier_ets_spec_exposes_periods_orders_and_ets_params():
    spec = get_model_spec("fourier-ets")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["trend"] is None
    assert spec.default_params["damped_trend"] is False
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "trend" in spec.param_help
    assert "damped_trend" in spec.param_help


def test_fourier_ets_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_fourier_ets_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        trend,
        damped_trend,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["trend"] = trend
        captured["damped_trend"] = damped_trend
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "fourier_ets_forecast", _fake_fourier_ets_forecast)

    yhat = make_forecaster(
        "fourier-ets",
        periods=(7, 30),
        orders=(2, 1),
        trend="none",
        damped_trend="true",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["trend"] is None
    assert captured["damped_trend"] is True


def test_fourier_uc_spec_exposes_periods_orders_and_uc_params():
    spec = get_model_spec("fourier-uc")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["level"] == "local level"
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "level" in spec.param_help


def test_fourier_uc_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_fourier_uc_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        level,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["level"] = level
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "fourier_uc_forecast", _fake_fourier_uc_forecast)

    yhat = make_forecaster(
        "fourier-uc",
        periods=(7, 30),
        orders=(2, 1),
        level="local linear trend",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["level"] == "local linear trend"


def test_uc_seasonal_spec_exposes_level_and_seasonal_params():
    spec = get_model_spec("uc-seasonal")

    assert spec.default_params["level"] == "local level"
    assert spec.default_params["seasonal"] == 12
    assert "level" in spec.param_help
    assert "seasonal" in spec.param_help


def test_uc_seasonal_factory_normalizes_seasonal_int(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_unobserved_components_forecast(
        train,
        horizon,
        *,
        level,
        seasonal,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["level"] = level
        captured["seasonal"] = seasonal
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        registry_mod, "unobserved_components_forecast", _fake_unobserved_components_forecast
    )

    yhat = make_forecaster(
        "uc-seasonal",
        level="local linear trend",
        seasonal="7",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["level"] == "local linear trend"
    assert captured["seasonal"] == 7


def test_fourier_autoreg_spec_exposes_periods_orders_and_autoreg_params():
    spec = get_model_spec("fourier-autoreg")

    assert spec.default_params["periods"] == (12,)
    assert spec.default_params["orders"] == 2
    assert spec.default_params["lags"] == 0
    assert spec.default_params["trend"] == "c"
    assert "periods" in spec.param_help
    assert "orders" in spec.param_help
    assert "lags" in spec.param_help
    assert "trend" in spec.param_help


def test_fourier_autoreg_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_fourier_autoreg_forecast(
        train,
        horizon,
        *,
        periods,
        orders,
        lags,
        trend,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["periods"] = periods
        captured["orders"] = orders
        captured["lags"] = lags
        captured["trend"] = trend
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "fourier_autoreg_forecast", _fake_fourier_autoreg_forecast)

    yhat = make_forecaster(
        "fourier-autoreg",
        periods=(7, 30),
        orders=(2, 1),
        lags="0",
        trend="ct",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["periods"] == (7, 30)
    assert captured["orders"] == (2, 1)
    assert captured["lags"] == 0
    assert captured["trend"] == "ct"


def test_stl_ets_spec_exposes_period_and_ets_params():
    spec = get_model_spec("stl-ets")

    assert spec.default_params["period"] == 12
    assert spec.default_params["trend"] == "add"
    assert spec.default_params["damped_trend"] is False
    assert spec.default_params["robust"] is False
    assert "period" in spec.param_help
    assert "trend" in spec.param_help
    assert "damped_trend" in spec.param_help
    assert "robust" in spec.param_help


def test_stl_ets_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_stl_ets_forecast(
        train,
        horizon,
        *,
        period,
        trend,
        damped_trend,
        robust,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["period"] = period
        captured["trend"] = trend
        captured["damped_trend"] = damped_trend
        captured["robust"] = robust
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "stl_ets_forecast", _fake_stl_ets_forecast)

    yhat = make_forecaster(
        "stl-ets",
        period="12",
        trend="add",
        damped_trend="true",
        robust="true",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["period"] == 12
    assert captured["trend"] == "add"
    assert captured["damped_trend"] is True
    assert captured["robust"] is True


def test_stl_autoreg_spec_exposes_period_lags_and_trend_params():
    spec = get_model_spec("stl-autoreg")

    assert spec.default_params["period"] == 12
    assert spec.default_params["lags"] == 1
    assert spec.default_params["trend"] == "c"
    assert spec.default_params["seasonal"] == 7
    assert spec.default_params["robust"] is False
    assert "period" in spec.param_help
    assert "lags" in spec.param_help
    assert "trend" in spec.param_help
    assert "seasonal" in spec.param_help
    assert "robust" in spec.param_help


def test_stl_autoreg_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_stl_autoreg_forecast(
        train,
        horizon,
        *,
        period,
        lags,
        trend,
        seasonal,
        robust,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["period"] = period
        captured["lags"] = lags
        captured["trend"] = trend
        captured["seasonal"] = seasonal
        captured["robust"] = robust
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "stl_autoreg_forecast", _fake_stl_autoreg_forecast)

    yhat = make_forecaster(
        "stl-autoreg",
        period="12",
        lags="1",
        trend="ct",
        seasonal="9",
        robust="true",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["period"] == 12
    assert captured["lags"] == 1
    assert captured["trend"] == "ct"
    assert captured["seasonal"] == 9
    assert captured["robust"] is True


def test_stl_uc_spec_exposes_stl_and_uc_params():
    spec = get_model_spec("stl-uc")

    assert spec.default_params["period"] == 12
    assert spec.default_params["level"] == "local level"
    assert spec.default_params["seasonal"] == 7
    assert spec.default_params["robust"] is False
    assert "period" in spec.param_help
    assert "level" in spec.param_help
    assert "seasonal" in spec.param_help
    assert "robust" in spec.param_help


def test_stl_uc_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_stl_uc_forecast(
        train,
        horizon,
        *,
        period,
        level,
        seasonal,
        robust,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["period"] = period
        captured["level"] = level
        captured["seasonal"] = seasonal
        captured["robust"] = robust
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "stl_uc_forecast", _fake_stl_uc_forecast)

    yhat = make_forecaster(
        "stl-uc",
        period="12",
        level="local linear trend",
        seasonal="9",
        robust="true",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["period"] == 12
    assert captured["level"] == "local linear trend"
    assert captured["seasonal"] == 9
    assert captured["robust"] is True


def test_stl_auto_arima_spec_exposes_stl_and_auto_arima_params():
    spec = get_model_spec("stl-auto-arima")

    assert spec.default_params["period"] == 12
    assert spec.default_params["seasonal"] == 7
    assert spec.default_params["robust"] is False
    assert spec.default_params["max_p"] == 3
    assert spec.default_params["max_d"] == 2
    assert spec.default_params["max_q"] == 3
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert spec.default_params["information_criterion"] == "aic"
    assert "period" in spec.param_help
    assert "seasonal" in spec.param_help
    assert "robust" in spec.param_help
    assert "max_p" in spec.param_help
    assert "max_d" in spec.param_help
    assert "max_q" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help
    assert "information_criterion" in spec.param_help


def test_stl_auto_arima_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_stl_auto_arima_forecast(
        train,
        horizon,
        *,
        period,
        seasonal,
        robust,
        max_p,
        max_d,
        max_q,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        information_criterion,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["period"] = period
        captured["seasonal"] = seasonal
        captured["robust"] = robust
        captured["max_p"] = max_p
        captured["max_d"] = max_d
        captured["max_q"] = max_q
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["information_criterion"] = information_criterion
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "stl_auto_arima_forecast", _fake_stl_auto_arima_forecast)

    yhat = make_forecaster(
        "stl-auto-arima",
        period="12",
        seasonal="9",
        robust="true",
        max_p="1",
        max_d="1",
        max_q="1",
        trend="none",
        enforce_stationarity="false",
        enforce_invertibility="false",
        information_criterion="bic",
    )([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)

    assert yhat.shape == (2,)
    assert captured["period"] == 12
    assert captured["seasonal"] == 9
    assert captured["robust"] is True
    assert captured["max_p"] == 1
    assert captured["max_d"] == 1
    assert captured["max_q"] == 1
    assert captured["trend"] is None
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["information_criterion"] == "bic"


def test_stl_sarimax_spec_exposes_stl_and_sarimax_params():
    spec = get_model_spec("stl-sarimax")

    assert spec.default_params["period"] == 12
    assert spec.default_params["order"] == (1, 0, 0)
    assert spec.default_params["seasonal_order"] == (0, 0, 0, 0)
    assert spec.default_params["trend"] is None
    assert spec.default_params["enforce_stationarity"] is True
    assert spec.default_params["enforce_invertibility"] is True
    assert spec.default_params["seasonal"] == 7
    assert spec.default_params["robust"] is False
    assert "period" in spec.param_help
    assert "order" in spec.param_help
    assert "seasonal_order" in spec.param_help
    assert "trend" in spec.param_help
    assert "enforce_stationarity" in spec.param_help
    assert "enforce_invertibility" in spec.param_help
    assert "seasonal" in spec.param_help
    assert "robust" in spec.param_help


def test_stl_sarimax_factory_normalizes_core_params(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_stl_sarimax_forecast(
        train,
        horizon,
        *,
        period,
        order,
        seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        seasonal,
        robust,
    ):
        captured["train"] = list(train)
        captured["horizon"] = horizon
        captured["period"] = period
        captured["order"] = order
        captured["seasonal_order"] = seasonal_order
        captured["trend"] = trend
        captured["enforce_stationarity"] = enforce_stationarity
        captured["enforce_invertibility"] = enforce_invertibility
        captured["seasonal"] = seasonal
        captured["robust"] = robust
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(registry_mod, "stl_sarimax_forecast", _fake_stl_sarimax_forecast)

    yhat = make_forecaster(
        "stl-sarimax",
        period="12",
        order=("1", "0", "0"),
        seasonal_order=("0", "0", "0", "0"),
        trend="c",
        enforce_stationarity="false",
        enforce_invertibility="false",
        seasonal="9",
        robust="true",
    )([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert yhat.shape == (2,)
    assert captured["period"] == 12
    assert captured["order"] == (1, 0, 0)
    assert captured["seasonal_order"] == (0, 0, 0, 0)
    assert captured["trend"] == "c"
    assert captured["enforce_stationarity"] is False
    assert captured["enforce_invertibility"] is False
    assert captured["seasonal"] == 9
    assert captured["robust"] is True
