import numpy as np
import pytest

from foresight import BaseForecaster, make_forecaster_object
from foresight.models import runtime as runtime_mod
from foresight.models.registry import make_forecaster


def test_local_forecaster_object_requires_fit_before_predict() -> None:
    f = make_forecaster_object("naive-last")
    assert isinstance(f, BaseForecaster)

    with pytest.raises(RuntimeError):
        f.predict(2)


def test_local_forecaster_object_supports_fit_then_predict() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    f = make_forecaster_object("moving-average", window=3)
    assert f.fit(y) is f
    assert f.model_key == "moving-average"
    assert f.model_params["window"] == 3

    yhat = f.predict(2)
    expected = make_forecaster("moving-average", window=3)(y, 2)

    assert yhat.shape == (2,)
    assert np.allclose(yhat, expected)


def test_local_forecaster_object_preserves_registry_defaults() -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)

    f = make_forecaster_object("naive-last")
    f.fit(y)

    yhat = f.predict(3)
    expected = make_forecaster("naive-last")(y, 3)

    assert np.allclose(yhat, expected)


def test_make_forecaster_sar_ols_accepts_legacy_uppercase_p_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    def _fake_sar_ols_forecast(
        train: object,
        horizon: int,
        *,
        p: int,
        season_length: int,
        **kwargs: object,
    ) -> np.ndarray:
        captured["p"] = p
        captured["P"] = int(kwargs["P"])
        captured["season_length"] = season_length
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(runtime_mod, "sar_ols_forecast", _fake_sar_ols_forecast)

    yhat = make_forecaster("sar-ols", p=0, P="2", season_length="7")(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        2,
    )

    assert yhat.shape == (2,)
    assert captured == {"p": 0, "P": 2, "season_length": 7}


def test_make_forecaster_svr_variants_accept_legacy_uppercase_c_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, float]] = {}

    def _fake_svr_lag_direct_forecast(
        train: object,
        horizon: int,
        *,
        lags: int,
        gamma: object,
        epsilon: float,
        **kwargs: object,
    ) -> np.ndarray:
        captured["svr-lag"] = {
            "lags": float(lags),
            "C": float(kwargs["C"]),
            "epsilon": float(epsilon),
        }
        return np.zeros(int(horizon), dtype=float)

    def _fake_linear_svr_lag_direct_forecast(
        train: object,
        horizon: int,
        *,
        lags: int,
        epsilon: float,
        max_iter: int,
        random_state: int,
        **kwargs: object,
    ) -> np.ndarray:
        captured["linear-svr-lag"] = {
            "lags": float(lags),
            "C": float(kwargs["C"]),
            "epsilon": float(epsilon),
            "max_iter": float(max_iter),
            "random_state": float(random_state),
        }
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(runtime_mod, "svr_lag_direct_forecast", _fake_svr_lag_direct_forecast)
    monkeypatch.setattr(
        runtime_mod,
        "linear_svr_lag_direct_forecast",
        _fake_linear_svr_lag_direct_forecast,
    )

    svr_yhat = make_forecaster("svr-lag", lags=3, C="1.5", epsilon="0.2")([1.0] * 12, 2)
    linear_yhat = make_forecaster(
        "linear-svr-lag",
        lags=4,
        C="2.5",
        epsilon="0.1",
        max_iter="123",
        random_state="9",
    )([1.0] * 12, 2)

    assert svr_yhat.shape == (2,)
    assert linear_yhat.shape == (2,)
    assert captured["svr-lag"] == {"lags": 3.0, "C": 1.5, "epsilon": 0.2}
    assert captured["linear-svr-lag"] == {
        "lags": 4.0,
        "C": 2.5,
        "epsilon": 0.1,
        "max_iter": 123.0,
        "random_state": 9.0,
    }


def test_make_forecaster_passive_aggressive_accepts_legacy_uppercase_c_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def _fake_passive_aggressive_lag_direct_forecast(
        train: object,
        horizon: int,
        *,
        lags: int,
        epsilon: float,
        max_iter: int,
        random_state: int,
        **kwargs: object,
    ) -> np.ndarray:
        captured["lags"] = float(lags)
        captured["C"] = float(kwargs["C"])
        captured["epsilon"] = float(epsilon)
        captured["max_iter"] = float(max_iter)
        captured["random_state"] = float(random_state)
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(
        runtime_mod,
        "passive_aggressive_lag_direct_forecast",
        _fake_passive_aggressive_lag_direct_forecast,
    )

    yhat = make_forecaster(
        "passive-aggressive-lag",
        lags=4,
        C="2.25",
        epsilon="0.3",
        max_iter="321",
        random_state="7",
    )([1.0] * 12, 2)

    assert yhat.shape == (2,)
    assert captured == {
        "lags": 4.0,
        "C": 2.25,
        "epsilon": 0.3,
        "max_iter": 321.0,
        "random_state": 7.0,
    }


def test_make_forecaster_tweedie_lag_coerces_string_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def _fake_tweedie_lag_direct_forecast(
        train: object,
        horizon: int,
        *,
        lags: int,
        power: float,
        alpha: float,
        max_iter: int,
        **kwargs: object,
    ) -> np.ndarray:
        captured["lags"] = float(lags)
        captured["power"] = float(power)
        captured["alpha"] = float(alpha)
        captured["max_iter"] = float(max_iter)
        captured["seasonal_orders"] = float(kwargs["fourier_orders"])
        return np.zeros(int(horizon), dtype=float)

    monkeypatch.setattr(runtime_mod, "tweedie_lag_direct_forecast", _fake_tweedie_lag_direct_forecast)

    yhat = make_forecaster(
        "tweedie-lag",
        lags="4",
        power="1.5",
        alpha="0.25",
        max_iter="123",
        fourier_orders="3",
    )([1.0] * 12, 2)

    assert yhat.shape == (2,)
    assert captured == {
        "lags": 4.0,
        "power": 1.5,
        "alpha": 0.25,
        "max_iter": 123.0,
        "seasonal_orders": 3.0,
    }
