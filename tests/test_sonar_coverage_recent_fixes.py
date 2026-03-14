from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from foresight.cli import build_parser
from foresight.models import global_regression as global_regression_mod
from foresight.models import regression as regression_mod
from foresight.models.global_regression import (
    decision_tree_step_lag_global_forecaster,
    rf_step_lag_global_forecaster,
    svr_step_lag_global_forecaster,
)
from foresight.models.regression import (
    decision_tree_lag_direct_forecast,
    rf_lag_direct_forecast,
    svr_lag_direct_forecast,
)
from foresight.docsgen.rnn import _metadata_primary_url, render_rnn_paper_zoo_doc, render_rnn_zoo_doc
from foresight.models.regression import (
    _augment_lag_feat_row,
    _lgbm_validate_common_regressor_params,
    _xgb_lag_direct_forecast,
    _xgb_lag_recursive_forecast,
    _xgb_validate_common_regressor_params,
)
from foresight.models.statsmodels_wrap import ets_forecast
from foresight.models.torch_rnn_paper_zoo import torch_rnnpaper_direct_forecast


def _install_fake_statsmodels(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]
) -> None:
    statsmodels = types.ModuleType("statsmodels")
    tsa = types.ModuleType("statsmodels.tsa")
    holtwinters = types.ModuleType("statsmodels.tsa.holtwinters")

    class _FakeFitted:
        def forecast(self, *, steps: int) -> np.ndarray:
            captured["steps"] = int(steps)
            return np.arange(int(steps), dtype=float)

    class _FakeExponentialSmoothing:
        def __init__(self, data: np.ndarray, **kwargs: object) -> None:
            captured["data"] = np.asarray(data, dtype=float)
            captured["kwargs"] = kwargs

        def fit(self, *, optimized: bool) -> _FakeFitted:
            captured["optimized"] = bool(optimized)
            return _FakeFitted()

    holtwinters.ExponentialSmoothing = _FakeExponentialSmoothing
    tsa.holtwinters = holtwinters
    statsmodels.tsa = tsa

    monkeypatch.setitem(sys.modules, "statsmodels", statsmodels)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.holtwinters", holtwinters)


def _install_fake_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    xgboost = types.ModuleType("xgboost")

    class _FakeXGBRegressor:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, X: np.ndarray, y: np.ndarray) -> _FakeXGBRegressor:
            self._y_mean = float(np.mean(y))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full((int(X.shape[0]),), getattr(self, "_y_mean", 0.0), dtype=float)

    xgboost.XGBRegressor = _FakeXGBRegressor
    monkeypatch.setitem(sys.modules, "xgboost", xgboost)


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    lightgbm = types.ModuleType("lightgbm")

    class _FakeLGBMRegressor:
        def __init__(self, **_: object) -> None:
            pass

    lightgbm.LGBMRegressor = _FakeLGBMRegressor
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)


def _install_fake_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    catboost = types.ModuleType("catboost")

    class _FakeCatBoostRegressor:
        def __init__(self, **_: object) -> None:
            pass

    catboost.CatBoostRegressor = _FakeCatBoostRegressor
    monkeypatch.setitem(sys.modules, "catboost", catboost)


def _install_fake_sklearn(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, list[dict[str, object]]]
) -> None:
    sklearn = types.ModuleType("sklearn")
    ensemble = types.ModuleType("sklearn.ensemble")
    kernel_ridge = types.ModuleType("sklearn.kernel_ridge")
    linear_model = types.ModuleType("sklearn.linear_model")
    multioutput = types.ModuleType("sklearn.multioutput")
    neural_network = types.ModuleType("sklearn.neural_network")
    svm = types.ModuleType("sklearn.svm")
    tree = types.ModuleType("sklearn.tree")

    class _CapturedEstimator:
        def __init__(self, **kwargs: object) -> None:
            captured.setdefault(type(self).__name__, []).append(dict(kwargs))
            self._target_ndim = 1
            self._n_outputs = 1

        def fit(self, X: np.ndarray, y: np.ndarray) -> _CapturedEstimator:
            self._target_ndim = int(np.ndim(y))
            self._n_outputs = int(y.shape[1]) if np.ndim(y) > 1 else 1
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._target_ndim > 1:
                return np.zeros((int(X.shape[0]), int(self._n_outputs)), dtype=float)
            return np.zeros((int(X.shape[0]),), dtype=float)

    class RandomForestRegressor(_CapturedEstimator):
        pass

    class ExtraTreesRegressor(_CapturedEstimator):
        pass

    class GradientBoostingRegressor(_CapturedEstimator):
        pass

    class AdaBoostRegressor(_CapturedEstimator):
        pass

    class HistGradientBoostingRegressor(_CapturedEstimator):
        pass

    class DecisionTreeRegressor(_CapturedEstimator):
        pass

    class Ridge(_CapturedEstimator):
        pass

    class Lasso(_CapturedEstimator):
        pass

    class ElasticNet(_CapturedEstimator):
        pass

    class KernelRidge(_CapturedEstimator):
        pass

    class HuberRegressor(_CapturedEstimator):
        pass

    class PoissonRegressor(_CapturedEstimator):
        pass

    class GammaRegressor(_CapturedEstimator):
        pass

    class TweedieRegressor(_CapturedEstimator):
        pass

    class QuantileRegressor(_CapturedEstimator):
        pass

    class SGDRegressor(_CapturedEstimator):
        pass

    class PassiveAggressiveRegressor(_CapturedEstimator):
        pass

    class MLPRegressor(_CapturedEstimator):
        pass

    class SVR(_CapturedEstimator):
        pass

    class LinearSVR(_CapturedEstimator):
        pass

    class MultiOutputRegressor:
        def __init__(self, estimator: object) -> None:
            self.estimator = estimator
            self.n_outputs = 1

        def fit(self, X: np.ndarray, y: np.ndarray) -> MultiOutputRegressor:
            self.n_outputs = int(y.shape[1]) if np.ndim(y) > 1 else 1
            if hasattr(self.estimator, "fit"):
                target = y[:, 0] if np.ndim(y) > 1 else y
                self.estimator.fit(X, target)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros((int(X.shape[0]), int(self.n_outputs)), dtype=float)

    ensemble.AdaBoostRegressor = AdaBoostRegressor
    ensemble.ExtraTreesRegressor = ExtraTreesRegressor
    ensemble.GradientBoostingRegressor = GradientBoostingRegressor
    ensemble.HistGradientBoostingRegressor = HistGradientBoostingRegressor
    ensemble.RandomForestRegressor = RandomForestRegressor
    kernel_ridge.KernelRidge = KernelRidge
    linear_model.ElasticNet = ElasticNet
    linear_model.GammaRegressor = GammaRegressor
    linear_model.HuberRegressor = HuberRegressor
    linear_model.Lasso = Lasso
    linear_model.PoissonRegressor = PoissonRegressor
    linear_model.PassiveAggressiveRegressor = PassiveAggressiveRegressor
    linear_model.QuantileRegressor = QuantileRegressor
    linear_model.Ridge = Ridge
    linear_model.SGDRegressor = SGDRegressor
    linear_model.TweedieRegressor = TweedieRegressor
    multioutput.MultiOutputRegressor = MultiOutputRegressor
    neural_network.MLPRegressor = MLPRegressor
    svm.LinearSVR = LinearSVR
    svm.SVR = SVR
    tree.DecisionTreeRegressor = DecisionTreeRegressor
    sklearn.ensemble = ensemble
    sklearn.kernel_ridge = kernel_ridge
    sklearn.linear_model = linear_model
    sklearn.multioutput = multioutput
    sklearn.neural_network = neural_network
    sklearn.svm = svm
    sklearn.tree = tree

    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble)
    monkeypatch.setitem(sys.modules, "sklearn.kernel_ridge", kernel_ridge)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", linear_model)
    monkeypatch.setitem(sys.modules, "sklearn.multioutput", multioutput)
    monkeypatch.setitem(sys.modules, "sklearn.neural_network", neural_network)
    monkeypatch.setitem(sys.modules, "sklearn.svm", svm)
    monkeypatch.setitem(sys.modules, "sklearn.tree", tree)


def _literal_occurrence_count(path: Path, literal: str) -> int:
    return path.read_text(encoding="utf-8").count(literal)


def _find_subparser(parser: argparse.ArgumentParser, *names: str) -> argparse.ArgumentParser:
    current = parser
    for name in names:
        action = next(
            action for action in current._actions if isinstance(action, argparse._SubParsersAction)
        )
        current = action.choices[name]
    return current


@pytest.mark.parametrize(
    ("url", "doi", "arxiv_id", "expected"),
    [
        ("https://example.com/paper", "", "", "https://example.com/paper"),
        ("", "10.1234/example", "", "https://doi.org/10.1234/example"),
        ("", "", "1234.5678", "https://arxiv.org/abs/1234.5678"),
        ("", "", "", "-"),
    ],
)
def test_metadata_primary_url_resolves_expected_fallbacks(
    url: str, doi: str, arxiv_id: str, expected: str
) -> None:
    assert _metadata_primary_url(url, doi, arxiv_id) == expected


def test_render_rnn_paper_zoo_doc_uses_doi_fallback_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta_path = tmp_path / "rnn_paper_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "elman-srn": {
                    "title": "Elman Network",
                    "year": "1990",
                    "doi": "10.1234/elman",
                    "arxiv_id": "",
                    "url": "",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FORESIGHT_RNN_PAPER_METADATA", str(meta_path))

    doc = render_rnn_paper_zoo_doc()

    assert "| `elman-srn` |" in doc
    assert "https://doi.org/10.1234/elman" in doc


def test_render_rnn_zoo_doc_uses_base_and_variant_url_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta_path = tmp_path / "rnn_paper_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "elman-srn": {
                    "title": "Elman Network",
                    "year": "1990",
                    "doi": "",
                    "arxiv_id": "1234.5678",
                    "url": "",
                },
                "bahdanau-attention": {
                    "title": "Bahdanau Attention",
                    "year": "2014",
                    "doi": "",
                    "arxiv_id": "",
                    "url": "https://example.com/bahdanau",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FORESIGHT_RNN_PAPER_METADATA", str(meta_path))

    doc = render_rnn_zoo_doc()

    assert "https://arxiv.org/abs/1234.5678" in doc
    assert "https://example.com/bahdanau" in doc


def test_ets_forecast_skips_seasonal_periods_when_seasonality_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_statsmodels(monkeypatch, captured)

    out = ets_forecast([1.0, 2.0, 3.0, 4.0], horizon=2, seasonal=None, seasonal_periods=12)

    assert out.shape == (2,)
    assert captured["kwargs"] == {
        "trend": "add",
        "damped_trend": False,
        "seasonal": None,
        "seasonal_periods": None,
    }
    assert captured["optimized"] is True
    assert captured["steps"] == 2


def test_ets_forecast_coerces_seasonal_periods_when_seasonality_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_statsmodels(monkeypatch, captured)

    out = ets_forecast([1.0, 2.0, 3.0, 4.0], horizon=3, seasonal="add", seasonal_periods="6")

    assert out.shape == (3,)
    assert captured["kwargs"] == {
        "trend": "add",
        "damped_trend": False,
        "seasonal": "add",
        "seasonal_periods": 6,
    }
    assert captured["steps"] == 3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize("paper", ["mut1", "mut2", "mut3"])
def test_mut_rnnpaper_variants_smoke(paper: str) -> None:
    y = np.sin(np.arange(40, dtype=float) / 3.0) + 0.02 * np.arange(40, dtype=float)

    out = torch_rnnpaper_direct_forecast(
        y,
        2,
        paper=paper,
        lags=8,
        hidden_size=4,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_augment_lag_feat_row_requires_history_for_seasonal_features() -> None:
    feat = np.asarray([[3.0, 2.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="t_next and history are required"):
        _augment_lag_feat_row(
            feat,
            roll_windows=(),
            roll_stats=(),
            diff_lags=(),
            seasonal_lags=(1,),
            seasonal_diff_lags=(),
            fourier_periods=(),
            fourier_orders=2,
            t_next=3,
            history=None,
        )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"n_estimators": 0}, "n_estimators must be >= 1"),
        ({"max_depth": 0}, "max_depth must be >= 1"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"subsample": 0.0}, "subsample must be in \\(0,1\\]"),
        ({"colsample_bytree": 0.0}, "colsample_bytree must be in \\(0,1\\]"),
    ],
)
def test_xgb_common_regressor_params_reject_invalid_scalars(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _xgb_validate_common_regressor_params(params)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"subsample": 0.0}, "subsample must be in \\(0,1\\]"),
        ({"colsample_bytree": 0.0}, "colsample_bytree must be in \\(0,1\\]"),
    ],
)
def test_lgbm_common_regressor_params_reject_invalid_scalars(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _lgbm_validate_common_regressor_params(params)


def test_rf_step_lag_global_forecaster_sets_explicit_rf_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    def _fake_run_point_global_model(*args: object, fit_model: object, **kwargs: object) -> dict[str, bool]:
        assert callable(fit_model)
        fit_model(np.ones((4, 2), dtype=float), np.arange(4, dtype=float))
        return {"ok": True}

    monkeypatch.setattr(global_regression_mod, "_run_point_global_model", _fake_run_point_global_model)

    forecaster = rf_step_lag_global_forecaster(lags=2, n_estimators=5)
    assert forecaster(None, None, 1) == {"ok": True}

    kwargs = captured["RandomForestRegressor"][0]
    assert kwargs["min_samples_leaf"] == 1
    assert kwargs["max_features"] == 1.0


def test_decision_tree_step_lag_global_forecaster_sets_explicit_ccp_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    def _fake_run_point_global_model(*args: object, fit_model: object, **kwargs: object) -> dict[str, bool]:
        assert callable(fit_model)
        fit_model(np.ones((4, 2), dtype=float), np.arange(4, dtype=float))
        return {"ok": True}

    monkeypatch.setattr(global_regression_mod, "_run_point_global_model", _fake_run_point_global_model)

    forecaster = decision_tree_step_lag_global_forecaster(lags=2)
    assert forecaster(None, None, 1) == {"ok": True}

    kwargs = captured["DecisionTreeRegressor"][0]
    assert kwargs["ccp_alpha"] == 0.0


def test_svr_step_lag_global_forecaster_sets_explicit_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    def _fake_run_point_global_model(*args: object, fit_model: object, **kwargs: object) -> dict[str, bool]:
        assert callable(fit_model)
        fit_model(np.ones((4, 2), dtype=float), np.arange(4, dtype=float))
        return {"ok": True}

    monkeypatch.setattr(global_regression_mod, "_run_point_global_model", _fake_run_point_global_model)

    forecaster = svr_step_lag_global_forecaster(lags=2)
    assert forecaster(None, None, 1) == {"ok": True}

    kwargs = captured["SVR"][0]
    assert kwargs["kernel"] == "rbf"


@pytest.mark.parametrize(
    ("factory_name", "kwargs", "message"),
    [
        ("ridge_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("rf_step_lag_global_forecaster", {"n_estimators": 0}, "n_estimators must be >= 1"),
        ("rf_step_lag_global_forecaster", {"max_depth": 0}, "max_depth must be >= 1 or None"),
        ("extra_trees_step_lag_global_forecaster", {"n_estimators": 0}, "n_estimators must be >= 1"),
        ("extra_trees_step_lag_global_forecaster", {"max_depth": 0}, "max_depth must be >= 1 or None"),
        ("decision_tree_step_lag_global_forecaster", {"max_depth": 0}, "max_depth must be >= 1 or None"),
        ("gbrt_step_lag_global_forecaster", {"n_estimators": 0}, "n_estimators must be >= 1"),
        ("gbrt_step_lag_global_forecaster", {"learning_rate": 0.0}, "learning_rate must be > 0"),
        ("lasso_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("lasso_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("elasticnet_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("elasticnet_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("kernel_ridge_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("linear_svr_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("huber_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("huber_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("poisson_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("poisson_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("gamma_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("gamma_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("tweedie_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("tweedie_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("quantile_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("sgd_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("sgd_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("adaboost_step_lag_global_forecaster", {"n_estimators": 0}, "n_estimators must be >= 1"),
        ("adaboost_step_lag_global_forecaster", {"learning_rate": 0.0}, "learning_rate must be > 0"),
        ("mlp_step_lag_global_forecaster", {"alpha": -0.1}, "alpha must be >= 0"),
        ("mlp_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("hgb_step_lag_global_forecaster", {"max_iter": 0}, "max_iter must be >= 1"),
        ("hgb_step_lag_global_forecaster", {"learning_rate": 0.0}, "learning_rate must be > 0"),
        ("hgb_step_lag_global_forecaster", {"max_depth": 0}, "max_depth must be >= 1 or None"),
    ],
)
def test_global_regression_forecasters_validate_shared_scalar_constraints(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_sklearn(monkeypatch, {})
    factory = getattr(global_regression_mod, factory_name)

    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("factory_name", "kwargs", "message"),
    [
        ("svr_step_lag_global_forecaster", {"C": 0.0}, global_regression_mod.SVR_C_ERROR),
        (
            "svr_step_lag_global_forecaster",
            {"epsilon": -0.1},
            global_regression_mod.SVR_EPSILON_ERROR,
        ),
        ("linear_svr_step_lag_global_forecaster", {"C": 0.0}, global_regression_mod.SVR_C_ERROR),
        (
            "linear_svr_step_lag_global_forecaster",
            {"epsilon": -0.1},
            global_regression_mod.SVR_EPSILON_ERROR,
        ),
        (
            "passive_aggressive_step_lag_global_forecaster",
            {"C": 0.0},
            global_regression_mod.SVR_C_ERROR,
        ),
        (
            "passive_aggressive_step_lag_global_forecaster",
            {"epsilon": -0.1},
            global_regression_mod.SVR_EPSILON_ERROR,
        ),
    ],
)
def test_global_regression_svr_family_reuses_shared_validation_messages(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_sklearn(monkeypatch, {})
    factory = getattr(global_regression_mod, factory_name)

    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_tweedie_step_lag_global_forecaster_validates_targets_via_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sklearn(monkeypatch, {})

    def _fake_run_point_global_model(*args: object, fit_model: object, **kwargs: object) -> dict[str, bool]:
        assert callable(fit_model)
        fit_model(np.ones((3, 2), dtype=float), np.array([1.0, 0.0, 2.0], dtype=float))
        return {"ok": True}

    monkeypatch.setattr(global_regression_mod, "_run_point_global_model", _fake_run_point_global_model)

    forecaster = global_regression_mod.tweedie_step_lag_global_forecaster(lags=2, power=1.5)

    with pytest.raises(ValueError, match="requires strictly positive training targets"):
        forecaster(None, None, 1)


@pytest.mark.parametrize(
    ("factory_name", "installer"),
    [
        ("xgb_step_lag_global_forecaster", _install_fake_xgboost),
        ("lgbm_step_lag_global_forecaster", _install_fake_lightgbm),
        ("catboost_step_lag_global_forecaster", _install_fake_catboost),
    ],
)
@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        ([0.0], global_regression_mod.QUANTILES_RANGE_ERROR),
        ([0.125], global_regression_mod.QUANTILES_ALIGN_ERROR),
        ([1e-9], global_regression_mod.QUANTILES_STRICT_ERROR),
    ],
)
def test_global_step_lag_quantile_factories_validate_quantiles_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    installer: object,
    quantiles: list[float],
    message: str,
) -> None:
    installer(monkeypatch)
    factory = getattr(global_regression_mod, factory_name)

    with pytest.raises(ValueError, match=re.escape(message)):
        factory(quantiles=quantiles)


@pytest.mark.parametrize(
    ("factory_name", "installer"),
    [
        ("xgb_step_lag_global_forecaster", _install_fake_xgboost),
        ("lgbm_step_lag_global_forecaster", _install_fake_lightgbm),
        ("catboost_step_lag_global_forecaster", _install_fake_catboost),
    ],
)
def test_global_step_lag_quantile_factories_normalize_valid_percentiles(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    installer: object,
) -> None:
    installer(monkeypatch)
    factory = getattr(global_regression_mod, factory_name)

    forecaster = factory(quantiles="0.1,0.9")

    assert callable(forecaster)


@pytest.mark.parametrize(
    ("factory_name", "horizon", "kwargs", "message"),
    [
        ("rf_lag_direct_forecast", 0, {"lags": 2}, "horizon must be >= 1"),
        ("rf_lag_direct_forecast", 1, {"lags": 0}, "lags must be >= 1"),
        ("rf_lag_direct_forecast", 1, {"lags": 2, "n_estimators": 0}, "n_estimators must be >= 1"),
        ("lasso_lag_direct_forecast", 1, {"lags": 2, "max_iter": 0}, "max_iter must be >= 1"),
        ("decision_tree_lag_direct_forecast", 1, {"lags": 2, "max_depth": 0}, "max_depth must be >= 1 or None"),
        ("extra_trees_lag_direct_forecast", 1, {"lags": 2, "max_depth": 0}, "max_depth must be >= 1 or None"),
        ("adaboost_lag_direct_forecast", 1, {"lags": 2, "n_estimators": 0}, "n_estimators must be >= 1"),
        ("hgb_lag_direct_forecast", 1, {"lags": 2, "max_iter": 0}, "max_iter must be >= 1"),
        ("hgb_lag_direct_forecast", 1, {"lags": 2, "max_depth": 0}, "max_depth must be >= 1 or None"),
        ("huber_lag_direct_forecast", 1, {"lags": 2, "alpha": -0.1}, "alpha must be >= 0"),
        ("huber_lag_direct_forecast", 1, {"lags": 2, "max_iter": 0}, "max_iter must be >= 1"),
        ("quantile_lag_direct_forecast", 1, {"lags": 2, "alpha": -0.1}, "alpha must be >= 0"),
        ("sgd_lag_direct_forecast", 1, {"lags": 2, "alpha": -0.1}, "alpha must be >= 0"),
        ("sgd_lag_direct_forecast", 1, {"lags": 2, "max_iter": 0}, "max_iter must be >= 1"),
    ],
)
def test_regression_forecasters_validate_shared_scalar_constraints(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    horizon: int,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_sklearn(monkeypatch, {})
    factory = getattr(regression_mod, factory_name)

    with pytest.raises(ValueError, match=message):
        factory([1.0, 2.0, 3.0, 4.0, 5.0], horizon, **kwargs)


@pytest.mark.parametrize(
    ("factory_name", "kwargs", "message"),
    [
        ("svr_lag_direct_forecast", {"C": 0.0}, regression_mod.SVR_C_ERROR),
        ("svr_lag_direct_forecast", {"epsilon": -0.1}, regression_mod.SVR_EPSILON_ERROR),
        ("linear_svr_lag_direct_forecast", {"C": 0.0}, regression_mod.SVR_C_ERROR),
        (
            "linear_svr_lag_direct_forecast",
            {"epsilon": -0.1},
            regression_mod.SVR_EPSILON_ERROR,
        ),
    ],
)
def test_regression_svr_family_reuses_shared_validation_messages(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_sklearn(monkeypatch, {})
    factory = getattr(regression_mod, factory_name)

    with pytest.raises(ValueError, match=message):
        factory([1.0, 2.0, 3.0, 4.0, 5.0], 1, lags=2, **kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (_xgb_lag_direct_forecast, {"learning_rate": 0.0}, "learning_rate must be > 0"),
        (_xgb_lag_direct_forecast, {"subsample": 0.0}, "subsample must be in \\(0,1\\]"),
        (_xgb_lag_direct_forecast, {"colsample_bytree": 0.0}, "colsample_bytree must be in \\(0,1\\]"),
        (_xgb_lag_direct_forecast, {"reg_lambda": -1.0}, "reg_lambda must be >= 0"),
        (_xgb_lag_direct_forecast, {"min_child_weight": -1.0}, "min_child_weight must be >= 0"),
        (_xgb_lag_direct_forecast, {"gamma": -1.0}, "gamma must be >= 0"),
        (_xgb_lag_direct_forecast, {"n_jobs": 0}, "n_jobs must be non-zero"),
        (_xgb_lag_recursive_forecast, {"learning_rate": 0.0}, "learning_rate must be > 0"),
        (_xgb_lag_recursive_forecast, {"subsample": 0.0}, "subsample must be in \\(0,1\\]"),
        (_xgb_lag_recursive_forecast, {"colsample_bytree": 0.0}, "colsample_bytree must be in \\(0,1\\]"),
        (_xgb_lag_recursive_forecast, {"reg_lambda": -1.0}, "reg_lambda must be >= 0"),
        (_xgb_lag_recursive_forecast, {"min_child_weight": -1.0}, "min_child_weight must be >= 0"),
        (_xgb_lag_recursive_forecast, {"gamma": -1.0}, "gamma must be >= 0"),
        (_xgb_lag_recursive_forecast, {"n_jobs": 0}, "n_jobs must be non-zero"),
    ],
)
def test_regression_xgb_internal_forecasters_validate_shared_scalar_constraints(
    monkeypatch: pytest.MonkeyPatch,
    factory: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_xgboost(monkeypatch)

    with pytest.raises(ValueError, match=message):
        factory(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            1,
            lags=2,
            booster="gbtree",
            objective="reg:squarederror",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("factory_name", "extra_kwargs"),
    [
        ("_xgb_lag_direct_forecast", {"booster": "gbtree"}),
        ("_xgb_lag_recursive_forecast", {"booster": "gbtree"}),
    ],
)
def test_regression_xgb_internal_forecasters_reject_empty_objective(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    extra_kwargs: dict[str, object],
) -> None:
    _install_fake_xgboost(monkeypatch)
    factory = getattr(regression_mod, factory_name)

    with pytest.raises(ValueError, match=regression_mod.XGB_OBJECTIVE_EMPTY_ERROR):
        factory([1.0, 2.0, 3.0, 4.0, 5.0], 1, lags=2, objective="", **extra_kwargs)


@pytest.mark.parametrize(
    "factory_name",
    [
        "_xgb_lag_direct_forecast_kwargs",
        "_xgb_lag_recursive_forecast_kwargs",
        "_xgb_lag_step_forecast_kwargs",
        "_xgb_lag_dirrec_forecast_kwargs",
        "_xgb_lag_mimo_forecast_kwargs",
    ],
)
def test_regression_xgb_kwargs_forecasters_reject_empty_objective(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    _install_fake_xgboost(monkeypatch)
    factory = getattr(regression_mod, factory_name)

    with pytest.raises(ValueError, match=regression_mod.XGB_OBJECTIVE_EMPTY_ERROR):
        factory([1.0, 2.0, 3.0, 4.0, 5.0], 1, lags=2, xgb_params={})


@pytest.mark.parametrize(
    "factory_name",
    ["xgb_logistic_lag_direct_forecast", "xgb_logistic_lag_recursive_forecast"],
)
def test_regression_xgb_logistic_forecasters_validate_unit_interval_targets(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    _install_fake_xgboost(monkeypatch)
    factory = getattr(regression_mod, factory_name)

    with pytest.raises(ValueError, match=r"reg:logistic requires series values in \[0,1\]"):
        factory([-0.1, 0.2, 0.4, 0.8], 1, lags=2)


def test_rf_lag_direct_forecast_sets_explicit_rf_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    out = rf_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2, n_estimators=3)

    assert out.shape == (2,)
    kwargs = captured["RandomForestRegressor"][0]
    assert kwargs["min_samples_leaf"] == 1
    assert kwargs["max_features"] == 1.0


def test_decision_tree_lag_direct_forecast_sets_explicit_ccp_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    out = decision_tree_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)

    assert out.shape == (2,)
    kwargs = captured["DecisionTreeRegressor"][0]
    assert kwargs["ccp_alpha"] == 0.0


def test_svr_lag_direct_forecast_sets_explicit_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = {}
    _install_fake_sklearn(monkeypatch, captured)

    out = svr_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)

    assert out.shape == (2,)
    kwargs = captured["SVR"][0]
    assert kwargs["kernel"] == "rbf"


def test_regression_source_extracts_repeated_xgb_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "foresight" / "models" / "regression.py"
    literals = [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:logistic",
        "count:poisson",
        "reg:gamma",
        "reg:tweedie",
        'xgboost lag models require xgboost. Install with: pip install -e ".[xgb]"',
        "objective must be non-empty",
        "C must be > 0",
        "epsilon must be >= 0",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_global_regression_source_extracts_repeated_quantile_and_svr_literals() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "src" / "foresight" / "models" / "global_regression.py"
    )
    literals = [
        "reg:squarederror",
        "C must be > 0",
        "epsilon must be >= 0",
        "quantiles must be in (0,1)",
        "quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)",
        "quantiles must be strictly between 0 and 1",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_statsmodels_wrap_source_extracts_repeated_validation_literals() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "src" / "foresight" / "models" / "statsmodels_wrap.py"
    )
    literals = [
        "horizon must be >= 1",
        "train_exog must have the same number of rows as train",
        "future_exog must have horizon rows",
        "train_exog and future_exog must either both be provided or both be omitted",
        "lags must be >= 0",
        "local level",
        "period must be >= 2",
        "orders must be an int or have the same length as periods",
        "periods must contain integers >= 2",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_docsgen_rnn_source_extracts_repeated_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "foresight" / "docsgen" / "rnn.py"
    literals = [
        "rnn_paper_metadata.json",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_generate_model_capability_docs_source_extracts_repeated_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_model_capability_docs.py"
    literals = [
        "Core classes",
        "Data preparation",
        "Hierarchical forecasting",
        "Intervals and tuning",
        "foresight.data",
        "foresight.eval_forecast",
        "foresight.serialization",
        "foresight.models.registry",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_fetch_rnn_paper_metadata_source_extracts_repeated_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "fetch_rnn_paper_metadata.py"
    literals = [
        "ForeSight-metadata/0.1",
        r"[^a-z0-9]+",
        "An Empirical Exploration of Recurrent Network Architectures",
        "https://proceedings.mlr.press/v37/jozefowicz15.html",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_cli_source_extracts_repeated_help_literals() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "foresight" / "cli.py"
    literals = [
        "Dataset key",
        "Optional target column name (default: use dataset spec default_y).",
        "Optional rolling train window size (default: expanding window).",
        "Model parameter as key=value (repeatable). Example: --model-param season_length=12",
        "Optional path to write metrics output",
    ]

    for literal in literals:
        assert _literal_occurrence_count(path, literal) <= 1


def test_build_parser_preserves_shared_help_texts() -> None:
    parser = build_parser()

    cv_run = _find_subparser(parser, "cv", "run")
    eval_run = _find_subparser(parser, "eval", "run")

    assert cv_run._option_string_actions["--dataset"].help == "Dataset key"
    assert cv_run._option_string_actions["--y-col"].help == (
        "Optional target column name (default: use dataset spec default_y)."
    )
    assert cv_run._option_string_actions["--max-train-size"].help == (
        "Optional rolling train window size (default: expanding window)."
    )
    assert eval_run._option_string_actions["--model-param"].help == (
        "Model parameter as key=value (repeatable). Example: --model-param season_length=12"
    )
    assert eval_run._option_string_actions["--output"].help == "Optional path to write metrics output"


def test_xgb_lag_direct_forecast_validates_labels_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_xgboost(monkeypatch)

    with pytest.raises(ValueError, match="requires strictly positive series values"):
        _xgb_lag_direct_forecast(
            [1.0, -1.0, 2.0, 3.0],
            1,
            lags=2,
            booster="gbtree",
            objective="reg:gamma",
        )


def test_xgb_lag_recursive_forecast_validates_labels_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_xgboost(monkeypatch)

    with pytest.raises(ValueError, match="requires strictly positive series values"):
        _xgb_lag_recursive_forecast(
            [1.0, -1.0, 2.0, 3.0],
            1,
            lags=2,
            booster="gbtree",
            objective="reg:gamma",
        )
