from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from foresight.models import global_regression as global_regression_mod
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


def _install_fake_sklearn(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, list[dict[str, object]]]
) -> None:
    sklearn = types.ModuleType("sklearn")
    ensemble = types.ModuleType("sklearn.ensemble")
    multioutput = types.ModuleType("sklearn.multioutput")
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

    class DecisionTreeRegressor(_CapturedEstimator):
        pass

    class SVR(_CapturedEstimator):
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

    ensemble.RandomForestRegressor = RandomForestRegressor
    multioutput.MultiOutputRegressor = MultiOutputRegressor
    svm.SVR = SVR
    tree.DecisionTreeRegressor = DecisionTreeRegressor
    sklearn.ensemble = ensemble
    sklearn.multioutput = multioutput
    sklearn.svm = svm
    sklearn.tree = tree

    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble)
    monkeypatch.setitem(sys.modules, "sklearn.multioutput", multioutput)
    monkeypatch.setitem(sys.modules, "sklearn.svm", svm)
    monkeypatch.setitem(sys.modules, "sklearn.tree", tree)


def _literal_occurrence_count(path: Path, literal: str) -> int:
    return path.read_text(encoding="utf-8").count(literal)


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
