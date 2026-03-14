from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

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
