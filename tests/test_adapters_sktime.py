from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import foresight.adapters.sktime as sktime_adapter_mod
from foresight.base import BaseForecaster
from foresight.pipeline import make_pipeline_object


def _patch_sktime_loader(monkeypatch: pytest.MonkeyPatch, result: object | None = None) -> None:
    fake_result = object() if result is None else result
    monkeypatch.setattr(sktime_adapter_mod, "_require_sktime", lambda: fake_result, raising=False)
    monkeypatch.setattr(
        sktime_adapter_mod,
        "require_dependency",
        lambda name, **kwargs: fake_result,
        raising=False,
    )


class _FakeLocalXRegForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(model_key="fake-local-xreg", model_params={"x_cols": ("promo",)})
        self.fit_y: np.ndarray | None = None
        self.fit_X: pd.DataFrame | None = None
        self.predict_X: pd.DataFrame | None = None

    def fit(self, y: Any, X: Any = None) -> _FakeLocalXRegForecaster:
        self.fit_y = np.asarray(y, dtype=float)
        self.fit_X = None if X is None else pd.DataFrame(X).copy()
        self._is_fitted = True
        return self

    def predict(self, horizon: int, X: Any = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before predict")
        self.predict_X = None if X is None else pd.DataFrame(X).copy()
        base = float(self.fit_y[-1]) if self.fit_y is not None else 0.0
        bonus = 0.0 if self.predict_X is None else float(self.predict_X.to_numpy(dtype=float).sum())
        return np.asarray([base + bonus] * int(horizon), dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        return {"kind": "local", "n_obs": 0 if self.fit_y is None else int(self.fit_y.size)}


class _FakeHistoricOnlyForecaster(_FakeLocalXRegForecaster):
    def __init__(self) -> None:
        super().__init__()
        self.model_key = "fake-historic-only"
        self.model_params = {"historic_x_cols": ("promo_hist",)}


def test_sktime_adapter_predicts_series_from_local_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(
        make_pipeline_object(base="naive-last", transforms=("standardize",))
    )
    y = pd.Series([2.0, 4.0, 6.0, 8.0], index=pd.RangeIndex(start=0, stop=4))

    yhat = adapter.fit(y).predict([1, 2])

    assert isinstance(yhat, pd.Series)
    assert yhat.index.tolist() == [4, 5]
    assert yhat.tolist() == pytest.approx([8.0, 8.0])


def test_sktime_adapter_uses_fit_time_fh_when_predict_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))

    yhat = adapter.fit(y, fh=2).predict()

    assert yhat.index.tolist() == [3, 4]
    assert yhat.tolist() == pytest.approx([3.0, 3.0])


def test_sktime_adapter_supports_absolute_range_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))

    yhat = adapter.fit(y).predict(pd.Index([3, 4]))

    assert yhat.index.tolist() == [3, 4]
    assert yhat.tolist() == pytest.approx([3.0, 3.0])


def test_sktime_adapter_supports_absolute_datetime_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
    y = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    yhat = adapter.fit(y).predict(pd.DatetimeIndex(["2024-01-04", "2024-01-05"]))

    assert list(yhat.index) == list(pd.DatetimeIndex(["2024-01-04", "2024-01-05"]))
    assert yhat.tolist() == pytest.approx([3.0, 3.0])


def test_sktime_adapter_rejects_predict_x_for_unsupported_beta_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
    adapter.fit([1.0, 2.0, 3.0], fh=2)

    with pytest.raises(
        ValueError, match="supports X only for local single-series xreg forecasters in beta"
    ):
        adapter.predict(X=[[1.0], [2.0]])


def test_sktime_adapter_supports_local_single_series_x_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(_FakeLocalXRegForecaster())
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))
    fit_X = pd.DataFrame({"promo": [0.0, 1.0, 0.0]}, index=y.index)
    future_X = pd.DataFrame({"promo": [1.0, 2.0]}, index=pd.Index([3, 4]))

    yhat = adapter.fit(y, X=fit_X).predict([1, 2], X=future_X)

    assert isinstance(yhat, pd.Series)
    assert yhat.index.tolist() == [3, 4]
    assert yhat.tolist() == pytest.approx([6.0, 6.0])


def test_sktime_adapter_supports_array_like_x_inputs_for_datetime_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(_FakeLocalXRegForecaster())
    y = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    yhat = adapter.fit(y, X=[[0.0], [1.0], [0.0]]).predict([1, 2], X=[[1.0], [2.0]])

    assert isinstance(yhat, pd.Series)
    assert list(yhat.index) == list(pd.DatetimeIndex(["2024-01-04", "2024-01-05"]))
    assert yhat.tolist() == pytest.approx([6.0, 6.0])


def test_sktime_adapter_rejects_x_columns_that_do_not_match_configured_x_cols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(_FakeLocalXRegForecaster())
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))

    with pytest.raises(ValueError, match="X columns must match configured x_cols"):
        adapter.fit(y, X=pd.DataFrame({"wrong_col": [0.0, 1.0, 0.0]}))


def test_sktime_adapter_rejects_sparse_horizon_x_without_full_future_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(_FakeLocalXRegForecaster())
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))
    adapter.fit(y, X=pd.DataFrame({"promo": [0.0, 1.0, 0.0]}, index=y.index))

    with pytest.raises(ValueError, match="one row per step up to max fh"):
        adapter.predict([1, 3], X=pd.DataFrame({"promo": [1.0, 2.0]}, index=pd.Index([3, 5])))


def test_sktime_adapter_rejects_unsupported_x_beta_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")

    with pytest.raises(
        ValueError, match="supports X only for local single-series xreg forecasters in beta"
    ):
        adapter.fit([1.0, 2.0, 3.0], X=[[1.0], [2.0], [3.0]])


def test_sktime_adapter_rejects_historic_only_xreg_beta_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sktime_loader(monkeypatch)

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter(_FakeHistoricOnlyForecaster())

    with pytest.raises(
        ValueError, match="supports X only for local single-series xreg forecasters in beta"
    ):
        adapter.fit([1.0, 2.0, 3.0], X=[[1.0], [2.0], [3.0]])


def test_sktime_adapter_missing_dependency_uses_sktime_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sktime_adapter_mod,
        "require_dependency",
        lambda name, **kwargs: (_ for _ in ()).throw(
            ImportError(
                "sktime adapter requires sktime. Install with: "
                'pip install "foresight-ts[sktime]" or pip install -e ".[sktime]"'
            )
        ),
        raising=False,
    )

    with pytest.raises(ImportError, match='pip install "foresight-ts\\[sktime\\]"'):
        sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
