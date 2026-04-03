from __future__ import annotations

import pandas as pd
import pytest

import foresight.adapters.sktime as sktime_adapter_mod
from foresight.pipeline import make_pipeline_object


def test_sktime_adapter_predicts_series_from_local_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sktime_adapter_mod, "_require_sktime", lambda: object())

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
    monkeypatch.setattr(sktime_adapter_mod, "_require_sktime", lambda: object())

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
    y = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(start=0, stop=3))

    yhat = adapter.fit(y, fh=2).predict()

    assert yhat.index.tolist() == [3, 4]
    assert yhat.tolist() == pytest.approx([3.0, 3.0])


def test_sktime_adapter_rejects_exogenous_X_in_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sktime_adapter_mod, "_require_sktime", lambda: object())

    adapter = sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")

    with pytest.raises(ValueError, match="does not support X in v1"):
        adapter.fit([1.0, 2.0, 3.0], X=[[1.0], [2.0], [3.0]])


def test_sktime_adapter_missing_dependency_uses_sktime_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sktime_adapter_mod,
        "_require_sktime",
        lambda: (_ for _ in ()).throw(
            ImportError(
                'sktime adapter requires sktime. Install with: '
                'pip install "foresight-ts[sktime]" or pip install -e ".[sktime]"'
            )
        ),
    )

    with pytest.raises(ImportError, match='pip install "foresight-ts\\[sktime\\]"'):
        sktime_adapter_mod.make_sktime_forecaster_adapter("naive-last")
