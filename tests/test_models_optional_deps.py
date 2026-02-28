import importlib.util

import pytest

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
