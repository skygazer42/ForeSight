import importlib.util

import numpy as np
import pytest

from foresight.models.registry import get_model_spec, list_models, make_forecaster

TRANSFORMERS_LOCAL_KEYS = ("hf-timeseries-transformer-direct",)


def _transformers_model_keys() -> list[str]:
    return [k for k in list_models() if "transformers" in get_model_spec(k).requires]


def test_transformers_models_are_registered_as_optional() -> None:
    for key in _transformers_model_keys():
        spec = get_model_spec(key)
        assert "transformers" in spec.requires


def test_hf_timeseries_transformer_is_registered_as_transformers_optional() -> None:
    spec = get_model_spec("hf-timeseries-transformer-direct")
    assert spec.interface == "local"
    assert "transformers" in spec.requires


def test_transformers_models_raise_importerror_when_transformers_missing() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed; this test targets the missing-dep path")

    for key in TRANSFORMERS_LOCAL_KEYS:
        f = make_forecaster(key)
        with pytest.raises(ImportError):
            f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)


def test_hf_timeseries_transformer_smoke_when_installed() -> None:
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("requires transformers + torch")

    y = np.sin(np.arange(80, dtype=float) / 5.0) + 0.01 * np.arange(80, dtype=float)
    f = make_forecaster(
        "hf-timeseries-transformer-direct",
        context_length=24,
        d_model=16,
        num_samples=20,
        encoder_layers=1,
        decoder_layers=1,
        nhead=2,
        epochs=0,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 3)

    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))
