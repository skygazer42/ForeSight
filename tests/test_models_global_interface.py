import pytest

from foresight.models.registry import get_model_spec, make_forecaster, make_global_forecaster

GLOBAL_TORCH_MODELS = [
    "torch-tft-global",
    "torch-informer-global",
    "torch-autoformer-global",
]


def test_global_models_are_marked_interface_global():
    for key in GLOBAL_TORCH_MODELS:
        spec = get_model_spec(key)
        assert spec.interface == "global"
        assert "torch" in spec.requires


def test_make_forecaster_rejects_global_models():
    for key in GLOBAL_TORCH_MODELS:
        with pytest.raises(ValueError):
            make_forecaster(key)


def test_make_global_forecaster_rejects_local_models():
    with pytest.raises(ValueError):
        make_global_forecaster("naive-last")
