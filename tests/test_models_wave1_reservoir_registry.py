from foresight.models.registry import get_model_spec, list_models

RESERVOIR_KEYS = (
    "torch-esn-direct",
    "torch-deep-esn-direct",
    "torch-liquid-state-direct",
)


def test_wave1_reservoir_models_are_registered() -> None:
    keys = set(list_models())
    for key in RESERVOIR_KEYS:
        assert key in keys


def test_wave1_reservoir_models_are_torch_local_optional() -> None:
    for key in RESERVOIR_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires
