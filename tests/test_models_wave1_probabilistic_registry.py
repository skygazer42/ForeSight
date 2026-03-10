from foresight.models.registry import get_model_spec, list_models

PROBABILISTIC_KEYS = (
    "torch-timegrad-direct",
    "torch-tactis-direct",
)


def test_wave1_probabilistic_models_are_registered() -> None:
    keys = set(list_models())
    for key in PROBABILISTIC_KEYS:
        assert key in keys


def test_wave1_probabilistic_models_are_torch_local_optional() -> None:
    for key in PROBABILISTIC_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires
