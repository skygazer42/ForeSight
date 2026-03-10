from foresight.models.registry import get_model_spec, list_models

FOUNDATION_WRAPPER_A_KEYS = (
    "lag-llama",
    "chronos",
    "chronos-bolt",
    "timesfm",
)


def test_wave1_foundation_wrapper_a_models_are_registered() -> None:
    keys = set(list_models())
    for key in FOUNDATION_WRAPPER_A_KEYS:
        assert key in keys


def test_wave1_foundation_wrapper_a_model_specs_are_local_wrapper_scaffolds() -> None:
    for key in FOUNDATION_WRAPPER_A_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "wrapper" in spec.description.lower()
