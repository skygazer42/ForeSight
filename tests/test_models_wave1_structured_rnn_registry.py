from foresight.models.registry import get_model_spec, list_models

STRUCTURED_RNN_KEYS = (
    "torch-multidim-rnn-direct",
    "torch-grid-lstm-direct",
    "torch-structural-rnn-direct",
)


def test_wave1_structured_rnn_models_are_registered() -> None:
    keys = set(list_models())
    for key in STRUCTURED_RNN_KEYS:
        assert key in keys


def test_wave1_structured_rnn_models_are_torch_local_optional() -> None:
    for key in STRUCTURED_RNN_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires
