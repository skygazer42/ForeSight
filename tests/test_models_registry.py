from foresight.models.registry import get_model_spec, list_models


def test_list_models_contains_expected_keys():
    keys = set(list_models())
    assert "naive-last" in keys
    assert "seasonal-naive" in keys


def test_model_spec_has_description():
    spec = get_model_spec("naive-last")
    assert isinstance(spec.description, str)
    assert spec.description
