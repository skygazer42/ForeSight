from foresight.models.registry import get_model_spec, list_models

GRAPH_STRUCTURE_KEYS = (
    "torch-agcrn-multivariate",
    "torch-mtgnn-multivariate",
)


def test_wave1_graph_structure_models_are_registered() -> None:
    keys = set(list_models())
    for key in GRAPH_STRUCTURE_KEYS:
        assert key in keys


def test_wave1_graph_structure_models_are_multivariate_torch_optional() -> None:
    for key in GRAPH_STRUCTURE_KEYS:
        spec = get_model_spec(key)
        assert spec.interface == "multivariate"
        assert "torch" in spec.requires
