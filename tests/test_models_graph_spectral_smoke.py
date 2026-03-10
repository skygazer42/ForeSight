import importlib


def test_graph_spectral_scaffold_module_imports() -> None:
    module = importlib.import_module("foresight.models.torch_graph_spectral")
    assert module is not None
