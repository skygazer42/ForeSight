import importlib


def test_graph_structure_scaffold_module_imports() -> None:
    module = importlib.import_module("foresight.models.torch_graph_structure")
    assert module is not None
