import importlib


def test_graph_attention_scaffold_module_imports() -> None:
    module = importlib.import_module("foresight.models.torch_graph_attention")
    assert module is not None
