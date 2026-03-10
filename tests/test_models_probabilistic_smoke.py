import importlib


def test_probabilistic_scaffold_module_imports() -> None:
    module = importlib.import_module("foresight.models.torch_probabilistic")
    assert module is not None
