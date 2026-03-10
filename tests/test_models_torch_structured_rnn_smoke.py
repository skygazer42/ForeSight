import importlib


def test_torch_structured_rnn_scaffold_module_imports() -> None:
    module = importlib.import_module("foresight.models.torch_structured_rnn")
    assert module is not None
