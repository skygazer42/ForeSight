import importlib


def test_foundation_scaffold_modules_import() -> None:
    foundation = importlib.import_module("foresight.models.foundation")
    graph_attention = importlib.import_module("foresight.models.torch_graph_attention")
    graph_spectral = importlib.import_module("foresight.models.torch_graph_spectral")
    graph_structure = importlib.import_module("foresight.models.torch_graph_structure")
    probabilistic = importlib.import_module("foresight.models.torch_probabilistic")
    reservoir = importlib.import_module("foresight.models.torch_reservoir")
    structured = importlib.import_module("foresight.models.torch_structured_rnn")

    assert foundation is not None
    assert graph_attention is not None
    assert graph_spectral is not None
    assert graph_structure is not None
    assert probabilistic is not None
    assert reservoir is not None
    assert structured is not None
