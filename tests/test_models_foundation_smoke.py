import importlib
import json

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


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


@pytest.mark.parametrize(
    "key",
    (
        "lag-llama",
        "chronos",
        "chronos-bolt",
        "timesfm",
    ),
)
def test_foundation_wrapper_a_fixture_json_smoke(tmp_path, key: str) -> None:
    checkpoint = tmp_path / f"{key}.json"
    checkpoint.write_text(
        json.dumps({"bias": 1.25, "scale": 1.0, "use_trend": True}),
        encoding="utf-8",
    )

    forecaster = make_forecaster(
        key,
        backend="fixture-json",
        checkpoint_path=str(checkpoint),
    )
    yhat = forecaster([1.0, 2.0, 3.0, 5.0, 8.0], 3)

    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))


def test_foundation_wrapper_a_requires_checkpoint_or_model_source() -> None:
    forecaster = make_forecaster("chronos")
    with pytest.raises(ValueError, match="requires checkpoint_path|requires model_source"):
        forecaster([1.0, 2.0, 3.0, 4.0], 2)


@pytest.mark.parametrize(
    "key",
    (
        "moirai",
        "moment",
        "time-moe",
        "timer-s1",
    ),
)
def test_foundation_wrapper_b_fixture_json_smoke(tmp_path, key: str) -> None:
    checkpoint = tmp_path / f"{key}.json"
    checkpoint.write_text(
        json.dumps({"bias": 0.75, "scale": 1.0, "use_trend": True, "trend_damp": 0.5}),
        encoding="utf-8",
    )

    forecaster = make_forecaster(
        key,
        backend="fixture-json",
        checkpoint_path=str(checkpoint),
    )
    yhat = forecaster([2.0, 3.0, 5.0, 8.0, 13.0], 3)

    assert yhat.shape == (3,)
    assert np.all(np.isfinite(yhat))


def test_foundation_wrapper_b_requires_checkpoint_or_model_source() -> None:
    forecaster = make_forecaster("moirai")
    with pytest.raises(ValueError, match="requires checkpoint_path|requires model_source"):
        forecaster([1.0, 2.0, 3.0, 4.0], 2)
