from __future__ import annotations

import numpy as np
import pytest

from foresight.pipeline import make_ensemble_object, make_pipeline_object
from foresight.serialization import load_forecaster, save_forecaster


def test_pipeline_object_diff1_naive_last_behaves_like_drift() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    forecaster = make_pipeline_object(base="naive-last", transforms=("diff1",))
    yhat = forecaster.fit(y).predict(3)

    assert yhat.shape == (3,)
    assert np.allclose(yhat, np.array([6.0, 7.0, 8.0]))


def test_ensemble_object_mean_matches_member_average() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])

    forecaster = make_ensemble_object(members=("mean", "naive-last"), agg="mean")
    yhat = forecaster.fit(y).predict(2)

    assert yhat.shape == (2,)
    assert np.allclose(yhat, np.array([3.25, 3.25]))


def test_pipeline_object_round_trips_through_artifact_serialization(tmp_path) -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
    artifact_path = tmp_path / "pipeline.pkl"

    forecaster = make_pipeline_object(base="naive-last", transforms=("standardize",)).fit(y)
    metadata = save_forecaster(forecaster, artifact_path)
    loaded = load_forecaster(artifact_path)

    assert metadata["model_key"] == "pipeline"
    assert metadata["model_params"]["base"] == "naive-last"
    assert metadata["model_params"]["transforms"] == ("standardize",)
    assert np.allclose(loaded.predict(2), np.array([10.0, 10.0]))


def test_pipeline_object_can_wrap_another_pipeline_object() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    inner = make_pipeline_object(base="naive-last", transforms=("diff1",))
    outer = make_pipeline_object(base=inner, transforms=("standardize",))
    yhat = outer.fit(y).predict(2)

    assert yhat.tolist() == pytest.approx([6.0, 7.0])


def test_ensemble_object_can_include_built_ensemble_member() -> None:
    y = np.array([1.0, 2.0, 3.0, 100.0], dtype=float)

    nested = make_ensemble_object(members=("mean", "median"), agg="mean")
    top = make_ensemble_object(members=(nested, "naive-last"), agg="mean")
    yhat = top.fit(y).predict(1)

    assert yhat.tolist() == pytest.approx([57.25])
