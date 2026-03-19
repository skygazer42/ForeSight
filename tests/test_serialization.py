import importlib.util
import pickle

import numpy as np
import pandas as pd
import pytest

from foresight import __version__
from foresight.models import make_forecaster_object, make_global_forecaster_object
from foresight.serialization import load_forecaster, load_forecaster_artifact, save_forecaster


def _small_panel_long_df() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=18, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append({"unique_id": uid, "ds": d, "y": float(bias + 0.5 * i)})
    return pd.DataFrame(rows)


def test_save_and_load_local_forecaster_round_trip(tmp_path) -> None:
    f = make_forecaster_object("moving-average", window=3).fit([1, 2, 3, 4, 5, 6])
    path = tmp_path / "local.pkl"

    meta = save_forecaster(f, path)
    loaded = load_forecaster(path)
    payload = load_forecaster_artifact(path)

    assert path.exists()
    assert payload["artifact_schema_version"] == 1
    assert meta["package_version"] == __version__
    assert meta["model_key"] == "moving-average"
    assert meta["model_params"]["window"] == 3
    assert meta["train_schema"]["n_obs"] == 6
    assert payload["metadata"]["package_version"] == __version__
    assert payload["metadata"]["model_params"]["window"] == 3
    assert np.allclose(loaded.predict(3), f.predict(3))


def test_load_forecaster_artifact_supports_legacy_artifact_without_schema_version(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                    "train_schema": {"kind": "local", "n_obs": 5},
                },
                "forecaster": f,
            },
            handle,
        )

    payload = load_forecaster_artifact(path)
    assert payload["artifact_schema_version"] == 0


def test_load_forecaster_artifact_rejects_missing_required_metadata_keys(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "broken.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "artifact_schema_version": 1,
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                },
                "forecaster": f,
            },
            handle,
        )

    with pytest.raises(KeyError, match="train_schema"):
        load_forecaster_artifact(path)


def test_load_forecaster_artifact_rejects_invalid_metadata_value_types(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "invalid-metadata.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "artifact_schema_version": 1,
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": [],
                    "train_schema": [],
                },
                "forecaster": f,
            },
            handle,
        )

    with pytest.raises(TypeError, match="model_params"):
        load_forecaster_artifact(path)


def test_load_forecaster_artifact_rejects_non_dict_runtime_summary(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "invalid-runtime.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "artifact_schema_version": 1,
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                    "train_schema": {
                        "kind": "local",
                        "n_obs": 5,
                        "runtime": [],
                    },
                },
                "forecaster": f,
            },
            handle,
        )

    with pytest.raises(TypeError, match="train_schema.runtime"):
        load_forecaster_artifact(path)


def test_load_forecaster_artifact_rejects_non_dict_runtime_section(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "invalid-runtime-section.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "artifact_schema_version": 1,
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                    "train_schema": {
                        "kind": "local",
                        "n_obs": 5,
                        "runtime": {
                            "family": "torch",
                            "training": [],
                            "prediction": {"mode": "point"},
                        },
                    },
                },
                "forecaster": f,
            },
            handle,
        )

    with pytest.raises(TypeError, match="train_schema.runtime.training"):
        load_forecaster_artifact(path)


def test_load_forecaster_artifact_rejects_non_dict_extra_payload(tmp_path) -> None:
    f = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
    path = tmp_path / "invalid-extra.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "artifact_schema_version": 1,
                "metadata": {
                    "package_version": __version__,
                    "model_key": "naive-last",
                    "model_params": {},
                    "train_schema": {
                        "kind": "local",
                        "n_obs": 5,
                    },
                },
                "extra": [],
                "forecaster": f,
            },
            handle,
        )

    with pytest.raises(TypeError, match="extra"):
        load_forecaster_artifact(path)


def test_save_forecaster_rejects_unfitted_forecaster(tmp_path) -> None:
    path = tmp_path / "unfitted.pkl"

    with pytest.raises(RuntimeError, match="fit"):
        save_forecaster(make_forecaster_object("naive-last"), path)


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_save_and_load_global_forecaster_round_trip(tmp_path) -> None:
    long_df = _small_panel_long_df()
    cutoff = pd.Timestamp("2020-01-16")
    f = make_global_forecaster_object(
        "ridge-step-lag-global",
        lags=5,
        alpha=0.5,
        add_time_features=True,
        id_feature="ordinal",
    ).fit(long_df)
    path = tmp_path / "global.pkl"

    meta = save_forecaster(f, path)
    loaded = load_forecaster(path)

    assert path.exists()
    assert meta["model_key"] == "ridge-step-lag-global"
    assert meta["train_schema"]["n_rows"] == len(long_df)
    pred = loaded.predict(cutoff, 2)
    expected = f.predict(cutoff, 2)
    assert pred.shape == expected.shape
    assert pred["unique_id"].tolist() == expected["unique_id"].tolist()
    assert pred["ds"].tolist() == expected["ds"].tolist()
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), expected["yhat"].to_numpy(dtype=float))
