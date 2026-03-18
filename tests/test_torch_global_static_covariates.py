from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.models import torch_global


def _make_panel_long_df(*, missing_future_static: bool = False) -> tuple[pd.DataFrame, pd.Timestamp]:
    ds = pd.date_range("2020-01-01", periods=14, freq="D")
    cutoff = ds[-4]
    rows: list[dict[str, object]] = []

    for uid, bias, store_size in (("s0", 0.0, 10.0), ("s1", 1.0, 20.0)):
        for i, d in enumerate(ds):
            is_future = d > cutoff
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": np.nan if is_future else float(bias + 0.1 * i),
                    "promo": float(i % 2),
                    "store_size": np.nan if missing_future_static and is_future else store_size,
                }
            )

    return pd.DataFrame(rows), cutoff


def test_build_panel_dataset_broadcasts_static_covariates() -> None:
    long_df, cutoff = _make_panel_long_df()

    x_train, ids_train, y_train, x_pred, ids_pred, *_rest = torch_global._build_panel_dataset(
        long_df,
        cutoff=cutoff,
        horizon=3,
        context_length=4,
        x_cols=("promo",),
        static_cols=("store_size",),
        normalize=False,
        max_train_size=None,
        sample_step=2,
        add_time_features=False,
    )

    assert x_train.shape[2] == 3
    assert y_train.shape[1] == 3
    assert x_pred.shape[2] == 3

    first_s0 = int(np.flatnonzero(ids_train == 0)[0])
    first_s1 = int(np.flatnonzero(ids_train == 1)[0])
    pred_s0 = int(np.flatnonzero(ids_pred == 0)[0])
    pred_s1 = int(np.flatnonzero(ids_pred == 1)[0])

    assert np.allclose(x_train[first_s0, :, 2], 10.0)
    assert np.allclose(x_train[first_s1, :, 2], 20.0)
    assert np.allclose(x_pred[pred_s0, :, 2], 10.0)
    assert np.allclose(x_pred[pred_s1, :, 2], 20.0)


def test_build_panel_dataset_allows_missing_future_static_covariates() -> None:
    long_df, cutoff = _make_panel_long_df(missing_future_static=True)

    x_train, ids_train, _y_train, x_pred, ids_pred, *_rest = torch_global._build_panel_dataset(
        long_df,
        cutoff=cutoff,
        horizon=3,
        context_length=4,
        x_cols=("promo",),
        static_cols=("store_size",),
        normalize=False,
        max_train_size=None,
        sample_step=1,
        add_time_features=False,
    )

    first_s0 = int(np.flatnonzero(ids_train == 0)[0])
    pred_s1 = int(np.flatnonzero(ids_pred == 1)[0])

    assert np.isfinite(x_train).all()
    assert np.isfinite(x_pred).all()
    assert np.allclose(x_train[first_s0, :, 2], 10.0)
    assert np.allclose(x_pred[pred_s1, :, 2], 20.0)


def test_build_panel_dataset_rejects_non_static_series_covariates() -> None:
    long_df, cutoff = _make_panel_long_df()
    mask = long_df["unique_id"].eq("s0") & long_df["ds"].le(cutoff)
    long_df.loc[mask, "store_size"] = np.linspace(10.0, 12.0, int(mask.sum()))

    with pytest.raises(ValueError, match="static_cols.*store_size.*s0"):
        torch_global._build_panel_dataset(
            long_df,
            cutoff=cutoff,
            horizon=3,
            context_length=4,
            x_cols=("promo",),
            static_cols=("store_size",),
            normalize=False,
            max_train_size=None,
            sample_step=1,
            add_time_features=False,
        )
