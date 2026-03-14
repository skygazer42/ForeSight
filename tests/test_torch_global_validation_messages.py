import importlib.util

import numpy as np
import pandas as pd
import pytest

from foresight.models.registry import make_global_forecaster

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _make_long_df() -> tuple[pd.DataFrame, pd.Timestamp, int]:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    return pd.DataFrame(rows), ds[-6], 5


def _assert_global_validation_error(key: str, *, params: dict[str, object], message: str) -> None:
    long_df, cutoff, horizon = _make_long_df()
    forecaster = make_global_forecaster(key, **params, seed=0, patience=2, device="cpu")
    with pytest.raises(ValueError) as exc_info:
        forecaster(long_df, cutoff, horizon)
    assert str(exc_info.value) == message


BASE_PARAMS = {
    "torch-timexer-global": {
        "context_length": 32,
        "x_cols": ("promo",),
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
    },
    "torch-xformer-performer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-patchtst-global": {
        "context_length": 32,
        "patch_len": 8,
        "stride": 4,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-crossformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "segment_len": 8,
        "stride": 8,
        "num_scales": 2,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-pyraformer-global": {
        "context_length": 32,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "segment_len": 8,
        "stride": 8,
        "num_levels": 3,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-tsmixer-global": {
        "context_length": 32,
        "d_model": 32,
        "num_blocks": 2,
        "token_mixing_hidden": 64,
        "channel_mixing_hidden": 64,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
    "torch-fedformer-global": {
        "context_length": 32,
        "d_model": 32,
        "num_layers": 1,
        "ffn_dim": 64,
        "modes": 8,
        "ma_window": 7,
        "dropout": 0.0,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
        "quantiles": "0.1,0.5,0.9",
    },
    "torch-retnet-global": {
        "context_length": 32,
        "d_model": 24,
        "nhead": 4,
        "num_layers": 1,
        "ffn_dim": 48,
        "sample_step": 4,
        "epochs": 1,
        "val_split": 0.0,
        "batch_size": 32,
        "x_cols": ("promo",),
    },
}

GLOBAL_VALIDATION_CASES = [
    ("timexer d_model", "torch-timexer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("timexer nhead", "torch-timexer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "timexer divisibility",
        "torch-timexer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("timexer num_layers", "torch-timexer-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("timexer dropout", "torch-timexer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("xformer d_model", "torch-xformer-performer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("xformer nhead", "torch-xformer-performer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "xformer divisibility",
        "torch-xformer-performer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "xformer num_layers",
        "torch-xformer-performer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "xformer dim_feedforward",
        "torch-xformer-performer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    (
        "xformer dropout",
        "torch-xformer-performer-global",
        {"dropout": 1.0},
        "dropout must be in [0,1)",
    ),
    ("patchtst d_model", "torch-patchtst-global", {"d_model": 0}, "d_model must be >= 1"),
    ("patchtst nhead", "torch-patchtst-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "patchtst divisibility",
        "torch-patchtst-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("patchtst num_layers", "torch-patchtst-global", {"num_layers": 0}, "num_layers must be >= 1"),
    (
        "patchtst dim_feedforward",
        "torch-patchtst-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("patchtst dropout", "torch-patchtst-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    (
        "crossformer d_model",
        "torch-crossformer-global",
        {"d_model": 0},
        "d_model must be >= 1",
    ),
    ("crossformer nhead", "torch-crossformer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "crossformer divisibility",
        "torch-crossformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "crossformer num_layers",
        "torch-crossformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "crossformer dim_feedforward",
        "torch-crossformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("crossformer dropout", "torch-crossformer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    (
        "pyraformer d_model",
        "torch-pyraformer-global",
        {"d_model": 0},
        "d_model must be >= 1",
    ),
    ("pyraformer nhead", "torch-pyraformer-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "pyraformer divisibility",
        "torch-pyraformer-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    (
        "pyraformer num_layers",
        "torch-pyraformer-global",
        {"num_layers": 0},
        "num_layers must be >= 1",
    ),
    (
        "pyraformer dim_feedforward",
        "torch-pyraformer-global",
        {"dim_feedforward": 0},
        "dim_feedforward must be >= 1",
    ),
    ("pyraformer dropout", "torch-pyraformer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("tsmixer d_model", "torch-tsmixer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("tsmixer num_blocks", "torch-tsmixer-global", {"num_blocks": 0}, "num_blocks must be >= 1"),
    (
        "tsmixer token hidden",
        "torch-tsmixer-global",
        {"token_mixing_hidden": 0},
        "token_mixing_hidden must be >= 1",
    ),
    (
        "tsmixer channel hidden",
        "torch-tsmixer-global",
        {"channel_mixing_hidden": 0},
        "channel_mixing_hidden must be >= 1",
    ),
    ("tsmixer dropout", "torch-tsmixer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("fedformer d_model", "torch-fedformer-global", {"d_model": 0}, "d_model must be >= 1"),
    ("fedformer num_layers", "torch-fedformer-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("fedformer ffn_dim", "torch-fedformer-global", {"ffn_dim": 0}, "ffn_dim must be >= 1"),
    ("fedformer modes", "torch-fedformer-global", {"modes": 0}, "modes must be >= 1"),
    ("fedformer ma_window", "torch-fedformer-global", {"ma_window": 0}, "ma_window must be >= 1"),
    ("fedformer dropout", "torch-fedformer-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
    ("retnet d_model", "torch-retnet-global", {"d_model": 0}, "d_model must be >= 1"),
    ("retnet nhead", "torch-retnet-global", {"nhead": 0}, "nhead must be >= 1"),
    (
        "retnet divisibility",
        "torch-retnet-global",
        {"d_model": 5, "nhead": 2},
        "d_model must be divisible by nhead",
    ),
    ("retnet num_layers", "torch-retnet-global", {"num_layers": 0}, "num_layers must be >= 1"),
    ("retnet ffn_dim", "torch-retnet-global", {"ffn_dim": 0}, "ffn_dim must be >= 1"),
    ("retnet dropout", "torch-retnet-global", {"dropout": 1.0}, "dropout must be in [0,1)"),
]


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize(
    ("case_id", "key", "overrides", "message"),
    GLOBAL_VALIDATION_CASES,
    ids=[case[0] for case in GLOBAL_VALIDATION_CASES],
)
def test_torch_global_validation_messages(
    case_id: str,
    key: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    params = dict(BASE_PARAMS[key])
    params.update(overrides)
    _assert_global_validation_error(key, params=params, message=message)
