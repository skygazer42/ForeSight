import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("key", "overrides", "call_overrides"),
    (
        ("torch-informer-sam-direct", {"warmup_epochs": 1}, {}),
        ("torch-autoformer-regularized-direct", {"warmup_epochs": 1}, {}),
        ("torch-fedformer-lookahead-direct", {"warmup_epochs": 1}, {}),
        ("torch-crossformer-ema-direct", {"warmup_epochs": 1}, {}),
        ("torch-itransformer-swa-direct", {"swa_start_epoch": 0}, {}),
        (
            "torch-timexer-ema-direct",
            {"warmup_epochs": 1, "x_cols": ("x1",)},
            {
                "train_exog": np.sin(np.arange(128, dtype=float) / 11.0).reshape(-1, 1),
                "future_exog": np.sin(np.arange(5, dtype=float) / 11.0).reshape(-1, 1),
            },
        ),
    ),
)
def test_wave106_transformer_refinement_local_strategy_presets_smoke(
    key: str, overrides: dict[str, object], call_overrides: dict[str, object]
) -> None:
    y = np.sin(np.arange(128, dtype=float) / 6.0) + 0.02 * np.arange(128, dtype=float)
    forecaster = make_forecaster(
        key,
        lags=48,
        epochs=1,
        batch_size=16,
        patience=1,
        val_split=0.0,
        device="cpu",
        seed=0,
        **overrides,
    )
    yhat = forecaster(y, 5, **call_overrides)

    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))
