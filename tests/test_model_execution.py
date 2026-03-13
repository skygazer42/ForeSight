from __future__ import annotations

import numpy as np
import pytest

from foresight.services import model_execution


def test_make_local_forecaster_runner_uses_registered_model() -> None:
    forecaster = model_execution.make_local_forecaster_runner("naive-last")

    yhat = np.asarray(forecaster([1.0, 2.0, 3.0], 2), dtype=float)

    assert yhat.shape == (2,)
    assert np.allclose(yhat, [3.0, 3.0])


def test_call_local_xreg_forecaster_validates_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_make_local_forecaster_runner(
        model: str,
        model_params: dict[str, object] | None = None,
    ):
        def _forecaster(
            train_y: np.ndarray,
            horizon: int,
            *,
            train_exog: np.ndarray,
            future_exog: np.ndarray,
        ) -> np.ndarray:
            return np.asarray([1.0], dtype=float)

        return _forecaster

    monkeypatch.setattr(
        model_execution,
        "make_local_forecaster_runner",
        _fake_make_local_forecaster_runner,
    )

    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        model_execution.call_local_xreg_forecaster(
            model="demo",
            train_y=np.asarray([1.0, 2.0, 3.0], dtype=float),
            horizon=2,
            train_exog=np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            future_exog=np.asarray([[4.0], [5.0]], dtype=float),
            model_params={},
        )
