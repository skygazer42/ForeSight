from __future__ import annotations

import numpy as np
import pytest


def _require_cuda_torch() -> object:
    torch = pytest.importorskip("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        pytest.skip("torch cuda runtime is not available in this interpreter")
    if not bool(cuda.is_available()):
        pytest.skip("cuda is not available for this torch runtime")
    return torch


@pytest.mark.parametrize(
    "paper",
    [
        "copynet",
        "goru",
    ],
)
def test_torch_rnnpaper_selected_cuda_models_return_forecast_and_checkpoints(
    tmp_path,
    paper: str,
) -> None:
    _require_cuda_torch()

    from foresight.models.torch_rnn_paper_zoo import torch_rnnpaper_direct_forecast

    y = np.arange(1, 80, dtype=float)
    checkpoint_dir = tmp_path / paper / "checkpoints"

    yhat = torch_rnnpaper_direct_forecast(
        y,
        3,
        paper=paper,
        epochs=1,
        device="cuda",
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=True,
        save_last_checkpoint=True,
    )

    assert yhat.shape == (3,)
    assert np.isfinite(yhat).all()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()
