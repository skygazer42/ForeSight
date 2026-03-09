from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .torch_nn import TorchTrainConfig, _normalize_series, _require_torch, _train_loop


def _as_2d_float_array(train: Any) -> np.ndarray:
    if isinstance(train, pd.DataFrame):
        arr = train.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(train, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D multivariate series, got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 target columns for multivariate forecasting, got {arr.shape[1]}"
        )
    return arr


def _make_lagged_xy_multivariate(
    x: np.ndarray, *, lags: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    c = int(x.shape[1])
    lag_count = int(lags)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")
    if n < lag_count + h:
        raise ValueError(f"Need >= lags+horizon rows (lags={lag_count}, horizon={h}), got {n}")

    rows = n - lag_count - h + 1
    X = np.empty((rows, lag_count, c), dtype=float)
    Y = np.empty((rows, h, c), dtype=float)
    for i in range(rows):
        t = i + lag_count
        X[i] = x[t - lag_count : t, :]
        Y[i] = x[t : t + h, :]
    return X, Y


def var_forecast(
    train: Any,
    horizon: int,
    *,
    maxlags: int = 1,
    trend: str = "c",
    ic: str | None = None,
) -> np.ndarray:
    """
    Vector autoregression forecast via statsmodels VAR (optional dependency).

    The input contract is a 2D matrix with shape (n_obs, n_targets), or a pandas
    DataFrame whose columns correspond to target series.
    """
    try:
        from statsmodels.tsa.api import VAR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'var_forecast requires statsmodels. Install with: pip install -e ".[stats]"'
        ) from e

    x = _as_2d_float_array(train)
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    maxlags_int = int(maxlags)
    if maxlags_int <= 0:
        raise ValueError("maxlags must be >= 1")
    if x.shape[0] <= maxlags_int:
        raise ValueError(
            f"var_forecast requires more rows than maxlags; got n_obs={x.shape[0]}, maxlags={maxlags_int}"
        )

    ic_final = None if ic is None or str(ic).strip().lower() in {"", "none", "null"} else str(ic)

    model = VAR(x)
    res = model.fit(maxlags=maxlags_int, ic=ic_final, trend=str(trend))
    if int(res.k_ar) <= 0:
        raise ValueError("VAR selected zero autoregressive lags; increase maxlags or disable ic")

    fc = res.forecast(x[-int(res.k_ar) :], steps=int(horizon))
    out = np.asarray(fc, dtype=float)
    if out.shape != (int(horizon), x.shape[1]):
        raise ValueError(
            f"VAR forecaster must return shape ({int(horizon)}, {x.shape[1]}), got {out.shape}"
        )
    return out


def torch_stid_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 24,
    d_model: int = 64,
    num_blocks: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch STID-style multivariate baseline (lite) on a wide target matrix.
    """
    torch = _require_torch()
    nn = torch.nn

    x = _as_2d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    d = int(d_model)
    blocks = int(num_blocks)
    drop = float(dropout)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if blocks <= 0:
        raise ValueError("num_blocks must be >= 1")
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0,1)")

    x_work = x.astype(float, copy=False)
    mean = np.zeros((int(x.shape[1]),), dtype=float)
    std = np.ones((int(x.shape[1]),), dtype=float)
    if bool(normalize):
        cols: list[np.ndarray] = []
        for j in range(int(x.shape[1])):
            col_scaled, mean_j, std_j = _normalize_series(x_work[:, j])
            cols.append(col_scaled)
            mean[j] = mean_j
            std[j] = std_j
        x_work = np.stack(cols, axis=1)

    X, Y = _make_lagged_xy_multivariate(x_work, lags=lag_count, horizon=h)
    c = int(x.shape[1])

    class _STIDBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(p=drop),
                nn.Linear(2 * d, d),
            )

        def forward(self, xb: Any) -> Any:
            return xb + self.ffn(self.norm(xb))

    class _STIDDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_emb = nn.Parameter(torch.zeros((c, d), dtype=torch.float32))
            self.history_proj = nn.Linear(lag_count, d)
            self.blocks = nn.ModuleList([_STIDBlock() for _ in range(blocks)])
            self.head = nn.Linear(d, h)

        def forward(self, xb: Any) -> Any:  # (B, T, C)
            x_nodes = xb.transpose(1, 2)  # (B, C, T)
            z = self.history_proj(x_nodes) + self.node_emb.unsqueeze(0)
            for blk in self.blocks:
                z = blk(z)
            yhat = self.head(z).transpose(1, 2)  # (B, h, C)
            return yhat

    cfg = TorchTrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        patience=int(patience),
        loss=str(loss),
        val_split=float(val_split),
        grad_clip_norm=float(grad_clip_norm),
        optimizer=str(optimizer),
        momentum=float(momentum),
        scheduler=str(scheduler),
        scheduler_step_size=int(scheduler_step_size),
        scheduler_gamma=float(scheduler_gamma),
        restore_best=bool(restore_best),
    )
    model = _train_loop(_STIDDirect(), X, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:, :].astype(float, copy=False).reshape(1, lag_count, c)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(h, c)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std.reshape(1, c) + mean.reshape(1, c)
    return np.asarray(yhat, dtype=float)
