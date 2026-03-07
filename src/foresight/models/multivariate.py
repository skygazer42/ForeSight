from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
