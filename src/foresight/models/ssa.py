from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(train: Any) -> np.ndarray:
    x = np.asarray(train, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def _trajectory_matrix(x: np.ndarray, *, window_length: int) -> np.ndarray:
    L = int(window_length)
    if L <= 1:
        raise ValueError("window_length must be >= 2")
    if L >= int(x.size):
        raise ValueError("window_length must be <= len(train)-1")

    try:
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(x, window_shape=L)
        # (K, L) -> (L, K)
        return np.asarray(windows, dtype=float).T
    except Exception:  # noqa: BLE001
        K = int(x.size) - L + 1
        X = np.empty((L, K), dtype=float)
        for i in range(K):
            X[:, i] = x[i : i + L]
        return X


def _diagonal_averaging(X: np.ndarray) -> np.ndarray:
    """
    Hankelization via diagonal averaging.

    For a trajectory matrix X of shape (L, K), returns a 1D series of length L+K-1.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    L, K = X.shape
    if L <= 0 or K <= 0:
        raise ValueError("X must be non-empty")

    n = int(L + K - 1)
    out = np.zeros((n,), dtype=float)
    counts = np.zeros((n,), dtype=float)

    # Each row contributes to a contiguous diagonal segment.
    for i in range(int(L)):
        out[i : i + int(K)] += X[i, :]
        counts[i : i + int(K)] += 1.0

    counts = np.where(counts <= 0.0, 1.0, counts)
    return out / counts


def _ssa_recurrent_coefficients(U: np.ndarray, *, eps: float = 1e-10) -> np.ndarray | None:
    """
    Compute SSA recurrent (LRF) coefficients from selected eigenvectors.

    Returns a vector `a` of length L-1 such that:
      y_t = dot(a, [y_{t-L+1}, ..., y_{t-1}])

    When coefficients are ill-conditioned (denominator ~ 0), returns None.
    """
    if U.ndim != 2:
        raise ValueError("U must be 2D")
    L, r = U.shape
    if L < 2 or r <= 0:
        raise ValueError("U must have shape (L>=2, r>=1)")

    pi = U[-1, :].astype(float, copy=False)  # shape: (r,)
    denom = 1.0 - float(np.sum(pi * pi))
    if not np.isfinite(denom) or denom <= float(eps):
        return None

    a = (U[:-1, :] @ pi) / float(denom)  # shape: (L-1,)
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.shape != (int(L - 1),):
        raise RuntimeError("Internal error: invalid SSA coefficient shape")
    if not np.all(np.isfinite(a)):
        return None
    return a


def ssa_forecast(
    train: Any,
    horizon: int,
    *,
    window_length: int = 24,
    rank: int = 5,
) -> np.ndarray:
    """
    Singular Spectrum Analysis (SSA) forecasting via recurrent (LRF) extension.

    Parameters:
      - window_length: embedding window length L (must satisfy 2 <= L <= n-1)
      - rank: truncated SVD rank r (1..min(L, K))

    Notes:
      - This implementation performs SSA denoising by reconstructing the rank-r
        trajectory matrix and then forecasts by extending the reconstructed series.
    """
    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if x.size < 3:
        raise ValueError("ssa_forecast requires at least 3 training points")

    L = int(window_length)
    if L <= 1:
        raise ValueError("window_length must be >= 2")
    if L >= int(x.size):
        raise ValueError("window_length must be <= len(train)-1")

    X = _trajectory_matrix(x, window_length=L)  # (L, K)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    m = int(min(U.shape[1], s.size, Vt.shape[0]))
    if m <= 0:
        raise ValueError("SSA SVD produced empty decomposition")

    r = int(rank)
    if r <= 0:
        raise ValueError("rank must be >= 1")
    r = min(int(r), int(m))

    Ur = U[:, :r]
    sr = s[:r]
    Vtr = Vt[:r, :]
    Xr = (Ur * sr.reshape(1, -1)) @ Vtr

    y_recon = _diagonal_averaging(Xr)
    if y_recon.shape != x.shape:
        # For Hankel trajectory matrices, diagonal averaging should recover length n.
        raise RuntimeError("Internal error: SSA reconstruction length mismatch")

    a = _ssa_recurrent_coefficients(Ur)
    if a is None:
        return np.full((h,), float(y_recon[-1]), dtype=float)

    order = int(L - 1)
    if int(y_recon.size) < order:
        raise ValueError("Not enough history for SSA recurrent forecast")

    ext = np.empty((int(y_recon.size) + h,), dtype=float)
    ext[: y_recon.size] = y_recon.astype(float, copy=False)
    base = int(y_recon.size)
    for k in range(h):
        t = base + k
        past = ext[t - order : t]
        ext[t] = float(np.dot(a, past))

    return ext[base:]

