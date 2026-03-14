from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..features.tabular import (
    build_column_lag_features,
    build_lag_derived_features,
    normalize_lag_steps,
)
from ..features.time import build_time_features


def _is_effectively_one(value: float) -> bool:
    return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)


def _validate_tweedie_targets(*, power: float, y_train: np.ndarray) -> None:
    power_f = float(power)
    power_is_one = _is_effectively_one(power_f)
    if power_is_one and np.any(y_train < 0.0):
        raise ValueError(
            "tweedie-step-lag-global with power=1 requires non-negative training targets"
        )
    if (power_f > 1.0) and (not power_is_one) and np.any(y_train <= 0.0):
        raise ValueError(
            "tweedie-step-lag-global with power>1 requires strictly positive training targets"
        )


def _normalize_x_cols(x_cols: Any) -> tuple[str, ...]:
    if x_cols is None:
        return ()
    if isinstance(x_cols, str):
        s = x_cols.strip()
        if not s:
            return ()
        return tuple([p.strip() for p in s.split(",") if p.strip()])
    if isinstance(x_cols, list | tuple):
        out = [str(c).strip() for c in x_cols if str(c).strip()]
        return tuple(out)
    s = str(x_cols).strip()
    return (s,) if s else ()


def _resolve_target_lags(*, lags: Any, target_lags: Any = ()) -> tuple[int, ...]:
    spec = target_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        spec = lags
    return normalize_lag_steps(spec, allow_zero=False, name="target_lags")


def _resolve_historic_x_lags(historic_x_lags: Any) -> tuple[int, ...]:
    spec = historic_x_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return ()
    return normalize_lag_steps(spec, allow_zero=False, name="historic_x_lags")


def _resolve_future_x_lags(*, x_cols: tuple[str, ...], future_x_lags: Any) -> tuple[int, ...]:
    if not x_cols:
        return ()
    spec = future_x_lags
    if isinstance(spec, str) and not spec.strip():
        spec = ()
    if spec in (None, (), []):
        return (0,)
    return normalize_lag_steps(spec, allow_zero=True, name="future_x_lags")


def _extract_step_lag_role_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_lags": params.get("target_lags", ()),
        "historic_x_lags": params.get("historic_x_lags", ()),
        "future_x_lags": params.get("future_x_lags", ()),
    }


def _find_cutoff_index(ds: np.ndarray, cutoff: Any) -> int | None:
    idx = pd.Index(ds).get_indexer([cutoff])[0]
    if int(idx) < 0:
        return None
    return int(idx)


def _validate_long_df(long_df: Any, *, x_cols: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"long_df missing required columns: {sorted(missing)}")
    for c in x_cols:
        if c in {"unique_id", "ds", "y"}:
            raise ValueError(f"x_cols cannot include reserved column name: {c!r}")
        if c not in long_df.columns:
            raise KeyError(f"x_col not found in long_df: {c!r}")
    return long_df


def _step_vector(horizon: int, *, step_scale: str) -> np.ndarray:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    scale = str(step_scale)
    if scale not in {"one_based", "zero_based", "unit"}:
        raise ValueError("step_scale must be one of: one_based, zero_based, unit")

    if scale == "zero_based":
        step = np.arange(h, dtype=float)
    elif scale == "unit":
        step = np.arange(1, h + 1, dtype=float) / float(h)
    else:
        step = np.arange(1, h + 1, dtype=float)
    return step.reshape(-1, 1)


def _make_lagged_xy_multi(
    x: np.ndarray,
    *,
    lags: Any,
    horizon: int,
    start_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.size)
    h = int(horizon)
    lag_steps = normalize_lag_steps(lags, allow_zero=False, name="target_lags")
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if not lag_steps:
        raise ValueError("target_lags must be >= 1")
    t0 = int(max(lag_steps)) if start_t is None else max(int(max(lag_steps)), int(start_t))
    if n < t0 + h:
        # No rows.
        return (
            np.empty((0, int(len(lag_steps))), dtype=float),
            np.empty((0, int(h)), dtype=float),
            np.empty((0,), dtype=int),
        )

    rows = n - t0 - h + 1
    X = np.empty((rows, len(lag_steps)), dtype=float)
    Y = np.empty((rows, h), dtype=float)
    t_idx = np.arange(t0, t0 + rows, dtype=int)
    for i, t in enumerate(t_idx):
        X[i, :] = np.asarray([x[t - lag] for lag in lag_steps], dtype=float)
        for j in range(h):
            Y[i, j] = x[t + j]
    return X, Y, t_idx


def _panel_step_lag_train_xy(
    df: pd.DataFrame,
    cutoff: Any,
    horizon: int,
    *,
    lags: Any,
    target_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    x_cols: tuple[str, ...],
    add_time_features: bool,
    id_feature: str,
    step_scale: str,
    max_train_size: int | None,
    sample_step: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], int, int, int]:
    if max_train_size is not None and int(max_train_size) <= 0:
        raise ValueError("max_train_size must be >= 1 or None")
    if int(sample_step) <= 0:
        raise ValueError("sample_step must be >= 1")

    id_mode = str(id_feature).lower().strip()
    if id_mode not in {"none", "ordinal"}:
        raise ValueError("id_feature must be one of: none, ordinal")

    step = _step_vector(int(horizon), step_scale=str(step_scale))  # (H,1)
    h = int(horizon)
    target_lag_steps = _resolve_target_lags(lags=lags, target_lags=target_lags)
    historic_lag_steps = _resolve_historic_x_lags(historic_x_lags)
    future_lag_steps = _resolve_future_x_lags(x_cols=x_cols, future_x_lags=future_x_lags)

    uid_to_val: dict[str, float] = {}
    next_uid = 0

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for uid, g in df.groupby("unique_id", sort=False):
        uid_s = str(uid)
        if uid_s not in uid_to_val:
            uid_to_val[uid_s] = float(next_uid)
            next_uid += 1

        ds_arr = g["ds"].to_numpy(copy=False)
        y_arr = g["y"].to_numpy(dtype=float, copy=False)
        if y_arr.size == 0:
            continue

        cutoff_idx = _find_cutoff_index(ds_arr, cutoff)
        if cutoff_idx is None:
            continue
        train_end = int(cutoff_idx) + 1
        train_start = 0
        if max_train_size is not None:
            train_start = max(0, train_end - int(max_train_size))

        y_train = y_arr[train_start:train_end]
        min_required_t = max(
            [
                int(max(target_lag_steps)),
                *([int(max(historic_lag_steps))] if historic_lag_steps else []),
                *([int(max(future_lag_steps))] if future_lag_steps else []),
            ]
        )
        if y_train.size < min_required_t + h:
            continue

        X_base, Y, t_idx = _make_lagged_xy_multi(
            y_train,
            lags=target_lag_steps,
            horizon=h,
            start_t=min_required_t,
        )
        if X_base.size == 0:
            continue

        if int(sample_step) > 1:
            X_base = X_base[:: int(sample_step), :]
            Y = Y[:: int(sample_step), :]
            t_idx = t_idx[:: int(sample_step)]

        rows = int(X_base.shape[0])
        derived, _derived_names = build_lag_derived_features(
            X_base,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
        )
        # Expanded dataset across steps.
        X_rep = np.repeat(X_base, repeats=h, axis=0)  # (rows*h, lags)
        derived_rep = np.repeat(derived, repeats=h, axis=0)  # (rows*h, k)
        step_rep = np.tile(step, (rows, 1))  # (rows*h, 1)

        # Target indices (global into g arrays) aligned with Y.reshape(-1)
        target_local = t_idx.reshape(-1, 1) + np.arange(h, dtype=int).reshape(1, -1)
        target_global = int(train_start) + target_local  # (rows, h)
        target_flat = target_global.reshape(-1)
        base_global = int(train_start) + t_idx

        extra_parts: list[np.ndarray] = []

        if id_mode == "ordinal":
            id_val = float(uid_to_val[uid_s])
            extra_parts.append(np.full((rows * h, 1), id_val, dtype=float))

        if bool(add_time_features):
            tf, _names = build_time_features(ds_arr)
            extra_parts.append(tf[target_flat, :].astype(float, copy=False))

        if x_cols:
            ex = g.loc[:, list(x_cols)].to_numpy(dtype=float, copy=False)
            if historic_lag_steps:
                hist_block, _ = build_column_lag_features(
                    ex,
                    t=base_global,
                    lags=historic_lag_steps,
                    column_names=x_cols,
                    prefix="historic_x",
                )
                extra_parts.append(np.repeat(hist_block, repeats=h, axis=0))
            if future_lag_steps:
                futr_block, _ = build_column_lag_features(
                    ex,
                    t=target_flat,
                    lags=future_lag_steps,
                    column_names=x_cols,
                    prefix="future_x",
                    allow_zero=True,
                )
                extra_parts.append(futr_block)

        X_core = np.concatenate([X_rep, derived_rep, step_rep], axis=1)
        if extra_parts:
            extra_mat = np.concatenate(extra_parts, axis=1)
            X_long = np.concatenate([X_core, extra_mat], axis=1)
        else:
            X_long = X_core

        y_long = Y.reshape(-1).astype(float, copy=False)
        if X_long.shape[0] != y_long.shape[0]:
            raise RuntimeError("Internal error: X/y row mismatch")

        if not np.all(np.isfinite(X_long)):
            raise ValueError("Non-finite values in generated features")
        if not np.all(np.isfinite(y_long)):
            raise ValueError("Non-finite values in training targets")

        Xs.append(X_long)
        ys.append(y_long)

    if not Xs:
        raise ValueError("No series had enough data to build a global training set at this cutoff")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    # Feature dimension bookkeeping for factories that want to sanity-check.
    time_dim = 0
    if bool(add_time_features):
        # build_time_features always includes time_idx by default.
        # Derive dimension from a tiny probe.
        probe_tf, _ = build_time_features(pd.Series([df["ds"].iloc[0]]))
        time_dim = int(probe_tf.shape[1])
    exog_dim = int(len(x_cols)) * (int(len(historic_lag_steps)) + int(len(future_lag_steps)))
    id_dim = 1 if str(id_feature).lower().strip() == "ordinal" else 0

    return X_all, y_all, uid_to_val, id_dim, time_dim, exog_dim


def _panel_step_lag_predict_X(
    g: pd.DataFrame,
    *,
    uid_val: float,
    cutoff: Any,
    horizon: int,
    lags: Any,
    target_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    x_cols: tuple[str, ...],
    add_time_features: bool,
    id_feature: str,
    step_scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    ds_arr = g["ds"].to_numpy(copy=False)
    y_arr = g["y"].to_numpy(dtype=float, copy=False)
    x_cols_tup = tuple(x_cols)
    target_lag_steps = _resolve_target_lags(lags=lags, target_lags=target_lags)
    historic_lag_steps = _resolve_historic_x_lags(historic_x_lags)
    future_lag_steps = _resolve_future_x_lags(x_cols=x_cols_tup, future_x_lags=future_x_lags)

    cutoff_idx = _find_cutoff_index(ds_arr, cutoff)
    if cutoff_idx is None:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=object)

    h = int(horizon)
    pred_start = int(cutoff_idx) + 1
    pred_end = pred_start + h

    if pred_end > int(y_arr.size):
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=object)
    required_start = max(
        [
            int(max(target_lag_steps)),
            *([int(max(historic_lag_steps))] if historic_lag_steps else []),
            *([int(max(future_lag_steps))] if future_lag_steps else []),
        ]
    )
    if pred_start < required_start:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=object)

    base = np.asarray([[y_arr[pred_start - lag] for lag in target_lag_steps]], dtype=float)
    X_rep = np.repeat(base, repeats=h, axis=0)
    derived, _derived_names = build_lag_derived_features(
        base,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
    )
    derived_rep = np.repeat(derived, repeats=h, axis=0)
    step = _step_vector(h, step_scale=str(step_scale))

    target_idx = np.arange(pred_start, pred_end, dtype=int)
    extra_parts: list[np.ndarray] = []

    if str(id_feature).lower().strip() == "ordinal":
        extra_parts.append(np.full((h, 1), float(uid_val), dtype=float))

    if bool(add_time_features):
        tf, _ = build_time_features(ds_arr)
        extra_parts.append(tf[target_idx, :].astype(float, copy=False))

    if x_cols_tup:
        ex = g.loc[:, list(x_cols_tup)].to_numpy(dtype=float, copy=False)
        if historic_lag_steps:
            hist_block, _ = build_column_lag_features(
                ex,
                t=[pred_start],
                lags=historic_lag_steps,
                column_names=x_cols_tup,
                prefix="historic_x",
            )
            extra_parts.append(np.repeat(hist_block, repeats=h, axis=0))
        if future_lag_steps:
            futr_block, _ = build_column_lag_features(
                ex,
                t=target_idx,
                lags=future_lag_steps,
                column_names=x_cols_tup,
                prefix="future_x",
                allow_zero=True,
            )
            extra_parts.append(futr_block)

    X_core = np.concatenate([X_rep, derived_rep, step], axis=1)
    if extra_parts:
        extra_mat = np.concatenate(extra_parts, axis=1)
        X_pred = np.concatenate([X_core, extra_mat], axis=1)
    else:
        X_pred = X_core

    if not np.all(np.isfinite(X_pred)):
        raise ValueError("Non-finite values in prediction features")

    ds_out = ds_arr[target_idx]
    return X_pred.astype(float, copy=False), ds_out


def _run_point_global_model(
    long_df: Any,
    cutoff: Any,
    horizon: int,
    *,
    lags: Any,
    target_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    roll_windows: Any,
    roll_stats: Any,
    diff_lags: Any,
    x_cols: tuple[str, ...],
    add_time_features: bool,
    id_feature: str,
    step_scale: str,
    max_train_size: int | None,
    sample_step: int,
    fit_model: Callable[[np.ndarray, np.ndarray], Any],
) -> pd.DataFrame:
    df = _validate_long_df(long_df, x_cols=x_cols)
    df = df.sort_values(["unique_id", "ds"], kind="mergesort")

    X_train, y_train, uid_to_val, _id_dim, _time_dim, _exog_dim = _panel_step_lag_train_xy(
        df,
        cutoff,
        int(horizon),
        lags=lags,
        target_lags=target_lags,
        historic_x_lags=historic_x_lags,
        future_x_lags=future_x_lags,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=bool(add_time_features),
        id_feature=str(id_feature),
        step_scale=str(step_scale),
        max_train_size=max_train_size,
        sample_step=int(sample_step),
    )

    model = fit_model(X_train, y_train)

    rows: list[dict[str, Any]] = []
    for uid, g in df.groupby("unique_id", sort=False):
        uid_s = str(uid)
        uid_val = float(uid_to_val.get(uid_s, 0.0))
        X_pred, ds_out = _panel_step_lag_predict_X(
            g,
            uid_val=uid_val,
            cutoff=cutoff,
            horizon=int(horizon),
            lags=lags,
            target_lags=target_lags,
            historic_x_lags=historic_x_lags,
            future_x_lags=future_x_lags,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
        )
        if X_pred.size == 0:
            continue
        pred = np.asarray(model.predict(X_pred), dtype=float).reshape(-1)
        if pred.shape[0] != int(horizon):
            raise RuntimeError("Unexpected prediction shape")
        for i in range(int(horizon)):
            rows.append({"unique_id": uid_s, "ds": ds_out[i], "yhat": float(pred[i])})

    if not rows:
        raise ValueError("Global model produced 0 predictions at this cutoff")
    return pd.DataFrame(rows)


def ridge_step_lag_global_forecaster(
    *,
    lags: Any = 24,
    target_lags: Any = (),
    historic_x_lags: Any = (),
    future_x_lags: Any = (),
    alpha: float = 1.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn Ridge regression."""
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'ridge-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    alpha_f = float(alpha)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = Ridge(alpha=alpha_f)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags,
            target_lags=target_lags,
            historic_x_lags=historic_x_lags,
            future_x_lags=future_x_lags,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
        )

    return _f


def rf_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn RandomForestRegressor."""
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'rf-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_depth_int = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_estimators_int <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth_int is not None and max_depth_int <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = RandomForestRegressor(
            n_estimators=n_estimators_int,
            max_depth=max_depth_int,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def extra_trees_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn ExtraTreesRegressor."""
    try:
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'extra-trees-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_depth_int = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)
    n_jobs_int = int(n_jobs)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_estimators_int <= 0:
        raise ValueError("n_estimators must be >= 1")
    if max_depth_int is not None and max_depth_int <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = ExtraTreesRegressor(
            n_estimators=n_estimators_int,
            max_depth=max_depth_int,
            random_state=random_state_int,
            n_jobs=n_jobs_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def decision_tree_step_lag_global_forecaster(
    *,
    lags: int = 24,
    max_depth: int | None = 5,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn DecisionTreeRegressor."""
    try:
        from sklearn.tree import DecisionTreeRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'decision-tree-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    max_depth_int = None if max_depth is None else int(max_depth)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if max_depth_int is not None and max_depth_int <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = DecisionTreeRegressor(max_depth=max_depth_int, random_state=random_state_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def bagging_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 200,
    max_samples: float = 0.8,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn BaggingRegressor."""
    try:
        from sklearn.ensemble import BaggingRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'bagging-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    max_samples_f = float(max_samples)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_estimators_int <= 0:
        raise ValueError("n_estimators must be >= 1")
    if not (0.0 < max_samples_f <= 1.0):
        raise ValueError("max_samples must be in (0,1]")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = BaggingRegressor(
            n_estimators=n_estimators_int,
            max_samples=max_samples_f,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def gbrt_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn GradientBoostingRegressor."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'gbrt-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    max_depth_int = int(max_depth)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_estimators_int <= 0:
        raise ValueError("n_estimators must be >= 1")
    if learning_rate_f <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if max_depth_int <= 0:
        raise ValueError("max_depth must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            max_depth=max_depth_int,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def lasso_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 0.001,
    max_iter: int = 5000,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn Lasso."""
    try:
        from sklearn.linear_model import Lasso  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'lasso-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = Lasso(alpha=alpha_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def elasticnet_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn ElasticNet."""
    try:
        from sklearn.linear_model import ElasticNet  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'elasticnet-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    l1_ratio_f = float(l1_ratio)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if not (0.0 <= l1_ratio_f <= 1.0):
        raise ValueError("l1_ratio must be in [0,1]")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = ElasticNet(alpha=alpha_f, l1_ratio=l1_ratio_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def knn_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_neighbors: int = 10,
    weights: str = "distance",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn KNeighborsRegressor."""
    try:
        from sklearn.neighbors import KNeighborsRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'knn-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_neighbors_int = int(n_neighbors)
    weights_s = str(weights).strip().lower()
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_neighbors_int <= 0:
        raise ValueError("n_neighbors must be >= 1")
    if weights_s not in {"uniform", "distance"}:
        raise ValueError("weights must be one of: uniform, distance")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = KNeighborsRegressor(n_neighbors=n_neighbors_int, weights=weights_s)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def kernel_ridge_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 1.0,
    kernel: str = "rbf",
    gamma: float | None = None,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn KernelRidge."""
    try:
        from sklearn.kernel_ridge import KernelRidge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'kernel-ridge-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    kernel_s = str(kernel).strip()
    gamma_opt = None if gamma is None else float(gamma)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if not kernel_s:
        raise ValueError("kernel must be a non-empty string")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = KernelRidge(alpha=alpha_f, kernel=kernel_s, gamma=gamma_opt)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def svr_step_lag_global_forecaster(
    *,
    lags: int = 24,
    C: float = 1.0,
    gamma: str | float = "scale",
    epsilon: float = 0.1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn SVR."""
    try:
        from sklearn.svm import SVR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'svr-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    C_f = float(C)
    epsilon_f = float(epsilon)
    if isinstance(gamma, str):
        gamma_s = gamma.strip().lower()
        if gamma_s in {"scale", "auto"}:
            gamma_v: str | float = gamma_s
        else:
            gamma_v = float(gamma_s)
    else:
        gamma_v = float(gamma)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if C_f <= 0.0:
        raise ValueError("C must be > 0")
    if epsilon_f < 0.0:
        raise ValueError("epsilon must be >= 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = SVR(C=C_f, gamma=gamma_v, epsilon=epsilon_f)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def linear_svr_step_lag_global_forecaster(
    *,
    lags: int = 24,
    C: float = 1.0,
    epsilon: float = 0.0,
    max_iter: int = 5000,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn LinearSVR."""
    try:
        from sklearn.svm import LinearSVR  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'linear-svr-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    C_f = float(C)
    epsilon_f = float(epsilon)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if C_f <= 0.0:
        raise ValueError("C must be > 0")
    if epsilon_f < 0.0:
        raise ValueError("epsilon must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = LinearSVR(
            C=C_f,
            epsilon=epsilon_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def huber_step_lag_global_forecaster(
    *,
    lags: int = 24,
    epsilon: float = 1.35,
    alpha: float = 0.0001,
    max_iter: int = 200,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn HuberRegressor."""
    try:
        from sklearn.linear_model import HuberRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'huber-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    epsilon_f = float(epsilon)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if epsilon_f <= 1.0:
        raise ValueError("epsilon must be > 1.0")
    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = HuberRegressor(epsilon=epsilon_f, alpha=alpha_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def bayesian_ridge_step_lag_global_forecaster(
    *,
    lags: int = 24,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn BayesianRidge."""
    try:
        from sklearn.linear_model import BayesianRidge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'bayesian-ridge-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    max_iter_int = int(max_iter)
    tol_f = float(tol)
    alpha_1_f = float(alpha_1)
    alpha_2_f = float(alpha_2)
    lambda_1_f = float(lambda_1)
    lambda_2_f = float(lambda_2)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")
    if tol_f <= 0.0:
        raise ValueError("tol must be > 0")
    if alpha_1_f <= 0.0 or alpha_2_f <= 0.0 or lambda_1_f <= 0.0 or lambda_2_f <= 0.0:
        raise ValueError("alpha_1, alpha_2, lambda_1, lambda_2 must be > 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = BayesianRidge(
            max_iter=max_iter_int,
            tol=tol_f,
            alpha_1=alpha_1_f,
            alpha_2=alpha_2_f,
            lambda_1=lambda_1_f,
            lambda_2=lambda_2_f,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def ard_step_lag_global_forecaster(
    *,
    lags: int = 24,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    threshold_lambda: float = 10000.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn ARDRegression."""
    try:
        from sklearn.linear_model import ARDRegression  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'ard-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    max_iter_int = int(max_iter)
    tol_f = float(tol)
    alpha_1_f = float(alpha_1)
    alpha_2_f = float(alpha_2)
    lambda_1_f = float(lambda_1)
    lambda_2_f = float(lambda_2)
    threshold_lambda_f = float(threshold_lambda)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")
    if tol_f <= 0.0:
        raise ValueError("tol must be > 0")
    if alpha_1_f <= 0.0 or alpha_2_f <= 0.0 or lambda_1_f <= 0.0 or lambda_2_f <= 0.0:
        raise ValueError("alpha_1, alpha_2, lambda_1, lambda_2 must be > 0")
    if threshold_lambda_f <= 0.0:
        raise ValueError("threshold_lambda must be > 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = ARDRegression(
            max_iter=max_iter_int,
            tol=tol_f,
            alpha_1=alpha_1_f,
            alpha_2=alpha_2_f,
            lambda_1=lambda_1_f,
            lambda_2=lambda_2_f,
            threshold_lambda=threshold_lambda_f,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def omp_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn OrthogonalMatchingPursuit."""
    try:
        from sklearn.linear_model import OrthogonalMatchingPursuit  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'omp-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    nnz_int = None if n_nonzero_coefs is None else int(n_nonzero_coefs)
    tol_f = None if tol is None else float(tol)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if nnz_int is not None and nnz_int <= 0:
        raise ValueError("n_nonzero_coefs must be >= 1 or None")
    if tol_f is not None and tol_f < 0.0:
        raise ValueError("tol must be >= 0 or None")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=nnz_int, tol=tol_f)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def passive_aggressive_step_lag_global_forecaster(
    *,
    lags: int = 24,
    C: float = 1.0,
    loss: str = "epsilon_insensitive",
    epsilon: float = 0.1,
    max_iter: int = 1000,
    random_state: int | None = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn PassiveAggressiveRegressor."""
    try:
        from sklearn.linear_model import PassiveAggressiveRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'passive-aggressive-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    C_f = float(C)
    loss_s = str(loss).strip().lower()
    epsilon_f = float(epsilon)
    max_iter_int = int(max_iter)
    random_state_int = None if random_state is None else int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if C_f <= 0.0:
        raise ValueError("C must be > 0")
    if loss_s not in {"epsilon_insensitive", "squared_epsilon_insensitive"}:
        raise ValueError("loss must be one of: epsilon_insensitive, squared_epsilon_insensitive")
    if epsilon_f < 0.0:
        raise ValueError("epsilon must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = PassiveAggressiveRegressor(
            C=C_f,
            loss=loss_s,
            epsilon=epsilon_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def poisson_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn PoissonRegressor."""
    try:
        from sklearn.linear_model import PoissonRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'poisson-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        if np.any(y_train < 0.0):
            raise ValueError("poisson-step-lag-global requires non-negative training targets")
        model = PoissonRegressor(alpha=alpha_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def gamma_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn GammaRegressor."""
    try:
        from sklearn.linear_model import GammaRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'gamma-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        if np.any(y_train <= 0.0):
            raise ValueError("gamma-step-lag-global requires strictly positive training targets")
        model = GammaRegressor(alpha=alpha_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def tweedie_step_lag_global_forecaster(
    *,
    lags: int = 24,
    power: float = 1.5,
    alpha: float = 1.0,
    max_iter: int = 100,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn TweedieRegressor."""
    try:
        from sklearn.linear_model import TweedieRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'tweedie-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    power_f = float(power)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if 0.0 < power_f < 1.0:
        raise ValueError("power must be <= 0 or >= 1")
    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        _validate_tweedie_targets(power=power_f, y_train=y_train)
        model = TweedieRegressor(power=power_f, alpha=alpha_f, max_iter=max_iter_int)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def quantile_step_lag_global_forecaster(
    *,
    lags: int = 24,
    quantile: float = 0.5,
    alpha: float = 0.0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn QuantileRegressor."""
    try:
        from sklearn.linear_model import QuantileRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'quantile-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    quantile_f = float(quantile)
    alpha_f = float(alpha)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)

    if not (0.0 < quantile_f < 1.0):
        raise ValueError("quantile must be in (0,1)")
    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = QuantileRegressor(quantile=quantile_f, alpha=alpha_f, fit_intercept=True)
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
        )

    return _f


def sgd_step_lag_global_forecaster(
    *,
    lags: int = 24,
    alpha: float = 0.0001,
    penalty: str = "l2",
    max_iter: int = 2000,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn SGDRegressor."""
    try:
        from sklearn.linear_model import SGDRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'sgd-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    penalty_s = str(penalty).strip().lower()
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if penalty_s not in {"l2", "l1", "elasticnet"}:
        raise ValueError("penalty must be one of: l2, l1, elasticnet")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = SGDRegressor(
            alpha=alpha_f,
            penalty=penalty_s,
            max_iter=max_iter_int,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def adaboost_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn AdaBoostRegressor."""
    try:
        from sklearn.ensemble import AdaBoostRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'adaboost-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    n_estimators_int = int(n_estimators)
    learning_rate_f = float(learning_rate)
    random_state_int = int(random_state)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if n_estimators_int <= 0:
        raise ValueError("n_estimators must be >= 1")
    if learning_rate_f <= 0.0:
        raise ValueError("learning_rate must be > 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = AdaBoostRegressor(
            n_estimators=n_estimators_int,
            learning_rate=learning_rate_f,
            random_state=random_state_int,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def mlp_step_lag_global_forecaster(
    *,
    lags: int = 24,
    hidden_layer_sizes: Any = (64, 64),
    alpha: float = 0.0001,
    max_iter: int = 300,
    random_state: int = 0,
    learning_rate_init: float = 0.001,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn MLPRegressor."""
    try:
        from sklearn.neural_network import MLPRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'mlp-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    alpha_f = float(alpha)
    max_iter_int = int(max_iter)
    random_state_int = int(random_state)
    learning_rate_init_f = float(learning_rate_init)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    sizes_raw = hidden_layer_sizes
    if isinstance(sizes_raw, tuple | list):
        sizes = tuple(int(s) for s in sizes_raw)
    else:
        sizes = (int(sizes_raw),)

    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("hidden_layer_sizes must contain positive integers")
    if alpha_f < 0.0:
        raise ValueError("alpha must be >= 0")
    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")
    if learning_rate_init_f <= 0.0:
        raise ValueError("learning_rate_init must be > 0")

    def _fit_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model = MLPRegressor(
            hidden_layer_sizes=sizes,
            alpha=alpha_f,
            max_iter=max_iter_int,
            random_state=random_state_int,
            learning_rate_init=learning_rate_init_f,
        )
        model.fit(X_train, y_train)
        return model

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        return _run_point_global_model(
            long_df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            fit_model=_fit_model,
            **lag_role_params,
        )

    return _f


def hgb_step_lag_global_forecaster(
    *,
    lags: int = 24,
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int | None = 3,
    random_state: int = 0,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    **_params: Any,
) -> Any:
    """Global panel step-lag model with sklearn HistGradientBoostingRegressor."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'hgb-step-lag-global requires scikit-learn. Install with: pip install -e ".[ml]"'
        ) from e

    lags_int = int(lags)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_iter_int = int(max_iter)
    lr_f = float(learning_rate)
    max_depth_int = None if max_depth is None else int(max_depth)
    rs_int = int(random_state)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    lag_role_params = _extract_step_lag_role_params(_params)

    if max_iter_int <= 0:
        raise ValueError("max_iter must be >= 1")
    if lr_f <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if max_depth_int is not None and max_depth_int <= 0:
        raise ValueError("max_depth must be >= 1 or None")

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        df = _validate_long_df(long_df, x_cols=x_cols_tup)
        df = df.sort_values(["unique_id", "ds"], kind="mergesort")

        X_train, y_train, uid_to_val, _id_dim, _time_dim, _exog_dim = _panel_step_lag_train_xy(
            df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=str(step_scale),
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            **lag_role_params,
        )

        model = HistGradientBoostingRegressor(
            max_iter=max_iter_int,
            learning_rate=lr_f,
            max_depth=max_depth_int,
            random_state=rs_int,
        )
        model.fit(X_train, y_train)

        rows: list[dict[str, Any]] = []
        for uid, g in df.groupby("unique_id", sort=False):
            uid_s = str(uid)
            uid_val = float(uid_to_val.get(uid_s, 0.0))
            X_pred, ds_out = _panel_step_lag_predict_X(
                g,
                uid_val=uid_val,
                cutoff=cutoff,
                horizon=int(horizon),
                lags=lags_int,
                roll_windows=roll_windows,
                roll_stats=roll_stats,
                diff_lags=diff_lags,
                x_cols=x_cols_tup,
                add_time_features=bool(add_time_features),
                id_feature=str(id_feature),
                step_scale=str(step_scale),
                **lag_role_params,
            )
            if X_pred.size == 0:
                continue
            pred = model.predict(X_pred)
            pred = np.asarray(pred, dtype=float).reshape(-1)
            if pred.shape[0] != int(horizon):
                raise RuntimeError("Unexpected prediction shape")
            for i in range(int(horizon)):
                rows.append({"unique_id": uid_s, "ds": ds_out[i], "yhat": float(pred[i])})

        if not rows:
            raise ValueError("Global model produced 0 predictions at this cutoff")
        return pd.DataFrame(rows)

    return _f


def _xgb_step_lag_global_forecaster_impl(
    *,
    model_key: str,
    estimator_kind: str,
    booster: str = "gbtree",
    allow_quantiles: bool = False,
    point_objective: str = "reg:squarederror",
    point_objective_params: dict[str, Any] | None = None,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            f'{model_key} requires xgboost. Install with: pip install -e ".[xgb]"'
        ) from e

    lags_int = int(lags)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    step_scale_s = str(step_scale)
    lag_role_params = _extract_step_lag_role_params(_params)

    q_items: list[float] = []
    if quantiles is not None:
        if isinstance(quantiles, list | tuple):
            q_items = [float(q) for q in quantiles]
        elif isinstance(quantiles, str):
            s = quantiles.strip()
            q_items = [] if not s else [float(p.strip()) for p in s.split(",") if p.strip()]
        else:
            q_items = [float(quantiles)]

    q_pcts: list[int] = []
    for q in q_items:
        if not (0.0 < float(q) < 1.0):
            raise ValueError("quantiles must be in (0,1)")
        pct_f = float(q) * 100.0
        pct = int(round(pct_f))
        if abs(pct_f - float(pct)) > 1e-6:
            raise ValueError("quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)")
        if pct <= 0 or pct >= 100:
            raise ValueError("quantiles must be strictly between 0 and 1")
        q_pcts.append(int(pct))

    q_pcts = sorted(set(q_pcts))
    q_vals = tuple([p / 100.0 for p in q_pcts])

    if q_vals and not allow_quantiles:
        raise ValueError(f"{model_key} does not support quantiles")

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        nonlocal q_vals

        df = _validate_long_df(long_df, x_cols=x_cols_tup)
        df = df.sort_values(["unique_id", "ds"], kind="mergesort")

        X_train, y_train, uid_to_val, _id_dim, _time_dim, _exog_dim = _panel_step_lag_train_xy(
            df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=step_scale_s,
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            **lag_role_params,
        )

        def _fit_model(*, objective: str, objective_params: dict[str, Any] | None = None) -> Any:
            if estimator_kind == "xgbrf":
                params: dict[str, Any] = {
                    "objective": str(objective),
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "subsample": float(subsample),
                    "colsample_bytree": float(colsample_bytree),
                    "reg_lambda": float(reg_lambda),
                    "min_child_weight": float(min_child_weight),
                    "gamma": float(gamma),
                    "random_state": int(random_state),
                    "n_jobs": int(n_jobs),
                    "tree_method": str(tree_method),
                    "verbosity": 0,
                }
                if objective_params:
                    params.update(objective_params)
                model = xgb.XGBRFRegressor(**params)
            else:
                params = {
                    "booster": str(booster),
                    "n_estimators": int(n_estimators),
                    "learning_rate": float(learning_rate),
                    # gblinear ignores tree-only params but accepts them through XGBRegressor.
                    "max_depth": 1 if str(booster) == "gblinear" else int(max_depth),
                    "subsample": float(subsample),
                    "colsample_bytree": float(colsample_bytree),
                    "reg_alpha": float(reg_alpha),
                    "reg_lambda": float(reg_lambda),
                    "min_child_weight": float(min_child_weight),
                    "gamma": float(gamma),
                    "random_state": int(random_state),
                    "n_jobs": int(n_jobs),
                    "tree_method": str(tree_method),
                    "verbosity": 0,
                    "objective": str(objective),
                }
                if objective_params:
                    params.update(objective_params)
                model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            return model

        models_by_q: dict[float, Any] = {}
        point_model: Any | None = None

        if q_vals:
            for q in q_vals:
                models_by_q[float(q)] = _fit_model(
                    objective="reg:quantileerror",
                    objective_params={"quantile_alpha": float(q)},
                )
        else:
            point_model = _fit_model(
                objective=str(point_objective),
                objective_params=None
                if point_objective_params is None
                else dict(point_objective_params),
            )

        point_q: float | None = None
        if q_vals:
            if 0.5 in q_vals:
                point_q = 0.5
            else:
                point_q = min(q_vals, key=lambda x: abs(x - 0.5))

        rows: list[dict[str, Any]] = []
        for uid, g in df.groupby("unique_id", sort=False):
            uid_s = str(uid)
            uid_val = float(uid_to_val.get(uid_s, 0.0))
            X_pred, ds_out = _panel_step_lag_predict_X(
                g,
                uid_val=uid_val,
                cutoff=cutoff,
                horizon=int(horizon),
                lags=lags_int,
                roll_windows=roll_windows,
                roll_stats=roll_stats,
                diff_lags=diff_lags,
                x_cols=x_cols_tup,
                add_time_features=bool(add_time_features),
                id_feature=str(id_feature),
                step_scale=step_scale_s,
                **lag_role_params,
            )
            if X_pred.size == 0:
                continue

            out_row: dict[str, Any]
            if q_vals:
                preds_q: dict[float, np.ndarray] = {}
                for q, m in models_by_q.items():
                    p = np.asarray(m.predict(X_pred), dtype=float).reshape(-1)
                    if p.shape[0] != int(horizon):
                        raise RuntimeError("Unexpected prediction shape")
                    preds_q[float(q)] = p

                assert point_q is not None
                yhat_point = preds_q[float(point_q)]

                for i in range(int(horizon)):
                    out_row = {"unique_id": uid_s, "ds": ds_out[i], "yhat": float(yhat_point[i])}
                    for q in q_vals:
                        pct = int(round(float(q) * 100.0))
                        out_row[f"yhat_p{pct}"] = float(preds_q[float(q)][i])
                    rows.append(out_row)
            else:
                assert point_model is not None
                pred = np.asarray(point_model.predict(X_pred), dtype=float).reshape(-1)
                if pred.shape[0] != int(horizon):
                    raise RuntimeError("Unexpected prediction shape")
                for i in range(int(horizon)):
                    rows.append({"unique_id": uid_s, "ds": ds_out[i], "yhat": float(pred[i])})

        if not rows:
            raise ValueError("Global model produced 0 predictions at this cutoff")
        return pd.DataFrame(rows)

    return _f


def xgb_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost (optional quantiles)."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=True,
        point_objective="reg:squarederror",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_dart_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost DART booster."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-dart-step-lag-global",
        estimator_kind="xgb",
        booster="dart",
        allow_quantiles=False,
        point_objective="reg:squarederror",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_linear_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost linear booster (gblinear)."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-linear-step-lag-global",
        estimator_kind="xgb",
        booster="gblinear",
        allow_quantiles=False,
        point_objective="reg:squarederror",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=1,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=1.0,
        gamma=0.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgbrf_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost random forest (XGBRFRegressor)."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgbrf-step-lag-global",
        estimator_kind="xgbrf",
        allow_quantiles=False,
        point_objective="reg:squarederror",
        lags=lags,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_msle_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost squared-log-error objective."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-msle-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:squaredlogerror",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_mae_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost MAE objective."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-mae-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:absoluteerror",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_huber_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    huber_slope: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost pseudo-Huber objective."""
    hs = float(huber_slope)
    if hs <= 0:
        raise ValueError("huber_slope must be > 0")
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-huber-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:pseudohubererror",
        point_objective_params={"huber_slope": hs},
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_poisson_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost Poisson objective."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-poisson-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="count:poisson",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_tweedie_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    tweedie_variance_power: float = 1.5,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost Tweedie objective."""
    tvp = float(tweedie_variance_power)
    if not (1.0 <= tvp < 2.0):
        raise ValueError("tweedie_variance_power must be in [1,2)")
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-tweedie-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:tweedie",
        point_objective_params={"tweedie_variance_power": tvp},
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_gamma_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost Gamma objective."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-gamma-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:gamma",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def xgb_logistic_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 0,
    n_jobs: int = 1,
    tree_method: str = "hist",
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with XGBoost logistic objective."""
    return _xgb_step_lag_global_forecaster_impl(
        model_key="xgb-logistic-step-lag-global",
        estimator_kind="xgb",
        booster="gbtree",
        allow_quantiles=False,
        point_objective="reg:logistic",
        lags=lags,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        roll_windows=roll_windows,
        roll_stats=roll_stats,
        diff_lags=diff_lags,
        x_cols=x_cols,
        add_time_features=add_time_features,
        id_feature=id_feature,
        step_scale=step_scale,
        max_train_size=max_train_size,
        sample_step=sample_step,
        quantiles=quantiles,
        **_params,
    )


def lgbm_step_lag_global_forecaster(
    *,
    lags: int = 24,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_weight: float = 0.001,
    random_state: int = 0,
    n_jobs: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with LightGBM (optional quantiles)."""
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'lgbm-step-lag-global requires lightgbm. Install with: pip install -e ".[lgbm]"'
        ) from e

    import warnings

    lags_int = int(lags)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    step_scale_s = str(step_scale)
    lag_role_params = _extract_step_lag_role_params(_params)

    q_items: list[float] = []
    if quantiles is not None:
        if isinstance(quantiles, list | tuple):
            q_items = [float(q) for q in quantiles]
        elif isinstance(quantiles, str):
            s = quantiles.strip()
            q_items = [] if not s else [float(p.strip()) for p in s.split(",") if p.strip()]
        else:
            q_items = [float(quantiles)]

    q_pcts: list[int] = []
    for q in q_items:
        if not (0.0 < float(q) < 1.0):
            raise ValueError("quantiles must be in (0,1)")
        pct_f = float(q) * 100.0
        pct = int(round(pct_f))
        if abs(pct_f - float(pct)) > 1e-6:
            raise ValueError("quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)")
        if pct <= 0 or pct >= 100:
            raise ValueError("quantiles must be strictly between 0 and 1")
        q_pcts.append(int(pct))
    q_pcts = sorted(set(q_pcts))
    q_vals = tuple([p / 100.0 for p in q_pcts])

    base_params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }

    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMRegressor was fitted with feature names"
                ),
                category=UserWarning,
            )
            return np.asarray(model.predict(X), dtype=float).reshape(-1)

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        nonlocal q_vals

        df = _validate_long_df(long_df, x_cols=x_cols_tup)
        df = df.sort_values(["unique_id", "ds"], kind="mergesort")

        X_train, y_train, uid_to_val, _id_dim, _time_dim, _exog_dim = _panel_step_lag_train_xy(
            df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=step_scale_s,
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            **lag_role_params,
        )

        def _fit_model(*, objective: str, objective_params: dict[str, Any] | None = None) -> Any:
            params = dict(base_params)
            params["objective"] = str(objective)
            if objective_params:
                params.update(objective_params)
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            return model

        models_by_q: dict[float, Any] = {}
        point_model: Any | None = None

        if q_vals:
            for q in q_vals:
                models_by_q[float(q)] = _fit_model(
                    objective="quantile",
                    objective_params={"alpha": float(q)},
                )
        else:
            point_model = _fit_model(objective="regression")

        point_q: float | None = None
        if q_vals:
            if 0.5 in q_vals:
                point_q = 0.5
            else:
                point_q = min(q_vals, key=lambda x: abs(x - 0.5))

        rows: list[dict[str, Any]] = []
        for uid, g in df.groupby("unique_id", sort=False):
            uid_s = str(uid)
            uid_val = float(uid_to_val.get(uid_s, 0.0))
            X_pred, ds_out = _panel_step_lag_predict_X(
                g,
                uid_val=uid_val,
                cutoff=cutoff,
                horizon=int(horizon),
                lags=lags_int,
                roll_windows=roll_windows,
                roll_stats=roll_stats,
                diff_lags=diff_lags,
                x_cols=x_cols_tup,
                add_time_features=bool(add_time_features),
                id_feature=str(id_feature),
                step_scale=step_scale_s,
                **lag_role_params,
            )
            if X_pred.size == 0:
                continue

            out_row: dict[str, Any]
            if q_vals:
                preds_q: dict[float, np.ndarray] = {}
                for q, m in models_by_q.items():
                    p = _predict(m, X_pred)
                    if p.shape[0] != int(horizon):
                        raise RuntimeError("Unexpected prediction shape")
                    preds_q[float(q)] = p

                assert point_q is not None
                yhat_point = preds_q[float(point_q)]

                for i in range(int(horizon)):
                    out_row = {"unique_id": uid_s, "ds": ds_out[i], "yhat": float(yhat_point[i])}
                    for q in q_vals:
                        pct = int(round(float(q) * 100.0))
                        out_row[f"yhat_p{pct}"] = float(preds_q[float(q)][i])
                    rows.append(out_row)
            else:
                assert point_model is not None
                pred = _predict(point_model, X_pred)
                if pred.shape[0] != int(horizon):
                    raise RuntimeError("Unexpected prediction shape")
                for i in range(int(horizon)):
                    rows.append({"unique_id": uid_s, "ds": ds_out[i], "yhat": float(pred[i])})

        if not rows:
            raise ValueError("Global model produced 0 predictions at this cutoff")
        return pd.DataFrame(rows)

    return _f


def catboost_step_lag_global_forecaster(
    *,
    lags: int = 24,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 0,
    thread_count: int = 1,
    roll_windows: Any = (),
    roll_stats: Any = (),
    diff_lags: Any = (),
    x_cols: Any = (),
    add_time_features: bool = True,
    id_feature: str = "ordinal",
    step_scale: str = "one_based",
    max_train_size: int | None = None,
    sample_step: int = 1,
    quantiles: Any = (),
    **_params: Any,
) -> Any:
    """Global panel step-lag model with CatBoost (optional quantiles)."""
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            'catboost-step-lag-global requires catboost. Install with: pip install -e ".[catboost]"'
        ) from e

    lags_int = int(lags)
    x_cols_tup = _normalize_x_cols(x_cols)
    max_train_size_int = None if max_train_size is None else int(max_train_size)
    sample_step_int = int(sample_step)
    step_scale_s = str(step_scale)
    lag_role_params = _extract_step_lag_role_params(_params)

    q_items: list[float] = []
    if quantiles is not None:
        if isinstance(quantiles, list | tuple):
            q_items = [float(q) for q in quantiles]
        elif isinstance(quantiles, str):
            s = quantiles.strip()
            q_items = [] if not s else [float(p.strip()) for p in s.split(",") if p.strip()]
        else:
            q_items = [float(quantiles)]

    q_pcts: list[int] = []
    for q in q_items:
        if not (0.0 < float(q) < 1.0):
            raise ValueError("quantiles must be in (0,1)")
        pct_f = float(q) * 100.0
        pct = int(round(pct_f))
        if abs(pct_f - float(pct)) > 1e-6:
            raise ValueError("quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)")
        if pct <= 0 or pct >= 100:
            raise ValueError("quantiles must be strictly between 0 and 1")
        q_pcts.append(int(pct))
    q_pcts = sorted(set(q_pcts))
    q_vals = tuple([p / 100.0 for p in q_pcts])

    base_params: dict[str, Any] = {
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "random_seed": int(random_seed),
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
    }

    def _f(long_df: Any, cutoff: Any, horizon: int) -> pd.DataFrame:
        nonlocal q_vals

        df = _validate_long_df(long_df, x_cols=x_cols_tup)
        df = df.sort_values(["unique_id", "ds"], kind="mergesort")

        X_train, y_train, uid_to_val, _id_dim, _time_dim, _exog_dim = _panel_step_lag_train_xy(
            df,
            cutoff,
            int(horizon),
            lags=lags_int,
            roll_windows=roll_windows,
            roll_stats=roll_stats,
            diff_lags=diff_lags,
            x_cols=x_cols_tup,
            add_time_features=bool(add_time_features),
            id_feature=str(id_feature),
            step_scale=step_scale_s,
            max_train_size=max_train_size_int,
            sample_step=sample_step_int,
            **lag_role_params,
        )

        def _fit_model(*, loss_function: str) -> Any:
            model = CatBoostRegressor(**dict(base_params, loss_function=str(loss_function)))
            model.fit(X_train, y_train)
            return model

        models_by_q: dict[float, Any] = {}
        point_model: Any | None = None

        if q_vals:
            for q in q_vals:
                models_by_q[float(q)] = _fit_model(loss_function=f"Quantile:alpha={float(q)}")
        else:
            point_model = _fit_model(loss_function="RMSE")

        point_q: float | None = None
        if q_vals:
            if 0.5 in q_vals:
                point_q = 0.5
            else:
                point_q = min(q_vals, key=lambda x: abs(x - 0.5))

        rows: list[dict[str, Any]] = []
        for uid, g in df.groupby("unique_id", sort=False):
            uid_s = str(uid)
            uid_val = float(uid_to_val.get(uid_s, 0.0))
            X_pred, ds_out = _panel_step_lag_predict_X(
                g,
                uid_val=uid_val,
                cutoff=cutoff,
                horizon=int(horizon),
                lags=lags_int,
                roll_windows=roll_windows,
                roll_stats=roll_stats,
                diff_lags=diff_lags,
                x_cols=x_cols_tup,
                add_time_features=bool(add_time_features),
                id_feature=str(id_feature),
                step_scale=step_scale_s,
                **lag_role_params,
            )
            if X_pred.size == 0:
                continue

            out_row: dict[str, Any]
            if q_vals:
                preds_q: dict[float, np.ndarray] = {}
                for q, m in models_by_q.items():
                    p = np.asarray(m.predict(X_pred), dtype=float).reshape(-1)
                    if p.shape[0] != int(horizon):
                        raise RuntimeError("Unexpected prediction shape")
                    preds_q[float(q)] = p

                assert point_q is not None
                yhat_point = preds_q[float(point_q)]

                for i in range(int(horizon)):
                    out_row = {"unique_id": uid_s, "ds": ds_out[i], "yhat": float(yhat_point[i])}
                    for q in q_vals:
                        pct = int(round(float(q) * 100.0))
                        out_row[f"yhat_p{pct}"] = float(preds_q[float(q)][i])
                    rows.append(out_row)
            else:
                assert point_model is not None
                pred = np.asarray(point_model.predict(X_pred), dtype=float).reshape(-1)
                if pred.shape[0] != int(horizon):
                    raise RuntimeError("Unexpected prediction shape")
                for i in range(int(horizon)):
                    rows.append({"unique_id": uid_s, "ds": ds_out[i], "yhat": float(pred[i])})

        if not rows:
            raise ValueError("Global model produced 0 predictions at this cutoff")
        return pd.DataFrame(rows)

    return _f
