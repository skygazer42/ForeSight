from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cli_runtime import compact_log_payload, emit_cli_event
from ..contracts.frames import require_long_df as _contracts_require_long_df
from ..contracts.params import normalize_covariate_roles as _normalize_covariate_roles
from ..contracts.params import normalize_static_cols as _normalize_static_cols
from ..cv import cross_validation_predictions_long_df
from ..dataset_long_df_cache import get_or_build_dataset_long_df
from ..long_df_cache import sorted_long_df
from . import model_execution as _model_execution

_SUPPORTED_SCORE_METHODS = {
    "forecast-residual",
    "rolling-mad",
    "rolling-zscore",
}
_SUPPORTED_THRESHOLD_METHODS = {"mad", "quantile", "zscore"}
_MAD_SCALE = 1.4826
_SCORE_EPS = 1e-12


def _require_long_df(long_df: Any) -> pd.DataFrame:
    return _contracts_require_long_df(long_df, require_non_empty=False)


def _normalize_score_method(
    *,
    score_method: str | None,
    model: str | None,
) -> str:
    raw = str(score_method or "").strip().lower()
    if not raw:
        return "forecast-residual" if str(model or "").strip() else "rolling-zscore"
    if raw not in _SUPPORTED_SCORE_METHODS:
        raise ValueError(f"score_method must be one of: {sorted(_SUPPORTED_SCORE_METHODS)}")
    return raw


def _normalize_threshold_method(
    *,
    score_method: str,
    threshold_method: str | None,
) -> str:
    raw = str(threshold_method or "").strip().lower()
    if not raw:
        return "mad" if score_method == "forecast-residual" else "zscore"
    if raw not in _SUPPORTED_THRESHOLD_METHODS:
        raise ValueError(f"threshold_method must be one of: {sorted(_SUPPORTED_THRESHOLD_METHODS)}")
    return raw


def _validate_detection_request(
    *,
    model: str | None,
    model_params: dict[str, Any] | None,
    score_method: str,
    threshold_method: str,
    threshold_k: float,
    threshold_quantile: float,
    window: int,
    min_history: int,
    min_train_size: int | None,
) -> str | None:
    model_name = str(model or "").strip() or None
    if model_name is None and model_params:
        raise ValueError("model_params requires model")

    if int(window) < 2:
        raise ValueError("window must be >= 2")
    if int(min_history) < 2:
        raise ValueError("min_history must be >= 2")
    if int(min_history) > int(window):
        raise ValueError("min_history must be <= window")
    if float(threshold_k) <= 0.0:
        raise ValueError("threshold_k must be > 0")
    if not (0.0 < float(threshold_quantile) < 1.0):
        raise ValueError("threshold_quantile must be in (0,1)")
    if threshold_method == "quantile" and not np.isfinite(float(threshold_quantile)):
        raise ValueError("threshold_quantile must be finite")

    if score_method == "forecast-residual":
        if model_name is None:
            raise ValueError("model is required when score_method='forecast-residual'")
        if min_train_size is None or int(min_train_size) < 1:
            raise ValueError("min_train_size must be >= 1 when score_method='forecast-residual'")

        model_spec = _model_execution.get_model_spec(str(model_name))
        interface = str(model_spec.interface).lower().strip()
        if interface == "multivariate":
            raise ValueError(
                "forecast-residual anomaly detection does not support multivariate models"
            )
        if interface != "global":
            historic_x_cols, future_x_cols = _normalize_covariate_roles(model_params)
            static_cols = _normalize_static_cols(model_params)
            capabilities = dict(model_spec.capabilities)
            if historic_x_cols:
                raise ValueError(
                    "forecast-residual anomaly detection does not yet support historic_x_cols"
                )
            if future_x_cols and not bool(capabilities.get("supports_x_cols", False)):
                raise ValueError(
                    f"Model {model_name!r} does not support x_cols in detect_anomalies_long_df"
                )
            if static_cols:
                if not bool(capabilities.get("supports_static_cols", False)):
                    raise ValueError(
                        f"Model {model_name!r} does not support static_cols in detect_anomalies_long_df"
                    )
                raise ValueError(
                    "forecast-residual anomaly detection does not yet support static_cols for local models"
                )
    return model_name


def _rolling_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    center = float(np.median(arr))
    return float(np.median(np.abs(arr - center)))


def _scaled_absolute_score(
    *,
    residual: pd.Series,
    scale: pd.Series,
    valid_mask: pd.Series,
) -> pd.Series:
    score = pd.Series(np.nan, index=residual.index, dtype=float)
    abs_residual = residual.abs()
    scale_positive = scale > float(_SCORE_EPS)
    score.loc[valid_mask & scale_positive] = (
        abs_residual.loc[valid_mask & scale_positive]
        / scale.loc[valid_mask & scale_positive]
    ).astype(float)
    zero_scale = valid_mask & ~scale_positive
    score.loc[zero_scale & (abs_residual <= float(_SCORE_EPS))] = 0.0
    score.loc[zero_scale & (abs_residual > float(_SCORE_EPS))] = np.inf
    return score


def _rolling_detection_frame_for_group(
    group: pd.DataFrame,
    *,
    score_method: str,
    window: int,
    min_history: int,
) -> pd.DataFrame:
    out = group.loc[:, ["unique_id", "ds", "y"]].copy()
    y = out["y"].astype(float)
    shifted_y = y.shift(1)
    cutoff = out["ds"].shift(1)

    if score_method == "rolling-zscore":
        center = shifted_y.rolling(window=int(window), min_periods=int(min_history)).mean()
        scale = shifted_y.rolling(window=int(window), min_periods=int(min_history)).std(ddof=0)
    else:
        center = shifted_y.rolling(window=int(window), min_periods=int(min_history)).median()
        scale = shifted_y.rolling(window=int(window), min_periods=int(min_history)).apply(
            _rolling_mad,
            raw=True,
        ) * float(_MAD_SCALE)

    residual = y - center
    valid_mask = center.notna()
    score = _scaled_absolute_score(residual=residual, scale=scale, valid_mask=valid_mask)
    out["cutoff"] = cutoff
    out["step"] = 1
    out["yhat"] = center.astype(float)
    out["residual"] = residual.astype(float)
    out["score"] = score.astype(float)
    out["model"] = None
    return out


def _rolling_detection_frame(
    df: pd.DataFrame,
    *,
    score_method: str,
    window: int,
    min_history: int,
) -> pd.DataFrame:
    frames = [
        _rolling_detection_frame_for_group(
            group.reset_index(drop=True),
            score_method=score_method,
            window=int(window),
            min_history=int(min_history),
        )
        for _, group in df.groupby("unique_id", sort=False)
    ]
    return pd.concat(frames, axis=0, ignore_index=True) if frames else df.iloc[0:0].copy()


def _forecast_residual_detection_frame(
    *,
    model: str,
    df: pd.DataFrame,
    model_params: dict[str, Any] | None,
    min_train_size: int,
    step_size: int,
    max_train_size: int | None,
    n_windows: int | None,
) -> pd.DataFrame:
    pred = cross_validation_predictions_long_df(
        model=str(model),
        long_df=df,
        horizon=1,
        step_size=int(step_size),
        min_train_size=int(min_train_size),
        model_params=model_params,
        max_train_size=max_train_size,
        n_windows=n_windows,
    ).copy()
    pred["residual"] = (pred["y"] - pred["yhat"]).astype(float)
    pred["score"] = pred["residual"].abs().astype(float)
    return pred


def _series_threshold_value(
    *,
    score: pd.Series,
    threshold_method: str,
    threshold_k: float,
    threshold_quantile: float,
) -> float:
    finite = score[np.isfinite(score.to_numpy(dtype=float, copy=False))].to_numpy(dtype=float, copy=False)
    if finite.size == 0:
        return float("inf")

    if threshold_method == "zscore":
        return float(threshold_k)
    if threshold_method == "quantile":
        return float(np.quantile(finite, float(threshold_quantile)))

    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    return float(center + float(threshold_k) * float(_MAD_SCALE) * mad)


def _apply_thresholds(
    df: pd.DataFrame,
    *,
    score_method: str,
    threshold_method: str,
    threshold_k: float,
    threshold_quantile: float,
    window_context: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("unique_id", sort=False):
        threshold = _series_threshold_value(
            score=group["score"],
            threshold_method=threshold_method,
            threshold_k=float(threshold_k),
            threshold_quantile=float(threshold_quantile),
        )
        scored = group.copy()
        scored["threshold"] = float(threshold)
        scored["is_anomaly"] = (
            scored["score"].notna() & (scored["score"].astype(float) > float(threshold))
        )
        scored["score_method"] = str(score_method)
        scored["threshold_method"] = str(threshold_method)
        scored["window_context"] = str(window_context)
        frames.append(scored)

    if not frames:
        return df.iloc[0:0].copy()

    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.loc[
        :,
        [
            "unique_id",
            "ds",
            "cutoff",
            "step",
            "y",
            "yhat",
            "residual",
            "score",
            "threshold",
            "is_anomaly",
            "score_method",
            "threshold_method",
            "window_context",
            "model",
        ],
    ]


def detect_anomalies_long_df(
    *,
    long_df: Any,
    model: str | None = None,
    model_params: dict[str, Any] | None = None,
    score_method: str | None = None,
    threshold_method: str | None = None,
    threshold_k: float = 3.0,
    threshold_quantile: float = 0.99,
    window: int = 12,
    min_history: int = 3,
    min_train_size: int | None = None,
    step_size: int = 1,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    df = sorted_long_df(_require_long_df(long_df), reset_index=True)
    if df.empty:
        raise ValueError("long_df is empty")

    score_method_final = _normalize_score_method(score_method=score_method, model=model)
    threshold_method_final = _normalize_threshold_method(
        score_method=score_method_final,
        threshold_method=threshold_method,
    )
    model_name = _validate_detection_request(
        model=model,
        model_params=model_params,
        score_method=score_method_final,
        threshold_method=threshold_method_final,
        threshold_k=float(threshold_k),
        threshold_quantile=float(threshold_quantile),
        window=int(window),
        min_history=int(min_history),
        min_train_size=min_train_size,
    )

    emit_cli_event(
        "DETECT start",
        event="detect_started",
        payload=compact_log_payload(
            model=model_name,
            score_method=score_method_final,
            threshold_method=threshold_method_final,
            rows=int(len(df)),
            n_series=int(df["unique_id"].nunique()),
        ),
    )

    if score_method_final == "forecast-residual":
        scored = _forecast_residual_detection_frame(
            model=str(model_name),
            df=df,
            model_params=model_params,
            min_train_size=int(min_train_size),
            step_size=int(step_size),
            max_train_size=max_train_size,
            n_windows=n_windows,
        )
        window_context = f"min_train_size={int(min_train_size)}|step_size={int(step_size)}"
    else:
        scored = _rolling_detection_frame(
            df,
            score_method=score_method_final,
            window=int(window),
            min_history=int(min_history),
        )
        window_context = f"window={int(window)}|min_history={int(min_history)}"

    out = _apply_thresholds(
        scored,
        score_method=score_method_final,
        threshold_method=threshold_method_final,
        threshold_k=float(threshold_k),
        threshold_quantile=float(threshold_quantile),
        window_context=window_context,
    )
    out.attrs["n_series"] = int(out["unique_id"].nunique()) if not out.empty else 0
    out.attrs["n_points"] = int(len(out))
    out.attrs["n_anomalies"] = int(out["is_anomaly"].sum()) if not out.empty else 0
    out.attrs["score_method"] = str(score_method_final)
    out.attrs["threshold_method"] = str(threshold_method_final)
    emit_cli_event(
        "DETECT done",
        event="detect_completed",
        payload=compact_log_payload(
            model=model_name,
            rows=int(len(out)),
            n_series=int(out.attrs["n_series"]),
            n_anomalies=int(out.attrs["n_anomalies"]),
        ),
    )
    return out


def detect_anomalies(
    *,
    dataset: str,
    y_col: str | None = None,
    model: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    score_method: str | None = None,
    threshold_method: str | None = None,
    threshold_k: float = 3.0,
    threshold_quantile: float = 0.99,
    window: int = 12,
    min_history: int = 3,
    min_train_size: int | None = None,
    step_size: int = 1,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> pd.DataFrame:
    frame_bundle = get_or_build_dataset_long_df(
        dataset=str(dataset),
        y_col=y_col,
        data_dir=data_dir,
        model_params=(model_params if str(model or "").strip() else None),
    )
    out = detect_anomalies_long_df(
        long_df=frame_bundle["long_df"],
        model=model,
        model_params=model_params,
        score_method=score_method,
        threshold_method=threshold_method,
        threshold_k=float(threshold_k),
        threshold_quantile=float(threshold_quantile),
        window=int(window),
        min_history=int(min_history),
        min_train_size=min_train_size,
        step_size=int(step_size),
        max_train_size=max_train_size,
        n_windows=n_windows,
    )
    out.attrs["dataset"] = str(dataset)
    out.attrs["y_col"] = str(frame_bundle["y_col_final"])
    return out


detect_dataset = detect_anomalies
detect_long_df = detect_anomalies_long_df


__all__ = [
    "detect_anomalies",
    "detect_anomalies_long_df",
    "detect_dataset",
    "detect_long_df",
]
