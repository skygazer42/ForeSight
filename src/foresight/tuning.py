from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from .cli_runtime import compact_log_payload, emit_cli_event
from .eval_forecast import eval_model, eval_model_long_df


def _normalize_search_space(search_space: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict of parameter grids")

    out: dict[str, tuple[Any, ...]] = {}
    for key, values in search_space.items():
        key_s = str(key).strip()
        if not key_s:
            raise ValueError("search_space keys must be non-empty strings")
        if isinstance(values, tuple | list):
            items = tuple(values)
        else:
            items = (values,)
        if not items:
            raise ValueError(f"search_space for {key_s!r} must be non-empty")
        out[key_s] = items
    return out


def _resolve_score(payload: dict[str, Any], *, metric: str) -> float:
    if metric not in payload:
        raise KeyError(f"Metric {metric!r} not found in evaluation payload")
    value = payload[metric]
    try:
        return float(value)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"Metric {metric!r} must be numeric, got: {value!r}") from e


def _grid_trials(
    *,
    model: str,
    search_space: dict[str, tuple[Any, ...]],
    metric: str,
    mode: str,
    evaluator: Any,
    base_params: dict[str, Any],
) -> list[dict[str, Any]]:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be one of: min, max")

    keys = sorted(search_space.keys())
    total_trials = 1
    for key in keys:
        total_trials *= int(len(search_space[key]))
    emit_cli_event(
        "TUNE start",
        event="tuning_started",
        payload=compact_log_payload(
            model=str(model),
            metric=str(metric),
            mode=str(mode),
            n_trials=int(total_trials),
        ),
    )
    trials: list[dict[str, Any]] = []
    for trial_idx, values in enumerate(itertools.product(*(search_space[k] for k in keys)), start=1):
        params = dict(base_params)
        params.update(dict(zip(keys, values, strict=True)))
        payload = evaluator(params)
        score = _resolve_score(payload, metric=metric)
        trials.append(
            {
                "model": str(model),
                "metric": str(metric),
                "params": {k: params[k] for k in keys},
                "score": float(score),
            }
        )
        emit_cli_event(
            f"TRIAL {trial_idx}/{total_trials}",
            event="tuning_trial_completed",
            payload=compact_log_payload(
                model=str(model),
                score=float(score),
                params={k: params[k] for k in keys},
            ),
            progress=True,
        )

    reverse = mode == "max"
    trials.sort(
        key=lambda row: (
            -float(row["score"]) if reverse else float(row["score"]),
            json.dumps(row["params"], sort_keys=True, default=str),
        )
    )
    return trials


def tune_model_long_df(
    *,
    model: str,
    long_df: Any,
    horizon: int,
    step: int,
    min_train_size: int,
    search_space: dict[str, Any],
    metric: str = "mae",
    mode: str = "min",
    model_params: dict[str, Any] | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict[str, Any]:
    """
    Deterministic grid search over model parameters for a canonical long-format DataFrame.
    """
    base_params = dict(model_params or {})
    grid = _normalize_search_space(search_space)

    trials = _grid_trials(
        model=str(model),
        search_space=grid,
        metric=str(metric),
        mode=str(mode),
        base_params=base_params,
        evaluator=lambda params: eval_model_long_df(
            model=str(model),
            long_df=long_df,
            horizon=int(horizon),
            step=int(step),
            min_train_size=int(min_train_size),
            model_params=params,
            max_windows=max_windows,
            max_train_size=max_train_size,
        ),
    )

    best = trials[0]
    return {
        "model": str(model),
        "metric": str(metric),
        "mode": str(mode),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_trials": int(len(trials)),
        "best_score": float(best["score"]),
        "best_params": dict(best["params"]),
        "trials": trials,
    }


def tune_model(
    *,
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    search_space: dict[str, Any],
    metric: str = "mae",
    mode: str = "min",
    y_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict[str, Any]:
    """
    Deterministic grid search over model parameters for a registered dataset key.
    """
    base_params = dict(model_params or {})
    grid = _normalize_search_space(search_space)

    trials = _grid_trials(
        model=str(model),
        search_space=grid,
        metric=str(metric),
        mode=str(mode),
        base_params=base_params,
        evaluator=lambda params: eval_model(
            model=str(model),
            dataset=str(dataset),
            horizon=int(horizon),
            step=int(step),
            min_train_size=int(min_train_size),
            y_col=y_col,
            model_params=params,
            data_dir=data_dir,
            max_windows=max_windows,
            max_train_size=max_train_size,
        ),
    )

    best = trials[0]
    return {
        "model": str(model),
        "dataset": str(dataset),
        "y_col": None if y_col is None else str(y_col),
        "metric": str(metric),
        "mode": str(mode),
        "horizon": int(horizon),
        "step": int(step),
        "min_train_size": int(min_train_size),
        "max_windows": None if max_windows is None else int(max_windows),
        "max_train_size": None if max_train_size is None else int(max_train_size),
        "n_trials": int(len(trials)),
        "best_score": float(best["score"]),
        "best_params": dict(best["params"]),
        "trials": trials,
    }
