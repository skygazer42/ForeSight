from __future__ import annotations

from typing import Any


def normalize_model_params(model_params: dict[str, Any] | None) -> dict[str, Any]:
    return dict(model_params or {})


def _normalize_name_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()

    if isinstance(raw, str):
        s = raw.strip()
        return tuple(part.strip() for part in s.split(",") if part.strip()) if s else ()

    if isinstance(raw, list | tuple):
        return tuple(str(value).strip() for value in raw if str(value).strip())

    s = str(raw).strip()
    return (s,) if s else ()


def normalize_x_cols(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        future = _normalize_name_list(raw.get("future_x_cols"))
        legacy = _normalize_name_list(raw.get("x_cols"))
        if legacy:
            future = tuple([*future, *[col for col in legacy if col not in future]])
        return future
    return _normalize_name_list(raw)


def normalize_covariate_roles(
    model_params: dict[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    params = normalize_model_params(model_params)
    future = _normalize_name_list(params.get("future_x_cols"))
    legacy = normalize_x_cols(params)
    if legacy:
        future = tuple([*future, *[col for col in legacy if col not in future]])
    historic = _normalize_name_list(params.get("historic_x_cols"))
    return historic, future


def parse_interval_levels(levels: Any) -> tuple[float, ...]:
    if levels is None:
        return ()

    if isinstance(levels, list | tuple):
        items = list(levels)
    elif isinstance(levels, str):
        s = levels.strip()
        items = [] if not s else [part.strip() for part in s.split(",") if part.strip()]
    else:
        items = [levels]

    out: list[float] = []
    for item in items:
        level = float(item)
        if level >= 1.0:
            level = level / 100.0
        if not (0.0 < level < 1.0):
            raise ValueError("interval_levels must be in (0,1) or percentages like 80,90")
        out.append(level)
    return tuple(sorted(set(out)))


def parse_quantiles(quantiles: Any) -> tuple[float, ...]:
    if quantiles is None:
        return ()

    if isinstance(quantiles, list | tuple):
        items = list(quantiles)
    elif isinstance(quantiles, str):
        s = quantiles.strip()
        items = [] if not s else [part.strip() for part in s.split(",") if part.strip()]
    else:
        items = [quantiles]

    out: list[float] = []
    for item in items:
        q = float(item)
        if q >= 1.0:
            q = q / 100.0
        if not (0.0 < q < 1.0):
            raise ValueError("quantiles must be in (0,1) or percentages like 10,50,90")
        pct = q * 100.0
        if abs(pct - round(pct)) > 1e-9:
            raise ValueError("quantiles must align to integer percentiles (e.g. 0.1,0.5,0.9)")
        out.append(q)
    return tuple(sorted(set(out)))


def required_quantiles_for_interval_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    normalized_levels = parse_interval_levels(levels)
    out: set[float] = {0.5}
    for level in normalized_levels:
        q_lo = (1.0 - float(level)) / 2.0
        q_hi = 1.0 - q_lo
        for q in (q_lo, q_hi):
            pct = q * 100.0
            if abs(pct - round(pct)) > 1e-9:
                raise ValueError(
                    "interval_levels for quantile global models must align to integer percentiles"
                )
            out.add(int(round(pct)) / 100.0)
    return tuple(sorted(out))
