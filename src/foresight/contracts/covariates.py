from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def _normalize_name_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()

    if isinstance(raw, str):
        s = raw.strip()
        return tuple(part.strip() for part in s.split(",") if part.strip()) if s else ()

    if isinstance(raw, Iterable):
        return tuple(str(value).strip() for value in raw if str(value).strip())

    s = str(raw).strip()
    return (s,) if s else ()


def _merge_unique_columns(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for col in group:
            if col not in seen:
                seen.add(col)
                out.append(col)
    return tuple(out)


@dataclass(frozen=True)
class CovariateSpec:
    historic_x_cols: tuple[str, ...] = ()
    future_x_cols: tuple[str, ...] = ()
    static_cols: tuple[str, ...] = ()

    @property
    def all_x_cols(self) -> tuple[str, ...]:
        return _merge_unique_columns(self.historic_x_cols, self.future_x_cols)

    @property
    def all_covariate_cols(self) -> tuple[str, ...]:
        return _merge_unique_columns(self.historic_x_cols, self.future_x_cols, self.static_cols)


def resolve_covariate_roles(
    *,
    x_cols: Iterable[str] | str | None = (),
    historic_x_cols: Iterable[str] | str | None = (),
    future_x_cols: Iterable[str] | str | None = (),
    static_cols: Iterable[str] | str | None = (),
) -> CovariateSpec:
    """
    Normalize covariate-role arguments.

    `x_cols` is a compatibility alias that merges into `future_x_cols`.
    """
    historic = _normalize_name_list(historic_x_cols)
    future = _merge_unique_columns(
        _normalize_name_list(future_x_cols),
        _normalize_name_list(x_cols),
    )
    static = _normalize_name_list(static_cols)
    return CovariateSpec(
        historic_x_cols=historic,
        future_x_cols=future,
        static_cols=static,
    )


def resolve_model_param_covariates(model_params: dict[str, Any] | None) -> CovariateSpec:
    params = dict(model_params or {})
    return resolve_covariate_roles(
        x_cols=params.get("x_cols"),
        historic_x_cols=params.get("historic_x_cols"),
        future_x_cols=params.get("future_x_cols"),
        static_cols=params.get("static_cols"),
    )
