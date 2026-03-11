from __future__ import annotations

from typing import Any


def require_x_cols_if_needed(
    *,
    model: str,
    capabilities: dict[str, Any],
    x_cols: tuple[str, ...],
    context: str,
) -> None:
    if bool(capabilities.get("requires_future_covariates", False)) and not x_cols:
        raise ValueError(f"Model {model!r} requires future covariates via x_cols in {context}")
