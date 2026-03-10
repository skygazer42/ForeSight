from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FoundationWrapperSpec:
    model_id: str
    backend: str
    source: str
    supports_online_loading: bool
    supports_offline_loading: bool
    inference_mode: str
    notes: str = ""


def foundation_wrapper_not_ready(model_id: str) -> NotImplementedError:
    return NotImplementedError(
        f"Foundation wrapper scaffold for {model_id!r} is present, but the implementation has not "
        "been attached yet."
    )


def normalize_foundation_source(value: Any) -> str:
    text = str(value or "").strip()
    return text
