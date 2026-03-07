from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from . import __version__
from .base import BaseForecaster, BaseGlobalForecaster

LEGACY_ARTIFACT_SCHEMA_VERSION = 0
ARTIFACT_SCHEMA_VERSION = 1
_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (
    LEGACY_ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_SCHEMA_VERSION,
)
_REQUIRED_METADATA_KEYS = (
    "package_version",
    "model_key",
    "model_params",
    "train_schema",
)


def _ensure_supported_forecaster(
    forecaster: Any,
) -> BaseForecaster | BaseGlobalForecaster:
    if isinstance(forecaster, BaseForecaster | BaseGlobalForecaster):
        return forecaster
    raise TypeError(
        "save_forecaster expects a fitted BaseForecaster or BaseGlobalForecaster instance"
    )


def save_forecaster(
    forecaster: Any,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj = _ensure_supported_forecaster(forecaster)
    if not bool(obj.is_fitted):
        raise RuntimeError(
            "save_forecaster requires a fitted forecaster; call fit(...) before saving"
        )
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "package_version": str(__version__),
        "model_key": str(obj.model_key),
        "model_params": dict(obj.model_params),
        "train_schema": obj.train_schema_summary(),
    }
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": metadata,
        "extra": dict(extra or {}),
        "forecaster": obj,
    }

    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return metadata


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(
            f"Serialized forecaster artifact must be a dict, got: {type(payload).__name__}"
        )
    if "metadata" not in payload or "forecaster" not in payload:
        raise KeyError(
            "Serialized forecaster artifact is missing required keys: metadata/forecaster"
        )

    validated = dict(payload)
    if "artifact_schema_version" not in validated:
        validated["artifact_schema_version"] = LEGACY_ARTIFACT_SCHEMA_VERSION

    schema_version = validated["artifact_schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError(
            "Serialized forecaster artifact field 'artifact_schema_version' must be an int"
        )
    if int(schema_version) not in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)
        raise ValueError(
            "Unsupported artifact schema version: "
            f"{schema_version}. Supported versions: {supported}. "
            "Re-save the artifact with the current foresight package."
        )

    metadata = validated["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Serialized forecaster artifact metadata must be a dict, got: {type(metadata).__name__}"
        )
    missing = [key for key in _REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise KeyError(
            "Serialized forecaster artifact metadata is missing required keys: " + ",".join(missing)
        )
    if not isinstance(metadata["package_version"], str):
        raise TypeError(
            "Serialized forecaster artifact metadata field 'package_version' must be a str"
        )
    if not isinstance(metadata["model_key"], str):
        raise TypeError("Serialized forecaster artifact metadata field 'model_key' must be a str")
    if not isinstance(metadata["model_params"], dict):
        raise TypeError(
            "Serialized forecaster artifact metadata field 'model_params' must be a dict"
        )
    if not isinstance(metadata["train_schema"], dict):
        raise TypeError(
            "Serialized forecaster artifact metadata field 'train_schema' must be a dict"
        )
    return validated


def load_forecaster_artifact(path: str | Path) -> dict[str, Any]:
    in_path = Path(path).expanduser()
    with in_path.open("rb") as f:
        payload = pickle.load(f)

    return _validate_payload(payload)


def load_forecaster(path: str | Path) -> BaseForecaster | BaseGlobalForecaster:
    payload = load_forecaster_artifact(path)
    return _ensure_supported_forecaster(payload["forecaster"])
