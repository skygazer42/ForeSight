from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import import_module
from typing import Any

_ALIAS_TO_IMPORT_NAME = {
    "catboost": "catboost",
    "darts": "darts",
    "gluonts": "gluonts",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "ml": "sklearn",
    "sktime": "sktime",
    "sklearn": "sklearn",
    "stats": "statsmodels",
    "statsmodels": "statsmodels",
    "torch": "torch",
    "transformers": "transformers",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}

_EXTRA_REQUIREMENTS = {
    "core": (),
    "ml": ("ml",),
    "xgb": ("xgb",),
    "lgbm": ("lgbm",),
    "catboost": ("catboost",),
    "darts": ("darts",),
    "gluonts": ("gluonts",),
    "stats": ("stats",),
    "torch": ("torch",),
    "transformers": ("transformers", "torch"),
    "sktime": ("sktime",),
    "all": (
        "ml",
        "xgb",
        "lgbm",
        "catboost",
        "stats",
        "torch",
        "transformers",
        "sktime",
        "darts",
        "gluonts",
    ),
}

_PREFERRED_EXTRA_FOR_DEPENDENCY = {
    "catboost": "catboost",
    "darts": "darts",
    "gluonts": "gluonts",
    "lgbm": "lgbm",
    "lightgbm": "lgbm",
    "ml": "ml",
    "sktime": "sktime",
    "sklearn": "ml",
    "stats": "stats",
    "statsmodels": "stats",
    "torch": "torch",
    "transformers": "transformers",
    "xgb": "xgb",
    "xgboost": "xgb",
}

_DEPENDENCY_DISPLAY_NAME = {
    "catboost": "catboost",
    "darts": "darts",
    "gluonts": "gluonts",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "ml": "scikit-learn",
    "sktime": "sktime",
    "sklearn": "scikit-learn",
    "stats": "statsmodels",
    "statsmodels": "statsmodels",
    "torch": "PyTorch",
    "transformers": "transformers",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}

_TORCH_REQUIRED_ATTRS = ("nn",)

_find_spec = importlib.util.find_spec
_import_module = import_module


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    import_name: str
    available: bool
    spec_found: bool
    version: str | None
    reason: str | None = None

    @property
    def recommended_extra(self) -> str:
        return preferred_extra_for_dependency(self.name)

    @property
    def package_install_command(self) -> str:
        return package_install_command(self.recommended_extra)

    @property
    def editable_install_command(self) -> str:
        return editable_install_command(self.recommended_extra)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_name": self.import_name,
            "available": self.available,
            "spec_found": self.spec_found,
            "version": self.version,
            "reason": self.reason,
            "recommended_extra": self.recommended_extra,
            "package_install_command": self.package_install_command,
            "editable_install_command": self.editable_install_command,
        }


@dataclass(frozen=True)
class ExtraStatus:
    name: str
    available: bool
    requirements: tuple[str, ...]
    details: dict[str, dict[str, Any]]

    @property
    def package_install_command(self) -> str:
        return package_install_command(self.name)

    @property
    def editable_install_command(self) -> str:
        return editable_install_command(self.name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "requirements": list(self.requirements),
            "details": dict(self.details),
            "package_install_command": self.package_install_command,
            "editable_install_command": self.editable_install_command,
        }


def _normalize_dependency_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _ALIAS_TO_IMPORT_NAME:
        raise KeyError(f"Unknown optional dependency key: {name!r}")
    return key


def _import_name_for_dependency(name: str) -> str:
    normalized = _normalize_dependency_name(name)
    return str(_ALIAS_TO_IMPORT_NAME[normalized])


def _status_from_failure(
    name: str, import_name: str, *, spec_found: bool, reason: str
) -> DependencyStatus:
    return DependencyStatus(
        name=name,
        import_name=import_name,
        available=False,
        spec_found=spec_found,
        version=None,
        reason=reason,
    )


def preferred_extra_for_dependency(name: str) -> str:
    normalized = _normalize_dependency_name(name)
    return str(_PREFERRED_EXTRA_FOR_DEPENDENCY[normalized])


def package_install_command(extra_name: str) -> str:
    extra = str(extra_name).strip().lower()
    if extra == "core":
        return "pip install foresight-ts"
    return f'pip install "foresight-ts[{extra}]"'


def editable_install_command(extra_name: str) -> str:
    extra = str(extra_name).strip().lower()
    if extra == "core":
        return "pip install -e ."
    return f'pip install -e ".[{extra}]"'


def dependency_display_name(name: str) -> str:
    normalized = _normalize_dependency_name(name)
    return str(_DEPENDENCY_DISPLAY_NAME[normalized])


def dependency_install_hint(name: str) -> str:
    normalized = _normalize_dependency_name(name)
    extra = preferred_extra_for_dependency(normalized)
    return f"{package_install_command(extra)} or {editable_install_command(extra)}"


def missing_dependency_message(name: str, *, subject: str | None = None) -> str:
    normalized = _normalize_dependency_name(name)
    display = dependency_display_name(normalized)
    target = str(subject).strip() if subject is not None else display
    return f"{target} requires {display}. Install with: {dependency_install_hint(normalized)}"


def get_dependency_status(name: str) -> DependencyStatus:
    normalized = _normalize_dependency_name(name)
    import_name = _import_name_for_dependency(normalized)
    spec = _find_spec(import_name)
    if spec is None:
        return _status_from_failure(
            normalized,
            import_name,
            spec_found=False,
            reason="module spec not found",
        )

    try:
        module = _import_module(import_name)
    except Exception as exc:  # noqa: BLE001
        return _status_from_failure(
            normalized,
            import_name,
            spec_found=True,
            reason=f"import failed: {type(exc).__name__}: {exc}",
        )

    if normalized == "torch":
        missing = [attr for attr in _TORCH_REQUIRED_ATTRS if not hasattr(module, attr)]
        if missing:
            return _status_from_failure(
                normalized,
                import_name,
                spec_found=True,
                reason=f"missing required attributes: {', '.join(missing)}",
            )

    version = getattr(module, "__version__", None)
    return DependencyStatus(
        name=normalized,
        import_name=import_name,
        available=True,
        spec_found=True,
        version=str(version) if version is not None else None,
        reason=None,
    )


def is_dependency_available(name: str) -> bool:
    return bool(get_dependency_status(name).available)


def require_dependency(name: str, *, install_hint: str | None = None) -> Any:
    normalized = _normalize_dependency_name(name)
    status = get_dependency_status(normalized)
    if not status.available:
        resolved_hint = install_hint or dependency_install_hint(normalized)
        hint = f" Install with: {resolved_hint}" if resolved_hint else ""
        raise ImportError(
            f"Optional dependency {normalized!r} is not available ({status.reason}).{hint}".strip()
        )
    return _import_module(status.import_name)


def get_extra_status(name: str) -> ExtraStatus:
    extra_name = str(name).strip().lower()
    try:
        requirements = tuple(_EXTRA_REQUIREMENTS[extra_name])
    except KeyError as exc:
        raise KeyError(f"Unknown extra name: {name!r}") from exc

    details = {req: get_dependency_status(req).as_dict() for req in requirements}
    available = all(bool(detail["available"]) for detail in details.values())
    if extra_name == "core":
        available = True
    return ExtraStatus(
        name=extra_name,
        available=available,
        requirements=requirements,
        details=details,
    )


def required_extra_name(requires: tuple[str, ...] | list[str]) -> str:
    reqs = [str(item).strip() for item in requires if str(item).strip()]
    return "core" if not reqs else "+".join(reqs)


__all__ = [
    "DependencyStatus",
    "ExtraStatus",
    "dependency_display_name",
    "dependency_install_hint",
    "editable_install_command",
    "get_dependency_status",
    "get_extra_status",
    "is_dependency_available",
    "missing_dependency_message",
    "package_install_command",
    "preferred_extra_for_dependency",
    "require_dependency",
    "required_extra_name",
]
