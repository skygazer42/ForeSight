from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "foresight"

CLI_FORBIDDEN_HELPERS = {
    "_normalize_covariate_roles",
    "_normalize_model_params",
    "_normalize_x_cols",
    "_require_future_df",
    "_require_long_df",
    "_require_x_cols_if_needed",
}
PUBLIC_FACADE_MODULES = (
    "foresight.forecast",
    "foresight.eval_forecast",
    "foresight.cli",
)
REGISTRY_FACADE_FORBIDDEN_PREFIXES = (
    "_build_",
    "_factory_",
    "_resolve_",
    "_runtime_",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(("foresight", *rel.parts))


def _resolve_import(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    package_parts = module_name.split(".")[:-1]
    drop = node.level - 1
    if drop > len(package_parts):
        return ""

    base_parts = package_parts[: len(package_parts) - drop]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _append_import_entries(
    imports: list[tuple[int, str]],
    node: ast.Import,
) -> None:
    imports.extend((node.lineno, alias.name) for alias in node.names)


def _append_import_from_entries(
    imports: list[tuple[int, str]],
    *,
    module_name: str,
    node: ast.ImportFrom,
) -> None:
    base = _resolve_import(module_name, node)
    if not base:
        return
    imports.append((node.lineno, base))
    imports.extend(
        (node.lineno, f"{base}.{alias.name}")
        for alias in node.names
        if alias.name != "*"
    )


def _imports_for(path: Path) -> list[tuple[int, str]]:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _append_import_entries(imports, node)
        elif isinstance(node, ast.ImportFrom):
            _append_import_from_entries(imports, module_name=module_name, node=node)
    return imports


def _function_defs_for(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs.append((node.lineno, node.name))
    return defs


def _exported_names_for(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = ast.literal_eval(node.value)
                if isinstance(value, list | tuple):
                    return [str(item) for item in value]
    return []


def _contracts_must_not_import_services(violations: list[str]) -> None:
    for path in sorted((SRC_ROOT / "contracts").rglob("*.py")):
        for lineno, imported in _imports_for(path):
            if imported == "foresight.services" or imported.startswith("foresight.services."):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: contracts must not import services "
                    f"({imported})"
                )


def _base_must_not_import_registry(violations: list[str]) -> None:
    path = SRC_ROOT / "base.py"
    for lineno, imported in _imports_for(path):
        if imported == "foresight.models.registry" or imported.startswith(
            "foresight.models.registry."
        ):
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: base.py must depend on models.factories, "
                f"not models.registry ({imported})"
            )


def _cli_must_not_redefine_contract_helpers(violations: list[str]) -> None:
    path = SRC_ROOT / "cli.py"
    for lineno, name in _function_defs_for(path):
        if name in CLI_FORBIDDEN_HELPERS:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: cli.py must not define duplicated helper "
                f"{name}()"
            )


def _facades_must_not_export_private_helpers(violations: list[str]) -> None:
    for rel_path in ("forecast.py", "eval_forecast.py"):
        path = SRC_ROOT / rel_path
        for name in _exported_names_for(path):
            if name.startswith("_"):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}: public facades must not export private helper "
                    f"{name!r} in __all__"
                )


def _services_must_not_import_public_facades(violations: list[str]) -> None:
    for path in sorted((SRC_ROOT / "services").rglob("*.py")):
        for lineno, imported in _imports_for(path):
            if imported in PUBLIC_FACADE_MODULES or imported.startswith(
                tuple(f"{name}." for name in PUBLIC_FACADE_MODULES)
            ):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: services must not import public "
                    f"facades ({imported})"
                )


def _registry_must_stay_facade(violations: list[str]) -> None:
    models_root = SRC_ROOT / "models"
    resolution_path = models_root / "resolution.py"
    runtime_path = models_root / "runtime.py"
    registry_path = models_root / "registry.py"

    # Future-facing guard: only enforce once the split modules exist.
    if not resolution_path.exists() or not runtime_path.exists():
        return

    for lineno, name in _function_defs_for(registry_path):
        if name.startswith(REGISTRY_FACADE_FORBIDDEN_PREFIXES):
            violations.append(
                f"{registry_path.relative_to(REPO_ROOT)}:{lineno}: models.registry should stay "
                f"a facade after the resolution/runtime split ({name})"
            )


def _run_architecture_checks() -> list[str]:
    violations: list[str] = []
    _contracts_must_not_import_services(violations)
    _base_must_not_import_registry(violations)
    _cli_must_not_redefine_contract_helpers(violations)
    _facades_must_not_export_private_helpers(violations)
    _services_must_not_import_public_facades(violations)
    _registry_must_stay_facade(violations)
    return violations


def _print_architecture_violations(violations: list[str]) -> None:
    print("Architecture import check failed:")
    for item in violations:
        print(f"- {item}")


def main() -> int:
    violations = _run_architecture_checks()

    if violations:
        _print_architecture_violations(violations)
        return 1

    print("Architecture import check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
