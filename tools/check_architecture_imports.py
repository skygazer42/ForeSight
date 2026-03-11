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


def _imports_for(path: Path) -> list[tuple[int, str]]:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import(module_name, node)
            if base:
                imports.append((node.lineno, base))
                for alias in node.names:
                    if alias.name != "*":
                        imports.append((node.lineno, f"{base}.{alias.name}"))
    return imports


def _function_defs_for(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs.append((node.lineno, node.name))
    return defs


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


def main() -> int:
    violations: list[str] = []
    _contracts_must_not_import_services(violations)
    _base_must_not_import_registry(violations)
    _cli_must_not_redefine_contract_helpers(violations)

    if violations:
        print("Architecture import check failed:")
        for item in violations:
            print(f"- {item}")
        return 1

    print("Architecture import check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
