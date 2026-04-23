import os
from collections.abc import Iterable
from pathlib import Path


def _default_candidate_roots(*, env: dict[str, str]) -> list[Path]:
    explicit = str(env.get("FORESIGHT_STORAGE_ROOT", "")).strip()
    user = str(env.get("USER") or env.get("USERNAME") or "user").strip() or "user"

    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    roots.append(Path("/raiddata") / user / "foresight")
    roots.append(Path("/data") / user / "foresight")
    return roots


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _is_creatable_root(path: Path) -> bool:
    parent = _nearest_existing_parent(path)
    return parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)


def prepare_storage_env(
    *,
    env: dict[str, str] | None = None,
    candidate_roots: Iterable[Path] | None = None,
) -> dict[str, str]:
    resolved = dict(os.environ if env is None else env)
    roots = (
        [Path(item).expanduser() for item in candidate_roots]
        if candidate_roots is not None
        else _default_candidate_roots(env=resolved)
    )

    storage_root = next((root for root in roots if _is_creatable_root(root)), None)
    if storage_root is None:
        return resolved

    tmp_dir = storage_root / "tmp"
    pip_cache_dir = storage_root / "cache" / "pip"
    uv_cache_dir = storage_root / "cache" / "uv"
    for path in (tmp_dir, pip_cache_dir, uv_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    tmp_value = str(resolved.get("TMPDIR") or tmp_dir)
    resolved.setdefault("TMPDIR", tmp_value)
    resolved.setdefault("TEMP", tmp_value)
    resolved.setdefault("TMP", tmp_value)
    resolved.setdefault("PIP_CACHE_DIR", str(pip_cache_dir))
    resolved.setdefault("UV_CACHE_DIR", str(uv_cache_dir))
    return resolved
