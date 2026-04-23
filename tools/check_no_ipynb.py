#!/usr/bin/env python3

import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/check_no_ipynb.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _is_ignored_dir(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts.intersection(
            {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                ".ipynb_checkpoints",
                ".worktrees",
                "worktrees",
            }
        )
    )


def find_notebooks(root: Path) -> list[Path]:
    hits: list[Path] = []
    for p in root.rglob("*.ipynb"):
        if _is_ignored_dir(p):
            continue
        hits.append(p.relative_to(root))
    return sorted(hits)


def main() -> int:
    root = _repo_root()
    hits = find_notebooks(root)

    if hits:
        print(
            "Found Jupyter notebooks (*.ipynb). This repo enforces 'no notebooks':", file=sys.stderr
        )
        for p in hits:
            print(f" - {p}", file=sys.stderr)
        return 1

    print("OK: no *.ipynb files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
