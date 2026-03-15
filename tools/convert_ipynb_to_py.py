#!/usr/bin/env python3

"""
Convert Jupyter notebooks (.ipynb) to runnable-ish Python scripts.

We keep the notebook's cell structure using VS Code / Jupytext-style markers:
  - `# %%` for code cells
  - `# %% [markdown]` for markdown cells (markdown is emitted as comments)

Notebook-specific magics (lines starting with `%` or `!`) are commented out so
the output can be executed with plain `python`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _comment_magic(line: str) -> str:
    # Preserve indentation by placing the comment after the original indent.
    indent = line[: len(line) - len(line.lstrip(" \t"))]
    rest = line[len(indent) :]
    if rest.startswith(("%", "!")):
        return f"{indent}# {rest}"
    return line


def _normalized_cell_source(cell: dict[str, object]) -> tuple[str, list[str]]:
    cell_type = str(cell.get("cell_type", ""))
    source = cell.get("source", [])
    if isinstance(source, str):
        return cell_type, [source]
    return cell_type, list(source)


def _normalized_cell_text(source: list[str]) -> list[str]:
    return "".join(source).replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _append_markdown_cell(lines: list[str], source: list[str]) -> None:
    lines.append("# %% [markdown]\n")
    for raw in _normalized_cell_text(source):
        if raw.strip() == "":
            lines.append("#\n")
        else:
            lines.append(f"# {raw.rstrip()}\n")
    lines.append("\n")


def _append_code_cell(lines: list[str], source: list[str]) -> None:
    lines.append("# %%\n")
    for raw in _normalized_cell_text(source):
        # Keep the trailing newline behavior consistent.
        lines.append(_comment_magic(raw.rstrip("\n")) + "\n")
    lines.append("\n")


def _append_unknown_cell(lines: list[str], *, cell_type: str, source: list[str]) -> None:
    lines.append(f"# %% [unknown:{cell_type}]\n")
    for raw in _normalized_cell_text(source):
        if raw.strip() == "":
            lines.append("#\n")
        else:
            lines.append(f"# {raw.rstrip()}\n")
    lines.append("\n")


def _append_cell(lines: list[str], *, cell_type: str, source: list[str]) -> None:
    if cell_type == "markdown":
        _append_markdown_cell(lines, source)
        return
    if cell_type == "code":
        _append_code_cell(lines, source)
        return
    _append_unknown_cell(lines, cell_type=cell_type, source=source)


def convert_one(ipynb_path: Path, out_path: Path | None = None) -> Path:
    out_path = out_path or ipynb_path.with_suffix(".py")

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    lines: list[str] = []
    lines.append("# -*- coding: utf-8 -*-\n")
    lines.append(f"# Converted from: {ipynb_path.as_posix()}\n")
    lines.append("\n")

    for cell in cells:
        cell_type, source = _normalized_cell_source(cell)
        _append_cell(lines, cell_type=cell_type, source=source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .ipynb notebooks to .py scripts.")
    ap.add_argument("paths", nargs="+", help="Notebook paths (files) or directories to scan.")
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Write next to each notebook as <name>.py (default).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional output directory. Keeps relative paths under this dir.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    notebooks: list[Path] = []
    for p in map(Path, args.paths):
        if p.is_dir():
            notebooks.extend(sorted(p.rglob("*.ipynb")))
        elif p.is_file() and p.suffix == ".ipynb":
            notebooks.append(p)

    if not notebooks:
        print("No .ipynb files found.")
    else:
        for nb_path in notebooks:
            if out_dir is None:
                out_path = nb_path.with_suffix(".py")
            else:
                out_path = out_dir / nb_path.with_suffix(".py")
            wrote = convert_one(nb_path, out_path=out_path)
            print(f"Wrote: {wrote}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
