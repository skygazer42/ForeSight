#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def convert_one(ipynb_path: Path, out_path: Path | None = None) -> Path:
    out_path = out_path or ipynb_path.with_suffix(".py")

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    lines: list[str] = []
    lines.append("# -*- coding: utf-8 -*-\n")
    lines.append(f"# Converted from: {ipynb_path.as_posix()}\n")
    lines.append("\n")

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]

        if cell_type == "markdown":
            lines.append("# %% [markdown]\n")
            md = "".join(source).replace("\r\n", "\n").replace("\r", "\n")
            for raw in md.split("\n"):
                if raw.strip() == "":
                    lines.append("#\n")
                else:
                    lines.append(f"# {raw.rstrip()}\n")
            lines.append("\n")
            continue

        if cell_type == "code":
            lines.append("# %%\n")
            code = "".join(source).replace("\r\n", "\n").replace("\r", "\n")
            for raw in code.split("\n"):
                # Keep the trailing newline behavior consistent.
                lines.append(_comment_magic(raw.rstrip("\n")) + "\n")
            lines.append("\n")
            continue

        # Unknown cell types are preserved as commented blocks.
        lines.append(f"# %% [unknown:{cell_type}]\n")
        blob = "".join(source).replace("\r\n", "\n").replace("\r", "\n")
        for raw in blob.split("\n"):
            if raw.strip() == "":
                lines.append("#\n")
            else:
                lines.append(f"# {raw.rstrip()}\n")
        lines.append("\n")

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
        return 0

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

