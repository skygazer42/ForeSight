from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(
    path: str | Path,
    *,
    nrows: int | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p, nrows=nrows, encoding=str(encoding))


def ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Missing datetime column: {col!r}")
    df[col] = pd.to_datetime(df[col], errors="coerce")


def parse_id_cols(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ()
        return tuple([p.strip() for p in s.split(",") if p.strip()])
    if isinstance(raw, list | tuple):
        out: list[str] = []
        for v in raw:
            s = str(v).strip()
            if s:
                out.append(s)
        return tuple(out)
    return (str(raw).strip(),)
