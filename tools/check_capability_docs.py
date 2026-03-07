#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))

    from foresight.models.registry import get_model_spec, list_models

    readme = (root / "README.md").read_text(encoding="utf-8")
    capability_keys = sorted(
        {key for model_key in list_models() for key in get_model_spec(model_key).capabilities}
    )
    missing = [key for key in capability_keys if f"`{key}`" not in readme]
    if missing:
        print(
            "Capability docs are out of sync with the model registry. "
            f"Missing README entries: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: capability docs mention all registry capability keys: " + ", ".join(capability_keys),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
