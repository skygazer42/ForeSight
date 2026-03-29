from __future__ import annotations

import importlib
import importlib.util as _importlib_util
import sys
from pathlib import Path

# Allow `import foresight` without requiring `pip install -e .` in every dev workflow.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

_optional_deps = importlib.import_module("foresight.optional_deps")

_ORIGINAL_FIND_SPEC = _importlib_util.find_spec


def _patched_find_spec(name: str, *args: object, **kwargs: object):
    if str(name).strip() == "torch" and not _optional_deps.is_dependency_available("torch"):
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


_importlib_util.find_spec = _patched_find_spec
