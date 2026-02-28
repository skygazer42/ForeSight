from __future__ import annotations

import sys
from pathlib import Path

# Allow `import foresight` without requiring `pip install -e .` in every dev workflow.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))
