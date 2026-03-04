import pathlib
import re

_FORBIDDEN_PATTERNS = [
    # torch.nn modules
    r"\bnn\.RNN\b",
    r"\bnn\.GRU\b",
    r"\bnn\.LSTM\b",
    r"\bnn\.RNNCell\b",
    r"\bnn\.GRUCell\b",
    r"\bnn\.LSTMCell\b",
    # explicit torch.nn.*
    r"\btorch\.nn\.RNN\b",
    r"\btorch\.nn\.GRU\b",
    r"\btorch\.nn\.LSTM\b",
    r"\btorch\.nn\.RNNCell\b",
    r"\btorch\.nn\.GRUCell\b",
    r"\btorch\.nn\.LSTMCell\b",
]


_EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".worktrees",
    "node_modules",
    "dist",
    "build",
}


def _iter_py_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    out: list[pathlib.Path] = []
    for p in root.rglob("*.py"):
        if not p.is_file():
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in p.parts):
            continue
        out.append(p)
    return out


def _scan_files(files: list[pathlib.Path]) -> list[tuple[pathlib.Path, int, str]]:
    hits: list[tuple[pathlib.Path, int, str]] = []
    patterns = [re.compile(p) for p in _FORBIDDEN_PATTERNS]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Some experimental areas may contain non-utf8; skip.
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in patterns:
                if pat.search(line):
                    hits.append((path, lineno, line.strip()))
    return hits


def test_no_builtin_torch_recurrent_modules_in_repo() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    files = _iter_py_files(repo_root)

    hits = _scan_files(files)
    if hits:
        preview = "\n".join([f"{p}:{ln}: {line}" for p, ln, line in hits[:10]])
        raise AssertionError(
            f"Found forbidden torch recurrent modules ({len(hits)} hits). Example(s):\n{preview}"
        )
