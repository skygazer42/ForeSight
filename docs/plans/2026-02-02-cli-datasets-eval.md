# CLI Datasets & Naive Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the `foresight` CLI with dataset discovery (`datasets list/preview`) and a minimal evaluation command (`eval naive-last`) that prints/writes JSON metrics using the existing backtesting + metrics utilities.

**Architecture:** Keep the CLI thin (`src/foresight/cli.py`) and reuse library functions in `src/foresight/datasets/*`, `src/foresight/backtesting.py`, and `src/foresight/metrics.py`. The CLI should not depend on external ML/DL libraries.

**Tech Stack:** Python 3.10+, `argparse`, `pytest`.

---

### Task 1: Add `foresight datasets list` and `foresight datasets preview`

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_cli_datasets.py`

**Step 1: Write the failing test**

Create `tests/test_cli_datasets.py`:
```python
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_datasets_list_outputs_store_sales():
    proc = _run_cli("datasets", "list")
    assert proc.returncode == 0
    assert "store_sales" in (proc.stdout + proc.stderr)


def test_datasets_preview_outputs_columns():
    proc = _run_cli("datasets", "preview", "store_sales", "--nrows", "50")
    assert proc.returncode == 0
    assert "sales" in (proc.stdout + proc.stderr)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_datasets.py -q`
Expected: FAIL (unknown CLI subcommands)

**Step 3: Write minimal implementation**

- Add `datasets` subcommands to `src/foresight/cli.py`.
- Add `load_dataset(key, **kwargs)` helper in `src/foresight/datasets/loaders.py` so CLI can load by key.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_datasets.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py src/foresight/datasets/loaders.py tests/test_cli_datasets.py
git commit -m "feat: add datasets list/preview CLI"
```

---

### Task 2: Add `foresight eval naive-last` (JSON metrics + optional output file)

**Files:**
- Modify: `src/foresight/cli.py:1`
- Create: `src/foresight/eval.py`
- Test: `tests/test_cli_eval.py`

**Step 1: Write the failing test**

Create `tests/test_cli_eval.py`:
```python
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_eval_naive_last_outputs_json(tmp_path: Path):
    out = tmp_path / "metrics.json"
    proc = _run_cli(
        "eval",
        "naive-last",
        "--dataset",
        "catfish",
        "--y-col",
        "Total",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["model"] == "naive-last"
    assert payload["dataset"] == "catfish"
    assert "mae" in payload
    assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_eval.py -q`
Expected: FAIL (unknown eval subcommand)

**Step 3: Write minimal implementation**

- Add `src/foresight/eval.py` containing a single `eval_naive_last(...)` helper.
- Implement CLI parsing and output JSON to stdout; if `--output` provided, write the same JSON to that path.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_eval.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py src/foresight/eval.py tests/test_cli_eval.py
git commit -m "feat: add naive-last eval CLI"
```

