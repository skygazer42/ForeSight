# ForeSight A-Track Optimization (20 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `src/foresight/` (package + CLI + datasets + eval) more robust and ergonomic for real usage, with better error handling, configurable dataset locations, richer eval outputs, and stronger CI/dev docs — without pulling in heavy ML/DL dependencies.

**Architecture:** Keep `foresight` as a lightweight core library + CLI. Add a small dataset “spec” layer (metadata: paths/columns/date parsing) to remove duplicated schemas. Route all CLI actions through library functions and standardize output formatting and exit codes.

**Tech Stack:** Python 3.10+, `argparse`, `numpy`, `pandas`, `pytest`, `ruff`, GitHub Actions.

---

### Task 1: Add global `--debug` and friendly CLI error handling

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_errors.py`

**Step 1: Write the failing test**

Create `tests/test_cli_errors.py`:
```python
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_unknown_dataset_key_is_friendly_without_debug():
    proc = _run_cli("datasets", "preview", "no_such_dataset")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "Unknown dataset key" in out
    assert "Traceback" not in out


def test_unknown_dataset_key_shows_traceback_with_debug():
    proc = _run_cli("--debug", "datasets", "preview", "no_such_dataset")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "Traceback" in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_errors.py -q`  
Expected: FAIL (current CLI prints raw traceback always; no `--debug` flag).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py`:
- Add global arg `--debug` (store_true).
- Wrap handler execution in `try/except Exception`:
  - If `args.debug`: re-raise.
  - Else: print `ERROR: <message>` to stderr and return `2`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_errors.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py tests/test_cli_errors.py
git commit -m "feat(cli): friendly errors and --debug"
```

---

### Task 2: Improve dataset key errors to list available keys

**Files:**
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_datasets_errors.py`

**Step 1: Write the failing test**

Create `tests/test_datasets_errors.py`:
```python
import pytest

from foresight.datasets.loaders import load_dataset


def test_load_dataset_unknown_key_mentions_available_keys():
    with pytest.raises(KeyError) as ei:
        load_dataset("no_such_dataset")
    msg = str(ei.value)
    assert "Unknown dataset key" in msg
    assert "store_sales" in msg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets_errors.py -q`  
Expected: FAIL (current error message does not suggest available keys).

**Step 3: Write minimal implementation**

In `src/foresight/datasets/loaders.py`:
- When raising on unknown key, include `registry.list_datasets()` in the message.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets_errors.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/datasets/loaders.py tests/test_datasets_errors.py
git commit -m "feat(datasets): better unknown-key error"
```

---

### Task 3: Support `FORESIGHT_DATA_DIR` in dataset loading

**Files:**
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_datasets_data_dir.py`

**Step 1: Write the failing test**

Create `tests/test_datasets_data_dir.py`:
```python
import shutil
from pathlib import Path

from foresight.datasets.loaders import load_store_sales


def test_load_store_sales_supports_env_data_dir(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    shutil.copyfile("data/store_sales.csv", root / "data" / "store_sales.csv")

    monkeypatch.setenv("FORESIGHT_DATA_DIR", str(root))
    df = load_store_sales(nrows=5)
    assert len(df) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets_data_dir.py -q`  
Expected: FAIL (loader ignores env var).

**Step 3: Write minimal implementation**

In `src/foresight/datasets/loaders.py`:
- Add helper `_data_dir_override()` reading `FORESIGHT_DATA_DIR` if set.
- All dataset path resolution should use that base dir when present.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets_data_dir.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/datasets/loaders.py tests/test_datasets_data_dir.py
git commit -m "feat(datasets): FORESIGHT_DATA_DIR support"
```

---

### Task 4: Add global CLI `--data-dir` (overrides env var)

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/eval.py:1`
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_cli_data_dir.py`

**Step 1: Write the failing test**

Create `tests/test_cli_data_dir.py`:
```python
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_data_dir_overrides_env(tmp_path: Path):
    root_env = tmp_path / "env_root"
    (root_env / "data").mkdir(parents=True)
    shutil.copyfile("data/store_sales.csv", root_env / "data" / "store_sales.csv")

    root_flag = tmp_path / "flag_root"
    (root_flag / "data").mkdir(parents=True)
    shutil.copyfile("data/store_sales.csv", root_flag / "data" / "store_sales.csv")

    # Make env root invalid by removing the file after setting it.
    env = {"FORESIGHT_DATA_DIR": str(root_env)}
    (root_env / "data" / "store_sales.csv").unlink()

    proc = _run_cli("--data-dir", str(root_flag), "datasets", "preview", "store_sales", "--nrows", "3", env_extra=env)
    assert proc.returncode == 0
    assert "sales" in (proc.stdout + proc.stderr)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_data_dir.py -q`  
Expected: FAIL (flag not supported; env var currently the only override after Task 3).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py`:
- Add global `--data-dir` argument (string, default "").
- Pass `data_dir=args.data_dir` into dataset/eval calls.

In `src/foresight/eval.py`:
- Add optional `data_dir` argument and forward into `load_dataset`.

In `src/foresight/datasets/loaders.py`:
- Make `data_dir` function arg take precedence over env var.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_data_dir.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py src/foresight/eval.py src/foresight/datasets/loaders.py tests/test_cli_data_dir.py
git commit -m "feat(cli): add --data-dir override"
```

---

### Task 5: Introduce `DatasetSpec` metadata in registry (paths + schemas)

**Files:**
- Modify: `src/foresight/datasets/registry.py:1`
- Modify: `src/foresight/datasets/loaders.py:1`
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_dataset_specs.py`

**Step 1: Write the failing test**

Create `tests/test_dataset_specs.py`:
```python
from foresight.datasets.registry import get_dataset_spec


def test_dataset_spec_contains_expected_fields():
    spec = get_dataset_spec("store_sales")
    assert spec.key == "store_sales"
    assert "sales" in spec.expected_columns
    assert str(spec.rel_path).endswith("data/store_sales.csv")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_specs.py -q`  
Expected: FAIL (no DatasetSpec/get_dataset_spec yet).

**Step 3: Write minimal implementation**

In `src/foresight/datasets/registry.py`:
- Add a small dataclass `DatasetSpec` with:
  - `key: str`
  - `description: str`
  - `rel_path: Path`
  - `expected_columns: set[str]`
  - `parse_dates: list[str]`
- Implement `get_dataset_spec(key)` and keep `list_datasets()` stable.

In `src/foresight/datasets/loaders.py`:
- Use `DatasetSpec.rel_path` to resolve the file path.

In `src/foresight/cli.py`:
- Replace hardcoded `expected_cols` dict in `datasets validate` with registry metadata.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_specs.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/datasets/registry.py src/foresight/datasets/loaders.py src/foresight/cli.py tests/test_dataset_specs.py
git commit -m "feat(datasets): add DatasetSpec metadata"
```

---

### Task 6: Ensure all datasets parse date columns (including `cashflow_data.date`)

**Files:**
- Modify: `src/foresight/datasets/registry.py:1`
- Modify: `src/foresight/datasets/loaders.py:1`
- Test: `tests/test_dataset_parse_dates.py`

**Step 1: Write the failing test**

Create `tests/test_dataset_parse_dates.py`:
```python
import pandas as pd

from foresight.datasets.loaders import load_cashflow_data


def test_cashflow_date_is_parsed_as_datetime():
    df = load_cashflow_data(nrows=20)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_parse_dates.py -q`  
Expected: FAIL (current loader does not parse `date`).

**Step 3: Write minimal implementation**

In `src/foresight/datasets/registry.py`:
- Add `parse_dates=["date"]` for `cashflow_data` (and keep other datasets’ parse_dates correct).

In `src/foresight/datasets/loaders.py`:
- Pass `parse_dates=spec.parse_dates` into `pd.read_csv(...)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_parse_dates.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/datasets/registry.py src/foresight/datasets/loaders.py tests/test_dataset_parse_dates.py
git commit -m "fix(datasets): parse date columns via spec"
```

---

### Task 7: Add `foresight datasets validate --dataset <key>`

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_datasets_validate_single.py`

**Step 1: Write the failing test**

Create `tests/test_cli_datasets_validate_single.py`:
```python
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


def test_datasets_validate_single_dataset():
    proc = _run_cli("datasets", "validate", "--dataset", "catfish", "--nrows", "5")
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "OK catfish" in out
    assert "store_sales" not in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_datasets_validate_single.py -q`  
Expected: FAIL (flag not supported).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py`:
- Add optional argument `--dataset` to `datasets validate`.
- If provided, validate only that dataset key.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_datasets_validate_single.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py tests/test_cli_datasets_validate_single.py
git commit -m "feat(cli): validate a single dataset"
```

---

### Task 8: Add `foresight datasets path <key>`

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/datasets/registry.py:1`
- Test: `tests/test_cli_datasets_path.py`

**Step 1: Write the failing test**

Create `tests/test_cli_datasets_path.py`:
```python
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


def test_datasets_path_prints_a_csv_path():
    proc = _run_cli("datasets", "path", "catfish")
    assert proc.returncode == 0
    p = Path(proc.stdout.strip())
    assert p.suffix == ".csv"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_datasets_path.py -q`  
Expected: FAIL (subcommand does not exist).

**Step 3: Write minimal implementation**

In `src/foresight/datasets/registry.py`:
- Add helper `resolve_dataset_path(key, data_dir: str | Path | None = None) -> Path`.

In `src/foresight/cli.py`:
- Add `datasets path` subcommand calling `resolve_dataset_path`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_datasets_path.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py src/foresight/datasets/registry.py tests/test_cli_datasets_path.py
git commit -m "feat(cli): datasets path"
```

---

### Task 9: Add `foresight datasets list --with-path`

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_datasets_list_with_path.py`

**Step 1: Write the failing test**

Create `tests/test_cli_datasets_list_with_path.py`:
```python
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


def test_datasets_list_with_path_mentions_csv_path():
    proc = _run_cli("datasets", "list", "--with-path")
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert ".csv" in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_datasets_list_with_path.py -q`  
Expected: FAIL (flag not supported).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py`:
- Add `--with-path` to datasets list.
- When set, print `key<TAB>path<TAB>description`, else keep existing output.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_datasets_list_with_path.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py tests/test_cli_datasets_list_with_path.py
git commit -m "feat(cli): datasets list --with-path"
```

---

### Task 10: Add `--format` to eval commands (`json`/`csv`/`md`)

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_eval_format.py`

**Step 1: Write the failing test**

Create `tests/test_cli_eval_format.py`:
```python
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


def test_eval_naive_last_csv_output(tmp_path: Path):
    out = tmp_path / "metrics.csv"
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
        "--format",
        "csv",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0].startswith("model,")
    assert out.exists()


def test_eval_naive_last_md_output(tmp_path: Path):
    out = tmp_path / "metrics.md"
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
        "--format",
        "md",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    assert "| model |" in proc.stdout
    assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_eval_format.py -q`  
Expected: FAIL (eval subcommands don’t accept `--format`).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py`:
- Add `--format` to both `eval naive-last` and `eval seasonal-naive`.
- Use `_emit(..., fmt=args.format)`.

In `_format_payload`:
- Allow dict payload for `csv`/`md` by formatting as a single-row list.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_eval_format.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py tests/test_cli_eval_format.py
git commit -m "feat(cli): eval output formats"
```

---

### Task 11: Include `n_obs` / `n_points` in eval outputs

**Files:**
- Modify: `src/foresight/eval.py:1`
- Test: `tests/test_eval_metadata.py`

**Step 1: Write the failing test**

Create `tests/test_eval_metadata.py`:
```python
from foresight.eval import eval_naive_last


def test_eval_includes_basic_metadata():
    out = eval_naive_last(
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step=3,
        min_train_size=12,
    )
    assert "n_obs" in out
    assert "n_points" in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_metadata.py -q`  
Expected: FAIL (keys absent).

**Step 3: Write minimal implementation**

In `src/foresight/eval.py`:
- Add:
  - `n_obs = len(y)`
  - `n_points = int(res.y_true.size)` (windows*horizon)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_metadata.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/eval.py tests/test_eval_metadata.py
git commit -m "feat(eval): include n_obs and n_points"
```

---

### Task 12: Add per-step metrics arrays in eval output

**Files:**
- Modify: `src/foresight/eval.py:1`
- Test: `tests/test_eval_by_step_metrics.py`

**Step 1: Write the failing test**

Create `tests/test_eval_by_step_metrics.py`:
```python
from foresight.eval import eval_naive_last


def test_eval_includes_metrics_by_step():
    out = eval_naive_last(
        dataset="catfish",
        y_col="Total",
        horizon=4,
        step=4,
        min_train_size=12,
    )
    assert len(out["mae_by_step"]) == 4
    assert len(out["rmse_by_step"]) == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_by_step_metrics.py -q`  
Expected: FAIL (keys absent).

**Step 3: Write minimal implementation**

In `src/foresight/eval.py`:
- Compute per-step metrics across windows:
  - `mae_by_step[i] = mae(res.y_true[:, i], res.y_pred[:, i])`
  - Same for rmse/mape/smape
- Add these lists to payload.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_by_step_metrics.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/eval.py tests/test_eval_by_step_metrics.py
git commit -m "feat(eval): per-step metrics arrays"
```

---

### Task 13: Enforce metric shape matching (raise on mismatch)

**Files:**
- Modify: `src/foresight/metrics.py:1`
- Test: `tests/test_metrics_shape.py`

**Step 1: Write the failing test**

Create `tests/test_metrics_shape.py`:
```python
import numpy as np
import pytest

from foresight.metrics import mae


def test_metrics_raise_on_shape_mismatch():
    with pytest.raises(ValueError):
        mae(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics_shape.py -q`  
Expected: FAIL (broadcasting currently produces a value).

**Step 3: Write minimal implementation**

In `src/foresight/metrics.py`:
- After coercion, check `yt.shape == yp.shape` for all metrics and raise `ValueError` if not.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics_shape.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/metrics.py tests/test_metrics_shape.py
git commit -m "fix(metrics): require matching shapes"
```

---

### Task 14: Add `max_windows` support to `walk_forward`

**Files:**
- Modify: `src/foresight/backtesting.py:1`
- Test: `tests/test_backtesting_max_windows.py`

**Step 1: Write the failing test**

Create `tests/test_backtesting_max_windows.py`:
```python
import numpy as np

from foresight.backtesting import walk_forward
from foresight.models.naive import naive_last


def test_walk_forward_honors_max_windows():
    y = np.arange(50, dtype=float)
    out = walk_forward(y, horizon=3, step=1, min_train_size=10, max_windows=5, forecaster=naive_last)
    assert out.y_true.shape[0] == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtesting_max_windows.py -q`  
Expected: FAIL (walk_forward doesn’t accept `max_windows`).

**Step 3: Write minimal implementation**

In `src/foresight/backtesting.py`:
- Add optional param `max_windows: int | None = None`.
- Validate positive int when provided.
- Break the rolling loop once `len(y_true_list) >= max_windows`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtesting_max_windows.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/backtesting.py tests/test_backtesting_max_windows.py
git commit -m "feat(backtesting): max_windows"
```

---

### Task 15: Thread `--max-windows` through eval + leaderboard CLI

**Files:**
- Modify: `src/foresight/cli.py:1`
- Modify: `src/foresight/eval.py:1`
- Test: `tests/test_cli_max_windows.py`

**Step 1: Write the failing test**

Create `tests/test_cli_max_windows.py`:
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


def test_eval_respects_max_windows():
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
        "1",
        "--min-train-size",
        "12",
        "--max-windows",
        "4",
    )
    payload = json.loads(proc.stdout)
    assert payload["n_windows"] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_max_windows.py -q`  
Expected: FAIL (flag not supported / not wired).

**Step 3: Write minimal implementation**

In `src/foresight/eval.py`:
- Add optional argument `max_windows` and forward into `walk_forward(..., max_windows=max_windows)`.

In `src/foresight/cli.py`:
- Add `--max-windows` int option to eval subcommands and leaderboard naive subcommand.
- Forward into eval calls.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_max_windows.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py src/foresight/eval.py tests/test_cli_max_windows.py
git commit -m "feat(cli): --max-windows for eval/leaderboard"
```

---

### Task 16: Strengthen `datasets validate` to also check date parsing

**Files:**
- Modify: `src/foresight/cli.py:1`
- Test: `tests/test_cli_datasets_validate_dates.py`

**Step 1: Write the failing test**

Create `tests/test_cli_datasets_validate_dates.py`:
```python
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


def test_datasets_validate_checks_parse_dates():
    proc = _run_cli("datasets", "validate", "--dataset", "cashflow_data", "--nrows", "5")
    assert proc.returncode == 0
    assert "OK cashflow_data" in (proc.stdout + proc.stderr)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_datasets_validate_dates.py -q`  
Expected: FAIL until validate checks parse_dates (or until load parses; once parse_dates exists, add explicit check).

**Step 3: Write minimal implementation**

In `src/foresight/cli.py` validate:
- For datasets with `spec.parse_dates`, assert those columns exist and are datetime dtype.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_datasets_validate_dates.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli.py tests/test_cli_datasets_validate_dates.py
git commit -m "feat(cli): validate parse_dates columns"
```

---

### Task 17: Add `docs/DEVELOPMENT.md` (dev setup + common commands)

**Files:**
- Create: `docs/DEVELOPMENT.md`

**Step 1: Write the failing “test” (doc check)**

Skip (documentation task).

**Step 2: Implement**

Create `docs/DEVELOPMENT.md` describing:
- `pip install -e ".[dev]"`
- `ruff check src tests tools`
- `ruff format src tests tools`
- `pytest -q`
- Example CLI usage (`datasets list`, `datasets validate`, `eval naive-last`, `leaderboard naive`)

**Step 3: Commit**

```bash
git add docs/DEVELOPMENT.md
git commit -m "docs: add development guide"
```

---

### Task 18: Update README contributing section for `ruff` (not flake8/black)

**Files:**
- Modify: `README.md:1`

**Steps:**
- Update “贡献指南” to reference `ruff` + `pytest` (matching CI).
- Add a short “Developer” snippet pointing to `docs/DEVELOPMENT.md`.

**Commit:**
```bash
git add README.md
git commit -m "docs: align contributing with ruff/pytest"
```

---

### Task 19: Add CI formatting check (`ruff format --check`)

**Files:**
- Modify: `.github/workflows/ci.yml:1`

**Step 1: Update workflow**

Add a step after lint:
```yaml
      - name: Format
        run: ruff format --check src tests tools
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: check ruff formatting"
```

---

### Task 20: Improve and test `tools/check_no_ipynb.py` (ignore worktrees)

**Files:**
- Modify: `tools/check_no_ipynb.py:1`
- Test: `tests/test_tools_check_no_ipynb.py`

**Step 1: Write the failing test**

Create `tests/test_tools_check_no_ipynb.py`:
```python
from __future__ import annotations

import runpy
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(root / "tools" / "check_no_ipynb.py")


def test_check_no_ipynb_ignores_worktrees(tmp_path: Path):
    (tmp_path / ".worktrees" / "w1").mkdir(parents=True)
    (tmp_path / ".worktrees" / "w1" / "nb.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "real.ipynb").write_text("{}", encoding="utf-8")

    mod = _load_tool()
    find_notebooks = mod["find_notebooks"]
    hits = find_notebooks(tmp_path)
    assert hits == [Path("real.ipynb")]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_check_no_ipynb.py -q`  
Expected: FAIL (tool has no `find_notebooks` and/or doesn’t ignore `.worktrees`).

**Step 3: Write minimal implementation**

In `tools/check_no_ipynb.py`:
- Add `find_notebooks(root: Path) -> list[Path]` returning relative paths.
- Extend ignored dirs set to include `.worktrees` and `worktrees`.
- Keep CLI behavior the same by calling `find_notebooks(_repo_root())`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_check_no_ipynb.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tools/check_no_ipynb.py tests/test_tools_check_no_ipynb.py
git commit -m "chore(tools): ignore worktrees in no-notebooks check"
```

