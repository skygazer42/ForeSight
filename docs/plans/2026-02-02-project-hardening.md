# ForeSight Project Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn ForeSight into a runnable, testable Python project with a minimal CLI, dataset registry, core metrics/backtesting utilities, and CI checks (including “no *.ipynb” enforcement).

**Architecture:** Add a small first-party package under `src/foresight/` that provides dataset loading, metrics, and backtesting. Keep existing model code/scripts as-is for now; the new package is a stable foundation for future unification.

**Tech Stack:** Python 3.10+, `pytest` (tests), `ruff` (lint, optional), GitHub Actions (CI).

---

### Task 1: Add first-party Python package skeleton + minimal CLI

**Files:**
- Create: `pyproject.toml`
- Create: `src/foresight/__init__.py`
- Create: `src/foresight/__main__.py`
- Create: `src/foresight/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Create `tests/test_cli.py`:
```python
import subprocess
import sys


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "foresight", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "ForeSight" in (proc.stdout + proc.stderr)
```

**Step 2: Run test to verify it fails**

Run: `pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'foresight'`

**Step 3: Write minimal implementation**

- Add `src/foresight/__init__.py` exposing `__version__`.
- Add `src/foresight/__main__.py` calling `foresight.cli.main()`.
- Add `src/foresight/cli.py` using `argparse` and printing a banner containing `ForeSight`.
- Add `pyproject.toml` so the project is installable (`pip install -e .`).

**Step 4: Run test to verify it passes**

Run: `pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/foresight tests/test_cli.py
git commit -m "feat: add foresight package skeleton and CLI"
```

---

### Task 2: Add dataset registry + loaders for existing local datasets

**Files:**
- Create: `src/foresight/datasets/__init__.py`
- Create: `src/foresight/datasets/registry.py`
- Create: `src/foresight/datasets/loaders.py`
- Test: `tests/test_datasets.py`

**Step 1: Write the failing test**

Create `tests/test_datasets.py`:
```python
from foresight.datasets.registry import list_datasets
from foresight.datasets.loaders import load_store_sales


def test_list_datasets_contains_store_sales():
    assert "store_sales" in list_datasets()


def test_load_store_sales_smoke():
    df = load_store_sales()
    assert {"store", "dept", "week", "sales"}.issubset(df.columns)
    assert len(df) > 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q`
Expected: FAIL with import error (`foresight.datasets...` missing)

**Step 3: Write minimal implementation**

- Implement `foresight.datasets.registry.list_datasets()` returning available dataset keys.
- Implement `foresight.datasets.loaders`:
  - `load_store_sales()` reading `data/store_sales.csv` with `parse_dates=["week"]`
  - `load_promotion_data()` reading `data/promotion_data.csv` with `parse_dates=["week"]`
  - `load_cashflow_data()` reading `data/cashflow_data.csv` with date parsing if present
  - Optional: `load_catfish()` / `load_ice_cream_interest()` from `statistics time series/*.csv`
- Use `pathlib.Path` anchored to the repo root (not cwd).

**Step 4: Run test to verify it passes**

Run: `pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/datasets tests/test_datasets.py
git commit -m "feat: add dataset registry and loaders"
```

---

### Task 3: Add core metrics + simple walk-forward backtesting utilities

**Files:**
- Create: `src/foresight/metrics.py`
- Create: `src/foresight/backtesting.py`
- Create: `src/foresight/models/__init__.py`
- Create: `src/foresight/models/naive.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_backtesting.py`

**Step 1: Write the failing tests**

Create `tests/test_metrics.py`:
```python
import numpy as np
from foresight.metrics import mae, rmse, mape, smape


def test_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 5.0])
    assert mae(y_true, y_pred) == 1.0
    assert round(rmse(y_true, y_pred), 6) == round(((0**2 + 1**2 + 2**2) / 3) ** 0.5, 6)
    assert mape(y_true, y_pred) > 0
    assert smape(y_true, y_pred) > 0
```

Create `tests/test_backtesting.py`:
```python
import numpy as np
from foresight.backtesting import walk_forward
from foresight.models.naive import naive_last


def test_walk_forward_shapes():
    y = np.arange(20, dtype=float)
    out = walk_forward(y, horizon=3, step=3, min_train_size=10, forecaster=naive_last)
    assert out.y_true.shape == out.y_pred.shape
    assert out.y_true.ndim == 2  # (n_windows, horizon)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q`
Expected: FAIL (missing modules)

**Step 3: Write minimal implementation**

- `foresight.metrics`: implement `mae`, `rmse`, `mape` (with epsilon for divide-by-zero), `smape`
- `foresight.models.naive`:
  - `naive_last(train, horizon) -> np.ndarray[horizon]` (repeat last value)
  - `seasonal_naive(train, horizon, season_length) -> np.ndarray[horizon]`
- `foresight.backtesting.walk_forward` returning a small dataclass with `y_true`, `y_pred`, and window metadata.

**Step 4: Run tests to verify they pass**

Run: `pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/metrics.py src/foresight/backtesting.py src/foresight/models tests/test_metrics.py tests/test_backtesting.py
git commit -m "feat: add metrics, naive baselines, and backtesting"
```

---

### Task 4: Enforce “no notebooks” + add CI smoke checks

**Files:**
- Create: `tools/check_no_ipynb.py`
- Create: `.github/workflows/ci.yml`
- (Optional) Modify: `pyproject.toml` (ruff config)

**Step 1: Write the failing test**

Create `tools/check_no_ipynb.py` that exits non-zero when any `*.ipynb` exists in the repo.

Temporarily verify behavior locally by creating a dummy `tmp.ipynb` and checking that the script fails, then delete it.

**Step 2: Implement CI**

Add `.github/workflows/ci.yml` to run:
- `python -m py_compile` on `tools/*.py` and `src/**/*.py`
- `PYTHONPATH=src pytest -q`
- `python tools/check_no_ipynb.py`

**Step 3: Verify locally**

Run:
```bash
python tools/check_no_ipynb.py
pytest -q
```
Expected: exit 0

**Step 4: Commit**

```bash
git add tools/check_no_ipynb.py .github/workflows/ci.yml pyproject.toml
git commit -m "chore: add no-ipynb guard and CI workflow"
```
