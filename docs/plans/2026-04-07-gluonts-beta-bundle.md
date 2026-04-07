# GluonTS Beta Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a richer GluonTS beta bundle API on top of the shared adapter contract without changing the existing simple `to_gluonts_list_dataset(...)` behavior.

**Architecture:** Reuse the shared adapter payload contract, export GluonTS-native bundle field names, and keep all richer payloads keyed by `unique_id` to stay aligned with the Darts richer adapter direction. This batch remains data/contract interop only.

**Tech Stack:** Python 3.10, pandas, pytest, existing ForeSight adapter/shared contract code

---

### Task 1: Add Failing Tests For The New GluonTS Beta Surface

**Files:**
- Modify: `tests/test_adapters_public_surface.py`
- Modify: `tests/test_adapters_darts_gluonts.py`

**Step 1: Add public-surface failing tests**

Expect `foresight.adapters.__all__` to include:

- `to_gluonts_bundle`
- `from_gluonts_bundle`

**Step 2: Add a richer GluonTS panel bundle export test**

Cover canonical long-format input with:

- `historic_x_cols`
- `future_x_cols`
- `static_cols`

Expect mapping-shaped payloads keyed by `unique_id`.

**Step 3: Add a richer GluonTS round-trip test**

Expect `from_gluonts_bundle(...)` to restore:

- canonical long-format rows
- `historic_x_cols`
- `future_x_cols`
- `static_cols`

**Step 4: Run targeted tests to verify RED**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py -k gluonts
```

Expected: FAIL on the new GluonTS bundle tests.

**Step 5: Commit**

```bash
git add tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "test: define richer gluonts beta bundle contracts"
```

### Task 2: Implement Richer GluonTS Bundle Export

**Files:**
- Modify: `src/foresight/adapters/gluonts.py`
- Modify: `src/foresight/adapters/__init__.py`

**Step 1: Add `to_gluonts_bundle(...)`**

Build richer bundle payloads from the shared adapter bundle contract.

**Step 2: Keep `to_gluonts_list_dataset(...)` unchanged**

Do not regress the simple target-only ListDataset path.

**Step 3: Export the new beta symbols**

Update `foresight.adapters.__all__`.

**Step 4: Run GluonTS-targeted tests**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py -k gluonts
```

Expected: export tests pass; round-trip import may still fail until Task 3.

**Step 5: Commit**

```bash
git add src/foresight/adapters/gluonts.py src/foresight/adapters/__init__.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "feat: add richer gluonts beta bundle export"
```

### Task 3: Implement GluonTS Bundle Import

**Files:**
- Modify: `src/foresight/adapters/gluonts.py`
- Modify: `tests/test_adapters_darts_gluonts.py`

**Step 1: Add `from_gluonts_bundle(...)`**

Convert richer GluonTS bundle payloads back into canonical long-format.

**Step 2: Restore covariate attrs**

Ensure the returned long DataFrame restores:

- `historic_x_cols`
- `future_x_cols`
- `static_cols`

**Step 3: Run the full adapter file**

Run:

```bash
pytest -q tests/test_adapters_darts_gluonts.py
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/foresight/adapters/gluonts.py tests/test_adapters_darts_gluonts.py
git commit -m "feat: add gluonts beta bundle round-trip"
```

### Task 4: Update Docs And Verify The Whole Surface

**Files:**
- Modify: `docs/adapters.md`

**Step 1: Update GluonTS docs**

Document:

- the richer beta bundle APIs
- the bundle field names
- the continued stability of `to_gluonts_list_dataset(...)`

**Step 2: Run final verification**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
mkdocs build --strict
```

Expected: PASS

**Step 3: Commit**

```bash
git add docs/adapters.md src/foresight/adapters/gluonts.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "docs: describe richer gluonts beta bundles"
```
