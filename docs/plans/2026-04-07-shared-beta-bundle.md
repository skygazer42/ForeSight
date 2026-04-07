# Shared Beta Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared ForeSight-centric richer beta bundle API that unifies bundle semantics across adapters without changing the existing Darts and GluonTS bundle contracts.

**Architecture:** Reuse the existing `AdapterFrameBundle` / `AdapterSeriesPayload` normalization layer and expose a new pair of adapter-agnostic APIs under `foresight.adapters`. Keep all payloads mapping-shaped and keyed by `unique_id`, and preserve the existing adapter-specific richer APIs as-is.

**Tech Stack:** Python 3.10, pandas, pytest, existing adapter shared-contract code

---

### Task 1: Add Failing Tests For The Shared Bundle Surface

**Files:**
- Modify: `tests/test_adapters_public_surface.py`
- Modify: `tests/test_adapters_darts_gluonts.py`

**Step 1: Add public-surface failing tests**

Expect `foresight.adapters.__all__` to include:

- `to_beta_bundle`
- `from_beta_bundle`

**Step 2: Add a shared bundle export test**

Use canonical long-format input with:

- `historic_x_cols`
- `future_x_cols`
- `static_cols`

Expect:

- `target`
- `historic_covariates`
- `future_covariates`
- `static_covariates`
- `freq`

all keyed by `unique_id`.

**Step 3: Add a shared bundle round-trip test**

Expect `from_beta_bundle(...)` to restore:

- canonical long-format rows
- `historic_x_cols`
- `future_x_cols`
- `static_cols`

**Step 4: Run targeted tests to verify RED**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py -k beta_bundle
```

Expected: FAIL on the new shared-bundle tests.

**Step 5: Commit**

```bash
git add tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "test: define shared beta bundle contracts"
```

### Task 2: Implement Shared Bundle Export/Import

**Files:**
- Modify: `src/foresight/adapters/shared.py`
- Modify: `src/foresight/adapters/__init__.py`

**Step 1: Add `to_beta_bundle(...)`**

Return the adapter-agnostic shared schema directly from `AdapterFrameBundle`.

**Step 2: Add `from_beta_bundle(...)`**

Reconstruct canonical long-format output and attrs from the shared schema.

**Step 3: Export the shared bundle symbols**

Update `foresight.adapters.__all__`.

**Step 4: Run targeted tests**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py -k beta_bundle
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/adapters/shared.py src/foresight/adapters/__init__.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "feat: add shared beta bundle apis"
```

### Task 3: Update Docs And Verify Non-Regression

**Files:**
- Modify: `docs/adapters.md`

**Step 1: Document the shared bundle APIs**

Describe:

- the shared ForeSight-centric bundle contract
- the relationship between shared bundle APIs and Darts/GluonTS bundle APIs
- the fact that adapter-specific bundle APIs remain available

**Step 2: Run final verification**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
mkdocs build --strict
```

Expected: PASS

**Step 3: Commit**

```bash
git add docs/adapters.md src/foresight/adapters/shared.py src/foresight/adapters/__init__.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "docs: describe shared beta bundle api"
```
