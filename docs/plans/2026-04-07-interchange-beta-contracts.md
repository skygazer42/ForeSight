# Interchange Beta Contracts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared adapter normalization layer, richer Darts bundle APIs, and local single-series `X` support in the sktime adapter while preserving existing simple adapter behavior.

**Architecture:** Keep the current simple adapter APIs stable and additive. Introduce one internal adapter bundle helper that understands canonical target/historic/future/static covariate roles, then build the richer Darts APIs and sktime `X` path on top of it. This batch remains beta-scoped under `foresight.adapters`.

**Tech Stack:** Python 3.10, pandas, numpy, pytest, existing ForeSight long-format/covariate contracts

---

### Task 1: Add Failing Tests For The New Beta Surface

**Files:**
- Modify: `tests/test_adapters_darts_gluonts.py`
- Modify: `tests/test_adapters_sktime.py`
- Modify: `tests/test_adapters_public_surface.py`

**Step 1: Write a failing Darts bundle export/import surface test**

Add tests that expect `foresight.adapters.__all__` to include:

- `to_darts_bundle`
- `from_darts_bundle`

**Step 2: Write a failing Darts bundle single-series covariate test**

Cover a single-series long-format style case with `future_x_cols`, expecting:

- `to_darts_bundle(...)` returns `target`, `past_covariates`, `future_covariates`, `freq`
- `future_covariates` is populated
- `from_darts_bundle(...)` restores long-format rows and attrs

**Step 3: Write a failing Darts bundle panel/static-covariate test**

Cover panel long-format input with:

- `historic_x_cols`
- `future_x_cols`
- `static_cols`

Expect mapping-shaped bundle outputs and round-trip restoration of covariate attrs.

**Step 4: Write a failing sktime `X` success-path test**

Use a fake local xreg-compatible forecaster object so `fit(y, X)` and `predict(fh, X)` are expected to succeed.

**Step 5: Write a failing sktime unsupported-`X` rejection test**

Expect a precise beta limitation error for unsupported wrapper shapes.

**Step 6: Run the targeted adapter tests to verify RED**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_sktime.py tests/test_adapters_darts_gluonts.py
```

Expected: FAIL on the new tests, pass on the old ones.

**Step 7: Commit**

```bash
git add tests/test_adapters_public_surface.py tests/test_adapters_sktime.py tests/test_adapters_darts_gluonts.py
git commit -m "test: add beta interchange adapter contract coverage"
```

### Task 2: Implement The Shared Adapter Bundle Layer And Darts Bundle APIs

**Files:**
- Create: `src/foresight/adapters/shared.py`
- Modify: `src/foresight/adapters/darts.py`
- Modify: `src/foresight/adapters/__init__.py`

**Step 1: Add the shared internal bundle helper module**

Implement internal helpers that normalize canonical inputs into:

- target data
- historic covariates
- future covariates
- static covariates
- frequency metadata

Do not expose this helper from the public namespace.

**Step 2: Implement `to_darts_bundle(...)`**

Support:

- single-series input
- canonical panel long-format input
- `historic_x_cols`, `future_x_cols`, `static_cols`

Return the explicit bundle shape from the design doc.

**Step 3: Implement `from_darts_bundle(...)`**

Restore canonical long-format output and reattach:

- `historic_x_cols`
- `future_x_cols`
- `static_cols`

**Step 4: Export the new Darts bundle APIs**

Update `src/foresight/adapters/__init__.py` so the new bundle functions are part of the beta adapter namespace.

**Step 5: Run targeted tests to verify GREEN for Darts work**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/adapters/shared.py src/foresight/adapters/darts.py src/foresight/adapters/__init__.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
git commit -m "feat: add richer beta darts interchange bundles"
```

### Task 3: Implement sktime Local `X` Support

**Files:**
- Modify: `src/foresight/adapters/sktime.py`
- Modify: `tests/test_adapters_sktime.py`

**Step 1: Add minimal local single-series `X` normalization**

Accept pandas or array-like `X` for:

- `fit(y, X=...)`
- `predict(fh, X=...)`

Normalize it into the shape expected by compatible local xreg ForeSight objects.

**Step 2: Gate `X` support behind compatibility checks**

Support `X` only when the wrapped forecaster is compatible with local xreg.

Reject unsupported shapes or workflows with explicit beta limitation messages.

**Step 3: Preserve all current no-`X` behavior**

Do not regress:

- fit-time `fh`
- absolute range/datetime horizon support
- existing no-`X` local point flow

**Step 4: Run targeted sktime tests**

Run:

```bash
pytest -q tests/test_adapters_sktime.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/adapters/sktime.py tests/test_adapters_sktime.py
git commit -m "feat: support local xreg sktime adapter inputs"
```

### Task 4: Update Adapter Docs And Verify End-To-End

**Files:**
- Modify: `docs/adapters.md`

**Step 1: Document the richer Darts bundle APIs**

Explain:

- additive beta API
- bundle fields
- panel/global + covariate use case

**Step 2: Update sktime adapter docs**

Document:

- local single-series `X` support
- remaining unsupported panel/global/static cases

**Step 3: Run the full targeted validation set**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_sktime.py tests/test_adapters_darts_gluonts.py
mkdocs build --strict
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs/adapters.md tests/test_adapters_public_surface.py tests/test_adapters_sktime.py tests/test_adapters_darts_gluonts.py src/foresight/adapters/shared.py src/foresight/adapters/darts.py src/foresight/adapters/sktime.py src/foresight/adapters/__init__.py
git commit -m "docs: describe richer beta interchange adapters"
```
