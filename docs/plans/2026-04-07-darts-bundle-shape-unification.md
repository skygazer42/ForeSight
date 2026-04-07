# Darts Bundle Shape Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify `to_darts_bundle(...)` to one mapping-based bundle schema and keep `from_darts_bundle(...)` backward-compatible with the older single-series beta shape.

**Architecture:** Keep the simple Darts APIs unchanged. Update only the richer beta bundle contract so export is always normalized while import remains permissive. This keeps the external beta surface easier to use without breaking previously emitted single-series bundle payloads.

**Tech Stack:** Python 3.10, pandas, pytest, existing Darts beta adapter code

---

### Task 1: Add Failing Tests For The Unified Bundle Schema

**Files:**
- Modify: `tests/test_adapters_darts_gluonts.py`

**Step 1: Write a failing single-series bundle shape test**

Update the single-series bundle test to expect:

- `target` is a dict keyed by `unique_id`
- `past_covariates` is a dict
- `future_covariates` is a dict
- `freq` is a dict keyed by `unique_id`

**Step 2: Write a failing backward-compat import test**

Add a test proving `from_darts_bundle(...)` still accepts the previous single-series shape:

- `target: TimeSeries`
- `past_covariates: None`
- `future_covariates: TimeSeries`
- `freq: "D"`

Expected: canonical long-format restoration still works.

**Step 3: Run targeted Darts adapter tests to verify RED**

Run:

```bash
pytest -q tests/test_adapters_darts_gluonts.py -k bundle
```

Expected: FAIL on the new/updated bundle tests.

**Step 4: Commit**

```bash
git add tests/test_adapters_darts_gluonts.py
git commit -m "test: define unified darts bundle schema"
```

### Task 2: Implement The Unified Export Shape

**Files:**
- Modify: `src/foresight/adapters/darts.py`

**Step 1: Remove the single-series special return shape in `to_darts_bundle(...)`**

Always return mapping-shaped fields keyed by `unique_id`.

**Step 2: Normalize empty covariate payloads**

Return empty dicts for missing `past_covariates` / `future_covariates`, not `None`.

**Step 3: Keep `freq` mapping-shaped even for single-series**

Single-series output should still be `{"<uid>": "<freq>"}`.

**Step 4: Run the targeted Darts bundle tests**

Run:

```bash
pytest -q tests/test_adapters_darts_gluonts.py -k bundle
```

Expected: export-shape tests pass, backward-compat import may still fail if not yet handled.

**Step 5: Commit**

```bash
git add src/foresight/adapters/darts.py
git commit -m "feat: normalize darts bundle export shape"
```

### Task 3: Implement Backward-Compatible Bundle Import

**Files:**
- Modify: `src/foresight/adapters/darts.py`

**Step 1: Keep `from_darts_bundle(...)` permissive**

Continue accepting:

- normalized dict-based bundle payloads
- legacy single-series bundle payloads

**Step 2: Normalize legacy inputs internally**

Translate legacy single-series bundle values into the new internal path before reconstruction.

**Step 3: Run the full Darts adapter test file**

Run:

```bash
pytest -q tests/test_adapters_darts_gluonts.py
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/foresight/adapters/darts.py tests/test_adapters_darts_gluonts.py
git commit -m "fix: keep darts bundle import backward compatible"
```

### Task 4: Update Docs And Verify The Surface

**Files:**
- Modify: `docs/adapters.md`

**Step 1: Update the Darts bundle docs**

Document that:

- `to_darts_bundle(...)` now always emits mapping-shaped bundle fields
- `from_darts_bundle(...)` still accepts the older single-series shape for backward compatibility

**Step 2: Run final verification**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
mkdocs build --strict
```

Expected: PASS

**Step 3: Commit**

```bash
git add docs/adapters.md src/foresight/adapters/darts.py tests/test_adapters_darts_gluonts.py
git commit -m "docs: clarify unified darts bundle contract"
```
