# TS Data Processing + Algorithms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TS data processing utilities (A/B/D), add two core TS algorithms (`holt-winters-mul`, `seasonal-naive-auto`), and extend the CLI with `foresight data ...` subcommands — with full tests and doc regeneration where required.

**Architecture:** Keep core utilities dependency-free (`numpy`, `pandas`). Add thin CLI wrappers in `src/foresight/cli.py` that reuse existing `foresight.data.*` and `foresight.splits.*`. Register new models via `src/foresight/models/registry.py` without introducing new capability keys.

**Tech Stack:** Python 3.10+, `numpy`, `pandas`, `argparse`, `pytest`.

---

## Scope (as confirmed)

- **TS data processing:** A + B + D
  - A: long/panel regularization + bridge to multivariate
  - B: sliding window supervised dataset builders
  - D: feature engineering utilities (extend + test coverage)
- **TS algorithms (default):** `holt-winters-mul` + `seasonal-naive-auto`
- **CLI:** add `foresight data ...` command group

---

## Task 1: Add `prepare_wide_df()` (wide multivariate regularization)

**Files:**
- Modify: `src/foresight/data/prep.py`
- Test: `tests/test_data_prep_wide.py`

**Step 1: Write the failing tests**

Add `tests/test_data_prep_wide.py`:
- `test_prepare_wide_df_inserts_missing_timestamps_and_fills()`
  - Build a 2-column wide DF with a missing date in the middle
  - Call `prepare_wide_df(..., freq="D", missing="zero")`
  - Assert ds range is continuous and missing row is inserted with zeros
- `test_prepare_wide_df_strict_freq_rejects_irregular()` (or equivalent strict failure)

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_data_prep_wide.py`

Expected: FAIL with `ImportError`/`AttributeError` for missing `prepare_wide_df`.

**Step 3: Implement minimal `prepare_wide_df()`**

Implementation requirements:
- Input: `pd.DataFrame` required and non-empty
- Support `ds_col="ds"` (datetime-like) or `ds_col=None` (use index)
- Optional `freq` forces reindexing to full `date_range`
- If `freq` omitted, attempt `infer_series_frequency(..., strict=strict_freq)`; if it returns None, skip reindex unless `strict_freq=True`
- Apply missing policy (`error|drop|ffill|zero|interpolate`) to `target_cols` (or all non-ds columns)
- Reject duplicate timestamps

**Step 4: Run tests to verify green**

Run: `PYTHONPATH=src pytest -q tests/test_data_prep_wide.py`

Expected: PASS.

---

## Task 2: Add `long_to_wide()` (long→wide bridge)

**Files:**
- Modify: `src/foresight/data/format.py`
- Test: `tests/test_data_long_to_wide.py`

**Step 1: Write the failing tests**

Add `tests/test_data_long_to_wide.py`:
- `test_long_to_wide_pivots_expected_shape_and_columns()`
  - Provide a 2-series long DF with aligned ds
  - Assert output columns include `ds` plus two series columns
- `test_long_to_wide_can_fill_missing_values_with_zero()`
  - Provide long DF missing a ds for one id
  - Call `long_to_wide(..., freq="D", missing="zero")`
  - Assert the inserted timestamp exists and missing cell is 0

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_data_long_to_wide.py`

Expected: FAIL (missing `long_to_wide`).

**Step 3: Implement minimal `long_to_wide()`**

Implementation requirements:
- Validate required columns: `id_col`, `ds_col`, `value_col`
- Reject duplicates in `(id_col, ds_col)`
- Pivot: index=`ds_col`, columns=`id_col`, values=`value_col`
- Reset index to a column named `ds_col`
- Optionally sort id columns for deterministic output
- Delegate reindexing + missing policy to `prepare_wide_df()` where possible

**Step 4: Run tests to verify green**

Run: `PYTHONPATH=src pytest -q tests/test_data_long_to_wide.py`

Expected: PASS.

---

## Task 3: Add `make_lagged_xy_multi()` (multi-horizon supervised builder)

**Files:**
- Modify: `src/foresight/features/lag.py`
- Modify: `src/foresight/features/__init__.py` (export)
- Test: `tests/test_features_lag.py`

**Step 1: Write failing tests**

Extend `tests/test_features_lag.py`:
- `test_make_lagged_xy_multi_shapes_and_values()`
  - Use a short deterministic series
  - Assert `X.shape == (rows, n_lags)`, `Y.shape == (rows, horizon)`, `t_index.shape == (rows,)`
  - Assert exact values for first row
- `test_make_lagged_xy_multi_rejects_not_enough_points()`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_features_lag.py -k lagged_xy_multi`

Expected: FAIL (missing function).

**Step 3: Implement `make_lagged_xy_multi()`**

Requirements:
- 1D float input only (match `make_lagged_xy` style)
- `lags` normalized with `normalize_lag_steps`
- Require `len(y) >= start_t + horizon`
- Return `X`, `Y`, and `t_index`

**Step 4: Re-run tests**

Run: `PYTHONPATH=src pytest -q tests/test_features_lag.py -k lagged_xy_multi`

Expected: PASS.

---

## Task 4: Add test coverage for `build_time_features()`

**Files:**
- Test: `tests/test_features_time.py`

**Step 1: Add tests**
- Parseable datetimes: output has expected feature count and finite values
- Non-parseable datetimes: fallback zeros keep shape stable

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest -q tests/test_features_time.py`

Expected: PASS (or identify bug and fix in `src/foresight/features/time.py`).

---

## Task 5: Add CLI `foresight data ...` command group

**Files:**
- Modify: `src/foresight/cli.py`
- Test: `tests/test_cli_data.py`
- (Optional) Docs: `README.md` CLI section

**Commands:**
- `foresight data to-long`
- `foresight data prepare-long`
- `foresight data infer-freq`
- `foresight data splits rolling-origin`
- (Optional) `foresight data validate-long`

**Step 1: Write failing CLI tests**

Add `tests/test_cli_data.py` (subprocess style):
- `test_cli_data_to_long_basic()`
- `test_cli_data_prepare_long_inserts_missing_ds()` (or similar)
- `test_cli_data_infer_freq_daily()`
- `test_cli_data_splits_rolling_origin_indices()`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_cli_data.py`

Expected: FAIL (unknown command).

**Step 3: Implement CLI subparsers + handlers**

Keep handlers thin:
- Use `load_csv`, `ensure_datetime`, `parse_id_cols`
- Use `to_long`, `prepare_long_df`, `infer_series_frequency`, `rolling_origin_splits`
- Emit via `_emit_dataframe`

**Step 4: Re-run CLI tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli_data.py`

Expected: PASS.

---

## Task 6: Add `holt-winters-mul` (+ optional `-auto`) model

**Files:**
- Modify: `src/foresight/models/smoothing.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_smoothing.py`

**Step 1: Write failing tests**

Extend `tests/test_models_smoothing.py`:
- `test_holt_winters_multiplicative_repeats_two_step_season()`
- Optional: `test_holt_winters_multiplicative_auto_runs_smoke()`

**Step 2: Run tests to verify fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_smoothing.py -k winters`

Expected: FAIL (missing function / registry key).

**Step 3: Implement multiplicative HW + register keys**

Requirements:
- Require strictly positive values (division-based seasonality)
- Same parameter conventions as additive: `season_length`, `alpha`, `beta`, `gamma`
- Forecast shape `(horizon,)`
- Auto uses small grid search similar to additive auto

**Step 4: Re-run tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_smoothing.py -k winters`

Expected: PASS.

---

## Task 7: Add `seasonal-naive-auto` model

**Files:**
- Modify: `src/foresight/models/naive.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_seasonal_naive_auto.py` (new)

**Step 1: Write failing tests**

Add `tests/test_models_seasonal_naive_auto.py`:
- Repeating pattern: ensure inferred season length matches and forecast repeats
- Fallback behavior: too-short series falls back to `naive-last`

**Step 2: Run tests to verify fail**

Run: `PYTHONPATH=src pytest -q tests/test_models_seasonal_naive_auto.py`

Expected: FAIL (missing key/function).

**Step 3: Implement season length inference + register**

Implementation idea:
- Compute simple ACF over candidate lags in `[min_season_length, max_season_length]`
- Pick lag with highest correlation; if weak, fallback to `naive_last`
- Forecast via `seasonal_naive(..., season_length=p)`

**Step 4: Re-run tests**

Run: `PYTHONPATH=src pytest -q tests/test_models_seasonal_naive_auto.py`

Expected: PASS.

---

## Task 8: Regenerate model/docs artifacts (if needed)

**Files:**
- Modify: `docs/models.md` (generated)
- (Only if root exports changed) Modify: `docs/api.md` (generated)

Run: `PYTHONPATH=src python tools/generate_model_capability_docs.py`

Verify: `PYTHONPATH=src pytest -q tests/test_docs_rnn_generated.py::test_model_capability_docs_are_up_to_date`

---

## Task 9: Final verification

Run the most relevant suites first:
- `PYTHONPATH=src pytest -q tests/test_data_prep_wide.py tests/test_data_long_to_wide.py`
- `PYTHONPATH=src pytest -q tests/test_features_lag.py -k lagged_xy_multi`
- `PYTHONPATH=src pytest -q tests/test_cli_data.py`
- `PYTHONPATH=src pytest -q tests/test_models_smoothing.py -k winters`
- `PYTHONPATH=src pytest -q tests/test_models_seasonal_naive_auto.py`
- `PYTHONPATH=src pytest -q tests/test_docs_rnn_generated.py`

Then (optional but ideal): `PYTHONPATH=src pytest -q`

