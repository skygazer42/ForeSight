# GluonTS Beta Bundle Design

**Goal:** Extend the adapter beta surface with a richer GluonTS bundle path that follows the same shared-contract direction as the Darts richer bundle APIs, while keeping the current `to_gluonts_list_dataset(...)` API unchanged.

## Summary

ForeSight now has two adapter tiers:

- simple conversion APIs such as `to_gluonts_list_dataset(...)`
- richer beta bundle APIs such as `to_darts_bundle(...)`

GluonTS is still missing the richer tier. This batch adds an additive beta bundle path rather than overloading the existing simple API.

## Design Decisions

### 1. Preserve `to_gluonts_list_dataset(...)`

The current `to_gluonts_list_dataset(...)` remains unchanged:

- single-series and panel long-format history conversion
- inferred or explicit `freq`
- target-only ListDataset output

This keeps the simple path stable for current users.

### 2. Add richer bundle APIs

Add:

- `to_gluonts_bundle(...)`
- `from_gluonts_bundle(...)`

These are beta APIs under `foresight.adapters`.

### 3. Reuse the shared adapter contract

The new GluonTS bundle path should build on `AdapterFrameBundle` / `AdapterSeriesPayload` from `foresight.adapters.shared`.

That means the richer GluonTS path inherits the same ForeSight-side source of truth:

- `target`
- `historic_covariates`
- `future_covariates`
- `static_covariates`
- `freq`

### 4. Use GluonTS-native field naming inside the richer bundle

The exported bundle should be GluonTS-oriented but still explicit:

- `target`
- `past_feat_dynamic_real`
- `feat_dynamic_real`
- `feat_static_real`
- `feature_names`
- `freq`

All payload-bearing fields are mapping-shaped and keyed by `unique_id`.

Rationale:

- this matches the Darts bundle direction of unified keyed payloads
- this keeps field names recognizable to GluonTS users
- this avoids returning partially-instantiated predictor objects in this batch

### 5. Keep scope to data/contract interop only

This batch does **not** add:

- GluonTS predictor wrappers
- estimator wrappers
- training/inference orchestration through the adapter

It only adds richer data-contract interop and round-trip support.

## Bundle Shape

`to_gluonts_bundle(...)` returns:

- `target: dict[str, dict[str, object]]`
- `past_feat_dynamic_real: dict[str, list[list[float]]]`
- `feat_dynamic_real: dict[str, list[list[float]]]`
- `feat_static_real: dict[str, list[float]]`
- `feature_names: dict[str, tuple[str, ...]]`
- `freq: dict[str, str]`

Rules:

- `target[uid]` contains GluonTS-style base fields such as `start`, `target`, `item_id`
- `past_feat_dynamic_real` stores historic covariates as feature-major 2D lists
- `feat_dynamic_real` stores future covariates as feature-major 2D lists
- `feat_static_real` stores ordered static covariates as a flat list
- `feature_names` stores the corresponding column ordering for:
  - `historic_x_cols`
  - `future_x_cols`
  - `static_cols`

### Backward compatibility

`from_gluonts_bundle(...)` only needs to support the richer bundle shape introduced here. Unlike Darts, there is no prior richer GluonTS bundle API to preserve.

## Test Strategy

Add tests for:

- public surface export of the new GluonTS bundle symbols
- richer single-series/panel bundle generation
- round-trip reconstruction into canonical long-format
- preservation of:
  - `historic_x_cols`
  - `future_x_cols`
  - `static_cols`
- clear dependency error path when GluonTS is unavailable

Regression requirements:

- existing `to_gluonts_list_dataset(...)` tests continue to pass
- Darts tests continue to pass
- docs build remains green
