# Shared Beta Bundle Design

**Goal:** Introduce an adapter-agnostic shared beta bundle API that unifies the richer adapter contract across Darts and GluonTS without changing their existing adapter-specific bundle shapes.

## Summary

ForeSight now has richer beta bundle APIs for Darts and GluonTS, but they use adapter-specific field names:

- Darts: `target`, `past_covariates`, `future_covariates`, `freq`
- GluonTS: `target`, `past_feat_dynamic_real`, `feat_dynamic_real`, `feat_static_real`, `feature_names`, `freq`

Those are both reasonable at the adapter boundary, but they do not give ForeSight users one common mental model for richer interoperability. This batch adds a shared beta bundle surface on top of the existing shared adapter normalization layer.

## Design Decisions

### 1. Keep adapter-specific bundle APIs unchanged

Do not change:

- `to_darts_bundle(...)`
- `from_darts_bundle(...)`
- `to_gluonts_bundle(...)`
- `from_gluonts_bundle(...)`

These remain useful adapter-specific beta APIs.

### 2. Add adapter-agnostic shared beta APIs

Add:

- `to_beta_bundle(...)`
- `from_beta_bundle(...)`

These APIs live under `foresight.adapters` and represent the ForeSight-centric richer bundle contract.

### 3. Shared bundle schema

`to_beta_bundle(...)` returns a single explicit schema:

- `target: dict[str, pd.DataFrame]`
- `historic_covariates: dict[str, pd.DataFrame]`
- `future_covariates: dict[str, pd.DataFrame]`
- `static_covariates: dict[str, pd.DataFrame]`
- `freq: dict[str, str]`

Rules:

- payloads are always keyed by `unique_id`
- missing covariate roles are empty dicts
- target frames always use `ds` + `y`
- historic/future/static payloads keep their original ForeSight column names

### 4. Shared import/export is ForeSight-first

This shared API is not tied to any external ecosystem object model.

- `to_beta_bundle(...)` consumes canonical ForeSight long-format input
- `from_beta_bundle(...)` produces canonical ForeSight long-format output

This makes it the canonical richer beta contract for adapter interop inside ForeSight.

### 5. Adapter-specific bundle APIs stay layered on top

This batch does not rewrite Darts/GluonTS to call the shared API internally if that adds churn. The minimum requirement is that the shared API be produced from the same source-of-truth semantics as `AdapterFrameBundle`.

## Test Strategy

Add tests for:

- public-surface exports of `to_beta_bundle(...)` / `from_beta_bundle(...)`
- shared bundle export from canonical long-format
- shared bundle import back to canonical long-format
- preservation of:
  - `historic_x_cols`
  - `future_x_cols`
  - `static_cols`

Regression requirements:

- Darts richer bundle tests keep passing
- GluonTS richer bundle tests keep passing
- docs build remains green
