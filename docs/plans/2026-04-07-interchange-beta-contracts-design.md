# Interchange Beta Contracts Design

**Goal:** Strengthen `foresight.adapters` as a beta interoperability surface for panel/global and covariate-rich workflows without breaking the existing simple adapter APIs.

## Summary

ForeSight already exposes beta adapters for `sktime`, `Darts`, and `GluonTS`, but today they mostly cover minimal data conversion or a very small local wrapper path. The next useful step is not adding external model wrappers; it is making the adapter layer capable of carrying the same covariate semantics ForeSight already uses internally.

This design keeps the current simple adapter functions intact and introduces richer beta APIs on top of a shared internal normalization layer. The first batch will cover:

- `Darts` richer bundle conversion for panel/global + covariates
- `sktime` local single-series `X` support for compatible local xreg models

`GluonTS` remains unchanged in this batch.

## Design Decisions

### 1. Preserve existing simple APIs

The following existing APIs remain behaviorally unchanged:

- `to_darts_timeseries(...)`
- `from_darts_timeseries(...)`
- `make_sktime_forecaster_adapter(...)` for current no-`X` usage
- `to_gluonts_list_dataset(...)`

This avoids a breaking change where existing Darts users suddenly receive bundle-shaped returns instead of `TimeSeries` or mapping-of-series.

### 2. Add richer Darts bundle APIs

Add new beta APIs:

- `to_darts_bundle(...)`
- `from_darts_bundle(...)`

The bundle shape is explicit and stable within the beta surface:

- `target`
- `past_covariates`
- `future_covariates`
- `freq`

For single-series input, each of those fields is a single Darts `TimeSeries` or `None`.

For panel/global input, `target`, `past_covariates`, and `future_covariates` are mappings keyed by `unique_id`.

Static covariates are not returned as a separate parallel structure. They are attached to each target `TimeSeries` when the backend object supports static covariates, and are restored back into canonical long-format columns and attrs on `from_darts_bundle(...)`.

### 3. Add a shared adapter normalization layer

Introduce one internal helper module under `foresight.adapters` that converts canonical ForeSight inputs into an explicit normalized bundle:

- target rows
- historic covariate columns
- future covariate columns
- static covariate columns
- per-series frequency metadata

This keeps Darts richer conversion and sktime `X` support from re-implementing covariate splitting separately.

This layer stays internal to adapters in this batch; it is not promoted to a stable public contract module.

### 4. Expand sktime adapter only for local single-series `X`

The sktime adapter will support:

- `fit(y, X=...)`
- `predict(fh, X=...)`

but only when all of the following are true:

- the wrapped model is a local forecaster
- the workflow is single-series
- the wrapped ForeSight object supports exogenous inputs through `x_cols`

This batch will not support:

- panel/global sktime wrapper semantics
- static covariate propagation through sktime
- a second richer sktime bundle API

If unsupported shapes are passed, the adapter should raise a precise error explaining the current beta limitation.

## Behavioral Notes

- `to_darts_bundle(...)` must preserve the distinction between `historic_x_cols` and `future_x_cols`; do not collapse them into one anonymous covariate table.
- `from_darts_bundle(...)` must restore canonical long-format output and reattach:
  - `historic_x_cols`
  - `future_x_cols`
  - `static_cols`
- Frequency should be explicit in the bundle output, even when inferred.
- The richer Darts API is additive beta surface, not a replacement.
- Existing simple Darts round-trips must remain unchanged.

## Test Strategy

Primary test additions:

- Darts richer bundle round-trip for:
  - single-series with future covariates
  - panel long-format with historic, future, and static covariates
- sktime adapter local xreg success path using `X`
- sktime adapter rejection for unsupported `X` cases
- public-surface tests for newly exported beta adapter symbols

Regression requirements:

- existing simple Darts tests still pass unchanged
- existing sktime no-`X` tests still pass unchanged
- docs build remains green
