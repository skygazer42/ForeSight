# Darts Bundle Shape Unification Design

**Goal:** Unify the richer beta Darts bundle API around one consistent return schema while keeping `from_darts_bundle(...)` backward-compatible with the previously shipped single-series bundle shape.

## Summary

The current beta Darts bundle API uses two shapes:

- single-series: `target` / `past_covariates` / `future_covariates` are `TimeSeries | None`, and `freq` is a string
- panel: those same fields are mappings keyed by `unique_id`, and `freq` is per-series metadata

That makes callers branch on cardinality even though they are using the same API. This batch standardizes the output contract of `to_darts_bundle(...)` so it always returns mapping-shaped fields keyed by `unique_id`. To reduce breakage risk, `from_darts_bundle(...)` stays tolerant of the legacy single-series shape.

## Design Decisions

### 1. Standardize `to_darts_bundle(...)` output

`to_darts_bundle(...)` will always return:

- `target: dict[str, TimeSeries]`
- `past_covariates: dict[str, TimeSeries]`
- `future_covariates: dict[str, TimeSeries]`
- `freq: dict[str, str]`

For single-series input, the maps contain exactly one key.

For missing covariate roles, return an empty dict instead of `None`.

### 2. Keep `from_darts_bundle(...)` backward-compatible

`from_darts_bundle(...)` should accept:

- the new normalized mapping schema
- the older single-series shape where payload values may be raw `TimeSeries` and `freq` may be a single string

This allows existing beta users to keep working while new callers get a cleaner contract.

### 3. Do not change simple Darts APIs

The following remain untouched:

- `to_darts_timeseries(...)`
- `from_darts_timeseries(...)`

Only the richer beta bundle path changes in this batch.

## Test Strategy

Add or update tests so that:

- single-series bundle export now expects mapping-shaped output
- panel bundle export still expects mapping-shaped output
- `from_darts_bundle(...)` accepts the legacy single-series shape
- round-trip behavior still preserves:
  - `historic_x_cols`
  - `future_x_cols`
  - `static_cols`

Regression checks:

- existing simple Darts conversion tests still pass
- adapter docs and site build stay green
