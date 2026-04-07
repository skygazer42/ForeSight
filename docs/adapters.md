# Adapters

ForeSight exposes interoperability bridges through the named
`foresight.adapters` module. These adapters are intentionally documented as
**beta integration surfaces** rather than part of the stable root-package API.

```python
from foresight.adapters import (
    from_darts_bundle,
    from_gluonts_bundle,
    make_sktime_forecaster_adapter,
    to_darts_bundle,
    to_gluonts_bundle,
    to_darts_timeseries,
    from_darts_timeseries,
    to_gluonts_list_dataset,
)
```

## sktime

Install:

```bash
pip install "foresight-ts[sktime]"
```

Use `make_sktime_forecaster_adapter(...)` to wrap a local point forecaster
object or model key behind a minimal `fit(...)` / `predict(...)` bridge.

Current v1 contract:

- local point forecasters only
- relative forecasting horizons
- absolute horizons when the training series uses a `RangeIndex` or regular `DatetimeIndex`
- beta `X` support for local single-series xreg-compatible forecasters
- no panel/global sktime wrapper support yet
- no static-covariate sktime support yet

The beta `X` path is intentionally narrow:

- `fit(y, X=...)` / `predict(fh, X=...)` are supported for compatible local xreg forecasters
- pandas and array-like `X` are accepted
- `X` must align with the training or forecast horizon rows
- `historic_x_cols`-only semantics are still rejected in the sktime beta path

## Darts

Install:

```bash
pip install "foresight-ts[darts]"
```

Use `to_darts_timeseries(...)` and `from_darts_timeseries(...)` to convert
between ForeSight single-series or canonical long-format panel data and Darts
`TimeSeries` objects.

Current v1 contract:

- single-series `pandas.Series` round-trip
- panel long-format `unique_id` / `ds` / `y` to mapping-of-series conversion
- data interop only; no Darts model wrapper

ForeSight now also exposes a richer beta bundle API for covariate-aware workflows:

```python
from foresight.adapters import to_darts_bundle, from_darts_bundle
```

Bundle contract:

- `target: dict[str, TimeSeries]`
- `past_covariates: dict[str, TimeSeries]`
- `future_covariates: dict[str, TimeSeries]`
- `freq: dict[str, str]`

Current beta bundle support:

- canonical long-format input with `historic_x_cols`, `future_x_cols`, and `static_cols`
- single-series and panel/global exports both use the same mapping-based schema keyed by `unique_id`
- empty covariate roles are emitted as empty dicts
- static covariates attached to target series metadata and restored on round-trip
- `from_darts_bundle(...)` remains backward-compatible with the older single-series beta shape
- additive beta API only; it does not replace the existing simple `to_darts_timeseries(...)` path

## GluonTS

Install:

```bash
pip install "foresight-ts[gluonts]"
```

Use `to_gluonts_list_dataset(...)` to convert canonical long-format history into
a GluonTS `ListDataset`.

Current v1 contract:

- single-series and panel long-format history conversion
- inferred or explicit `freq`
- data interop only; no GluonTS predictor wrapper

ForeSight now also exposes a richer beta bundle API for GluonTS-oriented workflows:

```python
from foresight.adapters import to_gluonts_bundle, from_gluonts_bundle
```

Bundle contract:

- `target: dict[str, dict[str, object]]`
- `past_feat_dynamic_real: dict[str, list[list[float]]]`
- `feat_dynamic_real: dict[str, list[list[float]]]`
- `feat_static_real: dict[str, list[float]]`
- `feature_names: dict[str, tuple[str, ...]]`
- `freq: dict[str, str]`

Current beta bundle support:

- canonical long-format input with `historic_x_cols`, `future_x_cols`, and `static_cols`
- panel/global payloads keyed by `unique_id`
- GluonTS-style dynamic/static feature naming without introducing predictor wrappers
- round-trip restoration back into canonical long-format plus covariate attrs
- additive beta API only; it does not replace `to_gluonts_list_dataset(...)`
