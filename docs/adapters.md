# Adapters

ForeSight exposes interoperability bridges through the named
`foresight.adapters` module. These adapters are intentionally documented as
**beta integration surfaces** rather than part of the stable root-package API.

```python
from foresight.adapters import (
    make_sktime_forecaster_adapter,
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
- relative forecasting horizons only
- no `X` exogenous support yet

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
