# Adapters

ForeSight exposes interoperability bridges through the named
`foresight.adapters` module. These adapters are intentionally documented as
**beta integration surfaces** rather than part of the stable root-package API.

```python
from foresight.adapters import (
    from_beta_bundle,
    from_darts_bundle,
    from_gluonts_bundle,
    make_sktime_forecaster_adapter,
    to_beta_bundle,
    to_darts_bundle,
    to_gluonts_bundle,
    to_darts_timeseries,
    from_darts_timeseries,
    to_gluonts_list_dataset,
)
```

## Shared Beta Bundle

ForeSight now exposes an adapter-agnostic richer beta bundle API:

```python
from foresight.adapters import to_beta_bundle, from_beta_bundle
```

Shared bundle contract:

- `target: dict[str, pd.DataFrame]`
- `historic_covariates: dict[str, pd.DataFrame]`
- `future_covariates: dict[str, pd.DataFrame]`
- `static_covariates: dict[str, pd.DataFrame]`
- `freq: dict[str, str]`

Design intent:

- all payloads are keyed by `unique_id`
- target frames use canonical `ds` / `y`
- covariate frames preserve ForeSight-native column names
- this is the ForeSight-centric richer bundle contract
- Darts and GluonTS richer bundle APIs remain available as adapter-specific projections of the same underlying semantics

Minimal example:

```python
import pandas as pd

from foresight.adapters import from_beta_bundle, to_beta_bundle

long_df = pd.DataFrame(
    {
        "unique_id": ["store_a", "store_a"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "y": [10.0, 11.0],
        "stock": [4.0, 5.0],
        "promo": [0.0, 1.0],
        "store_size": [100.0, 100.0],
    }
)
long_df.attrs["historic_x_cols"] = ("stock",)
long_df.attrs["future_x_cols"] = ("promo",)
long_df.attrs["static_cols"] = ("store_size",)

bundle = to_beta_bundle(long_df)
restored = from_beta_bundle(bundle)
```

Result:

- `bundle["target"]`, `bundle["historic_covariates"]`, and the other bundle
  roles are keyed by `unique_id`
- `restored` is canonical ForeSight long-format data with the covariate attrs
  restored

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

Minimal example:

```python
from typing import Any

import numpy as np
import pandas as pd

from foresight.adapters import make_sktime_forecaster_adapter
from foresight.base import BaseForecaster


class LastValueForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(model_key="example-last-value")
        self._last = 0.0

    def fit(self, y: Any, X: Any = None) -> "LastValueForecaster":
        self._last = float(np.asarray(y, dtype=float)[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int, X: Any = None) -> np.ndarray:
        return np.asarray([self._last] * int(horizon), dtype=float)

    def train_schema_summary(self) -> dict[str, Any]:
        return {"kind": "local"}


y = pd.Series([5.0, 6.0, 7.0], index=pd.RangeIndex(start=0, stop=3))
adapter = make_sktime_forecaster_adapter(LastValueForecaster())
yhat = adapter.fit(y).predict([1, 2])
```

Result:

- `yhat` is a pandas `Series`
- the output index follows the forecasting horizon
- the adapter stays on the named `foresight.adapters` beta surface

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

Minimal example:

```python
import pandas as pd

from foresight.adapters import from_darts_bundle, to_darts_bundle

long_df = pd.DataFrame(
    {
        "unique_id": ["store_a", "store_a"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "y": [10.0, 11.0],
        "stock": [4.0, 5.0],
        "promo": [0.0, 1.0],
    }
)
long_df.attrs["historic_x_cols"] = ("stock",)
long_df.attrs["future_x_cols"] = ("promo",)
long_df.attrs["static_cols"] = ()

bundle = to_darts_bundle(long_df)
restored = from_darts_bundle(bundle)
```

Result:

- `bundle["target"]`, `bundle["past_covariates"]`, and
  `bundle["future_covariates"]` are mapping-shaped and keyed by `unique_id`
- `restored` returns to canonical ForeSight long-format data

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

Minimal example:

```python
import pandas as pd

from foresight.adapters import from_gluonts_bundle, to_gluonts_bundle

long_df = pd.DataFrame(
    {
        "unique_id": ["store_a", "store_a"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "y": [10.0, 11.0],
        "stock": [4.0, 5.0],
        "promo": [0.0, 1.0],
        "store_size": [100.0, 100.0],
    }
)
long_df.attrs["historic_x_cols"] = ("stock",)
long_df.attrs["future_x_cols"] = ("promo",)
long_df.attrs["static_cols"] = ("store_size",)

bundle = to_gluonts_bundle(long_df)
restored = from_gluonts_bundle(bundle)
```

Result:

- `bundle["target"]`, `bundle["past_feat_dynamic_real"]`, and
  `bundle["feat_dynamic_real"]` stay keyed by `unique_id`
- `restored` reconstructs canonical ForeSight long-format data with covariate
  attrs
