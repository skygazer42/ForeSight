# How To Choose A Model

ForeSight has a wide registry, but most users do not need to browse all model
families manually. Start from the data shape and workflow constraint first,
then narrow by dependency and capability flags.

## Start With The Workflow

### 1. Single-series, dependency-light baseline

Choose a stable local core model when you want fast iteration, no heavy
dependencies, and a trustworthy baseline:

- `naive-last`, `seasonal-naive`, `drift`
- `moving-average`, `mean`, `theta`, `holt`
- `fft`, `fourier` when seasonality is strong and the series is regular

Good default shortlist:

```bash
foresight models list --requires core --interface local --stability stable --format json
```

### 2. Local model with future covariates / exogenous regressors

Choose local statsmodels or tree-based lag models when you already know future
promotion, price, calendar, or event features:

- `sarimax`, `auto-arima` for classical xreg workflows
- `xgb-*`, `lgbm-*`, `catboost-*`, and `*-lag` sklearn families for tabular lag features

Use the capability filter instead of guessing from the name:

```bash
foresight models list \
  --interface local \
  --capability supports_x_cols=true \
  --format json
```

### 3. Panel / multi-series forecasting

Choose global models when you have many related series in long format
(`unique_id`, `ds`, `y`) and want shared training across entities:

- `*-step-lag-global` families for fast sklearn / boosting style global baselines
- `torch-*-global` families when you need neural global models with static or future covariates

Good global shortlist:

```bash
foresight models list --interface global --stability stable --format json
```

### 4. Wide multivariate targets

Choose `multivariate` models only when the target itself is a synchronized
multi-column / graph-like system rather than a standard panel:

- `var` for classical statsmodels VAR
- `torch-*-multivariate` graph / spatiotemporal families for neural multivariate settings

```bash
foresight models list --interface multivariate --format json
```

## Then Filter By Constraint

### Need prediction intervals

Filter for:

```bash
foresight models list --capability supports_interval_forecast=true --format json
```

If you also need exogenous regressors and interval support together:

```bash
foresight models list \
  --capability supports_x_cols=true \
  --capability supports_interval_forecast_with_x_cols=true \
  --format json
```

### Need artifact save/load

Filter for:

```bash
foresight models list --capability supports_artifact_save=true --format json
```

This is the safest filter when you plan to train once and reuse the fitted
artifact from CLI or Python later.

### Need the smallest dependency footprint

Filter by extra:

```bash
foresight models list --extra core --format json
foresight models list --extra stats --format json
foresight models list --extra ml --format json
foresight models list --extra torch --format json
```

## Stability Guidance

- Prefer `stable` for production baselines, reproducible examples, and first-time package adoption.
- Use `beta` when you want broader torch coverage and can tolerate faster iteration in model internals.
- Use `experimental` when you are explicitly exploring frontier architectures, wrappers, or paper-zoo style models.

```bash
foresight models list --stability stable --format json
foresight models list --stability beta --format json
foresight models list --stability experimental --format json
```

## Practical Defaults

- If you need a first benchmark: start with `naive-last`, `seasonal-naive`, `theta`, `holt`.
- If you need local xreg: start with `sarimax` or one of the `xgb-*` lag models.
- If you need scalable panel forecasting without neural training: start with `xgb-step-lag-global` or another `*-step-lag-global`.
- If you need neural global forecasting: start with a `torch-*-global` model that already advertises the capabilities you need.

## Recommended Workflow

1. Use [`Compatibility guide`](compatibility.md) to pick the smallest install.
2. Run `foresight doctor` to confirm dependency and dataset resolution.
3. Use `foresight models list/info/search` with `--interface`, `--extra`, `--stability`, and `--capability`.
4. Validate candidate models with a small `eval run` or `leaderboard models` comparison before scaling up.

For the raw registry fields behind these filters, see the generated
[Model capability matrix](models.md).
