<div align="center">

# ForeSight

**A lightweight, batteries-included time-series forecasting toolkit for Python.**

Unified model registry &bull; Walk-forward backtesting &bull; Probabilistic forecasting &bull; CLI + Python API

[![PyPI](https://img.shields.io/pypi/v/foresight-ts?color=blue)](https://pypi.org/project/foresight-ts/)
[![Python](https://img.shields.io/pypi/pyversions/foresight-ts)](https://pypi.org/project/foresight-ts/)
[![License](https://img.shields.io/github/license/skygazer42/ForeSight)](https://github.com/skygazer42/ForeSight/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/skygazer42/ForeSight)](https://github.com/skygazer42/ForeSight)
[![Last commit](https://img.shields.io/github/last-commit/skygazer42/ForeSight)](https://github.com/skygazer42/ForeSight/commits/main)

[Installation](#-installation) &middot; [Quick Start](#-quick-start) &middot; [Docs](https://skygazer42.github.io/ForeSight/) &middot; [Models](#-model-zoo) &middot; [Evaluation](#-evaluation--backtesting) &middot; [Contributing](#-contributing)

</div>

---

ForeSight ships **250+ forecasting models** — from naive baselines and exponential smoothing to Transformers and Mamba — behind a **unified interface**, with built-in walk-forward backtesting, cross-validation, conformal intervals, hierarchical reconciliation, and a CLI-first workflow.

Core models require only `numpy` and `pandas`; heavier backends (PyTorch, XGBoost, LightGBM, CatBoost, statsmodels, scikit-learn) are opt-in extras.

### Why ForeSight?

- **250+ models, one interface** — statistical, ML, and deep learning models all share the same `forecaster(train, horizon) → yhat` contract
- **Backtesting-first** — walk-forward evaluation with expanding or rolling windows, cross-validation predictions table, per-step metrics
- **Panel / global models** — first-class multi-series support via `unique_id / ds / y` long format (compatible with StatsForecast / Prophet)
- **Probabilistic forecasting** — quantile regression, conformal intervals, bootstrap intervals, CRPS
- **Multivariate & hierarchical** — VAR models, top-down / bottom-up reconciliation
- **Tuning** — built-in grid search over model parameters with backtesting-based scoring
- **Production-friendly** — `fit` / `predict` object API, model artifact save/load, forecast CLI for direct predictions
- **Minimal by default** — core depends only on `numpy` + `pandas`; everything else is opt-in

> Design inspired by [StatsForecast](https://github.com/Nixtla/statsforecast), [Darts](https://github.com/unit8co/darts), [sktime](https://www.sktime.org/), [NeuralForecast](https://github.com/Nixtla/neuralforecast), and [Prophet](https://facebook.github.io/prophet/). See [Related Projects](#-related-projects).

---

## 📦 Installation

```bash
pip install foresight-ts                # core (numpy + pandas only)
```

Install optional backends as needed:

```bash
pip install "foresight-ts[ml]"          # scikit-learn models
pip install "foresight-ts[xgb]"         # XGBoost models
pip install "foresight-ts[lgbm]"        # LightGBM models
pip install "foresight-ts[catboost]"    # CatBoost models
pip install "foresight-ts[stats]"       # statsmodels (ARIMA, ETS, VAR, …)
pip install "foresight-ts[torch]"       # PyTorch neural models
pip install "foresight-ts[all]"         # everything above
```

<details>
<summary><b>Install from source (for development)</b></summary>

```bash
git clone https://github.com/skygazer42/ForeSight.git
cd ForeSight
pip install -e ".[dev]"     # editable install + pytest, ruff
```

</details>

---

## 🚀 Quick Start

### Python API

```python
from foresight import eval_model, make_forecaster, make_forecaster_object

# 1. Walk-forward evaluation on a built-in dataset
metrics = eval_model(
    model="theta", dataset="catfish", y_col="Total",
    horizon=3, step=3, min_train_size=12,
)
print(metrics)  # {'mae': ..., 'rmse': ..., 'mape': ..., 'smape': ...}

# 2. Functional API — stateless forecaster
f = make_forecaster("holt", alpha=0.3, beta=0.1)
yhat = f([112, 118, 132, 129, 121, 135, 148, 148], horizon=3)

# 3. Object API — fit / predict / save / load
obj = make_forecaster_object("moving-average", window=3)
obj.fit([1, 2, 3, 4, 5, 6])
yhat = obj.predict(3)
```

<details>
<summary><b>More Python API examples</b></summary>

```python
import pandas as pd
from foresight import (
    bootstrap_intervals, forecast_model, tune_model,
    save_forecaster, load_forecaster,
    forecast_model_long_df,
    make_global_forecaster, make_global_forecaster_object,
    make_multivariate_forecaster, eval_multivariate_model_df,
    to_long, prepare_long_df,
    build_hierarchy_spec, reconcile_hierarchical_forecasts,
    eval_hierarchical_forecast_df,
)

# Forecast with bootstrap prediction intervals
future_df = forecast_model(
    model="naive-last",
    y=[1, 2, 3, 4, 5, 6],
    ds=pd.date_range("2024-01-01", periods=6, freq="D"),
    horizon=3,
    interval_levels=(0.8, 0.9),
    interval_min_train_size=4,
)

# Save / load trained model artifacts
obj = make_forecaster_object("theta", alpha=0.3)
obj.fit([1, 2, 3, 4, 5])
save_forecaster(obj, "/tmp/theta.pkl")
loaded = load_forecaster("/tmp/theta.pkl")
yhat = loaded.predict(3)

# Serialized artifacts carry a schema version plus package/model metadata.
# If a future release rejects an older artifact as incompatible, re-save it
# with the current foresight package.

# Grid search tuning
result = tune_model(
    model="moving-average", dataset="catfish", y_col="Total",
    horizon=1, step=1, min_train_size=24, max_windows=8,
    search_space={"window": (1, 3, 6)},
)

# Global / panel model (trains across all series)
# g = make_global_forecaster("ridge-step-lag-global", lags=48, alpha=1.0, x_cols=("promo",))
# pred_df = g(long_df, cutoff=cutoff, horizon=14)  # returns: unique_id, ds, yhat
#
# Or keep observed history and known-future covariates in separate tables.
# pred_df = forecast_model_long_df(
#     model="ridge-step-lag-global",
#     long_df=history_df,
#     future_df=future_covariates_df,
#     horizon=14,
#     model_params={
#         "future_x_cols": ("promo", "price"),
#         "target_lags": (1, 7, 14),
#         "future_x_lags": (1, 0),
#     },
# )
#
# g_obj = make_global_forecaster_object("ridge-step-lag-global", lags=48, alpha=1.0)
# g_obj.fit(long_df)
# pred_df = g_obj.predict(cutoff=cutoff, horizon=14)

# Multivariate model (VAR)
mv = make_multivariate_forecaster("var", maxlags=1)
yhat_mv = mv(wide_df[["sales", "traffic"]], horizon=2)  # shape: (2, 2)

# Hierarchical reconciliation
hierarchy = build_hierarchy_spec(raw_df, id_cols=("region", "store"), root="total")
reconciled = reconcile_hierarchical_forecasts(
    forecast_df=pred_df, hierarchy=hierarchy,
    method="top_down", history_df=history_long,
    exog_agg={"promo": "sum", "temp": "mean"},
)

hier_payload = eval_hierarchical_forecast_df(
    forecast_df=pred_df,
    hierarchy=hierarchy,
    method="top_down",
    history_df=history_long,
)
```

</details>

### CLI

```bash
# Quick discovery shortcuts
foresight --list
foresight --list-models
foresight --list-datasets

# Browse built-in datasets and models
foresight datasets list
foresight datasets preview catfish --nrows 10
foresight models list
foresight models info theta

# Data utilities (format conversion / prep)
foresight data to-long --path ./my.csv --time-col ds --y-col y --parse-dates
foresight data prepare-long --path ./long.csv --parse-dates --freq D --y-missing ffill
foresight data infer-freq --path ./my.csv --time-col ds --parse-dates
foresight data splits rolling-origin --n-obs 100 --horizon 7 --min-train-size 28 --step-size 7

# Evaluate a single model
foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12

# Grid search tuning
foresight tuning run --model moving-average --dataset catfish --y-col Total \
    --horizon 1 --step 1 --min-train-size 24 --max-windows 8 \
    --grid-param window=1,3,6

# Compare multiple models on a leaderboard
foresight leaderboard models --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 \
    --models naive-last,seasonal-naive,theta,holt

# Multi-dataset sweep (parallel + resumable)
foresight leaderboard sweep \
    --datasets catfish,ice_cream_interest \
    --models naive-last,theta --horizon 3 --step 3 \
    --min-train-size 12 --jobs 4 --progress

# Cross-validation predictions table
foresight cv run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step-size 3 --min-train-size 12 --n-windows 30

# Conformal prediction intervals
foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 --conformal-levels 80,90

# Reproducible packaged benchmark smoke run
python benchmarks/run_benchmarks.py --smoke
```

Want the code-level reading order? See [`docs/SOURCE_ENTRYPOINTS.md`](docs/SOURCE_ENTRYPOINTS.md).

<details>
<summary><b>More CLI examples (forecast, artifacts, covariates)</b></summary>

```bash
# Direct forecast from any CSV
foresight forecast csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3

# Forecast with bootstrap prediction intervals
foresight forecast csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --interval-levels 80,90 --interval-min-train-size 12

# Save and reuse model artifacts
foresight forecast csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --save-artifact /tmp/naive-last.pkl
foresight forecast artifact --artifact /tmp/naive-last.pkl --horizon 3 --format json

# Artifact loads validate schema compatibility and fail early with a clear
# error if the saved payload comes from an unsupported artifact format.

# SARIMAX with future covariates
foresight forecast csv --model sarimax --path ./my_exog.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param order=0,0,0 --model-param seasonal_order=0,0,0,0 \
    --model-param trend=c --model-param x_cols=promo

# Seasonal auto-ARIMA
foresight forecast csv --model auto-arima --path ./monthly.csv \
    --time-col ds --y-col y --parse-dates --horizon 6 \
    --model-param max_p=2 --model-param max_d=1 --model-param max_q=2 \
    --model-param max_P=1 --model-param max_D=1 --model-param max_Q=1 \
    --model-param seasonal_period=12

# Evaluate any CSV (no dataset registry needed)
foresight eval csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 --step 1 --min-train-size 12

# Leaderboard: summarize across datasets
foresight leaderboard summarize --input /tmp/sweep.json --format md --min-datasets 2
```

</details>

### Reproducible benchmark smoke run

ForeSight ships a tiny regression benchmark harness for packaged demo datasets.
The checked-in config lives in `benchmarks/benchmark_config.json`, and the CI
lane runs the smoke subset:

```bash
python benchmarks/run_benchmarks.py --smoke
python benchmarks/run_benchmarks.py --config baseline --format md
```

The smoke config intentionally stays small: packaged datasets only, dependency-free
baseline models only, fixed backtest settings, and deterministic summary ordering.

---

## 🧠 Model Zoo

ForeSight organizes **250+** registered models into families. Core models are dependency-free; optional models are activated by installing the corresponding extra.

### Core Models (no extra dependencies)

| Family | Models | Key Parameters |
|--------|--------|---------------|
| **Naive / Baseline** | `naive-last`, `seasonal-naive`, `mean`, `median`, `drift`, `moving-average`, `weighted-moving-average`, `moving-median`, `seasonal-mean`, `seasonal-drift` | `season_length`, `window` |
| **Exponential Smoothing** | `ses`, `ses-auto`, `holt`, `holt-auto`, `holt-damped`, `holt-winters-add`, `holt-winters-add-auto` | `alpha`, `beta`, `gamma`, `season_length` |
| **Theta** | `theta`, `theta-auto` | `alpha`, `grid_size` |
| **AR / Regression** | `ar-ols`, `ar-ols-lags`, `sar-ols`, `ar-ols-auto`, `lr-lag`, `lr-lag-direct` | `p`, `lags`, `season_length` |
| **Fourier / Spectral** | `fourier`, `fourier-multi`, `poly-trend`, `fft` | `period`, `order`, `top_k` |
| **Kalman Filter** | `kalman-level`, `kalman-trend` | `process_variance`, `obs_variance` |
| **Analog** | `analog-knn` | `lags`, `k`, `weights` |
| **Intermittent Demand** | `croston`, `croston-sba`, `croston-sbj`, `croston-opt`, `tsb`, `les`, `adida` | `alpha`, `beta` |
| **Meta / Ensemble** | `pipeline`, `ensemble-mean`, `ensemble-median` | `base`, `members`, `transforms` |

### Optional Models

<details>
<summary><b>scikit-learn</b> — <code>pip install "foresight-ts[ml]"</code></summary>

**Local (lag-feature + direct multi-horizon):**

`ridge-lag`, `ridge-lag-direct`, `rf-lag`, `decision-tree-lag`, `extra-trees-lag`, `adaboost-lag`, `bagging-lag`, `lasso-lag`, `elasticnet-lag`, `knn-lag`, `gbrt-lag`, `hgb-lag`, `svr-lag`, `linear-svr-lag`, `kernel-ridge-lag`, `mlp-lag`, `huber-lag`, `quantile-lag`, `sgd-lag`

**Global/panel (step-lag, trains across all series):**

`ridge-step-lag-global`, `decision-tree-step-lag-global`, `bagging-step-lag-global`, `gbrt-step-lag-global`, `lasso-step-lag-global`, `elasticnet-step-lag-global`, `knn-step-lag-global`, `kernel-ridge-step-lag-global`, `svr-step-lag-global`, `linear-svr-step-lag-global`, `huber-step-lag-global`, `quantile-step-lag-global`, `sgd-step-lag-global`, `adaboost-step-lag-global`, `mlp-step-lag-global`, `rf-step-lag-global`, `extra-trees-step-lag-global`

These sklearn global step-lag models support `x_cols`, `add_time_features`, `id_feature`, explicit `target_lags`, covariate-aware lag blocks via `historic_x_lags` / `future_x_lags`, and derived lag features (`roll_windows`, `roll_stats`, `diff_lags`).

</details>

<details>
<summary><b>XGBoost</b> — <code>pip install "foresight-ts[xgb]"</code></summary>

**Local models:**

| Strategy | Models |
|----------|--------|
| Direct | `xgb-lag`, `xgb-dart-lag`, `xgbrf-lag`, `xgb-linear-lag` |
| Recursive | `xgb-lag-recursive`, `xgb-dart-lag-recursive`, `xgb-linear-lag-recursive` |
| Step-index | `xgb-step-lag` |
| DirRec | `xgb-dirrec-lag` |
| MIMO | `xgb-mimo-lag` |
| Custom objectives | `xgb-mae-lag(-recursive)`, `xgb-huber-lag(-recursive)`, `xgb-quantile-lag(-recursive)`, `xgb-poisson-lag(-recursive)`, `xgb-gamma-lag(-recursive)`, `xgb-tweedie-lag(-recursive)`, `xgb-msle-lag(-recursive)`, `xgb-logistic-lag(-recursive)` |
| Customizable | `xgb-custom-lag`, `xgb-custom-lag-recursive`, `xgb-custom-step-lag`, `xgb-custom-dirrec-lag`, `xgb-custom-mimo-lag` |

**Global/panel (step-lag, supports `quantiles` for probabilistic output):**

`xgb-step-lag-global`, `xgb-mae-step-lag-global`, `xgb-huber-step-lag-global`, `xgb-poisson-step-lag-global`, `xgb-gamma-step-lag-global`, `xgb-tweedie-step-lag-global`, `xgb-msle-step-lag-global`, `xgb-logistic-step-lag-global`, `xgb-dart-step-lag-global`, `xgb-linear-step-lag-global`, `xgbrf-step-lag-global`

</details>

<details>
<summary><b>LightGBM</b> — <code>pip install "foresight-ts[lgbm]"</code></summary>

**Local:** `lgbm-lag`, `lgbm-lag-recursive`, `lgbm-step-lag`, `lgbm-dirrec-lag`, `lgbm-custom-lag`, `lgbm-custom-lag-recursive`, `lgbm-custom-step-lag`, `lgbm-custom-dirrec-lag`

**Global/panel:** `lgbm-step-lag-global` (supports `quantiles`)

</details>

<details>
<summary><b>CatBoost</b> — <code>pip install "foresight-ts[catboost]"</code></summary>

**Local:** `catboost-lag`, `catboost-lag-recursive`, `catboost-step-lag`, `catboost-dirrec-lag`, `catboost-custom-lag`, `catboost-custom-lag-recursive`, `catboost-custom-step-lag`, `catboost-custom-dirrec-lag`

**Global/panel:** `catboost-step-lag-global` (supports `quantiles`)

</details>

<details>
<summary><b>statsmodels</b> — <code>pip install "foresight-ts[stats]"</code></summary>

| Family | Models |
|--------|--------|
| ARIMA / Harmonic Regression | `arima`, `auto-arima`, `fourier-arima`, `fourier-auto-arima`, `fourier-autoreg`, `fourier-sarimax`, `sarimax`, `autoreg` |
| Multivariate | `var` |
| Unobserved Components | `uc-local-level`, `uc-local-linear-trend`, `uc-seasonal` |
| Decomposition | `stl-arima`, `stl-autoreg`, `stl-ets`, `mstl-arima`, `mstl-autoreg`, `mstl-ets`, `mstl-auto-arima`, `tbats-lite`, `tbats-lite-autoreg`, `tbats-lite-auto-arima` |
| ETS | `ets` |

`sarimax` and `auto-arima` support exogenous covariates via `x_cols`. `fourier-arima`, `fourier-auto-arima`, `fourier-autoreg`, and `fourier-sarimax` add deterministic Fourier seasonality with residual modeling on top. `stl-arima`, `stl-autoreg`, and `stl-ets` apply STL decomposition before forecasting the remainder. `mstl-arima`, `mstl-autoreg`, `mstl-ets`, and `mstl-auto-arima` extend that pattern to multiple seasonal periods. `tbats-lite`, `tbats-lite-autoreg`, and `tbats-lite-auto-arima` fit multi-season Fourier structure with residual ARIMA-family errors.

</details>

<details>
<summary><b>PyTorch — Local models</b> — <code>pip install "foresight-ts[torch]"</code></summary>

| Category | Models |
|----------|--------|
| MLP / Linear | `torch-mlp-direct`, `torch-nlinear-direct`, `torch-dlinear-direct`, `torch-tide-direct`, `torch-kan-direct` |
| RNN | `torch-lstm-direct`, `torch-gru-direct`, `torch-bilstm-direct`, `torch-bigru-direct`, `torch-attn-gru-direct`, `torch-dilated-rnn-direct` |
| CNN | `torch-cnn-direct`, `torch-tcn-direct`, `torch-resnet1d-direct`, `torch-wavenet-direct`, `torch-inception-direct`, `torch-scinet-direct` |
| Transformer | `torch-transformer-direct`, `torch-patchtst-direct`, `torch-crossformer-direct`, `torch-pyraformer-direct`, `torch-fnet-direct`, `torch-linear-attn-direct`, `torch-gmlp-direct`, `torch-tsmixer-direct`, `torch-retnet-direct`, `torch-retnet-recursive` |
| Residual Blocks | `torch-nbeats-direct`, `torch-nhits-direct` |
| SSM / State-space | `torch-mamba-direct`, `torch-rwkv-direct`, `torch-hyena-direct` |
| Hybrid | `torch-etsformer-direct`, `torch-esrnn-direct`, `torch-lstnet-direct` |
| Probabilistic | `torch-deepar-recursive`, `torch-qrnn-recursive` |
| Seq2Seq | `torch-seq2seq-lstm-direct`, `torch-seq2seq-gru-direct` (optional Bahdanau attention + teacher forcing) |
| RNN Paper Zoo | 100 named paper architectures (`torch-rnnpaper-*-direct`) — manual unroll, no PyTorch built-in RNN modules |
| RNN Zoo | 100 combos: 20 bases × 5 wrappers (`torch-rnnzoo-*-direct`: direct/bidir/ln/attn/proj) |
| Configurable Transformer | `torch-xformer-*-direct` — attention: full/local/logsparse/longformer/performer/linformer/nystrom/probsparse/autocorr/reformer; position: RoPE/sincos/Time2Vec; extras: RMSNorm/SwiGLU/RevIN |

> Tip: `foresight models list --prefix torch-rnnpaper` to filter. Docs: `docs/rnn_paper_zoo.md`, `docs/rnn_zoo.md`

</details>

<details>
<summary><b>PyTorch — Global/panel models</b> — <code>pip install "foresight-ts[torch]"</code></summary>

Train across all series in long-format DataFrame; supports covariates (`x_cols`), time features, and optional quantile regression.

| Category | Models |
|----------|--------|
| Transformer | `torch-tft-global`, `torch-informer-global`, `torch-autoformer-global`, `torch-fedformer-global`, `torch-nonstationary-transformer-global`, `torch-patchtst-global`, `torch-crossformer-global`, `torch-pyraformer-global`, `torch-itransformer-global`, `torch-timesnet-global`, `torch-tsmixer-global`, `torch-transformer-encdec-global`, `torch-retnet-global` |
| MLP / Linear | `torch-nbeats-global`, `torch-nhits-global`, `torch-nlinear-global`, `torch-dlinear-global`, `torch-tide-global` |
| RNN | `torch-deepar-global`, `torch-lstnet-global`, `torch-esrnn-global`, `torch-rnn-*-global` |
| CNN | `torch-tcn-global`, `torch-wavenet-global`, `torch-resnet1d-global`, `torch-inception-global`, `torch-scinet-global` |
| SSM | `torch-ssm-global`, `torch-mamba-global`, `torch-rwkv-global`, `torch-hyena-global` |
| Other | `torch-fnet-global`, `torch-gmlp-global`, `torch-dilated-rnn-global`, `torch-kan-global`, `torch-etsformer-global`, `torch-seq2seq-*-global`, `torch-xformer-*-global` |

</details>

Discover all models and their parameters:

```bash
foresight models list                     # list all registered models
foresight models list --prefix xgb        # filter by prefix
foresight models info holt-winters-add    # show parameters & defaults
```

Model discovery output also surfaces machine-readable capability flags from the
registry. These fields are intended to stay stable across CLI JSON and Python
registry usage:

- `supports_x_cols`: model accepts future covariates / exogenous regressors
- `supports_quantiles`: model can emit quantile forecast columns directly
- `supports_interval_forecast`: model supports forecast intervals in at least one supported path
- `supports_interval_forecast_with_x_cols`: model supports intervals when `x_cols` are used
- `supports_artifact_save`: model can be saved and reused through the artifact workflow
- `requires_future_covariates`: model requires known future covariates instead of treating them as optional

### Derived Lag Features

All lag-based regression models support optional feature engineering — leakage-free by design:

```bash
--model-param roll_windows=3,7,14         # rolling statistics over lag window tail
--model-param roll_stats=mean,std,min,max,median,slope
--model-param diff_lags=1,7,14            # lag differences (last − lag(k+1))
```

### Multi-horizon Strategies

| Strategy | How it works | Example suffix |
|----------|-------------|---------------|
| **Direct** | One model per horizon step | `*-lag` |
| **Recursive** | One-step model, iteratively re-fed | `*-lag-recursive` |
| **Step-index** | Single model with step as a feature | `*-step-lag` |
| **DirRec** | Per-step model with previous-step features | `*-dirrec-lag` |
| **MIMO** | Single model predicts entire horizon | `*-mimo-lag` |

---

## 📐 Data Format

ForeSight uses a panel-friendly **long format** compatible with StatsForecast and Prophet:

| Column | Description |
|--------|-------------|
| `unique_id` | Series identifier (optional for single series) |
| `ds` | Timestamp (`datetime`) |
| `y` | Target value (`float`) |
| *extra columns* | Covariates / exogenous features |

### Covariate roles

| Surface | Meaning |
|--------|---------|
| `y` | Forecast target |
| `historic_x_cols` | Covariates observed only up to the forecast cutoff |
| `future_x_cols` | Covariates known through the forecast horizon |
| `x_cols` | Backward-compatible alias for `future_x_cols` |

```python
from foresight.data import to_long, validate_long_df, prepare_long_df

# Convert raw data to long format with role-aware covariates
long_df = to_long(
    raw_df, time_col="ds", y_col="y",
    id_cols=("store", "dept"),
    historic_x_cols=("promo_hist",),
    future_x_cols=("promo_plan", "price"),
)

# Optional: fill missing timestamps, handle NaN by role
prepared = prepare_long_df(
    long_df, freq="D",
    y_missing="interpolate",
    historic_x_missing="ffill",
    future_x_missing="ffill",
)

# Validate
validate_long_df(prepared)
```

If known-future covariates arrive separately from observed history,
`forecast_model_long_df(...)` also accepts `future_df=...` and validates
horizon coverage per series before forecasting.

---

## 📊 Evaluation & Backtesting

| Capability | Description |
|-----------|-------------|
| **Walk-forward backtesting** | Expanding or rolling-window evaluation (`horizon`, `step`, `min_train_size`, `max_train_size`) |
| **Cross-validation table** | Full predictions DataFrame: `unique_id, ds, cutoff, step, y, yhat, model` |
| **Leaderboard** | Multi-model comparison on a single dataset |
| **Sweep** | Multi-dataset × multi-model benchmark with parallel workers + resume |
| **Tuning** | Grid search over model parameters, scored via backtesting |
| **Conformal intervals** | Symmetric prediction intervals from backtesting residuals |
| **Bootstrap intervals** | Non-parametric prediction intervals |

**Point metrics:** MAE, RMSE, MAPE, sMAPE, WAPE, MASE, RMSSE, MSE

**Probabilistic metrics:** Pinball loss, CRPS, interval coverage, interval width, sharpness, interval score, Winkler score, weighted interval score, MSIS

When `--conformal-levels` is enabled, the evaluation payload additionally reports per-level coverage, calibration gap, mean width, sharpness, interval score, and Winkler score (both overall and per-step).

### Probabilistic Forecasting

Torch global models and gradient boosting global models support `quantiles` for pinball-loss-trained prediction intervals:

```bash
foresight eval run --model torch-itransformer-global \
    --dataset catfish --y-col Total --horizon 7 --step 7 --min-train-size 60 \
    --model-param quantiles=0.1,0.5,0.9
```

This produces `yhat_p10`, `yhat_p50`, `yhat_p90` columns alongside `yhat` (which defaults to the median quantile).

---

## 📦 Optional Dependencies

| Extra | Backend | Version | Example Models |
|-------|---------|---------|---------------|
| `[ml]` | scikit-learn | ≥ 1.0 | `ridge-lag`, `rf-lag`, `hgb-lag`, `mlp-lag`, `*-step-lag-global` |
| `[xgb]` | XGBoost | ≥ 2.0 | `xgb-lag`, `xgb-step-lag`, `xgb-mimo-lag`, `xgb-step-lag-global` |
| `[lgbm]` | LightGBM | ≥ 4.0 | `lgbm-lag`, `lgbm-step-lag-global` |
| `[catboost]` | CatBoost | ≥ 1.2 | `catboost-lag`, `catboost-step-lag-global` |
| `[stats]` | statsmodels | ≥ 0.14 | `arima`, `auto-arima`, `sarimax`, `ets`, `var`, `fourier-*`, `stl-*`, `mstl-*`, `tbats-lite-*`, `uc-*` |
| `[torch]` | PyTorch | ≥ 2.0 | `torch-transformer-direct`, `torch-tft-global`, `torch-mamba-global` |
| `[all]` | All of the above | — | — |

---

## 🗂️ Repository Structure

```
src/foresight/              Python package — CLI, model registry, eval, backtesting
examples/                   Runnable scripts (no notebooks)
data/                       Bundled CSV datasets (catfish, ice_cream_interest, …)
statistics time series/     Classic statistics experiments (ARMA, ARIMA, VAR)
ml time series/             ML feature-engineering scripts (Prophet, tree models)
transformer time series/    DL / Transformer experiments (Informer, Autoformer, …)
paper/                      Reading notes & paper summaries
```

**Example scripts:**
`examples/quickstart_eval.py`, `examples/leaderboard.py`, `examples/cv_and_conformal.py`, `examples/intermittent_demand.py`, `examples/torch_global_models.py`, `examples/rnn_paper_zoo.py`

---

## 🤝 Contributing

```bash
ruff check src tests tools    # lint
ruff format src tests tools   # format
pytest -q                     # test
```

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for detailed development guidelines.

---

## 🔗 Related Projects

ForeSight draws design inspiration from these excellent projects:

| Project | Highlights |
|---------|-----------|
| [StatsForecast](https://github.com/Nixtla/statsforecast) | Fast statistical baselines, `unique_id/ds/y` convention, cross-validation |
| [NeuralForecast](https://github.com/Nixtla/neuralforecast) | Modern neural architectures (TFT, Informer, Autoformer, …) |
| [Darts](https://github.com/unit8co/darts) | Unified model API + backtesting helpers |
| [sktime](https://www.sktime.org/) | Unified `fit`/`predict` interface and evaluation utilities |
| [GluonTS](https://github.com/awslabs/gluonts) | Probabilistic forecasting datasets + benchmarking |
| [Prophet](https://facebook.github.io/prophet/) | `ds/y` DataFrame convention for forecasting |
| [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) | Deep learning forecasting pipelines |
| [Kats](https://github.com/facebookresearch/Kats) | Time series analysis & forecasting toolbox (Meta) |

---

## ⚖️ License

[GPL-3.0-only](LICENSE)
