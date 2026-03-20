# ForeSight

ForeSight is a registry-driven time-series forecasting package with a stable
Python API, a CLI-first workflow, probabilistic forecasts, and a broad mix of
classical, ML, and deep-learning model families.

## Start here

- [Artifact workflow](artifacts.md)
- [CLI runtime logging](cli-logging.md)
- [Model capability matrix](models.md)
- [Python API reference](api.md)
- [Install guide](INSTALL.md)
- [Development guide](DEVELOPMENT.md)
- [Release guide](RELEASE.md)
- [RNN Paper Zoo](rnn_paper_zoo.md)
- [RNN Zoo](rnn_zoo.md)

## Stable entry points

The public import surface is the root package:

```python
from foresight import eval_model, forecast_model, make_forecaster_object
```

Use the generated model matrix for registry capabilities, and use the API page
for the supported forecast, evaluation, artifact, tuning, and data-prep entry
points.

## Artifact CLI

Use the artifact workflow to save, inspect, validate, and compare fitted models:

```bash
foresight forecast csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --save-artifact /tmp/naive-last.pkl
foresight artifact info --artifact /tmp/naive-last.pkl
foresight artifact info --artifact /tmp/naive-last.pkl --format markdown
foresight artifact validate --artifact /tmp/naive-last.pkl
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix metadata.train_schema.runtime --format csv
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix tracking_summary --format csv
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix future_override_schema --format markdown
```

## Cross-validation CLI

Use the CV workflow when you need row-level rolling-origin predictions from a
registered dataset or an arbitrary CSV file:

```bash
foresight cv run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step-size 3 --min-train-size 12 --n-windows 30

foresight cv csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates \
    --horizon 3 --step-size 1 --min-train-size 24
```

## Anomaly Detection CLI

Use the anomaly workflow when you need row-level residual or rolling-score flags:

```bash
foresight detect run --dataset catfish --y-col Total \
    --model naive-last --score-method forecast-residual \
    --min-train-size 24 --step-size 1

foresight detect csv --path ./anomaly.csv \
    --time-col ds --y-col y --parse-dates \
    --score-method rolling-zscore --threshold-method zscore
```

## CLI runtime logging

Long-running CLI commands now print runtime logs to `stderr` by default. That
keeps `stdout` stable for machine-readable `json` / `csv` output. See the full
guide at [CLI runtime logging](cli-logging.md).

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10 \
    --format json > /tmp/forecast.json

foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 \
    --no-progress --log-style plain --log-file /tmp/eval-log.jsonl
```
