# CLI Runtime Logging

ForeSight writes enhanced runtime logs to `stderr` for long-running CLI commands.
This keeps `stdout` stable for machine-readable `json` / `csv` output and shell
pipelines.

Typical commands that emit runtime logs:

- `foresight forecast csv`
- `foresight forecast artifact`
- `foresight eval run`
- `foresight eval csv`
- `foresight cv run`
- `foresight leaderboard models`
- `foresight leaderboard sweep`
- `foresight tuning run`
- `foresight artifact info`
- `foresight artifact validate`
- `foresight artifact diff`

## Why logs go to `stderr`

ForeSight's CLI is designed so structured results still flow through `stdout`.
That matters for:

- `--format json` / `--format csv`
- shell pipes such as `| jq` or `| head`
- redirecting forecast payloads into files
- automation that parses CLI output

Runtime logs therefore stay on `stderr`, where they remain visible in an
interactive terminal without breaking output contracts.

## Default behavior

By default:

- lifecycle events such as `RUN start` and `RUN done` are shown
- progress events such as `EPOCH 3/10`, CV windows, and tuning trials are shown
- the renderer prefers rich terminal output when the optional `rich` package is available
- if `rich` is unavailable, output falls back to plain text automatically

Example:

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10 \
    --format json > /tmp/forecast.json
```

The forecast payload lands in `/tmp/forecast.json`, while run and training logs
still appear in the terminal on `stderr`.

## Logging controls

### `--no-progress`

Suppresses progress-style events while keeping run lifecycle summaries.

```bash
foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 \
    --no-progress
```

Use this when you want:

- cleaner CI logs
- less per-epoch torch output
- only start / finish / failure summaries

### `--log-style {auto,rich,plain,quiet}`

Controls terminal rendering:

- `auto`: prefer rich output when available, else plain
- `rich`: force rich-style rendering
- `plain`: force plain text rendering
- `quiet`: suppress terminal logging entirely

Example:

```bash
foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 \
    --log-style plain
```

### `--log-level {info,debug}`

Selects verbosity for runtime events.

- `info` is the default
- `debug` is reserved for higher-detail troubleshooting as more debug events are added

### `--log-file`

Writes the same runtime events to a JSONL file.

```bash
foresight eval run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step 3 --min-train-size 12 \
    --log-file /tmp/eval-log.jsonl
```

This is useful for:

- collecting logs from CI runs
- post-run debugging
- feeding event streams into external tooling

When `--log-style quiet` is used together with `--log-file`, terminal output is
suppressed but JSONL logging still continues.

## Event types

Current event streams are organized by workflow stage.

### Command lifecycle

All instrumented commands emit:

- `RUN start`
- `RUN done`
- `RUN failed`

### Forecast / artifact workflows

Examples include:

- `LOAD csv`
- `FORECAST ready`
- `ARTIFACT saved`
- `ARTIFACT load`

### Evaluation / CV / tuning

Examples include:

- `PHASE params`
- `PHASE eval`
- `PHASE cv`
- `PHASE forecast`
- `PHASE emit`
- `EVAL series`
- `EVAL done`
- `CV start`
- `CV series`
- `CV cutoff 3/10`
- `CV done`
- `TUNE start`
- `TRIAL 4/16`

### Torch training

Torch-backed models emit trainer lifecycle events such as:

- `TRAIN start`
- `EPOCH 1/10`
- `EARLY stop`
- `CHECKPOINT saved`
- `CHECKPOINT resume`
- `TRAIN done`

Structured JSONL payloads for these events now include higher-value training
telemetry commonly expected from mature deep learning tooling, including:

- run identity metadata such as stable per-command `run_id`, `pid`, and `python_version`
- dataset and batch context such as `train_samples`, `val_samples`, and `effective_batch_size`
- model footprint signals such as `trainable_parameters` and `total_parameters`
- optimization context such as `optimizer`, `scheduler`, `amp`, current `lr`, and `avg_grad_norm`
- device context such as `device_type`, `cuda_available`, and CUDA memory snapshots when training on GPU
- per-epoch timing / throughput such as `epoch_seconds`, `step_seconds`,
  `samples_per_second`, and `batches_per_second`
- completion metadata such as `best_epoch`, `best_improved`, `stop_reason`, and `total_seconds`

For phase events, payloads also include:

- `phase`: normalized phase name for the current command stage
- `elapsed_ms`: wall-clock milliseconds spent in that stage

This is useful when you want a fast breakdown of parameter parsing, dataset
preparation, evaluation, forecasting, and output serialization without changing
the machine-readable payload written to `stdout`.

### Optional TensorBoard tracking

Torch-backed models can also export epoch metrics to TensorBoard through
model params:

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10 \
    --model-param tensorboard_log_dir=./runs \
    --model-param tensorboard_run_name=mlp-demo
```

When enabled, the shared trainer writes scalar series such as loss, learning
rate, gradient norm, throughput, and GPU memory (when training on CUDA). It
also writes:

- `foresight/hparams` text metadata for the full trainer config
- a TensorBoard HParams run summary when the backend supports `add_hparams(...)`
- `foresight/artifacts` text metadata with the resolved run directory plus
  resume / checkpoint artifact paths such as `best.pt` and `last.pt`

This path is optional and requires TensorBoard support to be installed in the
Python environment.

### Optional MLflow tracking

The same shared trainer can also publish params, metrics, and checkpoint
artifacts to MLflow:

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10 \
    --model-param mlflow_experiment_name=foresight-dev \
    --model-param mlflow_run_name=mlp-demo
```

Add `mlflow_tracking_uri=...` when you want to target a non-default MLflow
backend. When enabled, ForeSight logs trainer hyperparameters, per-epoch scalar
metrics, structured config/device payloads, and saved checkpoints such as
`best.pt` / `last.pt`. This path is optional and requires `mlflow` to be
installed in the Python environment.

### Optional W&B tracking

Weights & Biases is also supported through the same trainer contract:

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10 \
    --model-param wandb_project=foresight-dev \
    --model-param wandb_run_name=mlp-demo \
    --model-param wandb_mode=offline
```

When enabled, ForeSight publishes the trainer config into the W&B run config,
streams epoch metrics, records structured run summaries such as
`foresight/device` and `foresight/artifacts`, and uploads saved checkpoints as
model artifacts when checkpoint saving is enabled. This path is optional and
requires `wandb` to be installed in the Python environment.

## Recommended usage patterns

### Interactive model development

Use the default behavior:

```bash
foresight forecast csv --model torch-mlp-direct --path ./train.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param lags=24 --model-param epochs=10
```

This gives you live visibility into run phases and epoch progress.

### CI or batch automation

Prefer reduced terminal noise plus JSONL persistence:

```bash
foresight tuning run --model moving-average --dataset catfish --y-col Total \
    --horizon 1 --step 1 --min-train-size 24 \
    --grid-param window=1,3,6 \
    --no-progress --log-style plain --log-file /tmp/tuning-log.jsonl
```

### Pipe-safe structured output

Keep `stdout` reserved for payloads:

```bash
foresight cv run --model theta --dataset catfish --y-col Total \
    --horizon 3 --step-size 3 --min-train-size 12 --format json \
    > /tmp/cv.json
```

Runtime logs continue on `stderr`, so the JSON file remains parseable.
