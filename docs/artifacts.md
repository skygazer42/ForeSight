# Artifact Workflow

ForeSight artifacts persist a fitted forecaster together with structured metadata
about the training schema and artifact payload version. The stable Python API is:

```python
from foresight import load_forecaster, load_forecaster_artifact, save_forecaster
```

Use `save_forecaster(...)` to persist a fitted object, `load_forecaster(...)` to
reconstruct it for prediction, and `load_forecaster_artifact(...)` when you want
to inspect the structured payload before rebuilding the runtime object.

Only load artifacts from trusted sources. ForeSight artifacts use Python pickle
under the hood, so loading a malicious artifact can execute arbitrary code.

## Python example

```python
from foresight import make_forecaster_object, load_forecaster, load_forecaster_artifact, save_forecaster

obj = make_forecaster_object("naive-last").fit([1, 2, 3, 4, 5])
save_forecaster(obj, "/tmp/naive-last.pkl")

payload = load_forecaster_artifact("/tmp/naive-last.pkl")
loaded = load_forecaster("/tmp/naive-last.pkl")
```

The artifact payload includes:

- `artifact_schema_version`
- `metadata.model_key`
- `metadata.model_params`
- `metadata.train_schema`
- optional `metadata.train_schema.runtime` for torch-backed models
- optional `extra` values added by CLI workflows

When you inspect an artifact through `foresight artifact info`, any structured
runtime tracking config under `metadata.train_schema.runtime.tracking` is also
promoted into a top-level `tracking` section for easier inspection and diffing,
along with a compact `tracking_backends` list such as
`["mlflow", "tensorboard", "wandb"]` and a concise `tracking_summary` mapping
such as `{"mlflow": "exp / run", "wandb": "project / run [offline]"}`.

## CLI workflow

```bash
# Save an artifact from a forecast run
foresight forecast csv --model naive-last --path ./my.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --save-artifact /tmp/naive-last.pkl

# Save a local x_cols artifact and reuse the saved future covariate context
foresight forecast csv --model sarimax --path ./my_exog.csv \
    --time-col ds --y-col y --parse-dates --horizon 3 \
    --model-param order=0,0,0 --model-param seasonal_order=0,0,0,0 \
    --model-param trend=c --model-param x_cols=promo \
    --save-artifact /tmp/sarimax.pkl
foresight forecast artifact --artifact /tmp/sarimax.pkl --horizon 2

# Override the saved future covariates with a new future CSV
foresight forecast artifact --artifact /tmp/sarimax.pkl \
    --future-path ./my_exog_future.csv --time-col ds --parse-dates \
    --horizon 4

# Reuse a quantile-capable global artifact and derive interval columns
foresight forecast artifact --artifact /tmp/xgb-global.pkl \
    --horizon 2 --interval-levels 80

# Override a saved global artifact with new future covariates
# The override CSV can contain canonical unique_id values or the raw id columns
# that were used when the artifact was saved, plus ds and required x_cols.
# Single-series global artifacts can also omit id columns entirely.
foresight forecast artifact --artifact /tmp/ridge-global.pkl \
    --future-path ./my_global_future.csv --time-col ds --parse-dates \
    --horizon 2

# Inspect summary metadata
foresight artifact info --artifact /tmp/naive-last.pkl

# Render summary metadata as a grouped Markdown report
foresight artifact info --artifact /tmp/naive-last.pkl --format markdown

# Validate artifact schema / metadata contract
foresight artifact validate --artifact /tmp/naive-last.pkl

# Compare two artifacts
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl

# Render grouped Markdown sections for human review
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --format markdown

# Focus on torch runtime metadata differences only
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix metadata.train_schema.runtime --format csv

# Focus on promoted tracking metadata only
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix tracking --format csv

# Focus on the compact human-readable tracking summary only
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix tracking_summary --format csv

# Focus on future override contract differences only
foresight artifact diff \
    --left-artifact /tmp/naive-last.pkl \
    --right-artifact /tmp/naive-last-v2.pkl \
    --path-prefix future_override_schema --format markdown
```

## Notes

- `forecast artifact` reuses a saved fitted object for prediction.
- Local single-series artifacts saved from `forecast csv` with `x_cols` reuse the
  saved future covariate rows and future timestamps. Reuse horizons must be less
  than or equal to the horizon saved in the artifact.
- Those same local `x_cols` artifacts can also accept `--future-path` plus
  `--time-col` to override the saved future covariate context and forecast a new
  horizon from the same fitted history.
- Global artifacts can also accept `--future-path` plus `--time-col` to override
  the saved future context. Because artifact reuse no longer has the original
  CLI arguments, the override CSV must include either canonical `unique_id`
  values or the raw id columns captured when the artifact was saved, along with
  the time column and any required `x_cols`. When the artifact only contains a
  single series, the override CSV can omit id columns entirely and will bind to
  that sole saved series.
- Global artifacts saved from `forecast csv` also preserve future rows supplied
  through a separate `--future-path`, so a later `forecast artifact` call can
  reuse that saved context without passing the future CSV again.
- Global artifacts from interval-capable quantile models can accept
  `--interval-levels` as long as the saved artifact prediction already exposes
  the matching quantile columns.
- `artifact info` returns a JSON-safe summary of `metadata`, optional top-level
  `tracking`, optional `tracking_backends`, optional `tracking_summary`,
  optional derived `future_override_schema`, and `extra`; `--format markdown`
  renders the same payload as grouped summary, tracking, metadata, extra, and
  future-override sections.
- `artifact validate` also checks CLI forecast-artifact runtime contracts, such
  as saved local future covariate context and saved global cutoff/max-horizon
  context, instead of only validating the top-level serialization schema.
- `artifact diff --format markdown` also breaks out `future_override_schema`
  changes into a dedicated `Future Override` section instead of burying them in
  generic diff rows.
- `artifact validate` performs explicit artifact contract validation and exits
  non-zero on unsupported schema versions or malformed metadata.
- `artifact diff` compares `artifact_schema_version`, `forecaster_type`,
  `metadata`, optional `tracking`, optional `tracking_backends`, optional
  `tracking_summary`, and `extra`, with optional CSV output or grouped Markdown
  sections for summary, tracking, metadata, and extra differences; if no
  differences remain after filtering, Markdown output collapses to a short
  summary message.
