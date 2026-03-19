# Artifact Workflow

ForeSight artifacts persist a fitted forecaster together with structured metadata
about the training schema and artifact payload version. The stable Python API is:

```python
from foresight import load_forecaster, load_forecaster_artifact, save_forecaster
```

Use `save_forecaster(...)` to persist a fitted object, `load_forecaster(...)` to
reconstruct it for prediction, and `load_forecaster_artifact(...)` when you want
to inspect the structured payload before rebuilding the runtime object.

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
```

## Notes

- `forecast artifact` reuses a saved fitted object for prediction.
- `artifact info` returns a JSON-safe summary of `metadata`, optional top-level
  `tracking`, optional `tracking_backends`, optional `tracking_summary`, and
  `extra`; `--format markdown` renders the same payload as grouped summary,
  tracking, metadata, and extra sections.
- `artifact validate` performs explicit artifact contract validation and exits
  non-zero on unsupported schema versions or malformed metadata.
- `artifact diff` compares `artifact_schema_version`, `forecaster_type`,
  `metadata`, optional `tracking`, optional `tracking_backends`, optional
  `tracking_summary`, and `extra`, with optional CSV output or grouped Markdown
  sections for summary, tracking, metadata, and extra differences; if no
  differences remain after filtering, Markdown output collapses to a short
  summary message.
