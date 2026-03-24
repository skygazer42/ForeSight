# Model Training Validation Design

**Goal:** Define a full-registry validation flow that runs on the built-in `promotion_data` dataset, executes real training for every model, uses CUDA for torch-capable models, and leaves a per-model training artifact on disk.

**Context**

The existing `tools/validate_all_models.py` and `src/foresight/services/model_validation.py` perform backtest-style evaluation across the model registry. That is sufficient for metric validation, but it does not provide a consistent "trained artifact per model" contract, and it does not emit incremental progress suitable for long-running GPU jobs.

The user wants a stricter operational check:

- use the built-in dataset
- run every registered model
- torch and transformers models must use CUDA when available
- torch-capable models should train for one epoch and leave checkpoint files
- every model must leave some training artifact so the run is inspectable after completion
- the runner must report progress every 50 completed models

**Constraints**

1. Not every model in the registry uses the same runtime contract.
   - `local` and `global` models generally route through forecasting services.
   - `multivariate` models use direct callables; there is no object API for them.
2. Many local/global object wrappers are lazy: `fit()` stores training context, while actual model training happens when `predict()` runs.
3. Only torch-based models expose a real checkpoint contract (`checkpoint_dir`, `save_best_checkpoint`, `save_last_checkpoint`).
4. Non-torch models do not have a uniform "epoch" or "weight tensor" concept.

**Recommended Contract**

Use one execution contract, but map artifacts by runtime family:

- `torch` / `transformers` local-global-multivariate models:
  - run with `device='cuda'`
  - set `epochs=1`
  - set `checkpoint_dir`
  - require at least one checkpoint file (`best.pt` or `last.pt`)
- non-torch `local` / `global` models:
  - run a real end-of-series forecast on the built-in dataset so training actually happens
  - save a forecast artifact bundle via the existing artifact workflow
- non-torch `multivariate` `var`:
  - fit a real statsmodels VAR result and pickle the fitted result object

This is the closest possible match to the user's requirement without refactoring 199 non-torch models to expose explicit fitted-weight objects.

**Data Flow**

1. Load and regularize `promotion_data` exactly once.
2. For each model:
   - derive lightweight parameters
   - choose the training path by interface and backend
   - execute one real training/forecast pass
   - write per-model artifact metadata
   - append one result row
3. Every 50 models:
   - update a progress JSON file
   - flush a progress line to stdout
4. At the end:
   - write `rows.json`, `summary.json`, `summary.md`
   - keep artifact directories for post-run inspection

**Artifact Layout**

Under one output root:

- `artifacts/.../models/<model_key>/result.json`
- `artifacts/.../models/<model_key>/forecast_artifact.pkl` for local/global non-torch models
- `artifacts/.../models/<model_key>/checkpoints/best.pt` and/or `last.pt` for torch models
- `artifacts/.../models/<model_key>/var.pkl` for `var`
- `artifacts/.../progress.json`

**Runner Strategy**

Use the existing `foresight-gpu` conda environment via `conda run -n foresight-gpu ...`.

That environment already has:

- `torch 2.11.0+cu128`
- CUDA available on `NVIDIA RTX A6000`
- `xgboost`, `lightgbm`, `catboost`, `statsmodels`, and `transformers`

**Testing Strategy**

Add focused tests before implementation:

1. training artifact selection by interface/backend
2. per-50 progress checkpoint writing
3. torch checkpoint parameter injection
4. `var` fitted-result persistence
5. CLI option handling for training validation mode

**Open Limitation**

For non-torch local/global models, the saved `.pkl` artifact is a reusable forecast artifact, not a native framework checkpoint. This is an architecture limitation of the current registry design, not a validation-runner bug.
