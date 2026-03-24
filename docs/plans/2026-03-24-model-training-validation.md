# Model Training Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a full-registry training validation flow that runs every model on the built-in dataset, uses CUDA for torch-capable models, writes a per-model artifact, and reports progress every 50 completed models.

**Architecture:** Extend `model_validation.py` from evaluation-only behavior into a training-and-artifact runner. Use forecasting services for local/global execution, multivariate callable runners for multivariate models, and emit a uniform row schema plus per-model artifact metadata under one output directory. Run the finished tool from the `foresight-gpu` conda environment so torch models use CUDA.

**Tech Stack:** Python 3.10, pandas, numpy, statsmodels, PyTorch CUDA, existing ForeSight forecasting/model execution services.

---

### Task 1: Write failing tests for training artifact selection

**Files:**
- Modify: `tests/test_model_validation_tool.py`
- Modify: `src/foresight/services/model_validation.py`

**Step 1: Write the failing test**

Add tests that expect:

- torch local/global/multivariate params to include `checkpoint_dir`, `save_best_checkpoint=True`, `save_last_checkpoint=True`
- non-torch local/global models to route to forecast-artifact persistence
- `var` to route to a dedicated multivariate pickle path

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_validation_tool.py -k training_artifact -v`
Expected: FAIL because the new helpers do not exist yet.

**Step 3: Write minimal implementation**

Add helper functions in `src/foresight/services/model_validation.py` for:

- artifact directory resolution
- checkpoint param injection
- artifact kind selection

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_validation_tool.py -k training_artifact -v`
Expected: PASS

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 2: Write failing tests for progress checkpointing

**Files:**
- Modify: `tests/test_model_validation_tool.py`
- Modify: `src/foresight/services/model_validation.py`

**Step 1: Write the failing test**

Add tests that expect:

- `progress.json` to be written
- progress payload to update every `progress_every` models
- stdout progress payload formatting helper to include completed/total counts

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_validation_tool.py -k progress -v`
Expected: FAIL because progress writing does not exist.

**Step 3: Write minimal implementation**

Add:

- progress-state builder
- `progress.json` writer
- configurable `progress_every` parameter

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_validation_tool.py -k progress -v`
Expected: PASS

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 3: Write failing tests for real training execution paths

**Files:**
- Modify: `tests/test_model_validation_tool.py`
- Modify: `src/foresight/services/model_validation.py`

**Step 1: Write the failing test**

Add tests that expect:

- local/global models to call forecasting or artifact save helpers
- multivariate torch models to call the multivariate runner with checkpoint params
- `var` to pickle a fitted artifact file

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_validation_tool.py -k training_execution -v`
Expected: FAIL because the training runner does not exist.

**Step 3: Write minimal implementation**

Implement helper functions for:

- local/global training forecast execution
- multivariate training execution
- `var` fitted-result persistence
- per-model result row construction

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_validation_tool.py -k training_execution -v`
Expected: PASS

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 4: Wire the CLI tool to the new training validation flow

**Files:**
- Modify: `tools/validate_all_models.py`
- Modify: `src/foresight/services/model_validation.py`
- Test: `tests/test_model_validation_tool.py`

**Step 1: Write the failing test**

Add tests that expect CLI-level arguments for:

- output root
- model subset
- device
- `progress_every`

and that the tool prints the final summary from the training validation payload.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_validation_tool.py -k cli -v`
Expected: FAIL because the CLI is still wired only to evaluation-only behavior.

**Step 3: Write minimal implementation**

Update the CLI tool to call the new training validation entry point.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_validation_tool.py -k cli -v`
Expected: PASS

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 5: Run focused verification

**Files:**
- Test: `tests/test_model_validation_tool.py`

**Step 1: Run focused test file**

Run: `pytest tests/test_model_validation_tool.py -v`
Expected: PASS

**Step 2: Run a smoke training subset in the GPU env**

Run:

```bash
/home/kdsoft/miniconda3/bin/conda run -n foresight-gpu \
  python tools/validate_all_models.py \
  --device cuda \
  --progress-every 2 \
  --models theta,ridge-lag,var,torch-mlp-direct,torch-stid-multivariate
```

Expected: PASS, artifacts written, progress file updated.

**Step 3: Record operational command for the full run**

Run:

```bash
/home/kdsoft/miniconda3/bin/conda run -n foresight-gpu \
  python tools/validate_all_models.py \
  --device cuda \
  --output-dir artifacts/validate_all_models
```

Expected: long-running end-to-end execution command is ready.

**Step 4: Commit**

Skip commit in this session unless explicitly requested.
