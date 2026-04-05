# PyPI Package Validation 100 Tasks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the `foresight-ts` PyPI package end-to-end with 100 concrete execution tasks in a clean conda environment, including install, CLI, core models, artifacts, checkpoints, and torch-specific behavior.

**Architecture:** Use the existing clean environment `foresight-pypi310` as the execution target, generate temporary validation assets under `/tmp`, and run a single scripted matrix that records one result per task. Keep repository changes limited to this plan document; do not modify package code.

**Tech Stack:** Conda, pip, Python 3.10, PyPI `foresight-ts`, numpy, pandas, PyTorch CPU, package CLI and Python API.

## Recorded Execution Outcome

- Recorded validation artifacts: `/tmp/foresight_validation_100_20260328/summary.json` and `/tmp/foresight_validation_100_20260328/results.json`
- Environment snapshot: conda env `foresight-pypi310`, Python `3.10.20`, package `foresight-ts 0.2.11`, `torch 2.11.0+cpu`
- Outcome summary: `96 / 100` tasks passed; failed tasks were `13`, `14`, `15`, and `48`
- Scope note: this document records the completed validation run; no package source files were modified as part of this plan

---

### Task Group 1: Environment And Package Identity

- [x] Task 1: Verify `python --version` in `foresight-pypi310`
- [x] Task 2: Verify `pip show foresight-ts`
- [x] Task 3: Verify `foresight.__version__`
- [x] Task 4: Verify `foresight.__file__` points to `site-packages`
- [x] Task 5: Verify `numpy` import from the conda env
- [x] Task 6: Verify `pandas` import from the conda env
- [x] Task 7: Verify root CLI help
- [x] Task 8: Verify model registry count
- [x] Task 9: Verify datasets list command
- [x] Task 10: Verify `models info naive-last`

### Task Group 2: Dataset And Basic API Checks

- [x] Task 11: Load `catfish`
- [x] Task 12: Load `ice_cream_interest`
- [ ] Task 13: Attempt load `store_sales` (`FileNotFoundError`: dataset file not found; requires `--data-dir` or `FORESIGHT_DATA_DIR`)
- [ ] Task 14: Attempt load `promotion_data` (`FileNotFoundError`: dataset file not found; requires `--data-dir` or `FORESIGHT_DATA_DIR`)
- [ ] Task 15: Attempt load `cashflow_data` (`FileNotFoundError`: dataset file not found; requires `--data-dir` or `FORESIGHT_DATA_DIR`)
- [x] Task 16: Python API `eval_model` with `naive-last`
- [x] Task 17: Python API `make_forecaster`
- [x] Task 18: Python API `make_forecaster_object`
- [x] Task 19: Python API `forecast_model`
- [x] Task 20: Python API `bootstrap_intervals`

### Task Group 3: Core Functional Model Smoke Forecasts A

- [x] Task 21: `adida`
- [x] Task 22: `analog-knn`
- [x] Task 23: `ar-ols`
- [x] Task 24: `ar-ols-auto`
- [x] Task 25: `ar-ols-lags`
- [x] Task 26: `croston`
- [x] Task 27: `croston-opt`
- [x] Task 28: `croston-sba`
- [x] Task 29: `croston-sbj`
- [x] Task 30: `drift`
- [x] Task 31: `fft`
- [x] Task 32: `fourier`
- [x] Task 33: `fourier-multi`
- [x] Task 34: `holt`
- [x] Task 35: `holt-auto`
- [x] Task 36: `holt-damped`
- [x] Task 37: `holt-winters-add`
- [x] Task 38: `holt-winters-add-auto`
- [x] Task 39: `holt-winters-mul`
- [x] Task 40: `holt-winters-mul-auto`

### Task Group 4: Core Functional Model Smoke Forecasts B

- [x] Task 41: `kalman-level`
- [x] Task 42: `kalman-trend`
- [x] Task 43: `les`
- [x] Task 44: `lr-lag`
- [x] Task 45: `lr-lag-direct`
- [x] Task 46: `mean`
- [x] Task 47: `median`
- [ ] Task 48: `moment` (`ValueError`: `moment` requires `checkpoint_path` or `model_source`)
- [x] Task 49: `moving-average`
- [x] Task 50: `moving-median`
- [x] Task 51: `naive-last`
- [x] Task 52: `poly-trend`
- [x] Task 53: `sar-ols`
- [x] Task 54: `seasonal-drift`
- [x] Task 55: `seasonal-mean`
- [x] Task 56: `seasonal-naive`
- [x] Task 57: `seasonal-naive-auto`
- [x] Task 58: `ses`
- [x] Task 59: `ses-auto`
- [x] Task 60: `ssa`

### Task Group 5: Core Functional Model Smoke Forecasts C

- [x] Task 61: `theta`
- [x] Task 62: `theta-auto`
- [x] Task 63: `tsb`
- [x] Task 64: `weighted-moving-average`
- [x] Task 65: object API `moving-average`
- [x] Task 66: object API `naive-last`
- [x] Task 67: object API `theta`
- [x] Task 68: object API `holt`
- [x] Task 69: object API `ses`
- [x] Task 70: object API `seasonal-naive`

### Task Group 6: Extended API And CSV Workflow Checks

- [x] Task 71: object API `drift`
- [x] Task 72: object API `kalman-level`
- [x] Task 73: object API `fft`
- [x] Task 74: object API `fourier`
- [x] Task 75: CLI `eval run`
- [x] Task 76: CLI `leaderboard models`
- [x] Task 77: CLI `forecast csv`
- [x] Task 78: CLI `cv csv`
- [x] Task 79: CLI `detect csv`
- [x] Task 80: Python API `detect_anomalies`

### Task Group 7: Data Utility And Classical Artifact Checks

- [x] Task 81: Python API `to_long`
- [x] Task 82: Python API `validate_long_df`
- [x] Task 83: Python API `split_long_df`
- [x] Task 84: Python API `infer_series_frequency`
- [x] Task 85: classical `save_forecaster`
- [x] Task 86: classical `load_forecaster`
- [x] Task 87: classical artifact prediction match
- [x] Task 88: CLI `artifact validate` on classical artifact
- [x] Task 89: CLI `artifact info` on classical artifact
- [x] Task 90: CLI `forecast artifact` on classical artifact

### Task Group 8: Torch Presence, Training, And Checkpoints

- [x] Task 91: verify `torch` import
- [x] Task 92: verify torch CPU runtime
- [x] Task 93: torch model info
- [x] Task 94: torch training smoke run
- [x] Task 95: torch `best.pt` created
- [x] Task 96: torch `last.pt` created
- [x] Task 97: torch checkpoint has `state_dict`
- [x] Task 98: torch checkpoint has optimizer/training metadata
- [x] Task 99: torch checkpoint resume smoke run
- [x] Task 100: torch artifact round-trip plus CLI forecast reuse
