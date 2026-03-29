# PyPI Package Validation 100 Tasks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the `foresight-ts` PyPI package end-to-end with 100 concrete execution tasks in a clean conda environment, including install, CLI, core models, artifacts, checkpoints, and torch-specific behavior.

**Architecture:** Use the existing clean environment `foresight-pypi310` as the execution target, generate temporary validation assets under `/tmp`, and run a single scripted matrix that records one result per task. Keep repository changes limited to this plan document; do not modify package code.

**Tech Stack:** Conda, pip, Python 3.10, PyPI `foresight-ts`, numpy, pandas, PyTorch CPU, package CLI and Python API.

---

### Task Group 1: Environment And Package Identity

- [ ] Task 1: Verify `python --version` in `foresight-pypi310`
- [ ] Task 2: Verify `pip show foresight-ts`
- [ ] Task 3: Verify `foresight.__version__`
- [ ] Task 4: Verify `foresight.__file__` points to `site-packages`
- [ ] Task 5: Verify `numpy` import from the conda env
- [ ] Task 6: Verify `pandas` import from the conda env
- [ ] Task 7: Verify root CLI help
- [ ] Task 8: Verify model registry count
- [ ] Task 9: Verify datasets list command
- [ ] Task 10: Verify `models info naive-last`

### Task Group 2: Dataset And Basic API Checks

- [ ] Task 11: Load `catfish`
- [ ] Task 12: Load `ice_cream_interest`
- [ ] Task 13: Attempt load `store_sales`
- [ ] Task 14: Attempt load `promotion_data`
- [ ] Task 15: Attempt load `cashflow_data`
- [ ] Task 16: Python API `eval_model` with `naive-last`
- [ ] Task 17: Python API `make_forecaster`
- [ ] Task 18: Python API `make_forecaster_object`
- [ ] Task 19: Python API `forecast_model`
- [ ] Task 20: Python API `bootstrap_intervals`

### Task Group 3: Core Functional Model Smoke Forecasts A

- [ ] Task 21: `adida`
- [ ] Task 22: `analog-knn`
- [ ] Task 23: `ar-ols`
- [ ] Task 24: `ar-ols-auto`
- [ ] Task 25: `ar-ols-lags`
- [ ] Task 26: `croston`
- [ ] Task 27: `croston-opt`
- [ ] Task 28: `croston-sba`
- [ ] Task 29: `croston-sbj`
- [ ] Task 30: `drift`
- [ ] Task 31: `fft`
- [ ] Task 32: `fourier`
- [ ] Task 33: `fourier-multi`
- [ ] Task 34: `holt`
- [ ] Task 35: `holt-auto`
- [ ] Task 36: `holt-damped`
- [ ] Task 37: `holt-winters-add`
- [ ] Task 38: `holt-winters-add-auto`
- [ ] Task 39: `holt-winters-mul`
- [ ] Task 40: `holt-winters-mul-auto`

### Task Group 4: Core Functional Model Smoke Forecasts B

- [ ] Task 41: `kalman-level`
- [ ] Task 42: `kalman-trend`
- [ ] Task 43: `les`
- [ ] Task 44: `lr-lag`
- [ ] Task 45: `lr-lag-direct`
- [ ] Task 46: `mean`
- [ ] Task 47: `median`
- [ ] Task 48: `moment`
- [ ] Task 49: `moving-average`
- [ ] Task 50: `moving-median`
- [ ] Task 51: `naive-last`
- [ ] Task 52: `poly-trend`
- [ ] Task 53: `sar-ols`
- [ ] Task 54: `seasonal-drift`
- [ ] Task 55: `seasonal-mean`
- [ ] Task 56: `seasonal-naive`
- [ ] Task 57: `seasonal-naive-auto`
- [ ] Task 58: `ses`
- [ ] Task 59: `ses-auto`
- [ ] Task 60: `ssa`

### Task Group 5: Core Functional Model Smoke Forecasts C

- [ ] Task 61: `theta`
- [ ] Task 62: `theta-auto`
- [ ] Task 63: `tsb`
- [ ] Task 64: `weighted-moving-average`
- [ ] Task 65: object API `moving-average`
- [ ] Task 66: object API `naive-last`
- [ ] Task 67: object API `theta`
- [ ] Task 68: object API `holt`
- [ ] Task 69: object API `ses`
- [ ] Task 70: object API `seasonal-naive`

### Task Group 6: Extended API And CSV Workflow Checks

- [ ] Task 71: object API `drift`
- [ ] Task 72: object API `kalman-level`
- [ ] Task 73: object API `fft`
- [ ] Task 74: object API `fourier`
- [ ] Task 75: CLI `eval run`
- [ ] Task 76: CLI `leaderboard models`
- [ ] Task 77: CLI `forecast csv`
- [ ] Task 78: CLI `cv csv`
- [ ] Task 79: CLI `detect csv`
- [ ] Task 80: Python API `detect_anomalies`

### Task Group 7: Data Utility And Classical Artifact Checks

- [ ] Task 81: Python API `to_long`
- [ ] Task 82: Python API `validate_long_df`
- [ ] Task 83: Python API `split_long_df`
- [ ] Task 84: Python API `infer_series_frequency`
- [ ] Task 85: classical `save_forecaster`
- [ ] Task 86: classical `load_forecaster`
- [ ] Task 87: classical artifact prediction match
- [ ] Task 88: CLI `artifact validate` on classical artifact
- [ ] Task 89: CLI `artifact info` on classical artifact
- [ ] Task 90: CLI `forecast artifact` on classical artifact

### Task Group 8: Torch Presence, Training, And Checkpoints

- [ ] Task 91: verify `torch` import
- [ ] Task 92: verify torch CPU runtime
- [ ] Task 93: torch model info
- [ ] Task 94: torch training smoke run
- [ ] Task 95: torch `best.pt` created
- [ ] Task 96: torch `last.pt` created
- [ ] Task 97: torch checkpoint has `state_dict`
- [ ] Task 98: torch checkpoint has optimizer/training metadata
- [ ] Task 99: torch checkpoint resume smoke run
- [ ] Task 100: torch artifact round-trip plus CLI forecast reuse
