# Torch Lookahead Custom Trainers Wave 34 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining Lookahead parity gaps in ForeSight's custom Torch trainer loops so seq2seq and rnnpaper models persist slow weights, resume Lookahead state, and select the correct deploy model for validation and checkpoints.

**Architecture:** Reuse the shared Lookahead helpers already added in `src/foresight/models/torch_nn.py` instead of inventing trainer-local logic. Add focused regression tests that exercise real seq2seq and rnnpaper forecasting entrypoints with Lookahead-enabled checkpoints, then thread the shared helper calls through the two custom training loops so deploy-state precedence remains `SWA > EMA > Lookahead > raw model`.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing regression tests for custom trainer Lookahead checkpoints

**Files:**
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Add a seq2seq regression test that:
  - calls `torch_seq2seq_direct_forecast(...)`
  - enables `lookahead_steps`, `lookahead_alpha`, `checkpoint_dir`, `save_last_checkpoint`
  - asserts `last.pt` contains `lookahead_state`, `lookahead_step`, and `model_state`
- Add an rnnpaper regression test that:
  - calls `torch_rnnpaper_direct_forecast(...)`
  - enables the same Lookahead checkpoint knobs
  - asserts the same checkpoint payload keys exist

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k lookahead`

Expected: FAIL because the custom trainer loops do not yet persist Lookahead state.

### Task 2: Integrate shared Lookahead helpers into seq2seq and rnnpaper custom loops

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`

**Step 1: Write minimal implementation**

- Import and use:
  - `_make_torch_lookahead_model`
  - `_update_torch_lookahead_model`
  - `_select_torch_deploy_model`
  - `_maybe_torch_model_state_for_checkpoint`
- In each custom loop:
  - restore `lookahead_state` and `lookahead_step` from resume state
  - update Lookahead immediately after each optimizer step
  - use shared deploy-model selection for validation, best-state snapshots, and saved `last_state`
  - include `lookahead_state`, `lookahead_step`, and `model_state` in checkpoint payload snapshots

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k lookahead`

Expected: PASS

### Task 3: Run targeted verification

**Files:**
- Verify: `tests/test_sonar_torch_rename_coverage_smoke.py`
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`

**Step 1: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py`

Expected: PASS

**Step 2: Run static verification on changed files**

Run: `python -m py_compile src/foresight/models/torch_nn.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS
