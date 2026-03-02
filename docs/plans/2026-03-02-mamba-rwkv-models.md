# Mamba + RWKV (Torch) Model Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two modern sequence backbones (Mamba-style selective SSM and RWKV-style time-mix) to ForeSight's Torch model zoo for both `local/direct` and `global/panel` interfaces.

**Architecture:** Implement minimal, self-contained PyTorch networks (only `torch.nn` primitives; no `torchvision` / pretrained imports). Keep Torch optional via `pip install -e ".[torch]"`. Integrate via `src/foresight/models/registry.py` with stable model keys and smoke tests.

**Tech Stack:** Python, NumPy/Pandas, PyTorch (`.[torch]` optional), pytest, ruff

---

### Task 1: Local/direct models (`torch-mamba-direct`, `torch-rwkv-direct`)

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Steps:**
1. Implement `torch_mamba_direct_forecast(...)` and `torch_rwkv_direct_forecast(...)` in `src/foresight/models/torch_nn.py`.
2. Add factories + `ModelSpec` keys in `src/foresight/models/registry.py`.
3. Add a small smoke test that trains 1–2 epochs and returns finite predictions.

---

### Task 2: Global/panel models (`torch-mamba-global`, `torch-rwkv-global`)

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Steps:**
1. Implement `_predict_torch_mamba_global(...)` / `torch_mamba_global_forecaster(...)` with optional `quantiles`.
2. Implement `_predict_torch_rwkv_global(...)` / `torch_rwkv_global_forecaster(...)` with optional `quantiles`.
3. Register both keys in `src/foresight/models/registry.py` with defaults and param help.
4. Add global smoke cases (1 epoch, small `context_length`) to ensure predictions are finite.

---

### Task 3: Docs + examples

**Files:**
- Modify: `README.md`
- Modify: `examples/torch_global_models.py`

**Steps:**
1. Add `torch-mamba-direct` / `torch-rwkv-direct` to the local Torch list.
2. Add `torch-mamba-global` / `torch-rwkv-global` to the global Torch list.
3. Include the new global keys in `examples/torch_global_models.py` with short training params.

---

### Task 4: Verification + commit

**Steps:**
1. Format: `ruff format src tests examples docs`
2. Lint: `ruff check src tests examples docs`
3. Tests: `pytest -q`
4. Commit: `git add -A && git commit -m "feat: add torch mamba + rwkv models" && git push`

