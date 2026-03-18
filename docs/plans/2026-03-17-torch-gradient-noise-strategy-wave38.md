# Torch Gradient Noise Strategy Wave 38 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add gradient noise injection as another orthogonal Torch training strategy so ForeSight local/global/seq2seq trainers can optionally add Gaussian noise to gradients before AGC and existing clipping.

**Architecture:** Extend the shared Torch trainer config with a single `grad_noise_std` field instead of introducing another optimizer enum. Implement gradient noise as a shared gradient pre-processing helper in `src/foresight/models/torch_nn.py`, then reuse the existing shared clipping entrypoint so local/global/custom seq2seq loops inherit the behavior consistently.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for gradient noise config exposure, validation, runtime acceptance, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `grad_noise_std`
- Add local/global validation cases for:
  - `grad_noise_std >= 0`
- Add runtime smoke tests proving these entrypoints accept gradient noise controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
- Extend registry/default/help assertions with:
  - `grad_noise_std == 0.0`
  - help text for the new field

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k grad_noise`

Expected: FAIL because the new gradient-noise control is not implemented yet.

### Task 2: Implement shared gradient-noise config and helper

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `grad_noise_std: float = 0.0`
- Add shared validation:
  - `grad_noise_std >= 0`
- Add a shared helper that:
  - skips when `grad_noise_std == 0`
  - adds zero-mean Gaussian noise to each floating-point parameter gradient
- Update `_apply_torch_gradient_clipping(...)` so the strategy order becomes:
  - GC
  - gradient noise
  - AGC
  - existing norm/value clipping
- Extend runtime coercion/default/help metadata for `grad_noise_std`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k grad_noise`

Expected: PASS

### Task 3: Thread gradient noise through direct seq2seq-style entrypoints

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Thread `grad_noise_std` through the direct/public builders that construct `TorchTrainConfig(...)` or custom clip configs.
- Keep gradient noise orthogonal to the existing strategy stack:
  - gradient noise perturbs gradients before AGC / clipping
  - SAM still changes the optimizer-step procedure
  - EMA / SWA / Lookahead remain post-step layers

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k grad_noise`

Expected: PASS

### Task 4: Run targeted verification

**Files:**
- Verify: `tests/test_torch_nn_validation_messages.py`
- Verify: `tests/test_torch_global_validation_messages.py`
- Verify: `tests/test_models_registry.py`
- Verify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 2: Run static verification**

Run: `python -m py_compile src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS
