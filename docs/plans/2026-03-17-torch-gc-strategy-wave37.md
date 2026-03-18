# Torch Gradient Centralization Strategy Wave 37 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Gradient Centralization (GC) as another orthogonal Torch training strategy so ForeSight local/global/seq2seq trainers can optionally centralize weight gradients before AGC and existing norm/value clipping.

**Architecture:** Extend the shared Torch trainer config with a single `gc_mode` field instead of introducing another optimizer variant. Implement GC as a shared gradient pre-processing helper in `src/foresight/models/torch_nn.py` that supports `off`, `all`, and `conv_only`, then reuse the existing shared clipping entrypoint so local/global/custom seq2seq loops inherit the behavior consistently.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for GC config exposure, validation, runtime acceptance, and registry metadata

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `gc_mode`
- Add local/global validation cases for:
  - `gc_mode in {off, all, conv_only}`
- Add runtime smoke tests proving these entrypoints accept GC controls:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
- Extend registry/default/help assertions with:
  - `gc_mode == "off"`
  - help text for the new field

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k gc_mode`

Expected: FAIL because the new GC control is not implemented yet.

### Task 2: Implement shared GC config and gradient helper

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `gc_mode: str = "off"`
- Add shared validation:
  - `gc_mode in {off, all, conv_only}`
- Add shared GC helpers for:
  - detecting whether GC is enabled
  - mapping mode to minimum tensor rank (`all` => 2D+, `conv_only` => 3D+)
  - centralizing gradients across non-output dimensions
- Update `_apply_torch_gradient_clipping(...)` so the strategy order becomes:
  - GC
  - AGC
  - existing norm/value clipping
- Extend runtime coercion/default/help metadata for `gc_mode`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k gc_mode`

Expected: PASS

### Task 3: Thread GC through direct seq2seq-style entrypoints

**Files:**
- Modify: `src/foresight/models/torch_seq2seq.py`
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Thread `gc_mode` through the direct/public builders that construct `TorchTrainConfig(...)` or custom clip configs.
- Keep GC orthogonal to the existing strategy stack:
  - GC transforms gradients before AGC / clipping
  - SAM still changes the optimizer-step procedure
  - EMA / SWA / Lookahead remain post-step layers

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k gc_mode`

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
