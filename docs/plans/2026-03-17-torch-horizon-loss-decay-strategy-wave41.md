# Torch Horizon Loss Decay Strategy Wave 41 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add horizon-aware loss weighting as a shared Torch training strategy so ForeSight trainers can emphasize near-term or long-term forecast steps during optimization without changing model architectures.

**Architecture:** Extend the shared Torch trainer config with one scalar control, `horizon_loss_decay`, and implement weighting once in `src/foresight/models/torch_nn.py` as a reusable loss reducer/wrapper. Reuse that helper in shared local/global loops, seq2seq loops, and custom pinball / Gaussian losses so deterministic and probabilistic Torch trainers all get consistent behavior.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Add failing tests for horizon-loss-decay config exposure, validation, helper behavior, registry metadata, and runtime acceptance

**Files:**
- Modify: `tests/test_torch_nn_validation_messages.py`
- Modify: `tests/test_torch_global_validation_messages.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_sonar_torch_rename_coverage_smoke.py`

**Step 1: Write the failing test**

- Extend shared Torch config coverage with:
  - `horizon_loss_decay`
- Add local/global validation cases for:
  - `horizon_loss_decay > 0`
- Add helper-level behavior coverage proving:
  - `horizon_loss_decay=1.0` is uniform
  - `horizon_loss_decay<1` front-loads earlier horizon steps
- Add runtime smoke tests proving these entrypoints accept the new strategy:
  - `torch-mlp-direct`
  - `torch-timexer-global`
  - `torch-seq2seq-attn-lstm-direct`
  - `torch-rnnpaper-seq2seq-direct`
  - `torch-seq2seq-lstm-global`
- Extend registry/default/help assertions with:
  - `horizon_loss_decay == 1.0`
  - help text for the new field

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py -k horizon_loss_decay`

Expected: FAIL because the new loss-weighting control is not implemented yet.

### Task 2: Implement shared horizon-loss-decay config and weighted loss helper

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/runtime.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/torch_seq2seq.py`

**Step 1: Write minimal implementation**

- Extend `TorchTrainConfig` and `TorchGlobalTrainConfig` with:
  - `horizon_loss_decay: float = 1.0`
- Add shared validation:
  - `horizon_loss_decay > 0`
- Add shared helpers that:
  - create unreduced loss tensors for MSE / MAE / Huber
  - apply exponential horizon weights on the horizon axis
  - normalize weights to preserve overall loss scale
  - gracefully fall back to mean reduction for scalar / non-horizon losses
- Route shared local/global loops through the new weighted loss wrapper
- Reuse the helper for pinball and Gaussian-NLL style overrides, not only default losses
- Extend runtime coercion/default/help metadata for `horizon_loss_decay`

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py -k horizon_loss_decay`

Expected: PASS

### Task 3: Thread horizon-loss-decay through direct seq2seq-style entrypoints and custom Torch wrappers

**Files:**
- Modify: `src/foresight/models/torch_rnn_paper_zoo.py`
- Modify: `src/foresight/models/torch_probabilistic.py`
- Modify: `src/foresight/models/torch_ct_rnn.py`
- Modify: `src/foresight/models/torch_graph_attention.py`
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_reservoir.py`
- Modify: `src/foresight/models/torch_rnn_zoo.py`
- Modify: `src/foresight/models/torch_ssm.py`
- Modify: `src/foresight/models/torch_structured_rnn.py`
- Modify: `src/foresight/models/torch_xformer.py`
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Write minimal implementation**

- Thread `horizon_loss_decay` through the direct/public builders that construct `TorchTrainConfig(...)` or custom strategy configs
- Keep the strategy orthogonal to:
  - input dropout / temporal dropout
  - post-backward gradient strategies
  - post-step averaging / lookahead strategies

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_sonar_torch_rename_coverage_smoke.py -k horizon_loss_decay`

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

Run: `python -m py_compile src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py src/foresight/models/torch_probabilistic.py src/foresight/models/torch_ct_rnn.py src/foresight/models/torch_reservoir.py src/foresight/models/torch_rnn_zoo.py src/foresight/models/torch_ssm.py src/foresight/models/torch_structured_rnn.py src/foresight/models/torch_xformer.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS

**Step 3: Run lint**

Run: `ruff check src/foresight/models/torch_nn.py src/foresight/models/runtime.py src/foresight/models/torch_global.py src/foresight/models/torch_seq2seq.py src/foresight/models/torch_rnn_paper_zoo.py src/foresight/models/torch_probabilistic.py src/foresight/models/torch_ct_rnn.py src/foresight/models/torch_reservoir.py src/foresight/models/torch_rnn_zoo.py src/foresight/models/torch_ssm.py src/foresight/models/torch_structured_rnn.py src/foresight/models/torch_xformer.py tests/test_torch_nn_validation_messages.py tests/test_torch_global_validation_messages.py tests/test_models_registry.py tests/test_sonar_torch_rename_coverage_smoke.py`

Expected: PASS
