# RetNet Direct Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class local `torch-retnet-direct` forecasting family to ForeSight with registry integration, optional-dependency coverage, smoke tests, and documentation updates.

**Architecture:** Reuse the existing local Torch direct-model pattern in `src/foresight/models/torch_nn.py`: train on lag windows through `_fit_encoder_direct_model()` and implement a lite retention block with causal decay over key/value state. Integrate the model through `src/foresight/models/registry.py`, export it in `src/foresight/models/__init__.py`, and update the roadmap and model docs so the new family is discoverable.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest, MkDocs

---

### Task 1: Lock RetNet Behavior In Tests

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Add a registry test**

Require `torch-retnet-direct` to appear in `list_models()`.

**Step 2: Add an optional-dependency coverage test**

Require the new key to be discovered as a local Torch optional model.

**Step 3: Add a CPU smoke test**

Require a tiny `make_forecaster("torch-retnet-direct", ...)` run to emit a finite `(horizon,)` forecast.

**Step 4: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k retnet
```

---

### Task 2: Implement The RetNet Local Direct Model

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Add `torch_retnet_direct_forecast()`**

Implement a lite retention network that:
- projects scalar lag inputs into token embeddings
- applies stacked retention blocks with recurrent decay over key/value state
- predicts the whole horizon from the last token state

**Step 2: Keep the interface consistent**

Support the standard local Torch direct parameters:
- `lags`
- `d_model`
- `nhead`
- `num_layers`
- `ffn_dim`
- `dropout`
- common Torch training parameters

**Step 3: Keep the implementation honest**

Describe it as a lite RetNet-style model rather than full paper parity.

---

### Task 3: Integrate RetNet Into The Registry

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Import the new forecast function**

Wire the function into the Torch local import list.

**Step 2: Add a factory**

Create `_factory_torch_retnet_direct(...)` following the existing Torch direct pattern.

**Step 3: Add `ModelSpec` metadata**

Register:
- key: `torch-retnet-direct`
- requires: `("torch",)`
- interface: local
- parameter help and defaults aligned with the implementation

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k retnet
```

---

### Task 4: Update Roadmap And Model Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md`
- Modify: `docs/models.md`

**Step 1: Update roadmap status**

Mark the RetNet cluster as implemented.

**Step 2: Mention the model in the README Torch local table**

Place it with the transformer / long-sequence local models.

**Step 3: Regenerate capability docs**

Run:

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
```

---

### Task 5: Verify The Integrated Change

**Files:**
- Verify only

**Step 1: Run targeted tests**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_torch_xformer_seq2seq_smoke.py -k retnet
```

**Step 2: Run a direct smoke command**

```bash
PYTHONPATH=src python - <<'PY'
from foresight.models.registry import make_forecaster
import numpy as np
f = make_forecaster("torch-retnet-direct", lags=48, d_model=32, nhead=4, num_layers=1, ffn_dim=64, epochs=2, batch_size=16, device="cpu", seed=0)
y = np.sin(np.arange(128, dtype=float) / 7.0) + 0.03 * np.arange(128, dtype=float)
print(f(y, 5))
PY
```
