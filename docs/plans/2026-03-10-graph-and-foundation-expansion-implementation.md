# Graph + Foundation Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two graph forecasting baselines (STGCN + Graph WaveNet) under the `multivariate` interface, and add a lightweight foundation-style wrapper model (Hugging Face TimeSeriesTransformer) under the `local` interface.

**Architecture:** Implement graph models as multivariate forecasters that consume a wide `(T, N)` target matrix and optionally accept an adjacency spec (`adj` / `adj_path`). Keep optional dependencies lazy: graph models require only `torch`, while the foundation wrapper requires `transformers` + `torch` and must not import those modules at import time.

**Tech Stack:** Python 3.10+, NumPy, Pandas, PyTorch, Hugging Face Transformers (optional), pytest

---

### Task 1: Lock Expected Behavior In Tests (RED)

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_multivariate.py`
- Create: `tests/test_models_optional_deps_transformers.py`

**Steps:**
1. Require multivariate registry keys:
   - `torch-stgcn-multivariate`
   - `torch-graphwavenet-multivariate`
2. Add smoke tests for both models that assert `(horizon, n_nodes)` output shape and finite values.
3. Add a transformers optional-dep test suite for `hf-timeseries-transformer-direct`:
   - registered and marked optional
   - raises `ImportError` when `transformers` missing
   - emits finite `(horizon,)` output when installed (tiny config, CPU).

Run (RED):
```bash
PYTHONPATH=src pytest -q tests/test_models_multivariate.py
PYTHONPATH=src pytest -q tests/test_models_registry.py -k multivariate
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k multivariate
PYTHONPATH=src pytest -q tests/test_models_optional_deps_transformers.py
```

---

### Task 2: Implement Graph Multivariate Models (STGCN + Graph WaveNet)

**Files:**
- Modify: `src/foresight/models/multivariate.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/models/registry.py`

**Steps:**
1. Add adjacency helpers in `multivariate.py`:
   - accept `adj` as: `identity`, `ring`, `fully-connected`, `corr`, or a numeric matrix
   - optional `adj_path` for `.npy` / `.csv`
   - row-normalize and add self-loops.
2. Add `torch_stgcn_forecast(...)`:
   - temporal conv + graph mixing blocks over `(lags, N)`
   - head: last token per node -> `horizon` steps.
3. Add `torch_graphwavenet_forecast(...)`:
   - causal dilated gated temporal conv blocks
   - graph mixing with optional adaptive adjacency.
4. Register models with `interface="multivariate"` and `requires=("torch",)`.

---

### Task 3: Add a Foundation-Style Local Wrapper (HF TimeSeriesTransformer)

**Files:**
- Create: `src/foresight/models/hf_time_series.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `pyproject.toml`

**Steps:**
1. Implement `_require_transformers()` that imports lazily and raises a clean `ImportError`.
2. Implement `hf_timeseries_transformer_direct_forecast(...)`:
   - builds a tiny `TimeSeriesTransformerForPrediction` config or loads `from_pretrained` if provided
   - uses `generate(...)` and returns the mean across samples as `(horizon,)`.
3. Register key `hf-timeseries-transformer-direct` with `requires=("transformers","torch")`.
4. Add optional extras entry in `pyproject.toml`:
   - `transformers = ["transformers>=4.0"]`
   - include it in `all`.

---

### Task 4: Verification (GREEN)

Run:
```bash
PYTHONPATH=src pytest -q tests/test_models_multivariate.py
PYTHONPATH=src pytest -q tests/test_models_optional_deps_transformers.py
PYTHONPATH=src pytest -q tests/test_models_registry.py
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src pytest -q tests/test_docs_rnn_generated.py
```

