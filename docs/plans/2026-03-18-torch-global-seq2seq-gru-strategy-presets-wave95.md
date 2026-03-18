# Torch Global Seq2Seq GRU Strategy Presets Wave95 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the six global Torch Seq2Seq GRU training-strategy preset registry entries by keeping the existing `torch-seq2seq-gru-swa-global` preset and adding EMA, SAM, regularized, long-horizon, and Lookahead variants on top of `torch-seq2seq-gru-global`.

**Architecture:** Keep the existing global Seq2Seq forecaster untouched and extend the catalog by cloning the base GRU `ModelSpec` with merged strategy defaults for the missing presets. Add dedicated registry, optional-dependency, and smoke tests so the wave validates the full six-preset matrix without touching trainer internals.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional model registry

---

### Task 1: Add failing registry coverage

**Files:**
- Create: `docs/plans/2026-03-18-torch-global-seq2seq-gru-strategy-presets-wave95.md`
- Create: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py`

**Step 1: Write the failing test**

```python
GLOBAL_SEQ2SEQ_GRU_STRATEGY_PRESET_KEYS = (
    "torch-seq2seq-gru-ema-global",
    "torch-seq2seq-gru-swa-global",
    "torch-seq2seq-gru-sam-global",
    "torch-seq2seq-gru-regularized-global",
    "torch-seq2seq-gru-longhorizon-global",
    "torch-seq2seq-gru-lookahead-global",
)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py`
Expected: FAIL because the missing preset keys are not registered yet.

### Task 2: Add failing optional-dependency and smoke coverage

**Files:**
- Create: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`

**Step 1: Write the failing optional-dependency test**

```python
for key in GLOBAL_SEQ2SEQ_GRU_STRATEGY_PRESET_KEYS:
    forecaster = make_global_forecaster(key)
    with pytest.raises(ImportError):
        forecaster(long_df, ds[-4], 3)
```

**Step 2: Write the failing installed-path smoke test**

```python
forecaster = make_global_forecaster(
    key,
    context_length=24,
    hidden_size=16,
    num_layers=1,
    teacher_forcing=0.6,
    teacher_forcing_final=0.0,
    epochs=1,
    device="cpu",
    **overrides,
)
```

**Step 3: Run tests to verify they fail before implementation**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`
Expected: FAIL because missing preset keys are not registered yet.

### Task 3: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py`
- Test: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py`
- Test: `tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`

**Step 1: Add a helper that clones the base GRU global `ModelSpec` for the missing presets**

```python
def _make_wave95_global_seq2seq_gru_strategy_presets(
    context: Any, catalog: dict[str, Any]
) -> dict[str, Any]:
    ...
```

**Step 2: Define the five missing preset entries**

```python
"torch-seq2seq-gru-lookahead-global": (
    "torch-seq2seq-gru-global",
    "... Lookahead-optimized cosine ...",
    {"lookahead_steps": 5, "lookahead_alpha": 0.5},
),
```

**Step 3: Merge defaults without changing the base factory**

```python
default_params={**base_spec.default_params, **overrides}
```

**Step 4: Run focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`
Expected: PASS.

### Task 4: Verify adjacent behavior and hygiene

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Run adjacent Seq2Seq smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave54_global_seq2seq_strategy_presets_smoke.py`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`
Expected: PASS with no output.

**Step 3: Run lint verification**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_registry.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_optional_deps.py tests/test_models_wave95_global_seq2seq_gru_strategy_presets_smoke.py`
Expected: PASS.
