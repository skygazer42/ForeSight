# Torch Global PatchTST Strategy Presets Wave87 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the six global Torch PatchTST training-strategy preset registry entries by keeping the existing `torch-patchtst-ema-global` preset and adding SWA, SAM, regularized, long-horizon, and Lookahead-oriented variants on top of `torch-patchtst-global`.

**Architecture:** Keep the existing global PatchTST forecaster untouched and extend the catalog by cloning the base `ModelSpec` with merged strategy defaults for the missing presets. Add dedicated registry, optional-dependency, and smoke tests so the wave validates the full six-preset matrix without touching trainer internals.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional model registry

---

### Task 1: Add failing registry coverage

**Files:**
- Create: `docs/plans/2026-03-18-torch-global-patchtst-strategy-presets-wave87.md`
- Create: `tests/test_models_wave87_global_patchtst_strategy_presets_registry.py`

**Step 1: Write the failing test**

```python
GLOBAL_PATCHTST_STRATEGY_PRESET_KEYS = (
    "torch-patchtst-ema-global",
    "torch-patchtst-swa-global",
    "torch-patchtst-sam-global",
    "torch-patchtst-regularized-global",
    "torch-patchtst-longhorizon-global",
    "torch-patchtst-lookahead-global",
)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave87_global_patchtst_strategy_presets_registry.py`
Expected: FAIL because the missing preset keys are not registered yet.

**Step 3: Add defaults assertions**

```python
swa = get_model_spec("torch-patchtst-swa-global")
assert swa.default_params["swa_start_epoch"] == 18
```

**Step 4: Re-run and keep it red for missing keys**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave87_global_patchtst_strategy_presets_registry.py`
Expected: FAIL with missing-key assertions or lookups.

### Task 2: Add failing optional-dependency and smoke coverage

**Files:**
- Create: `tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`

**Step 1: Write the failing optional-dependency test**

```python
for key in GLOBAL_PATCHTST_STRATEGY_PRESET_KEYS:
    forecaster = make_global_forecaster(key)
    with pytest.raises(ImportError):
        forecaster(long_df, ds[-4], 3)
```

**Step 2: Write the failing installed-path smoke test**

```python
forecaster = make_global_forecaster(
    key,
    context_length=24,
    patch_len=8,
    stride=4,
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    epochs=1,
    device="cpu",
    **overrides,
)
```

**Step 3: Run tests to verify they fail before implementation**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`
Expected: FAIL because missing preset keys are not registered yet.

### Task 3: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_wave87_global_patchtst_strategy_presets_registry.py`
- Test: `tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py`
- Test: `tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`

**Step 1: Add a helper that clones the base PatchTST global `ModelSpec` for the missing presets**

```python
def _make_wave87_global_patchtst_strategy_presets(
    context: Any, catalog: dict[str, Any]
) -> dict[str, Any]:
    ...
```

**Step 2: Define the five missing preset entries**

```python
"torch-patchtst-lookahead-global": (
    "torch-patchtst-global",
    "... Lookahead-optimized cosine ...",
    {"optimizer": "adamw", "scheduler": "cosine", "lookahead_steps": 5},
),
```

**Step 3: Merge defaults without changing the base factory**

```python
default_params={**base_spec.default_params, **overrides}
```

**Step 4: Run focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave87_global_patchtst_strategy_presets_registry.py tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`
Expected: PASS.

### Task 4: Verify adjacent behavior and hygiene

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Run adjacent global smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave87_global_patchtst_strategy_presets_registry.py tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`
Expected: PASS with no output.

**Step 3: Run lint verification**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave87_global_patchtst_strategy_presets_registry.py tests/test_models_wave87_global_patchtst_strategy_presets_optional_deps.py tests/test_models_wave87_global_patchtst_strategy_presets_smoke.py`
Expected: PASS.
