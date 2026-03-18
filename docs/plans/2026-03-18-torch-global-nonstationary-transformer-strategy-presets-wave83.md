# Torch Global Nonstationary Transformer Strategy Presets Wave83 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six global Torch Nonstationary Transformer training-strategy preset registry entries that reuse the existing `torch-nonstationary-transformer-global` implementation with EMA, SWA, SAM, regularized, long-horizon, and Lookahead-oriented defaults.

**Architecture:** Keep the existing global Nonstationary Transformer forecaster untouched and extend the catalog by cloning the base `ModelSpec` with merged strategy defaults. Add dedicated registry, optional-dependency, and smoke tests so the wave stays isolated, fast, and safe to validate.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional model registry

---

### Task 1: Add failing registry coverage

**Files:**
- Create: `docs/plans/2026-03-18-torch-global-nonstationary-transformer-strategy-presets-wave83.md`
- Create: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py`

**Step 1: Write the failing test**

```python
GLOBAL_NONSTATIONARY_TRANSFORMER_STRATEGY_PRESET_KEYS = (
    "torch-nonstationary-transformer-ema-global",
    "torch-nonstationary-transformer-swa-global",
    "torch-nonstationary-transformer-sam-global",
    "torch-nonstationary-transformer-regularized-global",
    "torch-nonstationary-transformer-longhorizon-global",
    "torch-nonstationary-transformer-lookahead-global",
)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py`
Expected: FAIL because the new preset keys are not registered yet.

**Step 3: Add defaults assertions**

```python
ema = get_model_spec("torch-nonstationary-transformer-ema-global")
assert ema.default_params["ema_decay"] == 0.995
```

**Step 4: Re-run and keep it red for missing keys**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py`
Expected: FAIL with missing-key assertions or lookups.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-18-torch-global-nonstationary-transformer-strategy-presets-wave83.md tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py
git commit -m "test: add wave83 global nonstationary transformer strategy preset coverage"
```

### Task 2: Add failing optional-dependency and smoke coverage

**Files:**
- Create: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`

**Step 1: Write the failing optional-dependency test**

```python
for key in GLOBAL_NONSTATIONARY_TRANSFORMER_STRATEGY_PRESET_KEYS:
    forecaster = make_global_forecaster(key)
    with pytest.raises(ImportError):
        forecaster(long_df, ds[-4], 3)
```

**Step 2: Write the failing installed-path smoke test**

```python
forecaster = make_global_forecaster(
    key,
    context_length=24,
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

Run: `PYTHONPATH=src pytest -q tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`
Expected: FAIL because the new preset keys are missing.

**Step 4: Keep smoke overrides minimal**

Only override low-epoch stability values like `warmup_epochs=1` and `swa_start_epoch=0`.

**Step 5: Commit**

```bash
git add tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py
git commit -m "test: add wave83 global nonstationary transformer strategy preset smoke coverage"
```

### Task 3: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py`
- Test: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py`
- Test: `tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`

**Step 1: Add a helper that clones the base Nonstationary Transformer global `ModelSpec`**

```python
def _make_wave83_global_nonstationary_transformer_strategy_presets(
    context: Any, catalog: dict[str, Any]
) -> dict[str, Any]:
    ...
```

**Step 2: Define six preset entries**

```python
"torch-nonstationary-transformer-lookahead-global": (
    "torch-nonstationary-transformer-global",
    "... Lookahead-optimized cosine ...",
    {"optimizer": "adamw", "scheduler": "cosine", "lookahead_steps": 5},
),
```

**Step 3: Merge defaults without changing the base factory**

```python
default_params={**base_spec.default_params, **overrides}
```

**Step 4: Run focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py
git commit -m "feat: add wave83 global nonstationary transformer strategy presets"
```

### Task 4: Verify adjacent behavior and hygiene

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`

**Step 1: Run adjacent global smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`
Expected: PASS with no output.

**Step 3: Run lint verification**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py`
Expected: PASS.

**Step 4: Review worktree boundaries**

Confirm only intended wave83 files were added or modified.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_registry.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_optional_deps.py tests/test_models_wave83_global_nonstationary_transformer_strategy_presets_smoke.py
git commit -m "test: verify wave83 global nonstationary transformer strategy presets"
```
