# Torch Global DeepAR Strategy Presets Wave58 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six global Torch DeepAR training-strategy preset registry entries that reuse the existing `torch-deepar-global` implementation with EMA, SWA, SAM, regularized, long-horizon, and Lookahead-oriented defaults.

**Architecture:** Keep the existing global DeepAR forecaster implementation untouched and extend the catalog with cloned `ModelSpec` entries whose `default_params` merge strategy overrides into the base spec. Because DeepAR already trains through Gaussian NLL, the long-horizon preset should rely on `horizon_loss_decay` and EMA stabilization instead of changing `loss`.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional model registry

---

### Task 1: Add failing registry coverage

**Files:**
- Create: `docs/plans/2026-03-18-torch-global-deepar-strategy-presets-wave58.md`
- Create: `tests/test_models_wave58_global_deepar_strategy_presets_registry.py`

**Step 1: Write the failing test**

```python
GLOBAL_DEEPAR_STRATEGY_PRESET_KEYS = (
    "torch-deepar-ema-global",
    "torch-deepar-swa-global",
    "torch-deepar-sam-global",
    "torch-deepar-regularized-global",
    "torch-deepar-longhorizon-global",
    "torch-deepar-lookahead-global",
)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave58_global_deepar_strategy_presets_registry.py`
Expected: FAIL because the new preset keys are missing.

**Step 3: Add metadata and defaults assertions**

```python
ema = get_model_spec("torch-deepar-ema-global")
assert ema.default_params["ema_decay"] == 0.995
```

**Step 4: Re-run to keep it red for the right reason**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave58_global_deepar_strategy_presets_registry.py`
Expected: FAIL with missing-key assertions or lookups.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-18-torch-global-deepar-strategy-presets-wave58.md tests/test_models_wave58_global_deepar_strategy_presets_registry.py
git commit -m "test: add wave58 global deepar strategy preset coverage"
```

### Task 2: Add failing optional-dependency and smoke coverage

**Files:**
- Create: `tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`

**Step 1: Write the failing optional-dependency test**

```python
for key in GLOBAL_DEEPAR_STRATEGY_PRESET_KEYS:
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
    epochs=1,
    device="cpu",
    **overrides,
)
```

**Step 3: Run tests to verify they fail before implementation**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`
Expected: FAIL because the new registry entries do not exist yet.

**Step 4: Keep the long-horizon assertion aligned with Gaussian NLL**

Assert `horizon_loss_decay` rather than `loss == "huber"` because DeepAR uses the probabilistic override path.

**Step 5: Commit**

```bash
git add tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py
git commit -m "test: add wave58 global deepar strategy smoke coverage"
```

### Task 3: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_wave58_global_deepar_strategy_presets_registry.py`
- Test: `tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py`
- Test: `tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`

**Step 1: Add a helper that clones the base DeepAR global `ModelSpec`**

```python
def _make_wave58_global_deepar_strategy_presets(
    context: Any, catalog: dict[str, Any]
) -> dict[str, Any]:
    ...
```

**Step 2: Define six preset entries without altering the base factory**

```python
"torch-deepar-sam-global": (
    "torch-deepar-global",
    "... SAM plus cosine-warmup ...",
    {"optimizer": "adamw", "scheduler": "cosine", "sam_rho": 0.05},
),
```

**Step 3: Keep the long-horizon preset probabilistic**

Use `horizon_loss_decay`, `ema_decay`, and `warmup_epochs` while leaving the Gaussian NLL path intact.

**Step 4: Run the focused verification**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave58_global_deepar_strategy_presets_registry.py tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py
git commit -m "feat: add wave58 global deepar strategy presets"
```

### Task 4: Verify adjacent behavior and hygiene

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Run an adjacent global torch smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave58_global_deepar_strategy_presets_registry.py tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`
Expected: PASS with no output.

**Step 3: Run lint verification**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave58_global_deepar_strategy_presets_registry.py tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py`
Expected: PASS with no issues.

**Step 4: Review worktree boundaries**

Confirm the new wave58 files stay isolated from unrelated in-flight edits.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py tests/test_models_wave58_global_deepar_strategy_presets_registry.py tests/test_models_wave58_global_deepar_strategy_presets_optional_deps.py tests/test_models_wave58_global_deepar_strategy_presets_smoke.py
git commit -m "test: verify wave58 global deepar strategy presets"
```
