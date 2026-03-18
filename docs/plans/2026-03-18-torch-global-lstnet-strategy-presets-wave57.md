# Torch Global LSTNet Strategy Presets Wave57 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six global Torch LSTNet training-strategy preset registry entries that reuse the existing `torch-lstnet-global` implementation with EMA, SWA, SAM, regularized, long-horizon, and Lookahead-oriented defaults.

**Architecture:** Keep the existing global LSTNet forecaster implementation unchanged and extend the catalog by cloning the base `ModelSpec` with merged `default_params`. Add focused registry, optional-dependency, and smoke coverage in standalone wave57 tests so the new presets stay isolated from earlier waves.

**Tech Stack:** Python, pytest, pandas, numpy, PyTorch-optional model registry

---

### Task 1: Add failing registry coverage

**Files:**
- Create: `docs/plans/2026-03-18-torch-global-lstnet-strategy-presets-wave57.md`
- Create: `tests/test_models_wave57_global_lstnet_strategy_presets_registry.py`

**Step 1: Write the failing test**

```python
from foresight.models.registry import get_model_spec, list_models


GLOBAL_LSTNET_STRATEGY_PRESET_KEYS = (
    "torch-lstnet-ema-global",
    "torch-lstnet-swa-global",
    "torch-lstnet-sam-global",
    "torch-lstnet-regularized-global",
    "torch-lstnet-longhorizon-global",
    "torch-lstnet-lookahead-global",
)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave57_global_lstnet_strategy_presets_registry.py`
Expected: FAIL because the new preset keys are not registered yet.

**Step 3: Add assertions for defaults and metadata**

```python
def test_wave57_global_lstnet_strategy_preset_defaults() -> None:
    ema = get_model_spec("torch-lstnet-ema-global")
    assert ema.default_params["ema_decay"] == 0.995
```

**Step 4: Run test to verify it still fails for missing keys**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave57_global_lstnet_strategy_presets_registry.py`
Expected: FAIL with missing preset lookups, proving the tests are exercising new behavior.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-18-torch-global-lstnet-strategy-presets-wave57.md tests/test_models_wave57_global_lstnet_strategy_presets_registry.py
git commit -m "test: add wave57 global lstnet strategy preset coverage"
```

### Task 2: Add failing optional-dependency and smoke coverage

**Files:**
- Create: `tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py`
- Create: `tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`

**Step 1: Write the failing missing-dependency test**

```python
for key in GLOBAL_LSTNET_STRATEGY_PRESET_KEYS:
    forecaster = make_global_forecaster(key)
    with pytest.raises(ImportError):
        forecaster(long_df, ds[-4], 3)
```

**Step 2: Write the failing installed-path smoke test**

```python
forecaster = make_global_forecaster(
    key,
    context_length=32,
    cnn_channels=8,
    kernel_size=3,
    rnn_hidden=16,
    skip=8,
    highway_window=16,
    epochs=1,
    device="cpu",
    **overrides,
)
```

**Step 3: Run tests to verify they fail before implementation**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`
Expected: FAIL because the preset keys are not defined yet.

**Step 4: Keep smoke overrides minimal**

Use only test-local overrides like `warmup_epochs=1` and `swa_start_epoch=0` so production defaults remain untouched.

**Step 5: Commit**

```bash
git add tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py
git commit -m "test: add wave57 global lstnet strategy preset smoke coverage"
```

### Task 3: Implement the catalog presets

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_wave57_global_lstnet_strategy_presets_registry.py`
- Test: `tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py`
- Test: `tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`

**Step 1: Add a helper that clones the base LSTNet global `ModelSpec`**

```python
def _make_wave57_global_lstnet_strategy_presets(
    context: Any, catalog: dict[str, Any]
) -> dict[str, Any]:
    ...
```

**Step 2: Define six preset entries**

```python
"torch-lstnet-ema-global": (
    "torch-lstnet-global",
    "... EMA-stabilized cosine-warmup ...",
    {"optimizer": "adamw", "scheduler": "cosine", "ema_decay": 0.995},
),
```

**Step 3: Merge cloned defaults without changing factories or requirements**

```python
default_params={**base_spec.default_params, **overrides}
```

**Step 4: Register the helper in `build_torch_global_catalog`**

Run: `PYTHONPATH=src pytest -q tests/test_models_wave57_global_lstnet_strategy_presets_registry.py tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py
git commit -m "feat: add wave57 global lstnet strategy presets"
```

### Task 4: Verify adjacent behavior and hygiene

**Files:**
- Modify: `src/foresight/models/catalog/torch_global.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Run adjacent global torch smoke**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile src/foresight/models/catalog/torch_global.py tests/test_models_wave57_global_lstnet_strategy_presets_registry.py tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`
Expected: PASS with no output.

**Step 3: Run lint verification**

Run: `ruff check src/foresight/models/catalog/torch_global.py tests/test_models_wave57_global_lstnet_strategy_presets_registry.py tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py`
Expected: PASS with no issues.

**Step 4: Review dirty worktree boundaries**

Confirm only the intended wave57 files were added or modified for this wave.

**Step 5: Commit**

```bash
git add src/foresight/models/catalog/torch_global.py tests/test_models_wave57_global_lstnet_strategy_presets_registry.py tests/test_models_wave57_global_lstnet_strategy_presets_optional_deps.py tests/test_models_wave57_global_lstnet_strategy_presets_smoke.py
git commit -m "test: verify wave57 global lstnet strategy presets"
```
