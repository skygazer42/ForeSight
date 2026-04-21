# Wave 1A Torch Local Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the first high-ROI wave of missing local/direct Torch forecasting families to ForeSight by promoting several existing global-only transformer-era models to `interface="local"` and adding two cheap modern direct baselines that fit the current local lag-window API.

**Architecture:** Reuse the existing local Torch pattern in `src/foresight/models/torch_nn.py`: each model remains a stateless `(train_1d, horizon) -> yhat` forecaster trained on lag windows via `_make_lagged_xy_multi()` and `_train_loop()`. Implement Wave 1A in three batches: encoder-style direct models (`Informer`, `Autoformer`, `Non-stationary Transformer`), structural direct models (`FEDformer`, `iTransformer`, `TimesNet`, `TFT`), and cheap modern direct baselines (`TimeMixer`, `SparseTSF`). Do **not** force `TimeXer` into this plan: it depends on future covariates and therefore needs a separate local `x_cols` design rather than a fake no-exogenous version.

**Tech Stack:** Python 3.10+, NumPy, Pandas, PyTorch (`.[torch]` optional), pytest, ruff

---

## Scope Guardrails

- New model keys in this plan:
  - `torch-informer-direct`
  - `torch-autoformer-direct`
  - `torch-nonstationary-transformer-direct`
  - `torch-fedformer-direct`
  - `torch-itransformer-direct`
  - `torch-timesnet-direct`
  - `torch-tft-direct`
  - `torch-timemixer-direct`
  - `torch-sparsetsf-direct`
- Explicitly out of scope for Wave 1A:
  - `torch-timexer-direct`
  - generic local Torch `x_cols` support
  - graph-native models
  - pretrained / zero-shot foundation models

---

### Task 1: Lock Wave 1A Scope In Tests First

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Write the failing registry assertions**

Add the nine new keys to `test_models_registry.py`:

```python
assert "torch-informer-direct" in keys
assert "torch-autoformer-direct" in keys
assert "torch-nonstationary-transformer-direct" in keys
assert "torch-fedformer-direct" in keys
assert "torch-itransformer-direct" in keys
assert "torch-timesnet-direct" in keys
assert "torch-tft-direct" in keys
assert "torch-timemixer-direct" in keys
assert "torch-sparsetsf-direct" in keys
```

**Step 2: Write the failing missing-dependency assertions**

Extend `tests/test_models_optional_deps_torch.py` so the new local keys are included in the missing-Torch loop:

```python
for key in _torch_local_model_keys():
    f = make_forecaster(key)
    with pytest.raises(ImportError):
        f([1.0, 2.0, 3.0], 2)
```

**Step 3: Write the failing smoke cases**

Add compact smoke cases to `tests/test_models_torch_xformer_seq2seq_smoke.py` using the installed-Torch pattern already used by other direct models:

```python
f = make_forecaster(
    "torch-informer-direct",
    lags=48,
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    epochs=2,
    batch_size=16,
    patience=2,
    device="cpu",
    seed=0,
)
yhat = f(y, 5)
assert yhat.shape == (5,)
assert np.all(np.isfinite(yhat))
```

Repeat this pattern for the remaining new keys with the smallest sensible parameter sets.

**Step 4: Run the targeted tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py
```

Expected: FAIL because the new model keys are not registered and not implemented yet.

**Step 5: Commit**

```bash
git add tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_torch_xformer_seq2seq_smoke.py
git commit -m "test: add wave1a torch local parity cases"
```

---

### Task 2: Add Direct Informer, Autoformer, and Non-stationary Transformer

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Add the smallest failing implementation hook**

Add function declarations near the existing local transformer family:

```python
def torch_informer_direct_forecast(...): ...
def torch_autoformer_direct_forecast(...): ...
def torch_nonstationary_transformer_direct_forecast(...): ...
```

Keep signatures aligned with their global counterparts where possible, but use local naming conventions:
- `lags`
- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `dropout`
- common Torch train params

**Step 2: Extract or add a shared local encoder helper**

In `src/foresight/models/torch_nn.py`, add a small internal helper so these three models do not duplicate full training loops:

```python
def _fit_encoder_direct_model(
    x_work: np.ndarray,
    horizon: int,
    *,
    lags: int,
    build_model: Callable[[int, int], Any],
    normalize: bool,
    device: str,
    cfg: TorchTrainConfig,
) -> np.ndarray:
    ...
```

The helper should:
- build `X, Y` via `_make_lagged_xy_multi(...)`
- train with `_train_loop(...)`
- forecast from the final lag window
- restore scale when `normalize=True`

**Step 3: Implement `torch-informer-direct` minimally**

Use a lite encoder-only variant:
- standard token embedding over lag positions
- Transformer encoder stack
- final-token projection to `horizon`

It does **not** need true ProbSparse attention in Wave 1A; keep the implementation honest by calling it “Informer-style (lite)” in docs.

**Step 4: Implement `torch-autoformer-direct` minimally**

Add:
- moving-average decomposition on the input series
- separate trend / seasonal branches
- encoder stack over the seasonal branch
- additive trend projection head

**Step 5: Implement `torch-nonstationary-transformer-direct` minimally**

Add:
- RevIN-style instance normalization
- learned scale/shift restoration around the encoder
- the same direct-multi-horizon output contract

**Step 6: Run focused GREEN checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "informer_direct or autoformer_direct or nonstationary"
```

Expected: PASS for the new local smoke coverage.

**Step 7: Commit**

```bash
git add src/foresight/models/torch_nn.py src/foresight/models/__init__.py
git commit -m "feat: add informer autoformer and nonstationary direct models"
```

---

### Task 3: Add Direct FEDformer, iTransformer, and TimesNet

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Implement `torch-fedformer-direct`**

Mirror the global lite idea, but on local lag windows:
- moving-average decomposition
- FFT-based frequency mixing on the seasonal branch
- linear trend head
- direct horizon head

Use parameters:
- `lags`
- `d_model`
- `num_layers`
- `ffn_dim`
- `modes`
- `ma_window`
- `dropout`

**Step 2: Implement `torch-itransformer-direct`**

Use inverted tokens:
- treat channels/features as tokens
- use time axis as embedding dimension
- for the univariate local path, token count stays small and simple

The output head should still be `Linear(d_model, horizon)`.

**Step 3: Implement `torch-timesnet-direct`**

Reuse the period-detection idea already present in the global implementation:
- rFFT on the lag context
- detect top dominant periods
- reshape into period views
- apply lite Conv2D TimesBlock(s)
- project last `horizon` steps

**Step 4: Export these functions**

Add the new symbols to `src/foresight/models/__init__.py`:

```python
from .torch_nn import (
    torch_fedformer_direct_forecast,
    torch_itransformer_direct_forecast,
    torch_timesnet_direct_forecast,
)
```

and add them to `__all__`.

**Step 5: Run focused GREEN checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "fedformer_direct or itransformer_direct or timesnet_direct"
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/models/torch_nn.py src/foresight/models/__init__.py
git commit -m "feat: add fedformer itransformer and timesnet direct models"
```

---

### Task 4: Add Direct TFT

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Add the failing smoke case if not already present**

Use:

```python
f = make_forecaster(
    "torch-tft-direct",
    lags=48,
    d_model=32,
    nhead=4,
    lstm_layers=1,
    epochs=2,
    batch_size=16,
    patience=2,
    device="cpu",
    seed=0,
)
```

**Step 2: Implement a TFT-style local-lite model**

Do not attempt full Temporal Fusion Transformer fidelity. Keep the direct version small and explicit:
- scalar input projection
- LSTM encoder over lag tokens
- gating / residual block on encoder outputs
- single multi-head attention block over encoder outputs
- final-token or pooled projection to `horizon`

The important part is architectural identity and stable forecasting behavior, not complete TFT feature parity.

**Step 3: Keep parameter names aligned with `torch-tft-global`**

Use:
- `lags`
- `d_model`
- `nhead`
- `lstm_layers`
- `dropout`
- common Torch train params

Avoid `x_cols` in the local direct API.

**Step 4: Run the focused test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "tft_direct"
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/torch_nn.py src/foresight/models/__init__.py
git commit -m "feat: add tft direct model"
```

---

### Task 5: Add Direct TimeMixer and SparseTSF

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/__init__.py`

**Step 1: Implement `torch-timemixer-direct`**

Use a clear multiscale mixer-lite design rather than trying to copy a paper verbatim:
- build 2-3 temporal resolutions from the lag window
- mix within each scale
- fuse scales back into a direct horizon head

Keep parameters small:
- `lags`
- `d_model`
- `num_blocks`
- `multiscale_factors`
- `token_mixing_hidden`
- `channel_mixing_hidden`
- `dropout`

**Step 2: Implement `torch-sparsetsf-direct`**

This should be the cheapest direct model in the batch:
- sparse linear readout over lag positions
- optional block/group downsampling
- direct multi-horizon output

Good starter signature:

```python
def torch_sparsetsf_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 192,
    period_len: int = 24,
    d_model: int = 64,
    ...
) -> np.ndarray:
    ...
```

**Step 3: Export both functions**

Add the imports and `__all__` entries in `src/foresight/models/__init__.py`.

**Step 4: Run focused GREEN checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "timemixer_direct or sparsetsf_direct"
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/torch_nn.py src/foresight/models/__init__.py
git commit -m "feat: add timemixer and sparsetsf direct models"
```

---

### Task 6: Register The New Keys And Keep Metadata Consistent

**Files:**
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Test: `tests/test_models_registry.py`

**Step 1: Add local `ModelSpec` entries**

Add specs for all nine keys near the existing Torch direct block:

```python
"torch-informer-direct": ModelSpec(
    key="torch-informer-direct",
    description="Torch Informer-style (lite) direct multi-horizon forecaster. Requires PyTorch.",
    factory=_factory_torch_informer_direct,
    default_params={...},
    param_help={...},
    requires=("torch",),
),
```

Do the same for:
- `torch-autoformer-direct`
- `torch-nonstationary-transformer-direct`
- `torch-fedformer-direct`
- `torch-itransformer-direct`
- `torch-timesnet-direct`
- `torch-tft-direct`
- `torch-timemixer-direct`
- `torch-sparsetsf-direct`

**Step 2: Keep naming and parameter help aligned with neighboring Torch models**

Match existing registry conventions:
- shared training params from `_TORCH_COMMON_DEFAULTS`
- shared param help from `_TORCH_COMMON_PARAM_HELP`
- direct-model parameter names use `lags`, not `context_length`

**Step 3: Add missing factory wrappers**

If needed, create small `_factory_torch_*` wrappers in `src/foresight/models/registry.py` consistent with neighboring local Torch models.

**Step 4: Run registry tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/registry.py src/foresight/models/__init__.py tests/test_models_registry.py
git commit -m "feat: register wave1a torch local parity models"
```

---

### Task 7: Update Optional-Dependency Coverage And README Surfaces

**Files:**
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md`

**Step 1: Finalize optional-dependency smoke coverage**

Make sure the installed-Torch smoke list in `tests/test_models_optional_deps_torch.py` includes at least one of the new direct families with small settings.

**Step 2: Update the model zoo section in README**

Extend the local Torch categories so the new keys are discoverable:
- transformer group: add `torch-informer-direct`, `torch-autoformer-direct`, `torch-fedformer-direct`, `torch-nonstationary-transformer-direct`, `torch-itransformer-direct`, `torch-timesnet-direct`
- hybrid group: add `torch-tft-direct`
- modern/lightweight group: add `torch-timemixer-direct`, `torch-sparsetsf-direct`

**Step 3: Tighten the roadmap note**

Update [the research roadmap](2026-03-09-algorithm-clusters-2001-2026-roadmap.md) so Wave 1 explicitly says:
- `TimeXer` is deferred to a covariate-aware local interface phase
- Wave 1A covers the nine direct models in this plan

**Step 4: Regenerate generated docs**

Run:

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src python tools/generate_model_capability_docs.py --check
PYTHONPATH=src python tools/check_capability_docs.py
```

Expected: generated docs update cleanly and checks pass.

**Step 5: Commit**

```bash
git add README.md docs/models.md docs/api.md docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md tests/test_models_optional_deps_torch.py
git commit -m "docs: document wave1a torch local parity models"
```

---

### Task 8: Full Verification

**Files:**
- Verify only

**Step 1: Ruff format check**

Run:

```bash
ruff format --check \
  src/foresight/models/torch_nn.py \
  src/foresight/models/registry.py \
  src/foresight/models/__init__.py \
  tests/test_models_registry.py \
  tests/test_models_optional_deps_torch.py \
  tests/test_models_torch_xformer_seq2seq_smoke.py \
  README.md
```

Expected: PASS

**Step 2: Ruff lint**

Run:

```bash
ruff check \
  src/foresight/models/torch_nn.py \
  src/foresight/models/registry.py \
  src/foresight/models/__init__.py \
  tests/test_models_registry.py \
  tests/test_models_optional_deps_torch.py \
  tests/test_models_torch_xformer_seq2seq_smoke.py
```

Expected: PASS

**Step 3: Run the focused Torch suite**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/test_models_registry.py \
  tests/test_models_optional_deps_torch.py \
  tests/test_models_torch_xformer_seq2seq_smoke.py
```

Expected: PASS

**Step 4: Run the generated-doc sync test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_docs_rnn_generated.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: complete wave1a torch local parity expansion"
```

---

## Follow-up After Wave 1A

Start a separate plan for `torch-timexer-direct` and generic local Torch `x_cols` support. That work should cover:

- a shared local Torch forecaster API for observed + future exogenous arrays
- capability flags for `supports_x_cols` and `requires_future_covariates`
- `forecast_model_long_df` and `eval_model_long_df` integration
- realistic covariate-aware local smoke tests
