# RetNet Global Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class `torch-retnet-global` panel/global forecasting family to ForeSight, reusing the existing long-format global training pipeline and matching the current Torch global model surface.

**Architecture:** Implement `RetNet` global as a dedicated predictor in `src/foresight/models/torch_global.py` rather than extending the older `_predict_torch_global()` multiplexer. The new model should consume `_build_panel_dataset()` windows, append series-id embeddings, apply stacked retention blocks over the full `(context + horizon)` token sequence, and emit either direct point forecasts or quantile forecasts through the existing `_make_pred_df_from_scaled()` helper.

**Tech Stack:** Python 3.10+, NumPy, Pandas, PyTorch, pytest, MkDocs

---

### Task 1: Lock Global RetNet Behavior In Tests

**Files:**
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_global_interface.py`
- Modify: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Add registry coverage**

Require `torch-retnet-global` to appear in `list_models()` and to be treated as a Torch global model.

**Step 2: Add optional-dependency coverage**

Require the new key to appear in the Torch optional global model set.

**Step 3: Add global interface coverage**

Require `get_model_spec("torch-retnet-global").interface == "global"` and `make_forecaster("torch-retnet-global")` to reject it as a local model.

**Step 4: Add a CPU smoke test**

Require a tiny `make_global_forecaster("torch-retnet-global", ...)` run to emit finite `unique_id / ds / yhat` rows.

**Step 5: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k "retnet and global"
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k "retnet and global"
PYTHONPATH=src pytest -q tests/test_models_global_interface.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "retnet and global"
```

---

### Task 2: Implement The Global RetNet Predictor

**Files:**
- Modify: `src/foresight/models/torch_global.py`

**Step 1: Add `_predict_torch_retnet_global()`**

Use `_build_panel_dataset()` and the existing global train loop.

**Step 2: Keep the model contract aligned**

Support:
- `context_length`
- `x_cols`
- `add_time_features`
- `normalize`
- `sample_step`
- `max_train_size`
- standard Torch training params
- `d_model`
- `nhead`
- `num_layers`
- `ffn_dim`
- `id_emb_dim`
- `dropout`
- `quantiles`

**Step 3: Keep the implementation honest**

Describe the model as a lite retention network rather than full paper parity.

---

### Task 3: Register The Global Family

**Files:**
- Modify: `src/foresight/models/registry.py`

**Step 1: Import the global forecaster**

Wire `torch_retnet_global_forecaster` into the Torch global import list.

**Step 2: Add `ModelSpec` metadata**

Register:
- key: `torch-retnet-global`
- requires: `("torch",)`
- interface: `global`
- optional `quantiles` support

**Step 3: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k "retnet and global"
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k "retnet and global"
PYTHONPATH=src pytest -q tests/test_models_global_interface.py -k retnet
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "retnet and global"
```

---

### Task 4: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/models.md`
- Modify: `docs/api.md`

**Step 1: Add the model to the README global table**

Place `torch-retnet-global` with the Transformer / long-sequence global models.

**Step 2: Regenerate capability docs**

Run:

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src python tools/check_capability_docs.py
```

---

### Task 5: Verify The Integrated Change

**Files:**
- Verify only

**Step 1: Run the targeted RetNet global tests**

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_global_interface.py tests/test_models_torch_xformer_seq2seq_smoke.py -k retnet
```

**Step 2: Run a direct smoke command**

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
import pandas as pd
from foresight.models.registry import make_global_forecaster

ds = pd.date_range("2020-01-01", periods=72, freq="D")
rows = []
for uid, bias in [("s0", 0.0), ("s1", 0.4)]:
    promo = (np.arange(ds.size) % 7 == 0).astype(float)
    y = bias + np.sin(np.arange(ds.size, dtype=float) / 7.0) + 0.5 * promo
    for t, yv, pv in zip(ds, y, promo, strict=True):
        rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
long_df = pd.DataFrame(rows)
g = make_global_forecaster(
    "torch-retnet-global",
    context_length=32,
    d_model=24,
    nhead=4,
    num_layers=1,
    ffn_dim=48,
    sample_step=4,
    epochs=1,
    batch_size=32,
    patience=2,
    val_split=0.0,
    x_cols=("promo",),
    device="cpu",
    seed=0,
)
print(g(long_df, ds[-6], 5).head().to_string(index=False))
PY
```
