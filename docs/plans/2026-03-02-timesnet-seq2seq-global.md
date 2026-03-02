# TimesNet + Global Seq2Seq (Torch) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 继续扩充 `foresight` 的 Torch 全局/面板模型：新增 `TimesNet`（lite）与“真正的”encoder-decoder `Seq2Seq`（带 attention/teacher forcing），并保持 quantile 概率预测与 CV/eval 链路兼容。

**Architecture:** 复用 `src/foresight/models/torch_global.py` 的 panel 数据构造 `_build_panel_dataset()` 与训练循环 `_train_loop_global()`；新增模型遵循 `interface="global"`：`(long_df, cutoff, horizon) -> pred_df[unique_id, ds, yhat, (optional yhat_pXX...)]`。Seq2Seq 使用自定义训练 loop（需要 teacher forcing）但输出格式保持一致。

**Tech Stack:** Python, NumPy/Pandas, PyTorch (`.[torch]` optional), pytest, ruff

---

### Task 1: Add TimesNet-lite global model

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Write the failing smoke test (registration + run)**

Add a new smoke case that trains 1 epoch and returns finite predictions:

```python
g = make_global_forecaster("torch-timesnet-global", context_length=32, epochs=1, batch_size=32, patience=2)
pred = g(long_df, cutoff, horizon)
assert set(pred.columns) >= {"unique_id","ds","yhat"}
```

**Step 2: Implement `_predict_torch_timesnet_global` + `torch_timesnet_global_forecaster`**

- TimesBlock-lite: period candidates from rFFT on context y; reshape by period and apply Conv2d; residual add.
- Output head supports `quantiles` (pinball loss + `yhat_pXX` columns).

**Step 3: Register `torch-timesnet-global`**

Add `ModelSpec(interface="global")` with reasonable defaults and `quantiles` support.

**Step 4: Run targeted tests**

Run: `pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_timesnet_global_smoke`
Expected: PASS

---

### Task 2: Add global Seq2Seq (encoder-decoder) model

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Step 1: Add failing smoke tests for new keys**

```python
g = make_global_forecaster("torch-seq2seq-lstm-global", context_length=32, epochs=1, batch_size=32, patience=2)
pred = g(long_df, cutoff, horizon)
assert np.all(np.isfinite(pred["yhat"]))
```

**Step 2: Implement `_predict_torch_seq2seq_global` + forecaster wrappers**

- Encoder: LSTM/GRU over context steps (with id embedding concatenated).
- Decoder: step-wise RNN, uses known future covariates/time features; teacher forcing using `Y_train`.
- Optional Bahdanau attention from decoder state to encoder outputs.
- Output head supports `quantiles` (multi-quantile output + pinball loss).

**Step 3: Register keys**

- `torch-seq2seq-lstm-global`
- `torch-seq2seq-attn-lstm-global` (attention)

**Step 4: Run targeted tests**

Run: `pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_seq2seq_global_smoke`
Expected: PASS

---

### Task 3: Docs + examples polish

**Files:**
- Modify: `README.md`
- Modify: `examples/torch_global_models.py`

**Step 1: README**

- Add `torch-timesnet-global` and global seq2seq keys to model zoo section.
- Mention that these global models support `quantiles`.

**Step 2: Example**

- Extend `examples/torch_global_models.py` list with the new models (short epochs).

---

### Task 4: Verification + commit

**Step 1: Formatting**

Run: `ruff format src tests tools`

**Step 2: Lint**

Run: `ruff check src tests tools`

**Step 3: Tests**

Run: `pytest -q`

**Step 4: Commit**

Run:

```bash
git add -A
git commit -m "feat: add timesnet + global seq2seq models"
```

