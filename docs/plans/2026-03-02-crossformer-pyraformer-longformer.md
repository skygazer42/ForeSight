# Crossformer + Pyraformer + Longformer Attention — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 ForeSight 的 Torch model zoo 中补齐两类常见长序列时序 Transformer（Crossformer / Pyraformer）并为可配置 `torch-xformer-*` 增加 Longformer-style attention。

**Architecture:**
- **Longformer attention（xFormer 变体）**：在现有 `torch-xformer-*` 的 attention 选项中增加 `attn="longformer"`，采用“滑动窗口 local attention + 少量 global tokens”的稀疏 pattern；为了能做 direct multi-horizon 预测，将 **horizon query tokens** 视为 global queries（可访问全上下文）。
- **Crossformer-lite（local + global/panel）**：把输入序列做 multi-scale segmentation（`segment_len * 2^i`），每个 scale 的 segment flatten 后投影成 token，再把不同 scale token 拼接进同一个 Transformer encoder 做 cross-scale mixing，最后对 token 做 pooling → 直接输出 `horizon`。
- **Pyraformer-lite（local + global/panel）**：先在 level-0 从 segment token 构建基础 token 序列，再对 token 做 factor-2 pooling 得到更粗粒度 level-1..L token；将所有 level tokens concat 后交给 Transformer encoder，pooling → 直接输出 `horizon`。

**Tech Stack:** `numpy` / `pandas` + optional `torch` (`pip install -e ".[torch]"`).

---

### Task 1: Add `attn="longformer"` to configurable Torch xFormer (local + global)

**Files:**
- Modify: `src/foresight/models/torch_xformer.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Steps:**
1. Extend attention validation sets to include `"longformer"`.
2. Implement Longformer-style mask as `local_window` + global-mask.
3. Register new registry keys:
   - `torch-xformer-longformer-*-direct`
   - `torch-xformer-longformer-*-global`
4. Add minimal smoke tests for local/global longformer variants.

---

### Task 2: Add `torch-crossformer-direct` (lite)

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/registry.py`
- Test: `tests/test_models_registry.py`

**Steps:**
1. Implement `torch_crossformer_direct_forecast()` using multi-scale `unfold()` segmentation.
2. Add `_factory_torch_crossformer_direct()` and a new `ModelSpec` entry `torch-crossformer-direct`.
3. Add registry assertion test.

---

### Task 3: Add `torch-crossformer-global` (lite)

**Files:**
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `examples/torch_global_models.py`
- Modify: `README.md`
- Test: `tests/test_models_registry.py`

**Steps:**
1. Implement `_predict_torch_crossformer_global()` (panel dataset → multi-scale segment tokens → encoder → pooled head).
2. Add `torch_crossformer_global_forecaster()` wrapper.
3. Register `torch-crossformer-global` in the model zoo + docs/examples.

---

### Task 4: Add `torch-pyraformer-direct` + `torch-pyraformer-global` (lite)

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/torch_global.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `examples/torch_global_models.py`
- Modify: `README.md`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_torch_crossformer_pyraformer_smoke.py`

**Steps:**
1. Implement local Pyraformer-lite with token pyramid pooling (`level_sizes` halving).
2. Implement global/panel Pyraformer-lite with the same pyramid pattern over (context+horizon) sequence.
3. Register new keys + docs + examples.
4. Add smoke tests for both local/global.

---

### Task 5: Verify and ship

**Steps:**
1. Run: `pytest -q`
2. Run: `ruff check src tests tools` (if ruff is part of the dev environment)
3. Update docs references (Crossformer / Pyraformer repos).
4. Commit and push to `main`.

---

## References

- Crossformer (ICLR 2023): https://github.com/Thinklab-SJTU/Crossformer
- Pyraformer (ICLR 2022): https://github.com/ant-research/Pyraformer
- Longformer (attention pattern): https://arxiv.org/abs/2004.05150

