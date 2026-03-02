# Global DL Architectures + Probabilistic Forecasting (Torch) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `foresight` 的 Torch（可选依赖）全局/面板模型里继续扩充“论文级”架构（PatchTST / TSMixer / iTransformer），并补齐更现代的概率预测能力（quantile regression）与更顺滑的评测链路（保留额外预测列）。

**Architecture:** 基于现有 `interface="global"` 约定 `(long_df, cutoff, horizon) -> pred_df[unique_id, ds, yhat]`，新增模型可选输出 `yhat_pXX` 分位数列；CV / eval 侧保留这些列并可计算 pinball 与区间指标。

**Tech Stack:** Python, NumPy/Pandas, PyTorch（`.[torch]` 可选依赖），pytest, ruff

---

## Scope

### A. 新增全局/面板 Torch 模型（paper-level, lite）
- `torch-patchtst-global`：PatchTST-style patch tokenization + Transformer encoder（lite）
- `torch-tsmixer-global`：TSMixer-style token/channel mixing blocks（lite）
- `torch-itransformer-global`：iTransformer-style inverted tokens（variables as tokens, lite）

### B. 概率预测（quantiles）
- 全局模型支持 `quantiles=...`（如 `"0.1,0.5,0.9"`），输出：
  - `yhat`：默认取 `p50`（或最接近 0.5 的分位数）
  - `yhat_p10/yhat_p50/yhat_p90`：按 quantiles 生成
- 训练使用 pinball loss（多分位数平均）

### C. 评测链路增强（global CV 保留额外列）
- `cross_validation_predictions_long_df(interface=global)` 不再丢弃 `pred_df` 中额外预测列（例如 `yhat_pXX`）
- `eval_model_long_df(interface=global)` 在存在 quantile 列时，附加输出 pinball / interval metrics（不破坏现有点预测指标）

---

## Implementation Tasks (high-level)

1. 在 `src/foresight/models/torch_global.py` 中实现三个新 global forecaster，并复用现有 `_build_panel_dataset` / `_train_loop_global`
2. 为 `_train_loop_global` 增加可选的 `loss_fn_override`，以支持 pinball loss
3. 统一 quantile 参数解析与列命名（`yhat_p{pct}`）
4. 重构 `src/foresight/cv.py` 的 global 分支以保留额外预测列（保持输出 schema 兼容）
5. 在 `src/foresight/metrics.py` / `src/foresight/eval_predictions.py` 中增加 pinball/interval 的评估辅助
6. `src/foresight/models/registry.py` 注册新模型 + 参数帮助（仍保持 torch 为可选依赖）
7. 增加 examples + smoke tests（训练 epoch 少、保证 CI 可跑）
8. 更新 README：新增模型条目、概率预测说明、相关项目对比（轻量介绍）

