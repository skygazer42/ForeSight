# Deep Learning Model Zoo Expansion (100 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand ForeSight’s optional PyTorch model zoo into a richer set of modern deep learning time-series forecasting algorithms, while keeping the core package lightweight (`numpy/pandas` only) and maintaining a simple `(train_1d, horizon) -> yhat` interface for walk-forward backtests.

**Architecture:**
- **Per-series, lag-window training**: each model trains on a single 1D series using lag windows and predicts a multi-step horizon (direct) or predicts 1-step and rolls forward (recursive).
- **Optional dependency gate**: deep models are available behind `pip install -e ".[torch]"`.
- **Unified registry**: every model is accessible through `foresight.models.registry.make_forecaster`.

**Tech Stack:** Python 3.10+, `numpy`, `pandas`; optional `torch>=2.0`. Tooling: `pytest`, `ruff`.

---

## Task Checklist (100)

### A) Implement Torch Model Algorithms (24)
1. [x] Implement `torch-mlp-direct` (MLP direct multi-horizon)
2. [x] Implement `torch-lstm-direct` (LSTM direct multi-horizon)
3. [x] Implement `torch-gru-direct` (GRU direct multi-horizon)
4. [x] Implement `torch-tcn-direct` (TCN direct multi-horizon)
5. [x] Implement `torch-nbeats-direct` (N-BEATS-style direct multi-horizon)
6. [x] Implement `torch-nlinear-direct` (NLinear-style direct baseline)
7. [x] Implement `torch-dlinear-direct` (DLinear-style decomposition + linear)
8. [x] Implement `torch-transformer-direct` (Transformer encoder direct multi-horizon)
9. [x] Implement `torch-patchtst-direct` (PatchTST-style patching + encoder)
10. [x] Implement `torch-tsmixer-direct` (TSMixer-style token/channel mixing)
11. [x] Implement `torch-cnn-direct` (Conv1D stack)
12. [x] Implement `torch-resnet1d-direct` (ResNet-1D)
13. [x] Implement `torch-wavenet-direct` (WaveNet-style gated dilated CNN)
14. [x] Implement `torch-bilstm-direct` (Bidirectional LSTM)
15. [x] Implement `torch-bigru-direct` (Bidirectional GRU)
16. [x] Implement `torch-attn-gru-direct` (GRU + attention pooling)
17. [x] Implement `torch-fnet-direct` (FNet-style Fourier mixing)
18. [x] Implement `torch-linear-attn-direct` (Linear-attention encoder)
19. [x] Implement `torch-inception-direct` (InceptionTime-style Conv1D)
20. [x] Implement `torch-gmlp-direct` (gMLP-style token gating/mixing)
21. [x] Implement `torch-nhits-direct` (N-HiTS-style multi-rate residual MLP)
22. [x] Implement `torch-tide-direct` (TiDE-style encoder/decoder MLP)
23. [x] Implement `torch-deepar-recursive` (DeepAR-style Gaussian RNN, recursive)
24. [x] Implement `torch-qrnn-recursive` (Quantile RNN via pinball loss, recursive)

### B) Register Torch Models in `foresight` Registry (24)
25. [x] Register model spec for `torch-mlp-direct`
26. [x] Register model spec for `torch-lstm-direct`
27. [x] Register model spec for `torch-gru-direct`
28. [x] Register model spec for `torch-tcn-direct`
29. [x] Register model spec for `torch-nbeats-direct`
30. [x] Register model spec for `torch-nlinear-direct`
31. [x] Register model spec for `torch-dlinear-direct`
32. [x] Register model spec for `torch-transformer-direct`
33. [x] Register model spec for `torch-patchtst-direct`
34. [x] Register model spec for `torch-tsmixer-direct`
35. [x] Register model spec for `torch-cnn-direct`
36. [x] Register model spec for `torch-resnet1d-direct`
37. [x] Register model spec for `torch-wavenet-direct`
38. [x] Register model spec for `torch-bilstm-direct`
39. [x] Register model spec for `torch-bigru-direct`
40. [x] Register model spec for `torch-attn-gru-direct`
41. [x] Register model spec for `torch-fnet-direct`
42. [x] Register model spec for `torch-linear-attn-direct`
43. [x] Register model spec for `torch-inception-direct`
44. [x] Register model spec for `torch-gmlp-direct`
45. [x] Register model spec for `torch-nhits-direct`
46. [x] Register model spec for `torch-tide-direct`
47. [x] Register model spec for `torch-deepar-recursive`
48. [x] Register model spec for `torch-qrnn-recursive`

### C) Export Torch Models from `foresight.models` (24)
49. [x] Export `torch_mlp_lag_direct_forecast` in `src/foresight/models/__init__.py`
50. [x] Export `torch_lstm_direct_forecast` in `src/foresight/models/__init__.py`
51. [x] Export `torch_gru_direct_forecast` in `src/foresight/models/__init__.py`
52. [x] Export `torch_tcn_direct_forecast` in `src/foresight/models/__init__.py`
53. [x] Export `torch_nbeats_direct_forecast` in `src/foresight/models/__init__.py`
54. [x] Export `torch_nlinear_direct_forecast` in `src/foresight/models/__init__.py`
55. [x] Export `torch_dlinear_direct_forecast` in `src/foresight/models/__init__.py`
56. [x] Export `torch_transformer_direct_forecast` in `src/foresight/models/__init__.py`
57. [x] Export `torch_patchtst_direct_forecast` in `src/foresight/models/__init__.py`
58. [x] Export `torch_tsmixer_direct_forecast` in `src/foresight/models/__init__.py`
59. [x] Export `torch_cnn_direct_forecast` in `src/foresight/models/__init__.py`
60. [x] Export `torch_resnet1d_direct_forecast` in `src/foresight/models/__init__.py`
61. [x] Export `torch_wavenet_direct_forecast` in `src/foresight/models/__init__.py`
62. [x] Export `torch_bilstm_direct_forecast` in `src/foresight/models/__init__.py`
63. [x] Export `torch_bigru_direct_forecast` in `src/foresight/models/__init__.py`
64. [x] Export `torch_attn_gru_direct_forecast` in `src/foresight/models/__init__.py`
65. [x] Export `torch_fnet_direct_forecast` in `src/foresight/models/__init__.py`
66. [x] Export `torch_linear_attention_direct_forecast` in `src/foresight/models/__init__.py`
67. [x] Export `torch_inception_direct_forecast` in `src/foresight/models/__init__.py`
68. [x] Export `torch_gmlp_direct_forecast` in `src/foresight/models/__init__.py`
69. [x] Export `torch_nhits_direct_forecast` in `src/foresight/models/__init__.py`
70. [x] Export `torch_tide_direct_forecast` in `src/foresight/models/__init__.py`
71. [x] Export `torch_deepar_recursive_forecast` in `src/foresight/models/__init__.py`
72. [x] Export `torch_qrnn_recursive_forecast` in `src/foresight/models/__init__.py`

### D) Document Torch Models in `README.md` (24)
73. [x] Document `torch-mlp-direct` in README model zoo
74. [x] Document `torch-lstm-direct` in README model zoo
75. [x] Document `torch-gru-direct` in README model zoo
76. [x] Document `torch-tcn-direct` in README model zoo
77. [x] Document `torch-nbeats-direct` in README model zoo
78. [x] Document `torch-nlinear-direct` in README model zoo
79. [x] Document `torch-dlinear-direct` in README model zoo
80. [x] Document `torch-transformer-direct` in README model zoo
81. [x] Document `torch-patchtst-direct` in README model zoo
82. [x] Document `torch-tsmixer-direct` in README model zoo
83. [x] Document `torch-cnn-direct` in README model zoo
84. [x] Document `torch-resnet1d-direct` in README model zoo
85. [x] Document `torch-wavenet-direct` in README model zoo
86. [x] Document `torch-bilstm-direct` in README model zoo
87. [x] Document `torch-bigru-direct` in README model zoo
88. [x] Document `torch-attn-gru-direct` in README model zoo
89. [x] Document `torch-fnet-direct` in README model zoo
90. [x] Document `torch-linear-attn-direct` in README model zoo
91. [x] Document `torch-inception-direct` in README model zoo
92. [x] Document `torch-gmlp-direct` in README model zoo
93. [x] Document `torch-nhits-direct` in README model zoo
94. [x] Document `torch-tide-direct` in README model zoo
95. [x] Document `torch-deepar-recursive` in README model zoo
96. [x] Document `torch-qrnn-recursive` in README model zoo

### E) Improve Torch Training Loop + Optional-Dep UX (4)
97. [x] Add advanced training configuration to `TorchTrainConfig` (loss/opt/scheduler/val_split/grad_clip/restore_best)
98. [x] Upgrade `_train_loop` with val split + early stopping + best checkpoint restore
99. [x] Add `_train_loop` optimizer/loss/scheduler selection and gradient clipping
100. [x] Add `_train_loop` custom loss override for probabilistic/quantile models (DeepAR/QRNN)

