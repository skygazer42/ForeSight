# Deep Learning (Transformer + RNN) Expansion — 100 Tasks

**Goal:** 在 `foresight` 的 model zoo 中系统性扩充 **Transformer 系列** 与 **RNN 系列**（local + global / panel），并提供统一的可选依赖（torch）接口、可回测评测与最小可运行示例。

**Architecture (high-level):**
- 继续沿用现有两种接口：
  - `interface=local`：`(train_1d, horizon) -> yhat[horizon]`
  - `interface=global`：`(long_df, cutoff, horizon) -> pred_df[unique_id, ds, yhat]`
- 新增一套 “可配置的” Torch Transformer/RNN 基础模块（attention 变体、norm/ffn 变体、seq2seq 变体），再通过 **大量 registry keys** 暴露为具体算法/变体，保证：
  - 依赖仍然可选（`.[torch]`）
  - CLI / API 能列出并运行
  - 测试只做少量 smoke（避免把 CI 变成训练集）

---

## Tasks (1–100)

### A. Shared DL building blocks (1–20)
1. Add RMSNorm module for Torch sequence models
2. Add SwiGLU feed-forward option for Transformer blocks
3. Add sinusoidal positional encoding (parameter-free)
4. Add learned positional embedding option
5. Add rotary positional embedding (RoPE) option
6. Add Time2Vec positional features option
7. Add sliding-window (local) attention mask helper
8. Add Performer-style random-feature attention module (lite)
9. Add Linformer-style low-rank KV projection attention module (lite)
10. Add Nyström-style landmark attention approximation module (lite)
11. Add unified Transformer block builder (norm/ffn/attn pluggable)
12. Add encoder-only forecasting head (token-wise horizon head)
13. Add decoder-token horizon head (learned query tokens)
14. Add RevIN (reversible instance norm) option for 1D series
15. Add residual gating option for sequence blocks
16. Add dropout + stochastic depth option (lite)
17. Add global/panel series-id embedding reuse helper
18. Add “known future covariates” packing helper (x_cols + time feats)
19. Add training-window sampler stride for large panels
20. Add fast deterministic seeding utilities for Torch models

### B. Local Transformer models (21–60) — 40 new keys
21. Register local xformer: full attention + LayerNorm + GELU
22. Register local xformer: full attention + RMSNorm + GELU
23. Register local xformer: full attention + LayerNorm + SwiGLU
24. Register local xformer: full attention + RMSNorm + SwiGLU
25. Register local xformer: local-window attention + LayerNorm + GELU
26. Register local xformer: local-window attention + RMSNorm + GELU
27. Register local xformer: local-window attention + LayerNorm + SwiGLU
28. Register local xformer: local-window attention + RMSNorm + SwiGLU
29. Register local xformer: performer attention + LayerNorm + GELU
30. Register local xformer: performer attention + RMSNorm + GELU
31. Register local xformer: performer attention + LayerNorm + SwiGLU
32. Register local xformer: performer attention + RMSNorm + SwiGLU
33. Register local xformer: linformer attention + LayerNorm + GELU
34. Register local xformer: linformer attention + RMSNorm + GELU
35. Register local xformer: linformer attention + LayerNorm + SwiGLU
36. Register local xformer: linformer attention + RMSNorm + SwiGLU
37. Register local xformer: nystrom attention + LayerNorm + GELU
38. Register local xformer: nystrom attention + RMSNorm + GELU
39. Register local xformer: nystrom attention + LayerNorm + SwiGLU
40. Register local xformer: nystrom attention + RMSNorm + SwiGLU
41. Register local xformer: RoPE positional + full attention (LN+GELU)
42. Register local xformer: RoPE positional + performer (LN+GELU)
43. Register local xformer: RoPE positional + linformer (LN+GELU)
44. Register local xformer: RoPE positional + nystrom (LN+GELU)
45. Register local xformer: sinusoidal pos + full attention (LN+GELU)
46. Register local xformer: sinusoidal pos + performer (LN+GELU)
47. Register local xformer: sinusoidal pos + linformer (LN+GELU)
48. Register local xformer: sinusoidal pos + nystrom (LN+GELU)
49. Register local xformer: Time2Vec pos + full attention (LN+GELU)
50. Register local xformer: Time2Vec pos + performer (LN+GELU)
51. Register local xformer: Time2Vec pos + linformer (LN+GELU)
52. Register local xformer: Time2Vec pos + nystrom (LN+GELU)
53. Register local xformer: RevIN + full attention
54. Register local xformer: RevIN + performer
55. Register local xformer: RevIN + linformer
56. Register local xformer: RevIN + nystrom
57. Register local xformer: deeper config (4 layers) + full attention
58. Register local xformer: deeper config (4 layers) + performer
59. Register local xformer: wider config (d_model=128) + full attention
60. Register local xformer: wider config (d_model=128) + performer

### C. Global / panel Transformer models (61–80) — 20 new keys
61. Register global xformer: full attention baseline
62. Register global xformer: local-window attention
63. Register global xformer: performer attention
64. Register global xformer: linformer attention
65. Register global xformer: nystrom attention
66. Register global xformer: full + RMSNorm
67. Register global xformer: local + RMSNorm
68. Register global xformer: performer + RMSNorm
69. Register global xformer: linformer + RMSNorm
70. Register global xformer: nystrom + RMSNorm
71. Register global xformer: full + SwiGLU
72. Register global xformer: performer + SwiGLU
73. Register global xformer: linformer + SwiGLU
74. Register global xformer: nystrom + SwiGLU
75. Register global xformer: RoPE + full
76. Register global xformer: RoPE + performer
77. Register global xformer: sinusoidal pos + full
78. Register global xformer: Time2Vec pos + full
79. Register global xformer: deeper config (4 layers)
80. Register global xformer: wider config (d_model=128)

### D. Local RNN models (81–95) — 15 new keys
81. Add local seq2seq LSTM direct multi-horizon forecaster
82. Add local seq2seq GRU direct multi-horizon forecaster
83. Add local attention seq2seq (Bahdanau) LSTM
84. Add local attention seq2seq (Bahdanau) GRU
85. Add local scheduled teacher forcing option
86. Register local seq2seq-lstm key (baseline)
87. Register local seq2seq-gru key (baseline)
88. Register local seq2seq-attn-lstm key
89. Register local seq2seq-attn-gru key
90. Register local seq2seq-lstm deeper (2 layers)
91. Register local seq2seq-gru deeper (2 layers)
92. Register local seq2seq-lstm wider (hidden=128)
93. Register local seq2seq-gru wider (hidden=128)
94. Add local LSTNet-style CNN+RNN model (lite)
95. Register local lstnet key (baseline)

### E. Global / panel RNN models (96–100) — 5 new keys
96. Add global/panel LSTM backbone (uses covariates + time features)
97. Add global/panel GRU backbone (uses covariates + time features)
98. Register global rnn-lstm key
99. Register global rnn-gru key
100. Add one global seq2seq-lite key (encoder-only RNN horizon head)

