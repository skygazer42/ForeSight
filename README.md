# ForeSight

> A practical time-series forecasting playground + a lightweight benchmarking toolkit (`foresight`).

![GitHub last commit](https://img.shields.io/github/last-commit/jhlucc/ForeSight)
![GitHub stars](https://img.shields.io/github/stars/jhlucc/ForeSight)
![License](https://img.shields.io/github/license/jhlucc/ForeSight)

---

## 📖 项目简介 | Overview

这个仓库有两条主线：

1) **脚本/笔记集合（仓库根目录各子文件夹）**  
   - `statistics time series/`：经典统计模型与示例脚本（ARMA/ARIMA/VAR…）  
   - `ml time series/`：传统 ML 的时序建模脚本（Prophet/树模型特征工程…）  
   - `transformer time series/`：深度学习/Transformer 系列实验区（Informer/FEDformer/Autoformer…）  
   - `paper/`：阅读笔记与资料整理

2) **`foresight` Python 包（`src/foresight/`）**  
   一个“默认依赖很轻”的 forecasting toolkit：**模型注册表 + 统一评测/回测 + 指标 + CLI**。  
   目标是让你用一套统一接口快速跑 baseline、做 walk-forward backtesting、输出可对比的 leaderboard。

> 设计风格参考主流时间序列工具的常见模式：统一模型接口（sktime/Darts 风格）、长表/面板数据（StatsForecast/Prophet 常见列约定）、backtesting-first（walk-forward / cross-validation）。见文末 *Related Projects*。

---

## 🚀 快速开始 | Quick Start (CLI)

```bash
# 1) 环境
python -m venv .venv
source .venv/bin/activate

# 2) 安装（默认仅 numpy/pandas + dev 工具）
pip install -e ".[dev]"

# 3) 数据集
foresight datasets list
foresight datasets preview catfish --nrows 10

# 4) 模型注册表
foresight models list
foresight models info theta

# 5) 评测（任意注册模型）
foresight eval run --model naive-last --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12
foresight eval run --model seasonal-naive --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --model-param season_length=12

# 6) Leaderboard（多个模型）
foresight leaderboard models --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --models naive-last,seasonal-naive,theta,holt
foresight leaderboard models --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --models naive-last,seasonal-naive --format md

# 7) Cross-validation 预测明细表（用于分析/画图/区间校准）
foresight cv run --model theta --dataset catfish --y-col Total --horizon 3 --step-size 3 --min-train-size 12 --n-windows 30 --format csv

# 8) Conformal intervals（基于回测残差的对称区间；快速 baseline）
foresight eval run --model theta --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --conformal-levels 80,90

# 9) 直接评测任意 CSV（无需注册到 datasets registry）
foresight eval csv --model naive-last --path ./my.csv --time-col ds --y-col y --parse-dates --horizon 3 --step 1 --min-train-size 12
```

---

## 🧪 快速开始 | Quick Start (Python API)

```python
from foresight.eval_forecast import eval_model
from foresight.intervals import bootstrap_intervals
from foresight.models.registry import make_forecaster, make_global_forecaster

metrics = eval_model(
    model="theta",
    dataset="catfish",
    y_col="Total",
    horizon=3,
    step=3,
    min_train_size=12,
    model_params={"alpha": 0.2},
)

f = make_forecaster("moving-average", window=3)
intervals = bootstrap_intervals([1, 2, 3, 4, 5, 6], horizon=3, forecaster=f, min_train_size=4)

# Global / panel model (requires torch; trains across all series in a long-format DataFrame)
# g = make_global_forecaster("torch-informer-global", context_length=96, epochs=10, x_cols=("promo",))
# cutoff = sorted(long_df["ds"].unique())[-(14 + 1)]  # leave 14 steps for the horizon
# pred_df = g(long_df, cutoff=cutoff, horizon=14)  # returns: unique_id, ds, yhat
```

更多可运行示例见：
- `examples/quickstart_eval.py`
- `examples/leaderboard.py`
- `examples/cv_and_conformal.py`
- `examples/intermittent_demand.py`
- `examples/torch_global_models.py`

---

## 🗂️ 目录结构 | Repository Structure

| Path | What | Notes |
| --- | --- | --- |
| `src/foresight/` | Python package | CLI + model registry + eval/backtesting |
| `examples/` | Runnable scripts | No notebooks; quick smoke demos |
| `data/` | Local data files | Some datasets are bundled as CSV |
| `statistics time series/` | Classic stats scripts | Educational / experiments |
| `ml time series/` | ML scripts | Feature engineering & models |
| `transformer time series/` | Transformer experiments | Larger dependencies; separate area |
| `paper/` | Notes | Papers & summaries |

---

## 🧠 Model Zoo (`foresight.models`)

`foresight` 内置的是“轻量、可跑、可回测”的模型集合（多数为纯 numpy 实现；少数为可选依赖封装）。

### Core (no optional deps)

- `naive-last`：最后值重复
- `seasonal-naive`：季节性重复（`season_length`）
- `mean` / `median`
- `drift`：线性漂移外推
- `moving-average`：滑动均值（`window`）
- `seasonal-mean`：季节均值（`season_length`）
- `ses`：Simple Exponential Smoothing（`alpha`）
- `ses-auto`：SES 参数自动调参（`grid_size`）
- `holt`：Holt 线性趋势（`alpha`, `beta`）
- `holt-auto`：Holt 参数自动调参（`grid_size`）
- `holt-damped`：Holt damped trend（`alpha`, `beta`, `phi`）
- `holt-winters-add`：Holt-Winters 加性季节（`season_length`, `alpha`, `beta`, `gamma`）
- `holt-winters-add-auto`：HW Add 参数自动调参（`season_length`, `grid_size`）
- `theta`：Theta-style baseline（`alpha`）
- `theta-auto`：Theta 参数自动调参（`grid_size`）
- `ar-ols`：AR(p) by OLS（`p`）
- `ar-ols-lags`：自定义 lag 集的 AR-OLS（`lags=1,2,12`）
- `sar-ols`：季节性 AR-OLS（`p`, `P`, `season_length`）
- `ar-ols-auto`：AIC 自动选择 AR(p)（`max_p`）
- `lr-lag`：lag 特征 + 线性回归（`lags`）
- `lr-lag-direct`：lag 特征 + 线性回归（direct multi-horizon, `lags`）
- `fourier`：Fourier 回归季节项 + 趋势（`period`, `order`, `include_trend`）
- `fourier-multi`：多季节 Fourier 回归（`periods=7,365`, `orders=2`）
- `poly-trend`：多项式趋势回归（`degree`）
- `fft`：FFT 频域外推 baseline（`top_k`, `include_trend`）
- `analog-knn`：Analog kNN（非参数，lag window 最近邻）（`lags`, `k`, `normalize`, `weights`）
- `kalman-level`：Kalman local-level（`process_variance`, `obs_variance`）
- `kalman-trend`：Kalman local linear trend（`level_variance`, `trend_variance`, `obs_variance`）
- `croston`：Croston classic 间歇需求（`alpha`）
- `croston-sba`：Croston SBA bias correction（`alpha`）
- `croston-sbj`：Croston SBJ bias correction（`alpha`）
- `croston-opt`：Croston alpha 自动调参（`grid_size`）
- `tsb`：TSB 间歇需求（`alpha`, `beta`）
- `les`：LES（Linear-Exponential Smoothing，支持“无需求时线性衰减”）（`alpha`, `beta`）
- `adida`：ADIDA 聚合/反聚合（`agg_period`, `base`, `alpha`）
- `pipeline`：meta 模型：先做变换再跑 base 模型（`base`, `transforms`, 其余参数透传）
- `ensemble-mean`：meta 模型：多个成员模型取平均（`members=naive-last,theta,...`）
- `ensemble-median`：meta 模型：多个成员模型取中位数（`members=...`）

### Optional extras

- `ridge-lag`（requires `.[ml]`）：lag 特征 + Ridge 回归（`lags`, `alpha`）
- `rf-lag`（requires `.[ml]`）：lag 特征 + RandomForest（direct multi-horizon, `lags`, `n_estimators`…）
- `lasso-lag`（requires `.[ml]`）：lag 特征 + Lasso（direct multi-horizon, `lags`, `alpha`…）
- `elasticnet-lag`（requires `.[ml]`）：lag 特征 + ElasticNet（direct multi-horizon, `lags`, `alpha`, `l1_ratio`…）
- `knn-lag`（requires `.[ml]`）：lag 特征 + KNN（direct multi-horizon, `lags`, `n_neighbors`…）
- `gbrt-lag`（requires `.[ml]`）：lag 特征 + GradientBoosting（direct multi-horizon, `lags`, `n_estimators`…）
- `torch-mlp-direct`（requires `.[torch]`）：Torch MLP（direct multi-horizon, `lags`, `hidden_sizes`, `epochs`…）
- `torch-lstm-direct`（requires `.[torch]`）：Torch LSTM（direct multi-horizon, `lags`, `hidden_size`, `epochs`…）
- `torch-gru-direct`（requires `.[torch]`）：Torch GRU（direct multi-horizon, `lags`, `hidden_size`, `epochs`…）
- `torch-tcn-direct`（requires `.[torch]`）：Torch TCN（direct multi-horizon, `lags`, `channels`, `epochs`…）
- `torch-nbeats-direct`（requires `.[torch]`）：Torch N-BEATS（direct multi-horizon, `lags`, `num_blocks`, `layer_width`, `epochs`…）
- `torch-nlinear-direct`（requires `.[torch]`）：Torch NLinear（last-value centering + linear, `lags`, `epochs`…）
- `torch-dlinear-direct`（requires `.[torch]`）：Torch DLinear（moving-average decomposition + linear, `lags`, `ma_window`, `epochs`…）
- `torch-transformer-direct`（requires `.[torch]`）：Torch Transformer encoder（direct multi-horizon, `lags`, `d_model`, `nhead`, `epochs`…）
- `torch-mamba-direct`（requires `.[torch]`）：Torch Mamba-style selective SSM（lite, `lags`, `d_model`, `num_layers`, `conv_kernel`, `epochs`…）
- `torch-rwkv-direct`（requires `.[torch]`）：Torch RWKV-style time-mix + channel-mix（lite, `lags`, `d_model`, `num_layers`, `ffn_dim`, `epochs`…）
- `torch-patchtst-direct`（requires `.[torch]`）：Torch PatchTST-style（patching + encoder, `lags`, `patch_len`, `stride`, `epochs`…）
- `torch-tsmixer-direct`（requires `.[torch]`）：Torch TSMixer-style（token/channel mixing, `lags`, `d_model`, `num_blocks`, `epochs`…）
- `torch-cnn-direct`（requires `.[torch]`）：Torch Conv1D stack（direct multi-horizon, `lags`, `channels`, `kernel_size`, `epochs`…）
- `torch-resnet1d-direct`（requires `.[torch]`）：Torch ResNet-1D（direct multi-horizon, `lags`, `channels`, `num_blocks`, `epochs`…）
- `torch-wavenet-direct`（requires `.[torch]`）：Torch WaveNet-style（gated dilated CNN, `lags`, `channels`, `num_layers`, `epochs`…）
- `torch-bilstm-direct`（requires `.[torch]`）：Torch BiLSTM（direct multi-horizon, `lags`, `hidden_size`, `epochs`…）
- `torch-bigru-direct`（requires `.[torch]`）：Torch BiGRU（direct multi-horizon, `lags`, `hidden_size`, `epochs`…）
- `torch-attn-gru-direct`（requires `.[torch]`）：Torch GRU + attention pooling（direct multi-horizon, `lags`, `hidden_size`, `epochs`…）
- `torch-fnet-direct`（requires `.[torch]`）：Torch FNet-style（Fourier mixing, `lags`, `d_model`, `epochs`…）
- `torch-linear-attn-direct`（requires `.[torch]`）：Torch linear attention encoder（`lags`, `d_model`, `epochs`…）
- `torch-inception-direct`（requires `.[torch]`）：Torch InceptionTime-style Conv1D（`lags`, `channels`, `epochs`…）
- `torch-gmlp-direct`（requires `.[torch]`）：Torch gMLP-style（token gating + mixing, `lags`, `d_model`, `epochs`…）
- `torch-nhits-direct`（requires `.[torch]`）：Torch N-HiTS-style（multi-rate residual MLP, `lags`, `pool_sizes`, `epochs`…）
- `torch-tide-direct`（requires `.[torch]`）：Torch TiDE-style（encoder/decoder MLP, `lags`, `d_model`, `epochs`…）
- `torch-deepar-recursive`（requires `.[torch]`）：Torch DeepAR-style（Gaussian RNN, recursive forecast, `lags`, `hidden_size`, `epochs`…）
- `torch-qrnn-recursive`（requires `.[torch]`）：Torch Quantile RNN（pinball loss, recursive forecast, `lags`, `q`, `epochs`…）
- `torch-xformer-*-direct`（requires `.[torch]`）：可配置 Transformer-family（full/local/performer/linformer/nystrom/probsparse/autocorr/reformer + RoPE/sincos/Time2Vec + RMSNorm/SwiGLU/RevIN…）
- `torch-seq2seq-*-direct`（requires `.[torch]`）：Seq2Seq RNN（LSTM/GRU，可选 Bahdanau attention，scheduled teacher forcing）
- `torch-lstnet-direct`（requires `.[torch]`）：LSTNet-style（CNN + GRU + skip + highway，lite）
- `torch-tft-global`（requires `.[torch]`）：Torch TFT（lite）全局/面板训练（`context_length`, `x_cols`, `add_time_features`, `d_model`, `epochs`…）
- `torch-informer-global`（requires `.[torch]`）：Torch Informer（lite）全局/面板训练（`context_length`, `x_cols`, `add_time_features`, `d_model`, `num_layers`, `epochs`…）
- `torch-nonstationary-transformer-global`（requires `.[torch]`）：Torch Non-stationary Transformer（lite，RevIN + de-stationary attention factors）全局/面板训练（`context_length`, `d_model`, `nhead`, `num_layers`, `epochs`…）
- `torch-autoformer-global`（requires `.[torch]`）：Torch Autoformer（lite）多尺度分解 + 全局训练（`context_length`, `x_cols`, `ma_window`, `epochs`…）
- `torch-fedformer-global`（requires `.[torch]`）：Torch FEDformer-style（lite，分解 + 频域混合/FFT）全局/面板训练（`context_length`, `d_model`, `num_layers`, `modes`, `ma_window`, `epochs`…）
- `torch-patchtst-global`（requires `.[torch]`）：Torch PatchTST-style（lite）全局/面板训练（patch tokens + encoder，`context_length`, `patch_len`, `stride`, `epochs`…）
- `torch-tsmixer-global`（requires `.[torch]`）：Torch TSMixer-style（lite）全局/面板训练（token/channel mixing，`context_length`, `num_blocks`, `epochs`…）
- `torch-itransformer-global`（requires `.[torch]`）：Torch iTransformer-style（lite）全局/面板训练（inverted tokens: variables-as-tokens，`context_length`, `d_model`, `epochs`…）
- `torch-timesnet-global`（requires `.[torch]`）：Torch TimesNet-style（lite）全局/面板训练（FFT period detection + period Conv2D blocks，`context_length`, `top_k`, `epochs`…）
- `torch-tcn-global`（requires `.[torch]`）：Torch TCN（causal dilated Conv1D residual blocks）全局/面板训练（`context_length`, `channels`, `epochs`…）
- `torch-nbeats-global`（requires `.[torch]`）：Torch N-BEATS-style（generic residual MLP blocks）全局/面板训练（`context_length`, `num_blocks`, `layer_width`, `epochs`…）
- `torch-nhits-global`（requires `.[torch]`）：Torch N-HiTS-style（multi-rate residual MLP, lite）全局/面板训练（`context_length`, `pool_sizes`, `num_blocks`, `epochs`…）
- `torch-nlinear-global`（requires `.[torch]`）：Torch NLinear-style（last-value centering + linear head, lite）全局/面板训练（`context_length`, `epochs`…）
- `torch-dlinear-global`（requires `.[torch]`）：Torch DLinear-style（moving-average decomposition + linear heads, lite）全局/面板训练（`context_length`, `ma_window`, `epochs`…）
- `torch-deepar-global`（requires `.[torch]`）：Torch DeepAR-style（Gaussian RNN NLL, lite）全局/面板训练（`context_length`, `hidden_size`, `epochs`…）
- `torch-tide-global`（requires `.[torch]`）：Torch TiDE-style（encoder/decoder MLP, lite）全局/面板训练（`context_length`, `d_model`, `epochs`…）
- `torch-wavenet-global`（requires `.[torch]`）：Torch WaveNet-style（gated dilated CNN, lite）全局/面板训练（`context_length`, `channels`, `num_layers`, `epochs`…）
- `torch-resnet1d-global`（requires `.[torch]`）：Torch ResNet-1D（Conv1D residual blocks, lite）全局/面板训练（`context_length`, `channels`, `num_blocks`, `epochs`…）
- `torch-inception-global`（requires `.[torch]`）：Torch InceptionTime-style（multi-kernel Conv1D, lite）全局/面板训练（`context_length`, `channels`, `kernel_sizes`, `epochs`…）
- `torch-lstnet-global`（requires `.[torch]`）：LSTNet-style（CNN + GRU + skip + highway, lite）全局/面板训练（`context_length`, `cnn_channels`, `rnn_hidden`, `epochs`…）
- `torch-fnet-global`（requires `.[torch]`）：Torch FNet-style（FFT token mixing, lite）全局/面板训练（`context_length`, `d_model`, `num_layers`, `epochs`…）
- `torch-gmlp-global`（requires `.[torch]`）：Torch gMLP-style（spatial gating, lite）全局/面板训练（`context_length`, `d_model`, `num_layers`, `epochs`…）
- `torch-ssm-global`（requires `.[torch]`）：Torch diagonal state-space（SSM, lite）全局/面板训练（`context_length`, `d_model`, `num_layers`, `epochs`…）
- `torch-mamba-global`（requires `.[torch]`）：Torch Mamba-style selective SSM（lite）全局/面板训练（`context_length`, `d_model`, `num_layers`, `conv_kernel`, `epochs`…）
- `torch-rwkv-global`（requires `.[torch]`）：Torch RWKV-style time-mix + channel-mix（lite）全局/面板训练（`context_length`, `d_model`, `num_layers`, `ffn_dim`, `epochs`…）
- `torch-transformer-encdec-global`（requires `.[torch]`）：Torch encoder-decoder Transformer（lite）全局/面板训练（`context_length`, `d_model`, `nhead`, `epochs`…）
- `torch-seq2seq-*-global`（requires `.[torch]`）：Torch Seq2Seq（encoder-decoder RNN，optional Bahdanau attention + teacher forcing，panel/global 训练）
- `torch-xformer-*-global`（requires `.[torch]`）：全局/面板 Transformer-family（支持 covariates + time features；attn 支持 probsparse/autocorr/reformer 等）
- `torch-rnn-*-global`（requires `.[torch]`）：全局/面板 RNN（LSTM/GRU，token-wise horizon head）
- `arima`（requires `.[stats]`）：ARIMA(p,d,q) via statsmodels（`order=1,0,0`）
- `auto-arima`（requires `.[stats]`）：AutoARIMA-style 网格搜索（`max_p`, `max_d`, `max_q`, `information_criterion`）
- `sarimax`（requires `.[stats]`）：SARIMAX / seasonal ARIMA（`order=...`, `seasonal_order=...`）
- `autoreg`（requires `.[stats]`）：AutoReg（`lags`, `trend`, `seasonal`, `period`）
- `uc-local-level`（requires `.[stats]`）：UnobservedComponents local level
- `uc-local-linear-trend`（requires `.[stats]`）：UnobservedComponents local linear trend
- `stl-arima`（requires `.[stats]`）：STL + ARIMA remainder forecasting（`period`, `order`, `seasonal`…）
- `mstl-arima`（requires `.[stats]`）：MSTL（多季节 STL）+ ARIMA remainder forecasting（`periods=7,365`, `order=...`）
- `mstl-auto-arima`（requires `.[stats]`）：MSTL + AutoARIMA-style 网格搜索（`periods=...`, `max_p/max_d/max_q`…）
- `tbats-lite`（requires `.[stats]`）：TBATS-like（multi-season Fourier + ARIMA residuals，可选 Box-Cox）（`periods`, `orders`, `arima_order`, `boxcox_lambda`）
- `ets`（requires `.[stats]`）：ETS / ExponentialSmoothing via statsmodels（`trend`, `seasonal`, `season_length`, `damped_trend`）

查看完整列表/参数说明：

```bash
foresight models list
foresight models info holt-winters-add
```

### Probabilistic Forecasting (Quantiles)

部分 **global/panel Torch 模型**支持 `quantiles` 参数（分位数回归 / pinball loss），例如：

```bash
foresight eval run --model torch-itransformer-global --dataset catfish --y-col Total --horizon 7 --step 7 --min-train-size 60 \
  --model-param quantiles=0.1,0.5,0.9
```

当 `quantiles` 非空时：
- 预测表会多输出 `yhat_p10 / yhat_p50 / yhat_p90 ...` 等列（与 quantiles 对应）
- `yhat` 默认取 `p50`（或最接近 0.5 的分位数）
- `eval` 的输出会附加 `q_` 前缀的指标（pinball / coverage / interval_score 等）

---

## 📐 Data Format (Panel-friendly)

`foresight` 的通用评测会把原始数据转换为一个 canonical long format：

- `unique_id`：序列 ID（单序列时为空）
- `ds`：时间列（datetime）
- `y`：目标值（float）

这个格式也支持 **额外的协变量列（covariates / exogenous features）**：只要保留在 long_df 里即可。  
当你需要把原始 DataFrame 转成 long_df 并保留协变量时，可使用 `x_cols=...`：

```python
from foresight.data import to_long

long_df = to_long(
    raw_df,
    time_col="ds",
    y_col="y",
    id_cols=("store", "dept"),
    x_cols=("promo", "price"),
)
```

你可以直接使用 `foresight.data.to_long()`：

```python
from foresight.data import to_long
```

也可以用 `foresight.data.validate_long_df()` 对 `unique_id/ds/y` 做基础校验（时间排序/重复时间戳等）：

```python
from foresight.data import validate_long_df
```

这个格式和很多主流库的约定兼容（例如 StatsForecast 使用 `unique_id/ds/y`，Prophet 使用 `ds/y`）。

---

## 📊 Evaluation & Backtesting

- **Walk-forward backtesting**（`foresight.backtesting.walk_forward`）
- **Rolling/expanding windows**：支持 `max_train_size`（rolling window）或默认 expanding window
- **Cross-validation predictions table**：`foresight.cv.cross_validation_predictions` / `foresight cv run`
- 支持 **单序列** 与 **面板数据（按 `unique_id` 分组）**
- 输出包含 **整体指标** + **按预测步长（horizon step）的指标列表**

当前 CLI 默认输出的点预测指标：
- MAE / RMSE / MAPE / sMAPE

同时支持一个“快速 baseline”的 **conformal 对称区间**（从回测残差估计半径）：

```bash
foresight eval run --model theta --dataset catfish --y-col Total --horizon 3 --step 3 --min-train-size 12 --conformal-levels 80,90
```

更多可用指标（供自定义评测使用）见 `foresight.metrics`：
- MSE, WAPE, MASE, RMSSE
- Pinball loss / Interval coverage / Interval width / Interval score / MSIS

---

## 📦 Optional Dependencies

默认安装只需要 `numpy` / `pandas`。

```bash
# ML extras (ridge-lag)
pip install -e ".[ml]"

# torch extras (torch-mlp-direct / torch-transformer-direct / torch-patchtst-direct / ...)
pip install -e ".[torch]"

# stats extras (arima / ets)
pip install -e ".[stats]"

# everything above
pip install -e ".[all]"
```

---

## 🤝 Contributing

```bash
ruff check src tests tools
ruff format src tests tools
pytest -q
```

开发说明见 `docs/DEVELOPMENT.md`。

---

## 🔎 Related Projects / Inspiration

ForeSight 的 `foresight` toolkit 参考了这些主流项目的接口/数据形态与评测习惯（强烈推荐）：

- **StatsForecast (Nixtla)**: fast statistical baselines + `unique_id/ds/y` + cross-validation  
  https://github.com/Nixtla/statsforecast
- **Prophet**: `ds/y` dataframe convention for forecasting  
  https://facebook.github.io/prophet/
- **sktime**: unified forecasting interface (`fit`/`predict`) and evaluation utilities  
  https://www.sktime.org/
- **Darts**: unified model API + backtesting helpers  
  https://github.com/unit8co/darts
- **GluonTS**: probabilistic forecasting datasets + benchmarking  
  https://github.com/awslabs/gluonts
- **NeuralForecast (Nixtla)**: modern neural forecasting models (TFT / Informer / Autoformer / …)  
  https://github.com/Nixtla/neuralforecast
- **PyTorch Forecasting**: deep learning forecasting pipelines  
  https://pytorch-forecasting.readthedocs.io/
- **Kats**: (Meta/Facebook) time series analysis & forecasting toolbox  
  https://github.com/facebookresearch/Kats

---

## ⚖️ License

GPL-3.0-only. See `LICENSE`.
