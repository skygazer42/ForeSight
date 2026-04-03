# 数据工程管道

在建模之前，高质量的数据准备至关重要。ForeSight 提供了一系列长格式 DataFrame 工具函数，可以灵活组合为端到端的数据工程管道。

---

## 核心工具一览

| 函数 | 用途 |
|------|------|
| `to_long` | 将宽格式转为长格式 |
| `prepare_long_df` | 校验并标准化长格式 DataFrame |
| `align_long_df` | 按指定频率对齐时间索引，填补缺失时间点 |
| `infer_series_frequency` | 自动推断序列频率 |
| `clip_long_df_outliers` | 基于统计方法裁剪离群值 |
| `enrich_long_df_calendar` | 添加日历特征与周期性时间编码 |
| `fit_long_df_scaler` / `transform_long_df_with_scaler` / `inverse_transform_long_df_with_scaler` | 按序列拟合、应用和反转缩放 |
| `split_long_df` | 按比例拆分训练集、验证集与测试集 |

---

## 逐步详解

### 转为长格式

```python
from foresight import to_long

long_df = to_long(raw_df, time_col="ds", y_col="y", id_cols=("store",))
```

### 校验与准备

```python
from foresight import prepare_long_df

prepared = prepare_long_df(long_df, freq="D")
```

### 频率对齐

```python
from foresight import align_long_df

aligned = align_long_df(prepared, freq="D")
```

对于未知频率的数据集，可先用 `infer_series_frequency` 自动推断：

```python
from foresight import infer_series_frequency

freq = infer_series_frequency(prepared)
```

### 离群值裁剪

```python
from foresight import clip_long_df_outliers

clipped = clip_long_df_outliers(aligned)
```

### 日历特征

```python
from foresight import enrich_long_df_calendar

enriched = enrich_long_df_calendar(clipped)
```

添加的特征包括星期、月份、是否节假日等，以及正弦/余弦周期编码。

### 按序列缩放

ForeSight 支持对每条序列独立拟合缩放器，避免跨序列信息泄漏：

```python
from foresight import (
    fit_long_df_scaler,
    transform_long_df_with_scaler,
    inverse_transform_long_df_with_scaler,
)

scaler = fit_long_df_scaler(enriched)
scaled = transform_long_df_with_scaler(enriched, scaler)

# 预测完成后反转缩放
original_scale = inverse_transform_long_df_with_scaler(scaled, scaler)
```

### 数据拆分

```python
from foresight import split_long_df

train, val, test = split_long_df(scaled, val_size=0.1, test_size=0.1)
```

---

## 完整管道示例

```python
from foresight import (
    fit_long_df_scaler,
    transform_long_df_with_scaler,
    inverse_transform_long_df_with_scaler,
    enrich_long_df_calendar,
    clip_long_df_outliers,
    split_long_df,
    align_long_df,
    prepare_long_df,
    to_long,
)

# 1. Convert to long format
long_df = to_long(raw_df, time_col="ds", y_col="y", id_cols=("store",))

# 2. Validate and prepare
prepared = prepare_long_df(long_df, freq="D")

# 3. Align frequencies
aligned = align_long_df(prepared, freq="D")

# 4. Clip outliers
clipped = clip_long_df_outliers(aligned)

# 5. Add calendar features
enriched = enrich_long_df_calendar(clipped)

# 6. Fit scaler
scaler = fit_long_df_scaler(enriched)
scaled = transform_long_df_with_scaler(enriched, scaler)

# 7. Split
train, val, test = split_long_df(scaled, val_size=0.1, test_size=0.1)
```
