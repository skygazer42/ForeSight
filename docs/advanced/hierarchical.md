# 层级预测

层级预测（Hierarchical Forecasting）用于处理具有自然层级结构的时间序列，例如 **区域 > 门店** 或 **品类 > 产品**。ForeSight 提供了一套完整的工具链，从构建层级规格到协调预测结果，再到一致性校验与评估。

---

## 构建层级规格

使用 `build_hierarchy_spec` 从原始数据中自动推导层级结构：

```python
from foresight import build_hierarchy_spec

hierarchy = build_hierarchy_spec(
    raw_df,
    id_cols=("region", "store"),
    root="total",
)
```

| 参数 | 说明 |
|------|------|
| `raw_df` | 包含所有序列的长格式 DataFrame |
| `id_cols` | 从粗到细的层级列名元组 |
| `root` | 聚合根节点的名称，默认 `"total"` |

返回的 `hierarchy` 对象描述了节点间的父子关系以及聚合矩阵，可直接传入后续函数。

---

## 协调预测

`reconcile_hierarchical_forecasts` 将各层级的独立预测值协调为一致的结果：

```python
from foresight import reconcile_hierarchical_forecasts

reconciled = reconcile_hierarchical_forecasts(
    forecast_df=pred_df,
    hierarchy=hierarchy,
    method="top_down",       # 或 "bottom_up"
    history_df=history_long,
    exog_agg={"promo": "sum", "temp": "mean"},
)
```

| 参数 | 说明 |
|------|------|
| `forecast_df` | 各节点的原始预测 DataFrame |
| `hierarchy` | 由 `build_hierarchy_spec` 生成的层级规格 |
| `method` | 协调方法：`"top_down"` 自顶向下分解，`"bottom_up"` 自底向上汇总 |
| `history_df` | 历史数据，用于计算分解比例 |
| `exog_agg` | 外生变量在聚合时使用的函数映射 |

---

## 一致性校验

在协调前后均可调用 `check_hierarchical_consistency` 验证预测值是否满足层级加和约束：

```python
from foresight import check_hierarchical_consistency

check_hierarchical_consistency(forecast_df=reconciled, hierarchy=hierarchy)
```

若存在不一致，函数将抛出详细的诊断信息。

---

## 层级评估

`eval_hierarchical_forecast_df` 在每个层级节点上计算误差指标，方便定位薄弱环节：

```python
from foresight import eval_hierarchical_forecast_df

payload = eval_hierarchical_forecast_df(
    forecast_df=reconciled,
    hierarchy=hierarchy,
    y_col="y",
)
```

---

## 完整示例

```python
from foresight import (
    build_hierarchy_spec,
    reconcile_hierarchical_forecasts,
    eval_hierarchical_forecast_df,
)

# 1. 构建层级
hierarchy = build_hierarchy_spec(
    raw_df,
    id_cols=("region", "store"),
    root="total",
)

# 2. 协调预测
reconciled = reconcile_hierarchical_forecasts(
    forecast_df=pred_df,
    hierarchy=hierarchy,
    method="top_down",
    history_df=history_long,
)

# 3. 评估
payload = eval_hierarchical_forecast_df(
    forecast_df=reconciled,
    hierarchy=hierarchy,
    y_col="y",
)
```
