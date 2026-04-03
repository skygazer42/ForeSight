# 多变量预测

多变量预测（Multivariate Forecasting）适用于多个 **相互关联** 的目标列需要同步预测的场景，例如同时预测某门店的销售额与客流量。

!!! note "与面板预测的区别"
    **面板 / 全局预测（Panel / Global）**：多条独立序列共享同一模型，每条序列有各自的 ID。

    **多变量预测（Multivariate）**：单组时间索引下的多列目标彼此存在相关性，需要联合建模。

---

## 创建多变量预测器

使用 `make_multivariate_forecaster` 创建预测器。目前支持基于 statsmodels 的 VAR 模型：

```python
from foresight import make_multivariate_forecaster

mv = make_multivariate_forecaster("var", maxlags=1)
```

| 参数 | 说明 |
|------|------|
| `"var"` | 模型类型，使用 statsmodels 的 VAR 实现 |
| `maxlags` | VAR 模型的最大滞后阶数 |

!!! tip
    VAR 模型需要安装 statsmodels extra：`pip install foresight[statsmodels]`。

---

## 执行预测

将包含多列目标的宽格式 DataFrame 传入预测器：

```python
yhat = mv(wide_df[["sales", "traffic"]], horizon=2)
```

返回的 `yhat` 包含与输入相同的列，行数等于 `horizon`。

---

## 评估

使用 `eval_multivariate_model_df` 对多变量预测结果进行评估：

```python
from foresight import eval_multivariate_model_df

metrics = eval_multivariate_model_df(
    forecast_df=yhat,
    actuals_df=actuals,
)
```

---

## 完整示例

```python
from foresight import make_multivariate_forecaster, eval_multivariate_model_df

# 1. 创建 VAR 预测器
mv = make_multivariate_forecaster("var", maxlags=1)

# 2. 联合预测 sales 与 traffic
yhat = mv(wide_df[["sales", "traffic"]], horizon=2)

# 3. 评估
metrics = eval_multivariate_model_df(forecast_df=yhat, actuals_df=actuals)
```
