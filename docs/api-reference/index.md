# API 参考

ForeSight Python API 完整参考。所有公开函数均可从顶层包 `foresight` 直接导入。

```python
from foresight import make_forecaster, eval_model, forecast_model
```

---

## 核心预测

构建预测器并生成预测的核心接口。

### `make_forecaster`

创建无状态的预测函数。

```python
def make_forecaster(model: str, **params) -> Callable[[list, int], ndarray]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称（如 `"theta"`, `"holt"`, `"naive-last"`） |
| `**params` | — | | — | 模型特定参数（如 `alpha=0.3`） |

**返回值：** `Callable[[list, int], ndarray]` — 接收训练序列和 horizon，返回预测数组。

```python
from foresight import make_forecaster

f = make_forecaster("holt", alpha=0.3, beta=0.1)
yhat = f([112, 118, 132, 129, 121, 135], horizon=3)
# yhat: ndarray of shape (3,)
```

---

### `make_forecaster_object`

创建有状态的预测器对象，支持 `fit` / `predict` / `save` / `load`。

```python
def make_forecaster_object(model: str, **params) -> BaseForecaster
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `**params` | — | | — | 模型特定参数 |

**返回值：** `BaseForecaster` — 具有 `fit(y)` 和 `predict(horizon)` 方法的预测器对象。

```python
from foresight import make_forecaster_object

obj = make_forecaster_object("moving-average", window=3)
obj.fit([1, 2, 3, 4, 5, 6])
yhat = obj.predict(3)
```

---

### `make_global_forecaster`

创建全局模型的无状态预测函数（跨多序列训练）。

```python
def make_global_forecaster(model: str, **params) -> Callable
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 全局模型名称 |
| `**params` | — | | — | 模型特定参数 |

**返回值：** `Callable` — 全局预测函数。

---

### `make_global_forecaster_object`

创建全局模型的有状态预测器对象。

```python
def make_global_forecaster_object(model: str, **params) -> BaseGlobalForecaster
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 全局模型名称 |
| `**params` | — | | — | 模型特定参数 |

**返回值：** `BaseGlobalForecaster` — 全局预测器对象。

---

### `make_multivariate_forecaster`

创建多变量预测函数。

```python
def make_multivariate_forecaster(model: str, **params) -> Callable
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 多变量模型名称 |
| `**params` | — | | — | 模型特定参数 |

**返回值：** `Callable` — 多变量预测函数。

---

### `forecast_model`

对单序列生成预测，返回 DataFrame 格式的结果。

```python
def forecast_model(
    model: str,
    y: list | ndarray,
    horizon: int,
    ds: list | ndarray | None = None,
    unique_id: str = "series=0",
    model_params: dict | None = None,
    interval_levels: list[float] | None = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `y` | `list \| ndarray` | :white_check_mark: | — | 历史序列值 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `ds` | `list \| ndarray` | | `None` | 时间索引 |
| `unique_id` | `str` | | `"series=0"` | 序列标识 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `interval_levels` | `list[float]` | | `None` | 预测区间的置信水平（如 `[0.8, 0.95]`） |
| `interval_min_train_size` | `int` | | `None` | Bootstrap 区间的最小训练大小 |
| `interval_samples` | `int` | | `1000` | Bootstrap 采样次数 |
| `interval_seed` | `int` | | `None` | 随机种子 |

**返回值：** `DataFrame` — 包含 `unique_id`、`ds`、`yhat` 列，若指定了 `interval_levels` 则额外包含区间列。

```python
from foresight import forecast_model

df = forecast_model(
    model="theta",
    y=[112, 118, 132, 129, 121, 135, 148, 148],
    horizon=3,
    interval_levels=[0.8, 0.95],
)
```

---

### `forecast_model_long_df`

对长格式 DataFrame 中的多序列生成预测。

```python
def forecast_model_long_df(
    model: str,
    long_df: DataFrame,
    future_df: DataFrame | None = None,
    horizon: int,
    model_params: dict | None = None,
    interval_levels: list[float] | None = None,
    interval_min_train_size: int | None = None,
    interval_samples: int = 1000,
    interval_seed: int | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `long_df` | `DataFrame` | :white_check_mark: | — | 长格式历史数据（含 `unique_id` / `ds` / `y`） |
| `future_df` | `DataFrame` | | `None` | 未来协变量 DataFrame |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `interval_levels` | `list[float]` | | `None` | 预测区间置信水平 |
| `interval_min_train_size` | `int` | | `None` | Bootstrap 最小训练大小 |
| `interval_samples` | `int` | | `1000` | Bootstrap 采样次数 |
| `interval_seed` | `int` | | `None` | 随机种子 |

**返回值：** `DataFrame` — 所有序列的预测结果。

---

## 评估

滚动窗口评估与交叉验证 API。

### `eval_model`

在内置数据集上执行滚动窗口评估。

```python
def eval_model(
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    y_col: str | None = None,
    model_params: dict | None = None,
    data_dir: str | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: list[float] | None = None,
    conformal_per_step: bool = True,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `y_col` | `str` | | `None` | 目标列名 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `data_dir` | `str` | | `None` | 自定义数据目录 |
| `max_windows` | `int` | | `None` | 最大评估窗口数 |
| `max_train_size` | `int` | | `None` | 最大训练集大小（rolling window） |
| `conformal_levels` | `list[float]` | | `None` | Conformal 预测区间置信水平 |
| `conformal_per_step` | `bool` | | `True` | 是否按步计算 conformal 区间 |

**返回值：** `dict` — 包含 `mae`、`rmse`、`mape`、`smape` 等指标。

```python
from foresight import eval_model

metrics = eval_model(
    model="theta",
    dataset="catfish",
    horizon=3, step=3, min_train_size=12,
    y_col="Total",
)
print(metrics)  # {'mae': ..., 'rmse': ..., 'mape': ..., 'smape': ...}
```

---

### `eval_model_long_df`

在自定义长格式 DataFrame 上执行滚动窗口评估。

```python
def eval_model_long_df(
    model: str,
    long_df: DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    model_params: dict | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
    conformal_levels: list[float] | None = None,
    conformal_per_step: bool = True,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `long_df` | `DataFrame` | :white_check_mark: | — | 长格式数据 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `max_windows` | `int` | | `None` | 最大评估窗口数 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |
| `conformal_levels` | `list[float]` | | `None` | Conformal 区间置信水平 |
| `conformal_per_step` | `bool` | | `True` | 按步计算 conformal 区间 |

**返回值：** `dict` — 评估指标字典。

---

### `eval_multivariate_model_df`

对多变量模型执行滚动窗口评估。

```python
def eval_multivariate_model_df(
    model: str,
    df: DataFrame,
    target_cols: list[str],
    horizon: int,
    step: int,
    min_train_size: int,
    ds_col: str = "ds",
    model_params: dict | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 多变量模型名称 |
| `df` | `DataFrame` | :white_check_mark: | — | 包含时间列和多目标列的 DataFrame |
| `target_cols` | `list[str]` | :white_check_mark: | — | 目标列名列表 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `ds_col` | `str` | | `"ds"` | 时间列名 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `max_windows` | `int` | | `None` | 最大评估窗口数 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |

**返回值：** `dict` — 每个目标列的评估指标。

---

### `eval_hierarchical_forecast_df`

对层级预测结果执行调和并评估。

```python
def eval_hierarchical_forecast_df(
    forecast_df: DataFrame,
    hierarchy: dict,
    method: str,
    history_df: DataFrame | None = None,
    yhat_col: str = "yhat",
    exog_agg: dict | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `forecast_df` | `DataFrame` | :white_check_mark: | — | 预测结果 DataFrame |
| `hierarchy` | `dict` | :white_check_mark: | — | 层级关系字典 |
| `method` | `str` | :white_check_mark: | — | 调和方法（`bottom_up` / `top_down`） |
| `history_df` | `DataFrame` | | `None` | 历史数据 |
| `yhat_col` | `str` | | `"yhat"` | 预测值列名 |
| `exog_agg` | `dict` | | `None` | 外生变量聚合方式 |

**返回值：** `dict` — 调和后的评估指标。

---

## 数据处理

数据格式转换、验证和预处理工具。

### `to_long`

将宽格式 DataFrame 转换为 ForeSight 标准的长格式。

```python
def to_long(
    df: DataFrame,
    time_col: str,
    y_col: str,
    id_cols: list[str] | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `df` | `DataFrame` | :white_check_mark: | — | 输入 DataFrame |
| `time_col` | `str` | :white_check_mark: | — | 时间列名 |
| `y_col` | `str` | :white_check_mark: | — | 目标值列名 |
| `id_cols` | `list[str]` | | `None` | ID 列名列表（多序列时使用） |

**返回值：** `DataFrame` — 标准长格式（含 `unique_id` / `ds` / `y`）。

```python
from foresight import to_long

long_df = to_long(df, time_col="date", y_col="sales", id_cols=["store", "product"])
```

---

### `validate_long_df`

验证 DataFrame 是否符合 ForeSight 长格式标准。

```python
def validate_long_df(df: DataFrame) -> None
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `df` | `DataFrame` | :white_check_mark: | — | 待验证的 DataFrame |

**返回值：** `None` — 验证通过时无返回；不通过时抛出异常。

---

### `prepare_long_df`

预处理长格式 DataFrame（排序、类型转换等）。

```python
def prepare_long_df(df: DataFrame, ...) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `df` | `DataFrame` | :white_check_mark: | — | 长格式 DataFrame |

**返回值：** `DataFrame` — 预处理后的 DataFrame。

---

### `split_long_df`

按时间或比例拆分长格式 DataFrame。

```python
def split_long_df(df: DataFrame, ...) -> tuple[DataFrame, DataFrame]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `df` | `DataFrame` | :white_check_mark: | — | 长格式 DataFrame |

**返回值：** `tuple[DataFrame, DataFrame]` — 训练集和测试集。

---

### `long_to_wide`

将长格式 DataFrame 转换回宽格式。

```python
def long_to_wide(df: DataFrame, ...) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `df` | `DataFrame` | :white_check_mark: | — | 长格式 DataFrame |

**返回值：** `DataFrame` — 宽格式 DataFrame。

---

### `resolve_covariate_roles`

解析协变量角色配置。

```python
def resolve_covariate_roles(...) -> dict
```

**返回值：** `dict` — 协变量角色映射。

---

### `build_hierarchy_spec`

构建层级规格说明。

```python
def build_hierarchy_spec(...) -> dict
```

**返回值：** `dict` — 层级规格字典。

---

## 异常检测

基于预测残差或滚动统计的时间序列异常检测。

### `detect_anomalies`

在内置数据集上执行异常检测。

```python
def detect_anomalies(
    dataset: str,
    y_col: str | None = None,
    model: str | None = None,
    model_params: dict | None = None,
    score_method: str | None = None,
    threshold_method: str | None = None,
    threshold_k: float = 3.0,
    threshold_quantile: float = 0.99,
    window: int = 12,
    min_history: int = 3,
    min_train_size: int | None = None,
    step_size: int = 1,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `y_col` | `str` | | `None` | 目标列名 |
| `model` | `str` | | `None` | 用于残差计算的预测模型 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `score_method` | `str` | | `None` | 异常评分方法 |
| `threshold_method` | `str` | | `None` | 阈值确定方法 |
| `threshold_k` | `float` | | `3.0` | k-sigma 阈值系数 |
| `threshold_quantile` | `float` | | `0.99` | 分位数阈值 |
| `window` | `int` | | `12` | 滑动窗口大小 |
| `min_history` | `int` | | `3` | 最小历史数据量 |
| `min_train_size` | `int` | | `None` | 最小训练集大小 |
| `step_size` | `int` | | `1` | 步长 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |
| `n_windows` | `int` | | `None` | 窗口数 |

**返回值：** `DataFrame` — 包含异常评分和标记的 DataFrame。

```python
from foresight import detect_anomalies

anomalies = detect_anomalies(
    dataset="catfish",
    y_col="Total",
    model="theta",
    threshold_k=2.5,
    window=12,
)
```

---

### `detect_anomalies_long_df`

在自定义长格式 DataFrame 上执行异常检测。

```python
def detect_anomalies_long_df(
    long_df: DataFrame,
    model: str | None = None,
    model_params: dict | None = None,
    score_method: str | None = None,
    threshold_method: str | None = None,
    threshold_k: float = 3.0,
    threshold_quantile: float = 0.99,
    window: int = 12,
    min_history: int = 3,
    min_train_size: int | None = None,
    step_size: int = 1,
    max_train_size: int | None = None,
    n_windows: int | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `long_df` | `DataFrame` | :white_check_mark: | — | 长格式 DataFrame |
| `model` | `str` | | `None` | 预测模型 |
| `model_params` | `dict` | | `None` | 模型参数 |
| `score_method` | `str` | | `None` | 异常评分方法 |
| `threshold_method` | `str` | | `None` | 阈值确定方法 |
| `threshold_k` | `float` | | `3.0` | k-sigma 阈值系数 |
| `threshold_quantile` | `float` | | `0.99` | 分位数阈值 |
| `window` | `int` | | `12` | 滑动窗口大小 |
| `min_history` | `int` | | `3` | 最小历史数据量 |
| `min_train_size` | `int` | | `None` | 最小训练集大小 |
| `step_size` | `int` | | `1` | 步长 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |
| `n_windows` | `int` | | `None` | 窗口数 |

**返回值：** `DataFrame` — 包含异常评分和标记的 DataFrame。

---

## 序列化

模型工件的保存与加载。

### `save_forecaster`

将已训练的预测器保存为工件文件。

```python
def save_forecaster(
    forecaster: BaseForecaster | BaseGlobalForecaster,
    path: str,
    extra: dict | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `forecaster` | `BaseForecaster \| BaseGlobalForecaster` | :white_check_mark: | — | 已训练的预测器对象 |
| `path` | `str` | :white_check_mark: | — | 保存路径 |
| `extra` | `dict` | | `None` | 附加元数据 |

**返回值：** `dict` — 工件摘要信息。

```python
from foresight import make_forecaster_object, save_forecaster

obj = make_forecaster_object("theta")
obj.fit([112, 118, 132, 129, 121, 135, 148, 148])

info = save_forecaster(obj, "./model.artifact", extra={"version": "1.0"})
```

---

### `load_forecaster`

从工件文件加载预测器。

```python
def load_forecaster(path: str) -> BaseForecaster | BaseGlobalForecaster
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `path` | `str` | :white_check_mark: | — | 工件文件路径 |

**返回值：** `BaseForecaster | BaseGlobalForecaster` — 恢复的预测器对象。

```python
from foresight import load_forecaster

obj = load_forecaster("./model.artifact")
yhat = obj.predict(3)
```

---

### `load_forecaster_artifact`

加载完整工件载荷（会通过 Python pickle 反序列化模型对象）。Only load artifacts from trusted sources.

```python
def load_forecaster_artifact(path: str) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `path` | `str` | :white_check_mark: | — | 工件文件路径 |

**返回值：** `dict` — 完整工件载荷字典，包含元数据、附加信息和反序列化后的预测器对象。

---

## 层级预测

层级时间序列的调和与一致性检查。

### `reconcile_hierarchical_forecasts`

对预测结果执行层级调和。

```python
def reconcile_hierarchical_forecasts(
    forecast_df: DataFrame,
    hierarchy: dict,
    method: str,
    history_df: DataFrame | None = None,
    yhat_col: str = "yhat",
    exog_agg: dict | None = None,
) -> DataFrame
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `forecast_df` | `DataFrame` | :white_check_mark: | — | 预测结果 DataFrame |
| `hierarchy` | `dict` | :white_check_mark: | — | 层级关系字典（parent → children list） |
| `method` | `str` | :white_check_mark: | — | 调和方法（`bottom_up` / `top_down`） |
| `history_df` | `DataFrame` | | `None` | 历史数据（`top_down` 必需） |
| `yhat_col` | `str` | | `"yhat"` | 预测值列名 |
| `exog_agg` | `dict` | | `None` | 外生变量聚合方式 |

**返回值：** `DataFrame` — 调和后的预测 DataFrame。

!!! warning "top_down 方法"

    使用 `top_down` 方法时，必须提供 `history_df` 参数以计算各子节点的历史占比。

```python
from foresight import reconcile_hierarchical_forecasts

hierarchy = {"Total": ["East", "West"], "East": ["NY", "BOS"], "West": ["LA", "SF"]}

reconciled = reconcile_hierarchical_forecasts(
    forecast_df=forecast_df,
    hierarchy=hierarchy,
    method="bottom_up",
)
```

---

### `check_hierarchical_consistency`

检查预测结果是否满足层级一致性约束。

```python
def check_hierarchical_consistency(
    forecast_df: DataFrame,
    hierarchy: dict,
    yhat_col: str = "yhat",
    atol: float = 1e-8,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `forecast_df` | `DataFrame` | :white_check_mark: | — | 预测结果 DataFrame |
| `hierarchy` | `dict` | :white_check_mark: | — | 层级关系字典 |
| `yhat_col` | `str` | | `"yhat"` | 预测值列名 |
| `atol` | `float` | | `1e-8` | 绝对容差 |

**返回值：**

```python
{
    "is_consistent": bool,       # 是否一致
    "n_inconsistencies": int,    # 不一致数量
    "inconsistencies": list,     # 不一致详情
}
```

---

## 调优

超参数网格搜索。

### `tune_model`

在内置数据集上执行网格搜索调优。

```python
def tune_model(
    model: str,
    dataset: str,
    horizon: int,
    step: int,
    min_train_size: int,
    search_space: dict,
    metric: str = "mae",
    mode: str = "min",
    y_col: str | None = None,
    model_params: dict | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `search_space` | `dict` | :white_check_mark: | — | 参数搜索空间（param → list of values） |
| `metric` | `str` | | `"mae"` | 评估指标 |
| `mode` | `str` | | `"min"` | 优化方向 |
| `y_col` | `str` | | `None` | 目标列名 |
| `model_params` | `dict` | | `None` | 固定模型参数 |
| `max_windows` | `int` | | `None` | 最大评估窗口数 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |

**返回值：**

```python
{
    "best_score": float,         # 最优指标值
    "best_params": dict,         # 最优参数组合
    "trials": list[dict],        # 所有试验记录
    "n_trials": int,             # 总试验次数
}
```

---

### `tune_model_long_df`

在自定义长格式 DataFrame 上执行网格搜索调优。

```python
def tune_model_long_df(
    model: str,
    long_df: DataFrame,
    horizon: int,
    step: int,
    min_train_size: int,
    search_space: dict,
    metric: str = "mae",
    mode: str = "min",
    model_params: dict | None = None,
    max_windows: int | None = None,
    max_train_size: int | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |
| `long_df` | `DataFrame` | :white_check_mark: | — | 长格式数据 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `search_space` | `dict` | :white_check_mark: | — | 参数搜索空间 |
| `metric` | `str` | | `"mae"` | 评估指标 |
| `mode` | `str` | | `"min"` | 优化方向 |
| `model_params` | `dict` | | `None` | 固定模型参数 |
| `max_windows` | `int` | | `None` | 最大评估窗口数 |
| `max_train_size` | `int` | | `None` | 最大训练集大小 |

**返回值：** `dict` — 同 `tune_model`。

---

## 区间估计

预测区间的 bootstrap 估计。

### `bootstrap_intervals`

通过 bootstrap 方法估计预测区间。

```python
def bootstrap_intervals(
    train: list | ndarray,
    horizon: int,
    forecaster: Callable,
    min_train_size: int,
    n_samples: int = 1000,
    quantiles: tuple[float, ...] = (0.1, 0.9),
    seed: int | None = None,
) -> dict
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `train` | `list \| ndarray` | :white_check_mark: | — | 训练序列 |
| `horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `forecaster` | `Callable` | :white_check_mark: | — | 预测函数（由 `make_forecaster` 创建） |
| `min_train_size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `n_samples` | `int` | | `1000` | Bootstrap 采样次数 |
| `quantiles` | `tuple[float, ...]` | | `(0.1, 0.9)` | 分位数 |
| `seed` | `int` | | `None` | 随机种子 |

**返回值：** `dict` — 包含各分位数对应的预测区间数组。

```python
from foresight import make_forecaster, bootstrap_intervals

f = make_forecaster("theta")
intervals = bootstrap_intervals(
    train=[112, 118, 132, 129, 121, 135, 148, 148],
    horizon=3,
    forecaster=f,
    min_train_size=6,
    n_samples=2000,
    quantiles=(0.05, 0.25, 0.75, 0.95),
    seed=42,
)
# intervals: {'0.05': ndarray, '0.25': ndarray, '0.75': ndarray, '0.95': ndarray}
```

---

## 模型注册

查询可用模型和模型规格。

### `list_models`

列出所有已注册的模型。

```python
def list_models() -> list[str]
```

**返回值：** `list[str]` — 模型名称列表。

```python
from foresight import list_models

models = list_models()
print(len(models))   # 250+
print(models[:5])    # ['naive-last', 'naive-mean', 'theta', ...]
```

---

### `get_model_spec`

获取指定模型的规格信息。

```python
def get_model_spec(model: str) -> ModelSpec
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `model` | `str` | :white_check_mark: | — | 模型名称 |

**返回值：** `ModelSpec` — 包含模型的接口类型、所需 extras、参数定义等信息。

```python
from foresight import get_model_spec

spec = get_model_spec("theta")
print(spec)
```

---

## 模块导入速查

```python
# 核心预测
from foresight import make_forecaster, make_forecaster_object
from foresight import make_global_forecaster, make_global_forecaster_object
from foresight import make_multivariate_forecaster
from foresight import forecast_model, forecast_model_long_df

# 评估
from foresight import eval_model, eval_model_long_df
from foresight import eval_multivariate_model_df
from foresight import eval_hierarchical_forecast_df

# 数据处理
from foresight import to_long, validate_long_df

# 异常检测
from foresight import detect_anomalies, detect_anomalies_long_df

# 序列化
from foresight import save_forecaster, load_forecaster, load_forecaster_artifact

# 层级预测
from foresight import reconcile_hierarchical_forecasts, check_hierarchical_consistency

# 调优
from foresight import tune_model, tune_model_long_df

# 区间估计
from foresight import bootstrap_intervals

# 模型注册
from foresight import list_models, get_model_spec
```

---

## 下一步

- [:octicons-arrow-right-24: CLI 参考](../cli/index.md) — 命令行界面完整参数
- [:octicons-arrow-right-24: 使用指南](../guide/index.md) — 各功能模块的详细教程
- [:octicons-arrow-right-24: 5 分钟上手](../getting-started/quickstart.md) — 快速入门教程
