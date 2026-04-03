# 常见问题 (FAQ)

本页面汇总了 ForeSight 用户最常遇到的问题。点击问题即可展开查看答案。

---

## 安装与环境

???+ question "ForeSight 支持哪些 Python 版本?"

    ForeSight 支持 **Python 3.10+**。建议使用最新的稳定版本以获得最佳兼容性。

??? question "如何安装可选依赖?"

    ForeSight 采用模块化的可选依赖设计，按需安装即可：

    | Extra 标签 | 安装命令 | 说明 |
    |---|---|---|
    | `stats` | `pip install foresight-ts[stats]` | 统计模型 |
    | `torch` | `pip install foresight-ts[torch]` | PyTorch 深度学习模型 |
    | `xgb` | `pip install foresight-ts[xgb]` | XGBoost 模型 |
    | `lgbm` | `pip install foresight-ts[lgbm]` | LightGBM 模型 |
    | `catboost` | `pip install foresight-ts[catboost]` | CatBoost 模型 |
    | `ml` | `pip install foresight-ts[ml]` | scikit-learn 机器学习模型 |
    | `all` | `pip install foresight-ts[all]` | 安装全部可选依赖 |

    可以组合安装多个 extra，例如：

    ```bash
    pip install foresight-ts[xgb,lgbm,catboost]
    ```

??? question "如何检查环境是否正确?"

    使用内置的诊断命令：

    ```bash
    foresight doctor
    ```

    该命令会检查 Python 版本、已安装的依赖、可选包的可用性等，并报告潜在问题。

??? question "核心包依赖哪些库?"

    ForeSight 核心包的运行时依赖非常轻量，仅需：

    - **numpy**
    - **pandas**

    所有其他依赖（如 `scikit-learn`、`torch`、`xgboost` 等）均为可选依赖，按需安装。

---

## 模型选择

??? question "ForeSight 有多少个模型?"

    ForeSight 目前拥有 **250+ 注册模型**，涵盖统计模型、机器学习模型和深度学习模型。

    查看完整模型列表：

    === "CLI"

        ```bash
        foresight models list
        ```

    === "Python"

        ```python
        from foresight import list_models

        models = list_models()
        ```

??? question "如何选择适合的模型?"

    推荐的模型选择策略：

    1. **从基线模型开始** — 使用 `naive`、`theta` 等简单模型建立基准
    2. **尝试统计模型** — 如 `ets`、`arima` 等经典方法
    3. **引入 ML 模型** — 如 `xgb-lag`、`lgbm-lag` 等树模型
    4. **探索 DL 模型** — 如 RNN、Transformer 等深度学习架构

    详细指引请参见 [模型选择指南](guide/models.md)。

??? question "什么是 local、global、multivariate 接口?"

    ForeSight 提供三种模型接口：

    | 接口 | 说明 | 适用场景 |
    |---|---|---|
    | **local** | 单序列训练与预测 | 每条序列独立建模 |
    | **global** | 跨序列面板训练 | 多序列共享参数，适合面板数据 |
    | **multivariate** | 多目标列联合预测 | 多个目标变量的协同预测 |

??? question "深度学习模型需要 GPU 吗?"

    **推荐但非必需。** PyTorch 模型会自动检测 CUDA 可用性：

    - 有 GPU：自动使用 CUDA 加速训练和推理
    - 无 GPU：回退到 CPU 运行（速度较慢但功能完整）

    !!! tip "提示"
        对于大规模数据集或复杂模型（如 RNN Zoo 中的架构），强烈建议使用 GPU。

---

## 数据格式

??? question "长格式数据必须包含哪些列?"

    ForeSight 使用标准的长格式（long format）数据，必须包含以下三列：

    | 列名 | 类型 | 说明 |
    |---|---|---|
    | `unique_id` | `str` / `int` | 序列唯一标识符 |
    | `ds` | `datetime` / `int` | 时间戳 |
    | `y` | `float` | 目标值 |

    示例：

    ```python
    import pandas as pd

    df = pd.DataFrame({
        "unique_id": ["series_1", "series_1", "series_2", "series_2"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
        "y": [10.0, 12.0, 5.0, 6.5],
    })
    ```

??? question "如何处理多序列面板数据?"

    使用长格式，通过 `unique_id` 列区分不同序列。所有序列堆叠在同一个 DataFrame 中：

    ```python
    #  unique_id |     ds     |   y
    #  --------- | ---------- | -----
    #  store_1   | 2024-01-01 | 100.0
    #  store_1   | 2024-01-02 | 110.0
    #  store_2   | 2024-01-01 |  50.0
    #  store_2   | 2024-01-02 |  55.0
    ```

    global 模型会自动识别 `unique_id` 并跨序列训练。

??? question "支持缺失值吗?"

    **大多数模型不支持目标列 `y` 中的 `NaN`。** 建议在训练前进行预处理：

    - 使用插值填充（线性、前向填充等）
    - 移除包含缺失值的行
    - 使用专门处理缺失值的模型（如部分统计模型）

    !!! warning "注意"
        将含 `NaN` 的数据直接传入不支持缺失值的模型可能导致训练失败或产生无意义的预测结果。

---

## 性能与优化

??? question "如何加速评估?"

    两个关键参数可以显著减少评估时间：

    - **`max_windows`** — 限制回测滑动窗口数量，减少评估轮次
    - **`max_train_size`** — 控制训练窗口大小，避免在超长序列上训练过慢

    ```python
    from foresight import evaluate

    results = evaluate(
        df,
        model="theta",
        horizon=12,
        max_windows=3,       # 最多 3 个评估窗口
        max_train_size=500,  # 训练窗口最多 500 个时间步
    )
    ```

??? question "全局模型比局部模型快吗?"

    **通常是的。** 全局模型（global）一次训练所有序列，相比局部模型（local）逐序列训练有显著的速度优势，尤其在序列数量较多时。

    此外，全局模型还能利用跨序列信息，在数据量较少的序列上往往表现更好。

??? question "如何做超参数调优?"

    使用内置的网格搜索接口：

    === "标准格式"

        ```python
        from foresight import tune_model

        best_params = tune_model(df, model="theta", horizon=12)
        ```

    === "长格式面板"

        ```python
        from foresight import tune_model_long_df

        best_params = tune_model_long_df(df, model="xgb-lag", horizon=12)
        ```

---

## 生产部署

??? question "如何保存和加载模型?"

    使用 `save_forecaster` 和 `load_forecaster` 进行模型持久化：

    ```python
    from foresight import save_forecaster, load_forecaster

    # 保存
    save_forecaster(forecaster, "model_artifact.pkl")

    # 加载
    forecaster = load_forecaster("model_artifact.pkl")
    ```

??? question "模型工件兼容旧版本吗?"

    ForeSight 通过 `artifact_schema_version` 控制工件格式的兼容性：

    | Schema 版本 | 状态 | 说明 |
    |---|---|---|
    | **1**（当前） | 活跃 | 当前默认格式 |
    | **0** | 兼容 | 旧格式仍可加载 |

    !!! info "提示"
        建议使用最新版本重新导出工件以确保长期兼容性。详见 [模型工件指南](guide/artifacts.md)。
