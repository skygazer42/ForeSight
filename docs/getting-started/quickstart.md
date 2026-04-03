# 5 分钟上手

本页通过最少代码展示 ForeSight 的核心功能：滚动窗口评估、函数式预测和对象式预测。

!!! info "前置条件"

    请先完成 [安装指南](installation.md) 中的基础安装。

---

## Python API

=== "滚动窗口评估"

    使用 `eval_model` 对内置数据集进行滚动窗口回测：

    ```python
    from foresight import eval_model

    metrics = eval_model(
        model="theta", dataset="catfish", y_col="Total",
        horizon=3, step=3, min_train_size=12,
    )
    print(metrics)  # {'mae': ..., 'rmse': ..., 'mape': ..., 'smape': ...}
    ```

    `eval_model` 自动执行 walk-forward 交叉验证并返回汇总指标字典。

=== "函数式 API"

    `make_forecaster` 返回一个无状态的预测函数，传入历史数据即可生成预测：

    ```python
    from foresight import make_forecaster

    f = make_forecaster("holt", alpha=0.3, beta=0.1)
    yhat = f([112, 118, 132, 129, 121, 135, 148, 148], horizon=3)
    ```

    适用于一次性预测场景，无需管理模型状态。

=== "对象式 API"

    `make_forecaster_object` 返回一个有状态的模型对象，支持 `fit` / `predict` / `save` / `load`：

    ```python
    from foresight import make_forecaster_object

    obj = make_forecaster_object("moving-average", window=3)
    obj.fit([1, 2, 3, 4, 5, 6])
    yhat = obj.predict(3)
    ```

    适用于生产环境中需要持久化模型的场景。

---

## CLI

=== "查看模型列表"

    ```bash
    foresight models list
    ```

    列出所有已注册的模型及其所属类别和所需 extras。

=== "评估模型"

    ```bash
    foresight eval run --model theta --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12
    ```

    输出与 `eval_model` 相同的指标汇总。

=== "从 CSV 预测"

    ```bash
    foresight forecast csv --model naive-last --path ./data.csv \
        --time-col ds --y-col y --parse-dates --horizon 7
    ```

    从本地 CSV 文件读取数据并生成未来 7 步预测。

---

## 下一步

- [使用指南](../guide/index.md) — 深入了解数据格式、评估策略、模型选择等
- [模型选择指南](../guide/models.md) — 了解 250+ 模型的能力矩阵
- [CLI 参考](../cli/index.md) — 所有子命令的完整参数文档
- [API 参考](../api-reference/index.md) — Python API 完整签名与示例
