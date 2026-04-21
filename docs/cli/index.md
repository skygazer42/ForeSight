# CLI 参考

ForeSight 提供完整的命令行界面，覆盖模型查询、评估、预测、调优、异常检测等全部核心功能。所有子命令均支持 `--output` 和 `--format` 参数，方便集成到自动化工作流中。

---

## 全局选项

```bash
foresight [--version] [--debug] [--data-dir DIR] <子命令>
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `--version` | flag | 显示版本号并退出 |
| `--debug` | flag | 启用调试日志输出 |
| `--data-dir` | `str` | 自定义数据目录路径 |

---

## models list

列出所有可用模型，支持按接口类型、依赖 extras 等过滤。

```bash
foresight models list [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--format` | `str` | | `tsv` | 输出格式（`tsv` / `json`） |
| `--prefix` | `str` | | — | 模型名称前缀过滤 |
| `--interface` | `str` | | `any` | 接口类型（`any` / `local` / `global` / `multivariate`） |
| `--requires` | `str` | | — | 仅显示需要指定 extras 的模型 |
| `--exclude-requires` | `str` | | — | 排除需要指定 extras 的模型 |
| `--columns` | `str` | | — | 指定输出列 |
| `--header` | flag | | — | 显示列头 |
| `--sort` | `str` | | — | 排序字段 |
| `--desc` | flag | | — | 降序排序 |
| `--limit` | `int` | | — | 限制输出行数 |
| `--output` | `str` | | — | 输出文件路径 |

=== "列出所有本地模型"

    ```bash
    foresight models list --interface local
    ```

=== "JSON 格式输出"

    ```bash
    foresight models list --format json --limit 10
    ```

=== "按前缀过滤"

    ```bash
    foresight models list --prefix arima --header
    ```

---

## models info

查看指定模型的详细信息。

```bash
foresight models info --model MODEL [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--format` | `str` | | `json` | 输出格式（`json` / `md`） |

```bash
foresight models info --model theta --format md
```

---

## datasets list

列出所有内置数据集。

```bash
foresight datasets list [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--format` | `str` | | `tsv` | 输出格式（`tsv` / `json`） |

```bash
foresight datasets list --format json
```

---

## eval run

在内置数据集上执行滚动窗口评估。

```bash
foresight eval run --model MODEL --dataset DATASET --horizon N --step N --min-train-size N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--y-col` | `str` | | — | 目标列名 |
| `--max-windows` | `int` | | — | 最大评估窗口数 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--model-param` | `key=val` | | — | 模型参数（可多次使用） |
| `--conformal-levels` | `str` | | — | Conformal 预测区间的置信水平 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `json` | 输出格式（`json` / `csv`） |
| `--metrics-output` | `str` | | — | 指标输出文件路径 |

=== "基本评估"

    ```bash
    foresight eval run \
        --model theta \
        --dataset catfish \
        --y-col Total \
        --horizon 3 --step 3 --min-train-size 12
    ```

=== "带 conformal 区间"

    ```bash
    foresight eval run \
        --model theta \
        --dataset catfish \
        --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --conformal-levels "0.8,0.95" \
        --format csv --output eval_results.csv
    ```

=== "带模型参数"

    ```bash
    foresight eval run \
        --model holt \
        --dataset catfish \
        --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --model-param alpha=0.3 \
        --model-param beta=0.1
    ```

---

## eval csv

在自定义 CSV 文件上执行滚动窗口评估。

```bash
foresight eval csv --model MODEL --path FILE --time-col COL --y-col COL --horizon N --step N --min-train-size N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--path` | `str` | :white_check_mark: | — | CSV 文件路径 |
| `--time-col` | `str` | :white_check_mark: | — | 时间列名 |
| `--y-col` | `str` | :white_check_mark: | — | 目标列名 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--id-cols` | `str` | | — | ID 列名（多序列时使用，逗号分隔） |
| `--parse-dates` | flag | | — | 自动解析日期列 |
| `--max-windows` | `int` | | — | 最大评估窗口数 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `json` | 输出格式（`json` / `csv`） |

```bash
foresight eval csv \
    --model naive-last \
    --path ./sales.csv \
    --time-col date --y-col revenue \
    --horizon 7 --step 7 --min-train-size 30 \
    --parse-dates \
    --format csv --output eval_output.csv
```

---

## cv run

在内置数据集上执行交叉验证。

```bash
foresight cv run --model MODEL --dataset DATASET --horizon N --min-train-size N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--step-size` | `int` | | — | 窗口滑动步长 |
| `--y-col` | `str` | | — | 目标列名 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--n-windows` | `int` | | — | 交叉验证窗口数 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |

```bash
foresight cv run \
    --model theta \
    --dataset catfish \
    --y-col Total \
    --horizon 3 --min-train-size 12 \
    --n-windows 5 \
    --format csv --output cv_results.csv
```

---

## cv csv

在自定义 CSV 文件上执行交叉验证。

```bash
foresight cv csv --model MODEL --path FILE --time-col COL --y-col COL --horizon N --min-train-size N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--path` | `str` | :white_check_mark: | — | CSV 文件路径 |
| `--time-col` | `str` | :white_check_mark: | — | 时间列名 |
| `--y-col` | `str` | :white_check_mark: | — | 目标列名 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--id-cols` | `str` | | — | ID 列名（逗号分隔） |
| `--parse-dates` | flag | | — | 自动解析日期列 |
| `--step-size` | `int` | | — | 窗口滑动步长 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--n-windows` | `int` | | — | 交叉验证窗口数 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |

```bash
foresight cv csv \
    --model holt \
    --path ./data.csv \
    --time-col ds --y-col y \
    --horizon 12 --min-train-size 36 \
    --parse-dates --n-windows 3 \
    --model-param alpha=0.5
```

---

## forecast csv

从 CSV 文件生成预测。

```bash
foresight forecast csv --model MODEL --path FILE --time-col COL --y-col COL --horizon N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--path` | `str` | :white_check_mark: | — | CSV 文件路径 |
| `--time-col` | `str` | :white_check_mark: | — | 时间列名 |
| `--y-col` | `str` | :white_check_mark: | — | 目标列名 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--id-cols` | `str` | | — | ID 列名（逗号分隔） |
| `--parse-dates` | flag | | — | 自动解析日期列 |
| `--future-path` | `str` | | — | 未来协变量 CSV 路径 |
| `--interval-levels` | `str` | | — | 预测区间的置信水平（逗号分隔） |
| `--interval-min-train-size` | `int` | | — | Bootstrap 区间的最小训练大小 |
| `--interval-samples` | `int` | | — | Bootstrap 采样次数 |
| `--interval-seed` | `int` | | — | 随机种子 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |
| `--save-artifact` | `str` | | — | 保存模型工件路径 |

=== "基本预测"

    ```bash
    foresight forecast csv \
        --model theta \
        --path ./sales.csv \
        --time-col date --y-col revenue \
        --horizon 30 --parse-dates
    ```

=== "带预测区间"

    ```bash
    foresight forecast csv \
        --model theta \
        --path ./sales.csv \
        --time-col date --y-col revenue \
        --horizon 30 --parse-dates \
        --interval-levels "0.8,0.95" \
        --interval-samples 2000 \
        --interval-seed 42
    ```

=== "保存工件"

    ```bash
    foresight forecast csv \
        --model holt \
        --path ./sales.csv \
        --time-col date --y-col revenue \
        --horizon 30 --parse-dates \
        --save-artifact ./model.artifact \
        --output forecast.csv
    ```

---

## forecast artifact

从已保存的模型工件生成预测。

Only load artifacts from trusted sources.

```bash
foresight forecast artifact --artifact PATH --horizon N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--artifact` | `str` | :white_check_mark: | — | 模型工件路径 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--interval-levels` | `str` | | — | 预测区间置信水平 |
| `--interval-min-train-size` | `int` | | — | Bootstrap 最小训练大小 |
| `--interval-samples` | `int` | | — | Bootstrap 采样次数 |
| `--interval-seed` | `int` | | — | 随机种子 |
| `--cutoff` | `str` | | — | 截止日期 |
| `--future-path` | `str` | | — | 未来协变量 CSV 路径 |
| `--time-col` | `str` | | — | 时间列名 |
| `--parse-dates` | flag | | — | 自动解析日期列 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |

```bash
foresight forecast artifact \
    --artifact ./model.artifact \
    --horizon 14 \
    --interval-levels "0.9" \
    --output forecast.csv
```

---

## artifact info

查看模型工件的详细信息。

Only load artifacts from trusted sources.

```bash
foresight artifact info --artifact PATH [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--artifact` | `str` | :white_check_mark: | — | 模型工件路径 |
| `--format` | `str` | | `json` | 输出格式（`json` / `md` / `markdown`） |
| `--output` | `str` | | — | 输出文件路径 |

```bash
foresight artifact info --artifact ./model.artifact --format md
```

---

## artifact validate

验证模型工件的完整性。

Only load artifacts from trusted sources.

```bash
foresight artifact validate --artifact PATH [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--artifact` | `str` | :white_check_mark: | — | 模型工件路径 |
| `--format` | `str` | | `json` | 输出格式（`json`） |
| `--output` | `str` | | — | 输出文件路径 |

```bash
foresight artifact validate --artifact ./model.artifact
```

---

## artifact diff

对比两个模型工件的差异。

Only load artifacts from trusted sources.

```bash
foresight artifact diff --left-artifact PATH --right-artifact PATH [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--left-artifact` | `str` | :white_check_mark: | — | 左侧工件路径 |
| `--right-artifact` | `str` | :white_check_mark: | — | 右侧工件路径 |
| `--path-prefix` | `str` | | — | 路径前缀过滤 |
| `--format` | `str` | | `json` | 输出格式（`json` / `csv` / `md` / `markdown`） |
| `--output` | `str` | | — | 输出文件路径 |

```bash
foresight artifact diff \
    --left-artifact ./v1.artifact \
    --right-artifact ./v2.artifact \
    --format md
```

---

## tuning run

执行网格搜索超参数调优。

```bash
foresight tuning run --model MODEL --dataset DATASET --horizon N --step N --min-train-size N --grid-param key=v1,v2 [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--model` | `str` | :white_check_mark: | — | 模型名称 |
| `--dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--grid-param` | `key=v1,v2` | :white_check_mark: | — | 搜索参数（可多次使用） |
| `--y-col` | `str` | | — | 目标列名 |
| `--metric` | `str` | | `mae` | 评估指标（`mae` / `rmse` / `mape` / `smape`） |
| `--mode` | `str` | | `min` | 优化方向（`min` / `max`） |
| `--model-param` | `key=val` | | — | 固定模型参数 |
| `--max-windows` | `int` | | — | 最大评估窗口数 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `json` | 输出格式（`json` / `csv` / `md`） |

=== "单参数调优"

    ```bash
    foresight tuning run \
        --model theta \
        --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --grid-param theta=0.5,1.0,1.5,2.0
    ```

=== "多参数调优"

    ```bash
    foresight tuning run \
        --model holt \
        --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --grid-param alpha=0.1,0.3,0.5 \
        --grid-param beta=0.01,0.05,0.1 \
        --metric rmse --format md
    ```

---

## doctor

诊断当前环境的安装状态和依赖可用性。

```bash
foresight doctor [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--format` | `str` | | `text` | 输出格式（`json` / `text`） |
| `--output` | `str` | | — | 输出文件路径 |
| `--strict` | flag | | — | 严格模式（缺少依赖时返回非零退出码） |
| `--require-extra` | `str` | | — | 检查指定 extras 是否已安装 |

=== "基本诊断"

    ```bash
    foresight doctor
    ```

=== "检查特定 extras"

    ```bash
    foresight doctor --strict --require-extra torch --require-extra xgboost
    ```

---

## detect run

在内置数据集上执行异常检测。

```bash
foresight detect run --dataset DATASET [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `--y-col` | `str` | | — | 目标列名 |
| `--model` | `str` | | — | 用于残差计算的预测模型 |
| `--score-method` | `str` | | — | 异常评分方法 |
| `--threshold-method` | `str` | | — | 阈值确定方法 |
| `--threshold-k` | `float` | | — | k-sigma 阈值系数 |
| `--threshold-quantile` | `float` | | — | 分位数阈值 |
| `--window` | `int` | | — | 滑动窗口大小 |
| `--min-history` | `int` | | — | 最小历史数据量 |
| `--min-train-size` | `int` | | — | 最小训练集大小 |
| `--step-size` | `int` | | — | 步长 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--n-windows` | `int` | | — | 窗口数 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |

```bash
foresight detect run \
    --dataset catfish --y-col Total \
    --model theta \
    --threshold-k 2.5 \
    --window 12 \
    --format csv --output anomalies.csv
```

---

## detect csv

在自定义 CSV 文件上执行异常检测。

```bash
foresight detect csv --path FILE --time-col COL --y-col COL [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--path` | `str` | :white_check_mark: | — | CSV 文件路径 |
| `--time-col` | `str` | :white_check_mark: | — | 时间列名 |
| `--y-col` | `str` | :white_check_mark: | — | 目标列名 |
| `--model` | `str` | | — | 预测模型 |
| `--score-method` | `str` | | — | 异常评分方法 |
| `--threshold-method` | `str` | | — | 阈值确定方法 |
| `--threshold-k` | `float` | | — | k-sigma 阈值系数 |
| `--threshold-quantile` | `float` | | — | 分位数阈值 |
| `--window` | `int` | | — | 滑动窗口大小 |
| `--min-history` | `int` | | — | 最小历史数据量 |
| `--min-train-size` | `int` | | — | 最小训练集大小 |
| `--step-size` | `int` | | — | 步长 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--n-windows` | `int` | | — | 窗口数 |
| `--model-param` | `key=val` | | — | 模型参数 |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `csv` | 输出格式（`csv` / `json`） |

```bash
foresight detect csv \
    --path ./metrics.csv \
    --time-col timestamp --y-col cpu_usage \
    --model moving-average \
    --threshold-quantile 0.99 \
    --parse-dates \
    --output anomalies.csv
```

---

## leaderboard run

在内置数据集上运行模型排行榜评估。

```bash
foresight leaderboard run --dataset DATASET --horizon N --step N --min-train-size N [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--dataset` | `str` | :white_check_mark: | — | 内置数据集名称 |
| `--horizon` | `int` | :white_check_mark: | — | 预测步长 |
| `--step` | `int` | :white_check_mark: | — | 滚动窗口步长 |
| `--min-train-size` | `int` | :white_check_mark: | — | 最小训练集大小 |
| `--y-col` | `str` | | — | 目标列名 |
| `--models` | `str` | | — | 指定模型列表（逗号分隔） |
| `--prefix` | `str` | | — | 模型名称前缀过滤 |
| `--interface` | `str` | | — | 接口类型过滤 |
| `--requires` | `str` | | — | 仅包含需要指定 extras 的模型 |
| `--exclude-requires` | `str` | | — | 排除需要指定 extras 的模型 |
| `--max-windows` | `int` | | — | 最大评估窗口数 |
| `--max-train-size` | `int` | | — | 最大训练集大小 |
| `--metric` | `str` | | `mae` | 排序指标（`mae` / `rmse` / `mape` / `smape`） |
| `--format` | `str` | | `json` | 输出格式（`json` / `csv` / `md`） |
| `--output` | `str` | | — | 输出文件路径 |
| `--limit` | `int` | | — | 限制输出行数 |

=== "全部模型排行"

    ```bash
    foresight leaderboard run \
        --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --format md --limit 20
    ```

=== "指定模型子集"

    ```bash
    foresight leaderboard run \
        --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --models theta,holt,naive-last,moving-average \
        --metric rmse
    ```

=== "排除重依赖模型"

    ```bash
    foresight leaderboard run \
        --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 \
        --exclude-requires torch,tensorflow \
        --format csv --output leaderboard.csv
    ```

---

## pipeline run

执行数据工程管道配置。

```bash
foresight pipeline run --config PATH [选项]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `--config` | `str` | :white_check_mark: | — | 管道配置文件路径（YAML/JSON） |
| `--output` | `str` | | — | 输出文件路径 |
| `--format` | `str` | | `json` | 输出格式（`json`） |

```bash
foresight pipeline run --config ./pipeline.yaml --output results.json
```

---

## 命令速查表

!!! tip "常用命令组合"

    ```bash
    # 环境诊断
    foresight doctor

    # 查看可用模型
    foresight models list --interface local --header

    # 快速评估
    foresight eval run --model theta --dataset catfish \
        --y-col Total --horizon 3 --step 3 --min-train-size 12

    # 从 CSV 预测并保存工件
    foresight forecast csv --model theta --path data.csv \
        --time-col ds --y-col y --horizon 12 --parse-dates \
        --save-artifact model.artifact

    # 从工件重新预测
    foresight forecast artifact --artifact model.artifact --horizon 12

    # 超参数调优
    foresight tuning run --model holt --dataset catfish \
        --y-col Total --horizon 3 --step 3 --min-train-size 12 \
        --grid-param alpha=0.1,0.3,0.5 --grid-param beta=0.01,0.1

    # 模型排行榜
    foresight leaderboard run --dataset catfish --y-col Total \
        --horizon 3 --step 3 --min-train-size 12 --format md

    # 异常检测
    foresight detect run --dataset catfish --y-col Total \
        --model theta --threshold-k 3.0
    ```

---

## 下一步

- [:octicons-arrow-right-24: API 参考](../api-reference/index.md) — Python API 完整签名
- [:octicons-arrow-right-24: 使用指南](../guide/index.md) — 各功能模块的详细教程
- [:octicons-arrow-right-24: 5 分钟上手](../getting-started/quickstart.md) — 快速入门教程
