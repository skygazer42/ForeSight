# 安装指南

本页介绍 ForeSight 的安装方式、可选依赖配置、环境验证以及常见问题排查。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.10 |
| 核心依赖 | `numpy` + `pandas`（随主包自动安装） |
| 操作系统 | Linux / macOS / Windows |

---

## 基础安装

```bash
pip install foresight-ts
```

核心包仅依赖 `numpy` 和 `pandas`，安装体积小、速度快。所有重量级后端均通过可选依赖按需安装。

---

## 可选依赖

ForeSight 将重量级后端拆分为多个 extras，按需安装即可：

| Extra 名称 | 安装命令 | 包含依赖 | 最低版本 |
|------------|---------|----------|---------|
| `ml` | `pip install foresight-ts[ml]` | scikit-learn | >= 1.0 |
| `xgb` | `pip install foresight-ts[xgb]` | XGBoost | >= 2.0 |
| `lgbm` | `pip install foresight-ts[lgbm]` | LightGBM | >= 4.0 |
| `catboost` | `pip install foresight-ts[catboost]` | CatBoost | >= 1.2 |
| `stats` | `pip install foresight-ts[stats]` | statsmodels | >= 0.14 |
| `torch` | `pip install foresight-ts[torch]` | PyTorch | >= 2.0 |
| `sktime` | `pip install foresight-ts[sktime]` | sktime | >= 0.30 |
| `darts` | `pip install foresight-ts[darts]` | u8darts | >= 0.30 |
| `gluonts` | `pip install foresight-ts[gluonts]` | gluonts | >= 0.15 |
| `all` | `pip install foresight-ts[all]` | 以上全部 | — |

可以组合多个 extras：

```bash
pip install foresight-ts[ml,xgb,stats]
```

安装全部可选依赖：

```bash
pip install foresight-ts[all]
```

适配器相关 extras 可以单独安装，例如：

```bash
pip install foresight-ts[sktime]
pip install foresight-ts[darts]
pip install foresight-ts[gluonts]
```

---

## 源码安装

适用于开发者或需要最新未发布功能的用户。

```bash
git clone https://github.com/skygazer42/ForeSight.git
cd ForeSight
pip install -e ".[dev]"
```

`dev` extra 包含开发所需的测试、lint、文档构建等工具。

---

## 从 TestPyPI 安装

如需测试预发布版本：

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    foresight-ts
```

!!! warning "注意"

    TestPyPI 上的版本可能不稳定，仅建议用于测试目的。

---

## 环境验证

安装完成后，使用 `foresight doctor` 命令验证环境是否正确配置：

```bash
foresight doctor
```

`doctor` 会检查 Python 版本、核心依赖、已安装的可选后端，并报告任何兼容性问题。

### 常用选项

| 选项 | 说明 |
|------|------|
| `--format text` | 以纯文本格式输出诊断结果（默认） |
| `--strict` | 严格模式：任何警告都视为错误并以非零状态码退出 |
| `--require-extra <name>` | 要求指定的 extra 必须已安装，否则报错 |

**示例：** 验证 PyTorch 后端是否已正确安装：

```bash
foresight doctor --require-extra torch
```

**示例：** CI 中使用严格模式：

```bash
foresight doctor --strict --require-extra ml --require-extra xgb
```

---

## 数据集路径配置

ForeSight 内置数据集默认存储在 `~/.foresight/data/` 目录下。可通过以下方式自定义路径：

### 环境变量

```bash
export FORESIGHT_DATA_DIR=/path/to/custom/data
```

### 命令行参数

在支持 `--data-dir` 的命令中直接指定：

```bash
foresight eval run --model theta --dataset catfish --data-dir /path/to/data ...
```

!!! tip "优先级"

    命令行 `--data-dir` 参数优先于环境变量 `FORESIGHT_DATA_DIR`。

---

## 常见问题

??? question "安装时报错 `Python version not supported`"

    ForeSight 要求 Python >= 3.10。请使用 `python --version` 确认版本，必要时使用 `pyenv` 或 `conda` 管理多版本环境。

??? question "`import foresight` 报 `ModuleNotFoundError`"

    确认 pip 安装的环境与运行 Python 的环境一致。在虚拟环境中工作时，确保已激活对应的 venv 或 conda env。

??? question "使用某模型时报 `MissingExtraError`"

    该模型依赖未安装的可选后端。根据错误提示安装对应 extra，例如：

    ```bash
    pip install foresight-ts[torch]
    ```

??? question "PyTorch 模型运行缓慢 / 未使用 GPU"

    ForeSight 的 `torch` extra 安装的是 CPU 版 PyTorch。如需 GPU 支持，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 单独安装 CUDA 版本。

??? question "`foresight doctor` 报告依赖版本冲突"

    尝试在干净的虚拟环境中重新安装：

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install foresight-ts[all]
    foresight doctor --strict
    ```

---

下一步：前往 [5 分钟上手](quickstart.md) 开始你的第一次预测。
