# 更新日志

本页面记录 ForeSight 各版本的主要变更。完整历史请参见 [GitHub Releases](https://github.com/skygazer42/ForeSight/releases)。

!!! info "版本说明"
    ForeSight 遵循 [语义版本控制](https://semver.org/lang/zh-CN/)。

## 未发布

- 新增 artifact composition summary 输出
- 新增 weighted named ensemble 组合预测
- 支持嵌套 composition 对象
- 支持 sktime 绝对 horizon 格式
- 新增 darts 和 gluonts 数据适配器（adapter）
- 新增 sktime adapter bridge

## 0.2.11

- 新增可选 `scikit-learn` (`.[ml]`) 全局/面板 step-lag 回归模型：
  adaboost, mlp, huber, quantile, sgd, kernel-ridge, svr, linear-svr, lasso, elasticnet, knn, decision-tree, bagging, gbrt, ridge, rf, extra-trees
- 新增可选 `xgboost` (`.[xgb]`) 全局 step-lag 变体：
  xgb-gamma, xgb-logistic, xgb-msle, xgb-mae, xgb-huber, xgb-poisson, xgb-tweedie, xgb-dart, xgb-linear, xgbrf
- 新增完整模型训练验证工作流 (`tools/validate_all_models.py`)，使用内置 `promotion_data`，支持逐模型 artifact 输出、进度 checkpoint 和 CUDA 加速验证

## 0.2.9

- 新增 `xgboost` 自定义多步策略模型：
    - `xgb-custom-step-lag` — step-index 单模型直接多步预测
    - `xgb-custom-dirrec-lag` — DirRec 策略（逐步模型 + 上一步特征）
    - `xgb-custom-mimo-lag` — MIMO 多输出回归（单模型预测完整 horizon）
- 新增 `lightgbm` (`.[lgbm]`) lag 特征模型族：`lgbm-lag`, `lgbm-lag-recursive`, `lgbm-step-lag`, `lgbm-dirrec-lag` 及对应的 custom 变体
- 新增 `catboost` (`.[catboost]`) lag 特征模型族：`catboost-lag`, `catboost-lag-recursive`, `catboost-step-lag`, `catboost-dirrec-lag` 及对应的 custom 变体

## 0.2.7

- 新增 xgboost 多步 horizon 策略模型：
    - `xgb-step-lag` — 带 step-index 特征的单模型直接多步预测
    - `xgb-dirrec-lag` — DirRec（直接-递归混合）逐步模型
    - `xgb-mimo-lag` — MIMO 多输出回归

## 0.2.6

- 新增 xgboost 可定制模型：
    - `xgb-custom-lag` — 直接多步预测，支持自定义超参数
    - `xgb-custom-lag-recursive` — 单步训练、递归预测，支持自定义超参数

## 0.2.5

- 新增多个 xgboost 递归 lag 特征模型变体，覆盖不同损失函数：
  `xgb-msle-lag-recursive`, `xgb-logistic-lag-recursive`, `xgb-mae-lag-recursive`, `xgb-huber-lag-recursive`, `xgb-quantile-lag-recursive`, `xgb-poisson-lag-recursive`, `xgb-gamma-lag-recursive`, `xgb-tweedie-lag-recursive`
- 新增随机森林变体 `xgbrf-lag-recursive`

## 0.2.4

- 新增 xgboost 递归变体：`xgb-lag-recursive`, `xgb-dart-lag-recursive`, `xgb-linear-lag-recursive`
- 新增目标函数变体：`xgb-msle-lag`（squared log error, y>=0）, `xgb-logistic-lag`（logistic, y in [0,1]）

## 0.2.3

- 新增 xgboost 目标函数/booster 变体：`xgb-linear-lag`, `xgb-mae-lag`, `xgb-huber-lag`, `xgb-quantile-lag`, `xgb-poisson-lag`, `xgb-gamma-lag`, `xgb-tweedie-lag`

## 0.2.2

- 新增可选 `xgboost` (`.[xgb]`) 基础模型：`xgb-lag`, `xgb-dart-lag`, `xgbrf-lag`

## 0.2.1

- 新增多个 `scikit-learn` (`.[ml]`) lag 特征模型，包括 tree/ensemble/SVR/MLP/robust/quantile 回归器（直接多步预测）

## 0.2.0

- 打包加固：`pip install foresight-ts` 支持 wheel/sdist 分发，完善 package data 和 CI 安装冒烟测试
- Torch RNN 模型大幅扩展：
    - **RNN Paper Zoo** — 100 个以论文命名的 RNN 架构
    - **RNN Zoo** — 20 个基础 RNN × 5 种 wrapper = 100 种组合

## 0.1.0

- ForeSight 初始公开版本，包含核心预测工具、CLI 命令行界面和模型注册表
