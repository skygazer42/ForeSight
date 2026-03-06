# Changelog

This project follows a lightweight changelog format.

## Unreleased

- Add optional `scikit-learn` (`.[ml]`) global/panel step-lag regressors:
  - `adaboost-step-lag-global`
  - `mlp-step-lag-global`
  - `huber-step-lag-global`
  - `quantile-step-lag-global`
  - `sgd-step-lag-global`
  - `kernel-ridge-step-lag-global`
  - `svr-step-lag-global`
  - `linear-svr-step-lag-global`
  - `lasso-step-lag-global`
  - `elasticnet-step-lag-global`
  - `knn-step-lag-global`
  - `decision-tree-step-lag-global`
  - `bagging-step-lag-global`
  - `gbrt-step-lag-global`
  - `ridge-step-lag-global`
  - `rf-step-lag-global`
  - `extra-trees-step-lag-global`
- Add optional `xgboost` (`.[xgb]`) global/panel step-lag regressors:
  - `xgb-gamma-step-lag-global`
  - `xgb-logistic-step-lag-global`
  - `xgb-msle-step-lag-global`
  - `xgb-mae-step-lag-global`
  - `xgb-huber-step-lag-global`
  - `xgb-poisson-step-lag-global`
  - `xgb-tweedie-step-lag-global`
  - `xgb-dart-step-lag-global`
  - `xgb-linear-step-lag-global`
  - `xgbrf-step-lag-global`

## 0.2.9

- Add optional `xgboost` (`.[xgb]`) customizable multi-horizon strategy models:
  - `xgb-custom-step-lag`: step-index single-model direct multi-horizon.
  - `xgb-custom-dirrec-lag`: DirRec strategy (per-step models with previous-step features).
  - `xgb-custom-mimo-lag`: MIMO multi-output regression (single model predicts the full horizon).
- Add optional `lightgbm` (`.[lgbm]`) lag-feature models:
  - `lgbm-lag`, `lgbm-lag-recursive`, `lgbm-step-lag`, `lgbm-dirrec-lag`.
  - Customizable variants: `lgbm-custom-lag`, `lgbm-custom-lag-recursive`, `lgbm-custom-step-lag`,
    `lgbm-custom-dirrec-lag`.
- Add optional `catboost` (`.[catboost]`) lag-feature models:
  - `catboost-lag`, `catboost-lag-recursive`, `catboost-step-lag`, `catboost-dirrec-lag`.
  - Customizable variants: `catboost-custom-lag`, `catboost-custom-lag-recursive`,
    `catboost-custom-step-lag`, `catboost-custom-dirrec-lag`.

## 0.2.7

- Add more optional `xgboost` (`.[xgb]`) multi-horizon strategies on lag features:
  - `xgb-step-lag`: single-model direct multi-horizon with an extra step-index feature.
  - `xgb-dirrec-lag`: DirRec (direct-recursive) per-step models with previous-step features.
  - `xgb-mimo-lag`: MIMO multi-output regression (single model predicts the full horizon).

## 0.2.6

- Add optional `xgboost` (`.[xgb]`) customizable models:
  - `xgb-custom-lag` (direct multi-horizon).
  - `xgb-custom-lag-recursive` (one-step trained, recursive forecast).

## 0.2.5

- Add more optional `xgboost` (`.[xgb]`) recursive lag-feature models:
  - Objectives: `xgb-msle-lag-recursive`, `xgb-logistic-lag-recursive`, `xgb-mae-lag-recursive`,
    `xgb-huber-lag-recursive`, `xgb-quantile-lag-recursive`, `xgb-poisson-lag-recursive`,
    `xgb-gamma-lag-recursive`, `xgb-tweedie-lag-recursive`.
  - Random forest: `xgbrf-lag-recursive`.

## 0.2.4

- Add more optional `xgboost` (`.[xgb]`) models:
  - Recursive variants: `xgb-lag-recursive`, `xgb-dart-lag-recursive`, `xgb-linear-lag-recursive`.
  - New objectives: `xgb-msle-lag` (squared log error, y>=0), `xgb-logistic-lag` (logistic, y in [0,1]).

## 0.2.3

- Add more optional `xgboost` (`.[xgb]`) objectives/boosters: `xgb-linear-lag`, `xgb-mae-lag`,
  `xgb-huber-lag`, `xgb-quantile-lag`, `xgb-poisson-lag`, `xgb-gamma-lag`, `xgb-tweedie-lag`.

## 0.2.2

- Add optional `xgboost` (`.[xgb]`) models: `xgb-lag`, `xgb-dart-lag`, `xgbrf-lag`.

## 0.2.1

- Add more `scikit-learn` lag-feature models (`.[ml]`), including trees/ensembles/SVR/MLP/robust/quantile
  regressors (direct multi-horizon).

## 0.2.0

- Packaging hardening for `pip install foresight-ts` (wheel/sdist, package data, CI install smoke).
- Torch RNN model expansions:
  - RNN Paper Zoo (100 paper-named architectures).
  - RNN Zoo (20 bases × 5 wrappers = 100 combos).

## 0.1.0

- Initial public version of `foresight` (core forecasting utilities + CLI + model registry).
