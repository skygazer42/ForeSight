# Tags / Release Notes

This file records **what changed for each Git tag** in this repository.

- Canonical source of truth: `CHANGELOG.md`
- This file is a **tag-indexed view** of the changelog, plus tag metadata (date / commit).
- If a version exists in `CHANGELOG.md` but **does not** have a Git tag, it will not appear here.

## Quick Index

| Tag | Date (commit) | Commit |
| --- | --- | --- |
| `v0.2.9` | 2026-03-05 | `5fdd998` |
| `v0.2.7` | 2026-03-05 | `a4364b1` |
| `v0.2.6` | 2026-03-05 | `2b8cf95` |
| `v0.2.5` | 2026-03-05 | `de9409f` |
| `v0.2.4` | 2026-03-05 | `9b70d4e` |
| `v0.2.3` | 2026-03-05 | `f32b38a` |
| `v0.2.2` | 2026-03-05 | `3766f1f` |
| `v0.2.1` | 2026-03-05 | `7aff89b` |
| `v0.2.0` | 2026-03-04 | `2c35740` |

## v0.2.9 (2026-03-05)

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

## v0.2.7 (2026-03-05)

- Add more optional `xgboost` (`.[xgb]`) multi-horizon strategies on lag features:
  - `xgb-step-lag`: single-model direct multi-horizon with an extra step-index feature.
  - `xgb-dirrec-lag`: DirRec (direct-recursive) per-step models with previous-step features.
  - `xgb-mimo-lag`: MIMO multi-output regression (single model predicts the full horizon).

## v0.2.6 (2026-03-05)

- Add optional `xgboost` (`.[xgb]`) customizable models:
  - `xgb-custom-lag` (direct multi-horizon).
  - `xgb-custom-lag-recursive` (one-step trained, recursive forecast).

## v0.2.5 (2026-03-05)

- Add more optional `xgboost` (`.[xgb]`) recursive lag-feature models:
  - Objectives: `xgb-msle-lag-recursive`, `xgb-logistic-lag-recursive`, `xgb-mae-lag-recursive`,
    `xgb-huber-lag-recursive`, `xgb-quantile-lag-recursive`, `xgb-poisson-lag-recursive`,
    `xgb-gamma-lag-recursive`, `xgb-tweedie-lag-recursive`.
  - Random forest: `xgbrf-lag-recursive`.

## v0.2.4 (2026-03-05)

- Add more optional `xgboost` (`.[xgb]`) models:
  - Recursive variants: `xgb-lag-recursive`, `xgb-dart-lag-recursive`, `xgb-linear-lag-recursive`.
  - New objectives: `xgb-msle-lag` (squared log error, y>=0), `xgb-logistic-lag` (logistic, y in [0,1]).

## v0.2.3 (2026-03-05)

- Add more optional `xgboost` (`.[xgb]`) objectives/boosters: `xgb-linear-lag`, `xgb-mae-lag`,
  `xgb-huber-lag`, `xgb-quantile-lag`, `xgb-poisson-lag`, `xgb-gamma-lag`, `xgb-tweedie-lag`.

## v0.2.2 (2026-03-05)

- Add optional `xgboost` (`.[xgb]`) models: `xgb-lag`, `xgb-dart-lag`, `xgbrf-lag`.

## v0.2.1 (2026-03-05)

- Add more `scikit-learn` lag-feature models (`.[ml]`), including trees/ensembles/SVR/MLP/robust/quantile
  regressors (direct multi-horizon).

## v0.2.0 (2026-03-04)

- Packaging hardening for `pip install foresight-ts` (wheel/sdist, package data, CI install smoke).
- Torch RNN model expansions:
  - RNN Paper Zoo (100 paper-named architectures).
  - RNN Zoo (20 bases × 5 wrappers = 100 combos).

