# ForeSight Package Maturity Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the product, API-consistency, and engineering-quality gaps that still keep `foresight` from behaving like a stable, mature time-series package.

**Architecture:** Use the central model registry as the single source of truth for capability metadata, then drive forecast/eval validation, CLI help, and generated docs from that metadata. Close the remaining forecast/eval/artifact workflow gaps before adding more algorithms, then harden compatibility guarantees, CI quality gates, and public documentation.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, ruff, optional statsmodels/scikit-learn/xgboost/lightgbm/catboost/torch, GitHub Actions.

---

## Priority Order

- **P0:** Capability consistency and compatibility guarantees
- **P1:** CI, typing, and release confidence
- **P2:** Documentation and benchmarking needed for adoption

### Task 1: Add Explicit Model Capability Metadata (P0)

**Files:**
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/cli.py`
- Modify: `src/foresight/models/__init__.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_cli_models.py`
- Test: `tests/test_cli_models_list_extended.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests that require every `ModelSpec` to expose machine-readable capability fields such as:
- `supports_x_cols`
- `supports_quantiles`
- `supports_interval_forecast`
- `supports_artifact_save`
- `interface`
- `requires_future_covariates`

Add CLI tests that require `foresight models info ...` and `foresight models list ...` to surface those capabilities.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_models_registry.py tests/test_cli_models.py tests/test_cli_models_list_extended.py`

Expected: FAIL because the registry only exposes `requires`, `param_help`, and `interface`, but no normalized capability metadata.

**Step 3: Write minimal implementation**

Extend `ModelSpec` with a compact `capabilities` mapping and helper accessors. Populate it first for the forecast/eval-critical families:
- local statsmodels wrappers
- global regression models
- torch global models
- multivariate models

Update CLI output to render capability flags directly from registry data instead of hand-written assumptions.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_models_registry.py tests/test_cli_models.py tests/test_cli_models_list_extended.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/registry.py src/foresight/cli.py src/foresight/models/__init__.py tests/test_models_registry.py tests/test_cli_models.py tests/test_cli_models_list_extended.py README.md
git commit -m "feat(registry): add model capability metadata"
```

### Task 2: Close Forecast and Eval Capability Gaps (P0)

**Files:**
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/eval_forecast.py`
- Modify: `src/foresight/intervals.py`
- Modify: `src/foresight/models/global_regression.py`
- Modify: `src/foresight/models/torch_global.py`
- Test: `tests/test_forecast_api.py`
- Test: `tests/test_eval_local_xreg.py`
- Test: `tests/test_eval_panel.py`
- Test: `tests/test_cli_eval_run_intervals.py`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add tests for the currently missing workflow combinations:
- `forecast_model_long_df(..., interval_levels=...)` works for supported global models
- `forecast_model_long_df(..., model_params={"x_cols": ...}, interval_levels=...)` works for supported local xreg models
- `eval_model_long_df(...)` supports `x_cols` for the same supported families that forecast supports
- unsupported combinations fail with capability-driven error messages instead of family-specific hard-coded messages

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_eval_panel.py tests/test_cli_eval_run_intervals.py`

Expected: FAIL at the current explicit guards in `forecast.py` and `eval_forecast.py`.

**Step 3: Write minimal implementation**

Refactor forecast/eval validation to consult registry capabilities first. Implement the smallest useful interval path:
- bootstrap or quantile-derived intervals for eligible global models
- interval support for eligible local xreg models with `x_cols`
- consistent error messages for unsupported model families

Do not promise universal interval support in this phase. Only support combinations that are backed by real implementations and clearly marked in registry metadata.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_eval_panel.py tests/test_cli_eval_run_intervals.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/forecast.py src/foresight/eval_forecast.py src/foresight/intervals.py src/foresight/models/global_regression.py src/foresight/models/torch_global.py tests/test_forecast_api.py tests/test_eval_local_xreg.py tests/test_eval_panel.py tests/test_cli_eval_run_intervals.py README.md
git commit -m "feat(forecast): close interval and x_cols capability gaps"
```

### Task 3: Version Artifact Schema and Compatibility Checks (P0)

**Files:**
- Modify: `src/foresight/serialization.py`
- Modify: `src/foresight/base.py`
- Modify: `src/foresight/cli.py`
- Test: `tests/test_serialization.py`
- Test: `tests/test_cli_forecast.py`
- Docs: `README.md`
- Docs: `docs/RELEASE.md`

**Step 1: Write the failing tests**

Add tests that require serialized artifacts to include:
- artifact schema version
- package version
- model key
- model params
- train schema summary

Add compatibility tests for:
- loading current-version artifacts
- rejecting malformed payloads with clear errors
- warning or blocking when schema version is unsupported

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_serialization.py tests/test_cli_forecast.py`

Expected: FAIL because artifacts currently store metadata but no explicit schema version or compatibility enforcement.

**Step 3: Write minimal implementation**

Introduce an artifact schema constant in `serialization.py`, validate it on load, and expose a concise compatibility message in CLI forecast-artifact paths. Keep pickle as the transport format for now; add schema/version guardrails before considering alternate formats.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_serialization.py tests/test_cli_forecast.py`

Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/serialization.py src/foresight/base.py src/foresight/cli.py tests/test_serialization.py tests/test_cli_forecast.py README.md docs/RELEASE.md
git commit -m "feat(serialization): version artifact schema and validate compatibility"
```

### Task 4: Add Typing, Coverage, and CI Matrix Gates (P1)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Create: `tools/check_capability_docs.py`
- Test: `tests/test_root_import.py`
- Test: `tests/test_models_registry.py`
- Docs: `docs/DEVELOPMENT.md`
- Docs: `README.md`

**Step 1: Write the failing tests**

Add lightweight tests or checks that require:
- registry capability docs to stay in sync
- public imports to remain stable

Prepare CI expectations for:
- type checking
- coverage reporting
- multiple Python versions
- optional dependency smoke coverage for at least core + stats + ml

**Step 2: Run checks to verify they fail**

Run:
- `python tools/check_capability_docs.py`
- `pytest -q tests/test_root_import.py tests/test_models_registry.py`

Expected: FAIL because the capability-doc sync check and CI/type/coverage configuration do not exist yet.

**Step 3: Write minimal implementation**

Add:
- `mypy` or `pyright` configuration in `pyproject.toml`
- `pytest-cov` configuration with a minimum threshold
- GitHub Actions matrix for Python versions supported by package policy
- at least one optional dependency matrix lane beyond core
- a small sync checker to ensure generated capability docs are not stale

Keep the first threshold pragmatic; do not set an unrealistically high coverage floor on day one.

**Step 4: Run checks to verify they pass**

Run:
- `python tools/check_capability_docs.py`
- `pytest -q tests/test_root_import.py tests/test_models_registry.py`
- `ruff check src tests tools`

Expected: PASS locally, with CI prepared to enforce the broader matrix.

**Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml .github/workflows/release.yml tools/check_capability_docs.py tests/test_root_import.py tests/test_models_registry.py docs/DEVELOPMENT.md README.md
git commit -m "chore(ci): add typing coverage and matrix quality gates"
```

### Task 5: Generate Public Capability Docs and Stable API Reference (P1)

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/models.md`
- Create: `docs/api.md`
- Create: `tools/generate_model_capability_docs.py`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Test: `tests/test_cli_models_info_paper_metadata.py`
- Test: `tests/test_docs_rnn_generated.py`

**Step 1: Write the failing tests**

Add tests or doc-generation checks that require:
- generated model capability tables to include capability metadata from registry
- API reference pages to mention the supported forecast/eval/artifact entry points
- doc generation to be reproducible from code state

**Step 2: Run checks to verify they fail**

Run:
- `python tools/generate_model_capability_docs.py --check`
- `pytest -q tests/test_docs_rnn_generated.py tests/test_cli_models_info_paper_metadata.py`

Expected: FAIL because there is no generated capability-doc pipeline or docs site config yet.

**Step 3: Write minimal implementation**

Set up a minimal docs site with:
- landing page
- model capability matrix
- public API reference pages
- generation script that derives model docs from registry metadata

Update package metadata to point `Documentation` at the docs site once published.

**Step 4: Run checks to verify they pass**

Run:
- `python tools/generate_model_capability_docs.py`
- `python tools/generate_model_capability_docs.py --check`
- `pytest -q tests/test_docs_rnn_generated.py tests/test_cli_models_info_paper_metadata.py`

Expected: PASS

**Step 5: Commit**

```bash
git add mkdocs.yml docs/index.md docs/models.md docs/api.md tools/generate_model_capability_docs.py README.md pyproject.toml tests/test_docs_rnn_generated.py tests/test_cli_models_info_paper_metadata.py
git commit -m "docs: generate capability matrix and publish stable API reference"
```

### Task 6: Add Reproducible Benchmark and Regression Tracking (P2)

**Files:**
- Create: `benchmarks/run_benchmarks.py`
- Create: `benchmarks/benchmark_config.json`
- Modify: `src/foresight/datasets.py`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_packaged_datasets_smoke.py`
- Docs: `README.md`
- Docs: `docs/DEVELOPMENT.md`

**Step 1: Write the failing tests**

Add smoke checks that require the benchmark runner to:
- load packaged benchmark datasets
- execute a tiny benchmark sweep on a fixed seed/config
- emit a deterministic summary table

**Step 2: Run checks to verify they fail**

Run:
- `pytest -q tests/test_packaged_datasets_smoke.py`
- `python benchmarks/run_benchmarks.py --smoke`

Expected: FAIL because no benchmark runner or config exists yet.

**Step 3: Write minimal implementation**

Create a small benchmark harness focused on stability rather than breadth:
- 2 to 4 packaged datasets
- a narrow baseline model set
- wall-clock timing
- point metrics and interval metrics where supported

Wire a smoke-mode run into CI; keep full benchmark runs manual at first.

**Step 4: Run checks to verify they pass**

Run:
- `pytest -q tests/test_packaged_datasets_smoke.py`
- `python benchmarks/run_benchmarks.py --smoke`

Expected: PASS

**Step 5: Commit**

```bash
git add benchmarks/run_benchmarks.py benchmarks/benchmark_config.json src/foresight/datasets.py .github/workflows/ci.yml tests/test_packaged_datasets_smoke.py README.md docs/DEVELOPMENT.md
git commit -m "feat(benchmarks): add reproducible benchmark harness"
```

---

## Suggested Execution Order

1. Task 1: capability metadata
2. Task 2: forecast/eval workflow closure
3. Task 3: artifact compatibility
4. Task 4: CI, typing, coverage
5. Task 5: generated docs
6. Task 6: benchmark harness

## Scope Notes

- Do not add more model families until Tasks 1 to 3 are complete.
- Keep all new validation and docs driven from registry metadata to avoid duplicate source-of-truth logic.
- Prefer extending existing tests over creating one-off smoke files when a nearby test module already covers the same public surface.
- Keep the first typing and coverage thresholds modest; the goal is to establish enforcement, not to stall the roadmap on cleanup debt.
- Preserve backward compatibility for the current public Python and CLI entry points unless a task explicitly introduces a versioned breaking change.

Plan complete and saved to `docs/plans/2026-03-07-package-maturity-hardening.md`. Two execution options:

**1. Subagent-Driven (this session)** - I execute the plan incrementally here, one task at a time, with review and verification between tasks.

**2. Parallel Session (same `main` branch)** - Start a fresh session and execute from this plan as a dedicated roadmap track while keeping this session for review.

Which approach?
