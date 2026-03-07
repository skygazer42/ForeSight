# Release

This is a lightweight checklist for publishing `foresight-ts` as a pip-installable package.

## 1) Bump version

`foresight-ts` uses a dynamic version sourced from `foresight.__version__`.

- Modify: `src/foresight/__init__.py:1`

## 2) Regenerate derived docs (if needed)

```bash
# Optional: refresh paper metadata (hits public APIs)
python tools/fetch_rnn_paper_metadata.py --refresh --sleep 0.02

# Required when code changes affect the generated tables/line numbers
python tools/generate_model_capability_docs.py
python tools/generate_rnn_docs.py
```

## 3) Run benchmark smoke + docs build locally

```bash
python benchmarks/run_benchmarks.py --smoke
mkdocs build --strict
```

## 4) Run release checks

```bash
python tools/release_check.py
```

## 5) Build artifacts

```bash
python -m build
```

## 6) Validate artifacts

```bash
twine check dist/*
```

## 7) Check serialization compatibility notes

If a release changes the saved forecaster artifact payload, make sure the
artifact schema/version guardrails are updated intentionally:

- update `src/foresight/serialization.py` schema constants if the on-disk payload changes
- keep `tests/test_serialization.py` and `tests/test_cli_forecast.py` green
- document whether users must re-save existing artifacts after upgrading

## 8) Publish (manual)

This step requires credentials and is intentionally not automated here.

```bash
twine upload dist/*
```

## Optional: GitHub Actions release

There is a `Release` workflow at `.github/workflows/release.yml` that can run the checks + build artifacts.

- To publish to **PyPI**, add a repository secret named `PYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=pypi`.
- To publish to **TestPyPI**, add a repository secret named `TESTPYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=testpypi`.

## Optional: GitHub Pages docs publish

There is a dedicated docs workflow at `.github/workflows/docs.yml`.

- It regenerates `docs/models.md`, `docs/api.md`, `docs/rnn_paper_zoo.md`, and `docs/rnn_zoo.md`
- It builds the site with `mkdocs build --strict`
- It deploys the generated `site/` output to GitHub Pages on pushes to `main`
