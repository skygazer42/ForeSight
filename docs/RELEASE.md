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
python tools/generate_rnn_docs.py
```

## 3) Run release checks

```bash
python tools/release_check.py
```

## 4) Build artifacts

```bash
python -m build
```

## 5) Validate artifacts

```bash
twine check dist/*
```

## 6) Publish (manual)

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
