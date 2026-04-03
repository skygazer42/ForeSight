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
python -m pytest -q tests/test_public_contract.py
python benchmarks/run_benchmarks.py --smoke
python tools/smoke_build_install.py --sdist
mkdocs build --strict
```

This contract suite is the release gate for the stable public surface:

- root `foresight` exports
- `models info` support metadata
- `doctor` machine-readable payload shape
- artifact schema compatibility
- support-policy docs

## 4) Run release checks

```bash
python tools/release_check.py
```

## 5) Build artifacts

```bash
python -m build
```

If `dist/` already contains older builds, remove them before publishing or use
version-scoped commands in the next steps so you only validate and upload the
current release artifacts.

## 6) Validate artifacts

```bash
python -m twine check dist/foresight_ts-<version>*
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
python -m twine upload dist/foresight_ts-<version>*
```

## Optional: GitHub Actions release

There is a `Release` workflow at `.github/workflows/release.yml` that can run the checks + build artifacts.

- To publish to **PyPI**, add a repository secret named `PYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=pypi`.
- To publish to **TestPyPI**, add a repository secret named `TESTPYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=testpypi`.

Before publishing, you can also sanity-check the installed artifact environment directly:

```bash
foresight doctor
python -m foresight doctor
python -m foresight doctor --format text
python -m foresight --data-dir /path/to/root doctor --format text --strict
python -m foresight doctor --require-extra torch --strict
```

Use `doctor --format text` for a human-readable release sanity check, and use
`doctor --strict` when you want warnings to fail the release checklist with exit
code `1`. Use `doctor --require-extra <extra>` when the release target must
include a specific optional backend.

The release checklist should stay aligned with the CI-backed support matrix in
`docs/compatibility.md`; do not publish a support claim that is not enforced by
the contract suite, packaging smoke, and docs build.

## Optional: GitHub Pages docs publish

There is a dedicated docs workflow at `.github/workflows/docs.yml`.

- It regenerates `docs/models.md`, `docs/api.md`, `docs/rnn_paper_zoo.md`, and `docs/rnn_zoo.md`
- It builds the site with `mkdocs build --strict`
- It deploys the generated `site/` output to GitHub Pages on pushes to `main`
