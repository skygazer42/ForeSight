# Release

This is the release checklist for publishing `foresight-ts` as a pip-installable package.

Do not treat repo-local tests as sufficient release proof. A release is only done
when a freshly uploaded artifact behaves correctly when installed from a package
index with plain `pip`.

## 1) Confirm the published baseline and bump version

`foresight-ts` uses a dynamic version sourced from `foresight.__version__`.

- Inspect the currently published version before touching artifacts:

```bash
python3 -m pip index versions foresight-ts
```

- Modify: `src/foresight/__init__.py:1`
- Rule: never reuse a version that already exists on PyPI or TestPyPI.
- Rule: if the published artifact has drifted from `main`, the fix still ships as
  a new version, not as a rebuild of the old one.

## 2) Regenerate derived docs (if needed)

```bash
# Optional: refresh paper metadata (hits public APIs)
python tools/fetch_rnn_paper_metadata.py --refresh --sleep 0.02

# Required when code changes affect the generated tables/line numbers
python tools/generate_model_capability_docs.py
python tools/generate_rnn_docs.py
```

## 3) Run the repo-local release gate

```bash
python -m pytest -q tests/test_public_contract.py
python benchmarks/run_benchmarks.py --smoke
python tools/smoke_build_install.py --sdist
mkdocs build --strict
python tools/release_check.py
```

This contract suite is the release gate for the stable public surface:

- root `foresight` exports
- `models info` support metadata
- `doctor` machine-readable payload shape
- artifact schema compatibility
- support-policy docs

## 4) Build artifacts

```bash
python -m build
```

If `dist/` already contains older builds, remove them before publishing or use
version-scoped commands in the next steps so you only validate and upload the
current release artifacts.

## 5) Validate artifacts

```bash
python -m twine check dist/foresight_ts-<version>*
```

## 6) Publish to TestPyPI first

This is the required dry run before any PyPI publish.

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/foresight_ts-<version>*
```

You can also use the `Release` workflow at `.github/workflows/release.yml` with:

- `publish=true`
- `repository=testpypi`

## 7) Verify the uploaded TestPyPI artifact with plain pip

Core install:

```bash
python3 -m virtualenv /tmp/foresight-testpypi-core
/tmp/foresight-testpypi-core/bin/python -m pip install --upgrade pip
/tmp/foresight-testpypi-core/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  foresight-ts==<version>
```

Required checks:

```bash
/tmp/foresight-testpypi-core/bin/python -m foresight --help
/tmp/foresight-testpypi-core/bin/python -m foresight doctor
/tmp/foresight-testpypi-core/bin/python -m foresight doctor --format text
/tmp/foresight-testpypi-core/bin/python -m foresight models info moment
```

Assert all of the following:

- `doctor` is present in `--help`
- `doctor` returns exit code `0`
- `models info moment` reports `stability=experimental`
- `models info moment` includes `required_extra=core`
- non-packaged datasets (`store_sales`, `promotion_data`, `cashflow_data`) fail
  with a helpful `FileNotFoundError` when no `data_dir` is supplied
- `moment` fails clearly without `checkpoint_path` / `model_source` and succeeds
  with `backend=fixture-json`

Optional-extra install:

```bash
python3 -m virtualenv /tmp/foresight-testpypi-ml-stats
/tmp/foresight-testpypi-ml-stats/bin/python -m pip install --upgrade pip
/tmp/foresight-testpypi-ml-stats/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  'foresight-ts[ml,stats]==<version>'
```

Required checks:

- `linear-svr-lag`
- `ets`
- `sarimax`

Each model should return a finite `(3,)` forecast in a fresh installed environment.

If any TestPyPI check fails, stop. Fix the code, bump the version again, and
repeat from the top.

## 8) Check serialization compatibility notes

If a release changes the saved forecaster artifact payload, make sure the
artifact schema/version guardrails are updated intentionally:

- update `src/foresight/serialization.py` schema constants if the on-disk payload changes
- keep `tests/test_serialization.py` and `tests/test_cli_forecast.py` green
- document whether users must re-save existing artifacts after upgrading

## 9) Publish to PyPI

Only publish after the TestPyPI installed-package checks pass.

```bash
python -m twine upload dist/foresight_ts-<version>*
```

You can also use the `Release` workflow with:

- `publish=true`
- `repository=pypi`

## 10) Re-verify the live PyPI artifact with plain pip

Repeat the same installed-package checks from Step 7, but install from
`https://pypi.org/simple` instead of TestPyPI.

The release is only complete when the PyPI-installed artifact matches the
expected CLI and model-metadata contract.

## Optional: GitHub Actions release

There is a `Release` workflow at `.github/workflows/release.yml` that can run the checks + build artifacts.

- To publish to **PyPI**, add a repository secret named `PYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=pypi`.
- To publish to **TestPyPI**, add a repository secret named `TESTPYPI_API_TOKEN`, then run the workflow with
  inputs `publish=true` and `repository=testpypi`.

Before publishing, you can also sanity-check the installed artifact environment directly:

```bash
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
the contract suite, packaging smoke, docs build, and real index-installed checks.

## Optional: GitHub Pages docs publish

There is a dedicated docs workflow at `.github/workflows/docs.yml`.

- It regenerates `docs/models.md`, `docs/api.md`, `docs/rnn_paper_zoo.md`, and `docs/rnn_zoo.md`
- It builds the site with `mkdocs build --strict`
- It deploys the generated `site/` output to GitHub Pages on pushes to `main`
