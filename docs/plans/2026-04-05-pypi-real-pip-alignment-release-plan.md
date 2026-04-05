# PyPI Real-Pip Alignment Release Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a new `foresight-ts` release whose installed PyPI/TestPyPI artifact matches the current `main` branch CLI and model-metadata contract.

**Architecture:** Treat package publication as a two-stage rollout: first prove the repo is releasable locally, then prove the uploaded artifact behaves correctly when installed from a package index with plain `pip` in fresh virtualenvs. This plan exists because the published `0.2.11` artifact was observed to drift from `main`, including a missing `doctor` CLI command and missing `ModelSpec.required_extra` metadata on installed objects.

**Tech Stack:** Python 3.10, pip, virtualenv, setuptools/build, twine, TestPyPI, PyPI, GitHub Actions release workflow

---

### Task 1: Freeze The Release Source And Choose A New Version

**Files:**
- Modify: `src/foresight/__init__.py`
- Check: `docs/RELEASE.md`
- Check: `.github/workflows/release.yml`

**Step 1: Confirm the release branch is clean and current**

Run: `git status --short --branch`
Expected: clean worktree on the branch you intend to release from.

**Step 2: Inspect the currently published PyPI version**

Run: `python3 -m pip index versions foresight-ts`
Expected: the latest published version is visible in the index response.

**Step 3: Choose the next version strictly greater than the published one**

Rule: never reuse a version that already exists on PyPI or TestPyPI, even if the repo already carries that version string.

**Step 4: Update the package version**

Modify `src/foresight/__init__.py` so `__version__` matches the new release version.

**Step 5: Commit the version bump before any artifact build**

```bash
git add src/foresight/__init__.py
git commit -m "chore: bump version for release"
```

### Task 2: Re-Run The Repo-Local Release Gate

**Files:**
- Check: `tools/release_check.py`
- Check: `tools/smoke_build_install.py`
- Check: `tests/test_release_tooling.py`
- Check: `docs/RELEASE.md`

**Step 1: Run release-tooling tests**

Run: `pytest -q tests/test_release_tooling.py`
Expected: PASS

**Step 2: Run the full release gate**

Run: `python3 tools/release_check.py`
Expected: final line `OK: release checks passed.`

**Step 3: Stop immediately if the release gate fails**

Rule: do not build or upload release artifacts until the full gate is green.

### Task 3: Build Versioned Artifacts And Validate Them

**Files:**
- Check: `pyproject.toml`
- Check: `MANIFEST.in`
- Check: `docs/RELEASE.md`

**Step 1: Build wheel and sdist**

Run: `python3 -m build`
Expected: both `dist/foresight_ts-<version>.tar.gz` and `dist/foresight_ts-<version>-py3-none-any.whl`

**Step 2: Validate the built artifacts**

Run: `python3 -m twine check dist/foresight_ts-<version>*`
Expected: both artifacts reported as `PASSED`

**Step 3: Keep the build outputs for index-install verification**

Rule: do not delete the built artifacts until both TestPyPI and PyPI verification rounds complete.

### Task 4: Publish The Candidate To TestPyPI First

**Files:**
- Check: `.github/workflows/release.yml`
- Check: `docs/RELEASE.md`

**Step 1: Upload to TestPyPI**

Use either:

```bash
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/foresight_ts-<version>*
```

or the `Release` GitHub Actions workflow with:

- `publish=true`
- `repository=testpypi`

**Step 2: Record the exact uploaded version**

Expected: one immutable TestPyPI version string you will install in the next tasks.

### Task 5: Verify The Core Contract From TestPyPI With Plain Pip

**Files:**
- Check: `src/foresight/cli.py`
- Check: `src/foresight/models/specs.py`
- Check: `docs/INSTALL.md`
- Check: `docs/RELEASE.md`

**Step 1: Create a fresh virtualenv**

```bash
python3 -m virtualenv /tmp/foresight-testpypi-core
/tmp/foresight-testpypi-core/bin/python -m pip install --upgrade pip
```

**Step 2: Install the exact candidate from TestPyPI using pip**

```bash
/tmp/foresight-testpypi-core/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  foresight-ts==<version>
```

**Step 3: Verify the installed CLI surface**

Run:

```bash
/tmp/foresight-testpypi-core/bin/python -m foresight --help
/tmp/foresight-testpypi-core/bin/python -m foresight doctor
/tmp/foresight-testpypi-core/bin/python -m foresight doctor --format text
```

Expected:

- `doctor` appears in `--help`
- `doctor` returns exit code `0`
- JSON and text output both render successfully

**Step 4: Verify the installed model metadata contract**

Run:

```bash
/tmp/foresight-testpypi-core/bin/python -m foresight models info moment
```

Expected:

- response includes `stability`
- `stability` equals `experimental`
- response includes `required_extra`
- `required_extra` equals `core`

**Step 5: Verify the dataset-resolution contract**

Run a short Python snippet that calls:

- `load_store_sales(nrows=5)`
- `load_promotion_data(nrows=5)`
- `load_cashflow_data(nrows=5)`

Expected: each call raises `FileNotFoundError` with a message instructing the user to provide `--data-dir` or `FORESIGHT_DATA_DIR`.

**Step 6: Verify the `moment` wrapper execution contract**

Run a short Python snippet that:

- confirms `make_forecaster("moment")(..., horizon=3)` raises `ValueError`
- confirms `make_forecaster("moment", backend="fixture-json", checkpoint_path=...)` returns a finite `(3,)` forecast

Expected: default failure is explicit; fixture-backed inference succeeds.

### Task 6: Verify Optional Extras From TestPyPI With Plain Pip

**Files:**
- Check: `docs/INSTALL.md`
- Check: `docs/compatibility.md`
- Check: `docs/RELEASE.md`

**Step 1: Create a second fresh virtualenv**

```bash
python3 -m virtualenv /tmp/foresight-testpypi-ml-stats
/tmp/foresight-testpypi-ml-stats/bin/python -m pip install --upgrade pip
```

**Step 2: Install the exact candidate plus extras from TestPyPI**

```bash
/tmp/foresight-testpypi-ml-stats/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  'foresight-ts[ml,stats]==<version>'
```

**Step 3: Verify representative extra-backed models**

Run a short Python snippet that builds finite `(3,)` forecasts for:

- `linear-svr-lag`
- `ets`
- `sarimax`

Expected: all three models return finite predictions without changing default params.

### Task 7: Publish To PyPI Only After TestPyPI Passes

**Files:**
- Check: `.github/workflows/release.yml`
- Check: `docs/RELEASE.md`

**Step 1: Stop if any TestPyPI contract check failed**

Rule: do not publish to PyPI until the index-installed checks pass exactly as written above.

**Step 2: Publish to PyPI**

Use either:

```bash
python3 -m twine upload dist/foresight_ts-<version>*
```

or the `Release` GitHub Actions workflow with:

- `publish=true`
- `repository=pypi`

### Task 8: Re-Run The Installed-Package Contract Against PyPI

**Files:**
- Check: `docs/RELEASE.md`
- Check: `docs/compatibility.md`

**Step 1: Repeat the core TestPyPI install checks against `https://pypi.org/simple`**

Expected: same successful CLI, dataset-contract, and `moment`-contract results as the TestPyPI candidate.

**Step 2: Repeat the `ml+stats` extra install checks against `https://pypi.org/simple`**

Expected: same successful extra-model smoke results as the TestPyPI candidate.

**Step 3: Record the evidence**

Capture:

- installed package version
- installed module path
- pass/fail status for the core contract checks
- pass/fail status for the extra-model checks
- any deviations from `main`

**Step 4: Announce the release only after the PyPI-installed checks pass**

Rule: the release is not complete when the upload finishes; it is complete when the PyPI-installed artifact matches the intended contract.
