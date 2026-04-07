# Adapter Runtime Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend installed-artifact adapter smoke so explicit `darts` and
`gluonts` extra runs validate real adapter round-trips in addition to package
installation and import smoke.

**Architecture:** Reuse `tools/smoke_build_install.py` as the single real-pip
artifact smoke entrypoint. Add small extra-specific runtime smoke commands for
`darts` and `gluonts`, lock them with release-tooling tests, and keep the
commands explicit in `docs/RELEASE.md`.

**Tech Stack:** Python 3.10, pandas, pytest, setuptools extras, existing
adapter bundle APIs, existing release smoke tooling

---

### Task 1: Add Failing Release-Tooling Tests For Adapter Runtime Smoke

**Files:**
- Modify: `tests/test_release_tooling.py`

**Step 1: Add failing assertions for Darts runtime smoke**

Expect `tools/smoke_build_install.py` to contain a Darts adapter runtime smoke
command that references:

- `to_darts_bundle`
- `from_darts_bundle`

**Step 2: Add failing assertions for GluonTS runtime smoke**

Expect `tools/smoke_build_install.py` to contain a GluonTS adapter runtime smoke
command that references:

- `to_gluonts_bundle`
- `from_gluonts_bundle`

**Step 3: Add failing release-doc assertions**

Expect `docs/RELEASE.md` to include:

- `python tools/smoke_build_install.py --sdist --require-extra darts`
- `python tools/smoke_build_install.py --sdist --require-extra gluonts`

**Step 4: Run targeted tests to verify RED**

Run:

```bash
pytest -q tests/test_release_tooling.py -k "darts or gluonts or smoke_build_install"
```

Expected: FAIL because runtime smoke commands are not present yet.

**Step 5: Commit**

```bash
git add tests/test_release_tooling.py
git commit -m "test: define adapter runtime smoke coverage"
```

### Task 2: Implement Darts And GluonTS Runtime Smoke Commands

**Files:**
- Modify: `tools/smoke_build_install.py`

**Step 1: Add Darts runtime smoke command**

Create a small installed-package Python command that:

- imports `darts`
- calls `to_darts_bundle(...)`
- calls `from_darts_bundle(...)`
- asserts the minimal round-trip contract

**Step 2: Add GluonTS runtime smoke command**

Create a small installed-package Python command that:

- imports `gluonts`
- calls `to_gluonts_bundle(...)`
- calls `from_gluonts_bundle(...)`
- asserts the minimal round-trip contract

**Step 3: Hook commands into explicit extra smoke**

When `--require-extra darts` or `--require-extra gluonts` is passed, run the
corresponding runtime smoke command after import smoke and
`doctor --require-extra`.

**Step 4: Run targeted tests**

Run:

```bash
pytest -q tests/test_release_tooling.py -k "darts or gluonts or smoke_build_install"
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/smoke_build_install.py tests/test_release_tooling.py
git commit -m "fix: add adapter runtime smoke commands"
```

### Task 3: Update Release Docs

**Files:**
- Modify: `docs/RELEASE.md`

**Step 1: Keep explicit runtime smoke commands documented**

Ensure the release checklist clearly includes the explicit Darts and GluonTS
artifact smoke commands.

**Step 2: Run targeted tests**

Run:

```bash
pytest -q tests/test_release_tooling.py -k release_docs_cover_docs_site_and_benchmark_smoke
```

Expected: PASS

**Step 3: Commit**

```bash
git add docs/RELEASE.md tests/test_release_tooling.py
git commit -m "docs: document adapter runtime smoke"
```

### Task 4: Run Final Verification

**Files:**
- Verify: `tools/smoke_build_install.py`
- Verify: `docs/RELEASE.md`
- Verify: `tests/test_release_tooling.py`

**Step 1: Run repo-local verification**

Run:

```bash
pytest -q tests/test_release_tooling.py tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py
```

Expected: PASS

**Step 2: Run real-pip Darts smoke**

Run:

```bash
python3 tools/smoke_build_install.py --sdist --require-extra darts
```

Expected: PASS

**Step 3: Run real-pip GluonTS smoke**

Run:

```bash
python3 tools/smoke_build_install.py --sdist --require-extra gluonts
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs tools tests
git commit -m "test: verify adapter runtime smoke batch"
```
