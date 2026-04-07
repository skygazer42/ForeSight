# Adapter Examples And Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add focused adapter examples, adapter extra installation docs, and opt-in real-pip adapter smoke validation without changing the current adapter feature surface.

**Architecture:** Keep examples and docs aligned around the public `foresight.adapters` beta surface, then extend the existing `tools/smoke_build_install.py` workflow so requested extras are installed from built artifacts and validated with adapter imports plus `foresight doctor --require-extra`.

**Tech Stack:** Python 3.10, pandas, pytest, MkDocs, setuptools extras, existing release smoke tooling

---

### Task 1: Add Failing Static Coverage For Adapter Examples And Docs

**Files:**
- Create: `tests/test_adapters_examples_docs.py`
- Modify: `tests/test_release_tooling.py`

**Step 1: Add failing tests for adapter examples**

Expect these files to exist:

- `examples/adapters_shared_bundle.py`
- `examples/adapters_sktime.py`
- `examples/adapters_darts.py`
- `examples/adapters_gluonts.py`

Expect them to import from `foresight.adapters`, not from adapter implementation
modules.

**Step 2: Add failing tests for adapter docs coverage**

Expect:

- `docs/adapters.md` to contain the new example file names or their public API
  calls
- `docs/getting-started/installation.md` to list `sktime`, `darts`, and
  `gluonts` extras

**Step 3: Add failing release-tooling assertions**

Expect:

- `docs/RELEASE.md` to mention adapter extra smoke invocations
- `tools/smoke_build_install.py` to support repeated `--require-extra`
- the smoke script to call `foresight doctor --require-extra <name>`

**Step 4: Run targeted tests to verify RED**

Run:

```bash
pytest -q tests/test_adapters_examples_docs.py tests/test_release_tooling.py -k adapter
```

Expected: FAIL because the new example files and release smoke support do not
exist yet.

**Step 5: Commit**

```bash
git add tests/test_adapters_examples_docs.py tests/test_release_tooling.py
git commit -m "test: define adapter examples and smoke coverage"
```

### Task 2: Add Adapter Example Scripts And Docs

**Files:**
- Create: `examples/adapters_shared_bundle.py`
- Create: `examples/adapters_sktime.py`
- Create: `examples/adapters_darts.py`
- Create: `examples/adapters_gluonts.py`
- Modify: `docs/adapters.md`
- Modify: `docs/getting-started/installation.md`

**Step 1: Add focused adapter example scripts**

Create one small script per adapter surface using inline demo data and public
imports from `foresight.adapters`.

**Step 2: Update adapter docs**

For shared bundle, sktime, Darts, and GluonTS sections, add:

- install command
- minimal example
- short output/round-trip explanation

**Step 3: Update installation docs**

Add `sktime`, `darts`, and `gluonts` extras to the installation matrix and add
a short adapter-install note.

**Step 4: Run targeted tests**

Run:

```bash
pytest -q tests/test_adapters_examples_docs.py tests/test_cli_eval.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add examples/adapters_shared_bundle.py examples/adapters_sktime.py examples/adapters_darts.py examples/adapters_gluonts.py docs/adapters.md docs/getting-started/installation.md tests/test_adapters_examples_docs.py tests/test_cli_eval.py
git commit -m "docs: add adapter examples"
```

### Task 3: Extend Real-Pip Smoke For Adapter Extras

**Files:**
- Modify: `tools/smoke_build_install.py`
- Modify: `docs/RELEASE.md`
- Modify: `tests/test_release_tooling.py`

**Step 1: Add repeated `--require-extra` support**

Let callers request zero or more extras such as:

```bash
python tools/smoke_build_install.py --sdist --require-extra sktime --require-extra darts
```

**Step 2: Install artifacts with requested extras**

When extras are requested, install the built wheel/sdist using the package
specifier with extras and then validate the corresponding adapter import plus
`foresight doctor --require-extra <name>`.

**Step 3: Update release docs**

Document the adapter extra smoke commands in `docs/RELEASE.md`.

**Step 4: Run targeted tests**

Run:

```bash
pytest -q tests/test_release_tooling.py -k smoke_build_install
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/smoke_build_install.py docs/RELEASE.md tests/test_release_tooling.py
git commit -m "fix: add adapter extra release smoke"
```

### Task 4: Run Final Verification

**Files:**
- Verify: `examples/adapters_shared_bundle.py`
- Verify: `examples/adapters_sktime.py`
- Verify: `examples/adapters_darts.py`
- Verify: `examples/adapters_gluonts.py`
- Verify: `docs/adapters.md`
- Verify: `docs/getting-started/installation.md`
- Verify: `tools/smoke_build_install.py`
- Verify: `docs/RELEASE.md`
- Verify: `tests/test_adapters_examples_docs.py`
- Verify: `tests/test_release_tooling.py`

**Step 1: Run focused verification**

Run:

```bash
pytest -q tests/test_adapters_public_surface.py tests/test_adapters_darts_gluonts.py tests/test_adapters_sktime.py tests/test_adapters_examples_docs.py tests/test_release_tooling.py tests/test_cli_eval.py
```

Expected: PASS

**Step 2: Build docs**

Run:

```bash
mkdocs build --strict
```

Expected: PASS

**Step 3: Run real-pip adapter smoke**

Run:

```bash
python tools/smoke_build_install.py --sdist --require-extra sktime
```

Expected: PASS

**Step 4: Commit**

```bash
git add examples docs tools tests
git commit -m "test: verify adapter examples smoke batch"
```
