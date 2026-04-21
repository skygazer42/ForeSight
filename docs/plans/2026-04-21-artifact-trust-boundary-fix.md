# Artifact Trust Boundary Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align ForeSight artifact docs and CLI help with the real pickle-backed trust boundary, then lock that behavior with regression tests.

**Architecture:** Keep the patch narrow. Tests define the corrected public contract first, then documentation and the top-level artifact parser are updated to match. No serialization behavior changes are included in this plan.

**Tech Stack:** Python, argparse, pytest, Markdown docs

---

### Task 1: Lock the public trust-boundary contract in tests

**Files:**
- Modify: `tests/test_cli_artifact_info.py`
- Modify: `tests/test_release_tooling.py`

**Step 1: Write the failing tests**

- Add a test that checks `docs/api.md` and `docs/api-reference/index.md` for wording that reflects pickle-backed loading rather than metadata-only inspection.
- Add a test that checks `docs/guide/artifacts.md` and the `artifact diff` section in `docs/cli/index.md` for the trusted-source warning.
- Add a test that checks `foresight artifact --help` for the trusted-source warning.
- Replace the hard-coded absolute-link assertion with a generalized workspace absolute-link guard.

**Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest -q tests/test_cli_artifact_info.py tests/test_release_tooling.py
```

Expected: failures in the new trust-boundary assertions and CLI top-level help assertion.

### Task 2: Apply the minimal CLI and documentation fixes

**Files:**
- Modify: `src/foresight/cli.py`
- Modify: `docs/api.md`
- Modify: `docs/api-reference/index.md`
- Modify: `docs/guide/artifacts.md`
- Modify: `docs/cli/index.md`

**Step 1: Write the minimal implementation**

- Add a trusted-source description to the top-level `artifact` parser.
- Rewrite the public API wording for `load_forecaster_artifact(...)` so it states that the full artifact payload is loaded from a pickle-backed file and should only be used with trusted artifacts.
- Add matching warnings to the guide sections for artifact inspection and CLI management.
- Add the missing trusted-source warning to the `artifact diff` command reference section.

**Step 2: Run tests to verify they pass**

Run:

```bash
python3 -m pytest -q tests/test_cli_artifact_info.py tests/test_release_tooling.py
```

Expected: PASS.

### Task 3: Run focused quality verification

**Files:**
- Modify: none

**Step 1: Run targeted checks**

Run:

```bash
ruff check src/foresight/cli.py tests/test_cli_artifact_info.py tests/test_release_tooling.py
ruff format --check src/foresight/cli.py tests/test_cli_artifact_info.py tests/test_release_tooling.py
python3 -m pytest -q tests/test_cli_artifact_info.py tests/test_release_tooling.py tests/test_serialization.py tests/test_public_contract.py
```

Expected: PASS.
