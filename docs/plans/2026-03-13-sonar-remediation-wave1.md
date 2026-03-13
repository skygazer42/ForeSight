# Sonar Remediation Wave 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CI-based Sonar analysis for branches and pull requests, and harden the docs deployment workflow permissions to address the current GitHub Actions vulnerability findings.

**Architecture:** Keep the existing CI structure intact and add the smallest set of repository-level artifacts needed for SonarCloud to analyze this project consistently. The implementation should preserve current build/test behavior, add regression tests for workflow expectations, and move write-level GitHub token permissions from workflow scope to the narrowest job scope that still allows Pages deployment.

**Tech Stack:** GitHub Actions, SonarCloud, Python 3.10, pytest

---

## Constraints

- Preserve the current CI jobs and their responsibilities.
- Do not require local-only secrets or machine-specific paths in committed workflow files.
- Keep Sonar setup compatible with branch and pull request analysis.
- Fix the docs workflow vulnerability without weakening Pages deploy behavior.
- Use TDD for repository policy changes by asserting workflow contents in tests first.

### Task 1: Lock In Workflow Expectations With Tests

**Files:**
- Modify: `tests/test_release_tooling.py`

**Step 1: Write failing workflow tests**

Add coverage for these repository guarantees:

- a Sonar workflow or Sonar CI job exists and references the SonarCloud project key
- Sonar analysis uses full git history checkout
- docs workflow no longer grants `pages: write` or `id-token: write` at workflow scope
- docs deploy job grants the minimum write permissions needed for GitHub Pages deploy

**Step 2: Run the targeted release-tooling tests**

Run: `PYTHONPATH=src pytest -q tests/test_release_tooling.py`
Expected: FAIL because the Sonar workflow does not exist yet and docs permissions are still scoped at workflow level

### Task 2: Add CI-Based Sonar Analysis

**Files:**
- Create: `sonar-project.properties`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_release_tooling.py`

**Step 1: Add the minimal Sonar project configuration**

Declare:

- `sonar.projectKey=skygazer42_ForeSight`
- `sonar.organization=skygazer42`
- source/test roots aligned with the Python package layout
- coverage report path for CI-generated XML coverage

**Step 2: Add a dedicated Sonar job to CI**

The job should:

- run on `push` and `pull_request` through the existing CI workflow
- use `actions/checkout` with `fetch-depth: 0`
- install project dev dependencies
- run a targeted coverage-producing pytest slice
- invoke the official Sonar scan action with the repository token secret

**Step 3: Re-run the release-tooling tests**

Run: `PYTHONPATH=src pytest -q tests/test_release_tooling.py`
Expected: PASS for Sonar workflow assertions

### Task 3: Harden Docs Workflow Permissions

**Files:**
- Modify: `.github/workflows/docs.yml`
- Modify: `tests/test_release_tooling.py`

**Step 1: Move write permissions down to job scope**

Required behavior:

- workflow scope should not hold `pages: write` or `id-token: write`
- build job should keep only the read permissions it needs
- deploy job should hold `pages: write` and `id-token: write`

**Step 2: Re-run the targeted release-tooling tests**

Run: `PYTHONPATH=src pytest -q tests/test_release_tooling.py`
Expected: PASS

### Task 4: Verify The Touched Surfaces

**Files:**
- Verify: `.github/workflows/ci.yml`
- Verify: `.github/workflows/docs.yml`
- Verify: `sonar-project.properties`
- Verify: `tests/test_release_tooling.py`

**Step 1: Run focused checks**

Run:

- `PYTHONPATH=src pytest -q tests/test_release_tooling.py`
- `ruff check tests/test_release_tooling.py`

Expected: PASS

**Step 2: Run broader workflow regression checks**

Run:

- `PYTHONPATH=src pytest -q tests/test_release_tooling.py tests/test_docs_rnn_generated.py`

Expected: PASS
