# Adapter Runtime Smoke Design

**Goal:** Extend real-pip adapter smoke coverage so installed-artifact validation
also exercises the Darts and GluonTS adapter runtime paths, not just extra
installation and import visibility.

## Summary

ForeSight now has:

- focused adapter examples
- adapter extra installation docs
- real-pip artifact smoke for `--require-extra sktime`

The remaining gap is that Darts and GluonTS are only validated at install/import
level. That proves the extras resolve, but it does not prove the installed
adapter functions can execute a minimal round-trip on real package artifacts.

This batch closes that gap by adding installed-package runtime smoke for the
existing Darts and GluonTS richer bundle APIs.

## Design Decisions

### 1. Keep this batch on runtime validation, not new APIs

Do not add:

- new adapter exports
- new bundle schema fields
- Darts model wrappers
- GluonTS predictor/trainer wrappers

This batch only strengthens release confidence for the APIs already shipped.

### 2. Keep release-gate commands explicit

Do not add Darts or GluonTS runtime smoke to the default lightweight
`python tools/smoke_build_install.py --sdist` path.

Instead, continue to expose explicit commands:

- `python3 tools/smoke_build_install.py --sdist --require-extra darts`
- `python3 tools/smoke_build_install.py --sdist --require-extra gluonts`

That preserves operator control over heavier extra installs.

### 3. Validate adapter runtime with minimal bundle round-trips

For `darts`:

- import `darts`
- construct a tiny canonical long DataFrame
- call `to_darts_bundle(...)`
- call `from_darts_bundle(...)`
- assert the returned payload has `target` / `freq`
- assert the restored frame has canonical columns

For `gluonts`:

- import `gluonts`
- construct a tiny canonical long DataFrame
- call `to_gluonts_bundle(...)`
- call `from_gluonts_bundle(...)`
- assert the returned payload has `target` / `freq`
- assert the restored frame has canonical columns

These checks are intentionally data-only and avoid training or external data
fetching.

### 4. Extend the existing smoke tool with small extra-specific commands

Reuse `tools/smoke_build_install.py`.

Add a narrow extra-to-command mapping that supports:

- import smoke commands
- installed adapter runtime smoke commands

Do not introduce a large generic command registry beyond what is needed for
`sktime`, `darts`, and `gluonts`.

### 5. Lock the new release contract statically

Update `tests/test_release_tooling.py` and `docs/RELEASE.md` so the Darts and
GluonTS runtime smoke commands stay present and auditable.

## Test Strategy

Add or update tests for:

- Darts runtime smoke commands in `tools/smoke_build_install.py`
- GluonTS runtime smoke commands in `tools/smoke_build_install.py`
- explicit release checklist commands for both extras

Regression requirements:

- release-tooling tests remain green
- adapter public-surface tests remain green
- Darts/GluonTS adapter tests remain green
- real-pip smoke succeeds for:
  - `python3 tools/smoke_build_install.py --sdist --require-extra darts`
  - `python3 tools/smoke_build_install.py --sdist --require-extra gluonts`
