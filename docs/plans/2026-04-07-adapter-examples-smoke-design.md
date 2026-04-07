# Adapter Examples And Smoke Design

**Goal:** Make the existing `foresight.adapters` beta surfaces easier to install, understand, and validate by adding focused examples, adapter extra installation docs, and real-pip release smoke coverage.

## Summary

ForeSight now exposes a useful beta adapter surface for shared richer bundles,
Darts richer bundles, GluonTS richer bundles, and the sktime local forecaster
bridge. The remaining gap is product usability rather than missing API
coverage:

- users do not yet have focused adapter examples in `examples/`
- the installation docs do not list the adapter extras alongside the other
  package extras
- the release smoke path verifies root imports and `foresight.adapters`, but it
  does not yet verify adapter extra installation through the existing real-pip
  artifact flow

This batch closes that gap without expanding the beta API surface.

## Design Decisions

### 1. Keep the scope on usability, not new adapter features

Do not add:

- new predictor wrappers
- new bundle schema fields
- new stable root-package exports
- new adapter semantics

This batch packages the current adapter beta surface more clearly instead of
making it larger.

### 2. Add one example file per adapter surface

Create focused examples in `examples/`:

- `adapters_shared_bundle.py`
- `adapters_sktime.py`
- `adapters_darts.py`
- `adapters_gluonts.py`

Each example should stay small, use inline demo data, and import only the
named public adapter helpers from `foresight.adapters`.

### 3. Unify adapter docs around install, shape, and minimal usage

`docs/adapters.md` should document each adapter section in the same order:

- install command
- intended use
- minimal input shape
- minimal example
- resulting output shape or round-trip behavior

`docs/getting-started/installation.md` should add `sktime`, `darts`, and
`gluonts` extras to the installation matrix so adapter-specific installation is
visible where users expect it.

### 4. Reuse the existing release smoke tool

Do not create a second install smoke script.

Extend `tools/smoke_build_install.py` so it can optionally install built
artifacts with requested extras and then verify:

- the extra package import succeeds
- `foresight.adapters` still imports
- `foresight doctor --require-extra <name>` succeeds

This keeps all real-pip artifact validation on the same release path the
project already trusts.

### 5. Keep adapter extras smoke opt-in

The default smoke path should stay lightweight. Adapter extra validation should
only run when the caller explicitly asks for it, for example with repeated
`--require-extra` flags. This avoids forcing heavy Darts/GluonTS installs into
every default smoke run.

## Test Strategy

Add or update tests for:

- adapter example scripts using the public `foresight.adapters` surface
- adapter docs covering shared bundle, sktime, Darts, and GluonTS examples
- installation docs listing adapter extras
- release docs and smoke tooling covering adapter extra validation

Regression requirements:

- current adapter pytest files remain green
- release tooling tests remain green
- docs site build remains green
- real-pip smoke can validate requested adapter extras from built artifacts
