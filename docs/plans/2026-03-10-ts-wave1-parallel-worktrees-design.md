# Time-Series Wave 1 Parallel Worktrees Design

**Date:** 2026-03-10

**Goal:** Expand ForeSight with eight parallel delivery lanes for missing time-series algorithm families while keeping merge risk controlled through isolated branches, isolated worktrees, and strict ownership boundaries.

**Scope:** This design covers branch topology, worktree layout, lane ownership, merge policy, verification policy, and documentation policy for the first parallel wave.

---

## 1. Context

ForeSight already has a large registered model surface, but the current roadmap still contains several missing or only partially covered algorithm families. The highest-risk part of parallel implementation is not model code by itself. It is the shared integration surface:

- `src/foresight/models/registry.py`
- generated docs such as `docs/models.md`
- top-level product docs such as `README.md`
- any shared wrapper or graph helper used by more than one family

The design therefore prioritizes isolation first, then controlled convergence.

---

## 2. Topology

### Baseline branch

- Source baseline: `main`

### Integration branch

- Integration branch: `feat/ts-wave1-integration`
- Purpose:
  - hold the approved design and implementation plan
  - hold integration-only scaffolding
  - receive the eight feature branches
  - run final verification and generated docs refresh before any merge upstream

### Worktree root

- Worktree directory: `.worktrees/`
- Reason:
  - project-local
  - already ignored by git in this repository
  - easy to inspect and clean up

### Feature branches and worktrees

Each lane gets one feature branch and one dedicated worktree:

| Lane | Branch | Worktree | Family |
| --- | --- | --- | --- |
| 01 | `feat/ts-lane-01-reservoir-lite` | `.worktrees/ts-lane-01-reservoir-lite` | ESN / reservoir / liquid-state lite |
| 02 | `feat/ts-lane-02-structured-rnn-lite` | `.worktrees/ts-lane-02-structured-rnn-lite` | structured / grid recurrent lite |
| 03 | `feat/ts-lane-03-graph-attention-lite` | `.worktrees/ts-lane-03-graph-attention-lite` | ASTGCN / GMAN style lite |
| 04 | `feat/ts-lane-04-graph-structure-lite` | `.worktrees/ts-lane-04-graph-structure-lite` | AGCRN / MTGNN style lite |
| 05 | `feat/ts-lane-05-graph-spectral-lite` | `.worktrees/ts-lane-05-graph-spectral-lite` | StemGNN / FourierGNN lite |
| 06 | `feat/ts-lane-06-probabilistic-native-lite` | `.worktrees/ts-lane-06-probabilistic-native-lite` | TimeGrad / TACTiS lite |
| 07 | `feat/ts-lane-07-foundation-wrappers-a` | `.worktrees/ts-lane-07-foundation-wrappers-a` | Lag-Llama / Chronos / Chronos-Bolt / TimesFM |
| 08 | `feat/ts-lane-08-foundation-wrappers-b` | `.worktrees/ts-lane-08-foundation-wrappers-b` | Moirai / MOMENT / Time-MoE / Timer-S1 |

All eight feature branches must be created from `feat/ts-wave1-integration`, not from `main`.

---

## 3. Existing Coverage Adjustments

The current roadmap predates some code already present in the repository. The following families already exist and are not part of this wave:

- FiLM
- MICN
- Koopa
- TimeXer
- SAMformer
- LMU
- LTC
- CfC
- S4
- S4D
- S5
- Mamba-2
- xLSTM
- Griffin
- Hawk
- STID
- STGCN
- Graph WaveNet

Wave 1 should target genuinely missing families rather than duplicating existing implementations.

---

## 4. Lane Ownership Boundaries

### Lane 01: reservoir lite

- Owns: `src/foresight/models/torch_reservoir.py`
- Owns tests:
  - `tests/test_models_torch_reservoir_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`

### Lane 02: structured recurrent lite

- Owns: `src/foresight/models/torch_structured_rnn.py`
- Owns tests:
  - `tests/test_models_torch_structured_rnn_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`

### Lane 03: graph attention lite

- Owns: `src/foresight/models/torch_graph_attention.py`
- Owns tests:
  - `tests/test_models_graph_attention_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`

### Lane 04: graph structure lite

- Owns: `src/foresight/models/torch_graph_structure.py`
- Owns tests:
  - `tests/test_models_graph_structure_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`

### Lane 05: graph spectral lite

- Owns: `src/foresight/models/torch_graph_spectral.py`
- Owns tests:
  - `tests/test_models_graph_spectral_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`

### Lane 06: probabilistic native lite

- Owns: `src/foresight/models/torch_probabilistic.py`
- Owns tests:
  - `tests/test_models_probabilistic_smoke.py`
- Allowed shared edits:
  - one import block in `src/foresight/models/__init__.py`
  - one isolated registry block in `src/foresight/models/registry.py`
  - minimal forecast output plumbing in `src/foresight/forecast.py`

### Lane 07: foundation wrappers A

- Owns wrapper families:
  - Lag-Llama
  - Chronos
  - Chronos-Bolt
  - TimesFM
- Owns tests:
  - `tests/test_models_foundation_smoke.py`
- Allowed shared edits:
  - wrapper-specific helpers in `src/foresight/models/foundation.py`
  - registry entries for lane 07 models
  - minimal inference plumbing in `src/foresight/forecast.py`
  - artifact or input helpers in `src/foresight/io.py` only when required

### Lane 08: foundation wrappers B

- Owns wrapper families:
  - Moirai
  - MOMENT
  - Time-MoE
  - Timer-S1
- Owns tests:
  - `tests/test_models_foundation_smoke.py`
- Allowed shared edits:
  - wrapper-specific helpers in `src/foresight/models/foundation.py`
  - registry entries for lane 08 models
  - minimal inference plumbing in `src/foresight/forecast.py`
  - artifact or input helpers in `src/foresight/io.py` only when required

---

## 5. Shared-File Policy

Feature lanes must not directly update:

- `README.md`
- `docs/models.md`
- `docs/api.md`
- `docs/index.md`
- roadmap status tables

Feature lanes may only touch `src/foresight/models/registry.py` in pre-marked family sections and only for:

- imports
- factories
- `ModelSpec` entries for their own models

If two or more lanes discover a missing common helper, that helper must first be promoted to the integration branch as a dedicated shared scaffold commit before the lanes continue.

---

## 6. Integration Scaffolding Before Parallel Work

Before opening the lanes for implementation, the integration branch should add minimal shared scaffolding:

- empty module shells for each new family area
- empty smoke-test shells for each lane
- explicit family anchors in `src/foresight/models/registry.py`
- minimal common helper shell in `src/foresight/models/foundation.py`

This scaffolding is not a user-facing feature. Its purpose is to reduce merge conflicts and make ownership explicit.

---

## 7. Merge Strategy

All feature branches merge into `feat/ts-wave1-integration` only.

Recommended merge order:

1. lane 01 reservoir lite
2. lane 02 structured recurrent lite
3. lane 03 graph attention lite
4. lane 04 graph structure lite
5. lane 05 graph spectral lite
6. lane 06 probabilistic native lite
7. lane 07 foundation wrappers A
8. lane 08 foundation wrappers B

Rationale:

- local low-coupling lanes go first
- graph lanes next because they share the multivariate surface
- probabilistic native after graph because it can affect forecast output semantics
- wrapper lanes last because they share checkpoint-facing plumbing

---

## 8. Verification Policy

### Per-lane merge gate

Each lane must pass:

- targeted registry assertions
- targeted smoke tests
- relevant optional-dependency coverage
- `ruff check` on touched files

Each lane must also keep its scope honest:

- no fake support claims
- no generated docs updates
- no README edits
- no unrelated cleanup

### Integration verification cadence

On the integration branch:

1. after each lane merge, run lane-specific tests
2. after each family cluster, run the relevant broader suite
3. after all merges, run:
   - `ruff check src tests`
   - targeted registry and smoke suites
   - wrapper and forecast API checks where needed
   - `PYTHONPATH=src python tools/generate_model_capability_docs.py`
   - `PYTHONPATH=src python tools/check_capability_docs.py`

---

## 9. Documentation Policy

Generated and product-facing docs are integration-only work:

- update roadmap statuses only after a family has landed
- regenerate `docs/models.md` and `docs/api.md` only on the integration branch
- refresh `README.md` only after the code and tests are green

This prevents eight lanes from fighting over the same generated outputs.

---

## 10. Completion Criteria

Wave 1 is complete when:

- all eight feature branches exist with dedicated worktrees
- all eight lanes have delivered their assigned families or honest minimal wrappers
- every lane has tests
- all lane work has been merged into `feat/ts-wave1-integration`
- integration verification is green
- generated docs are refreshed and validated
- the roadmap reflects the actual landed state

Until those conditions are met, the work remains in the integration branch.
