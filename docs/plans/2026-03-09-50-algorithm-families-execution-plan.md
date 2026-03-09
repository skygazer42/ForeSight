# Fifty Algorithm Families 2001-2026 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand ForeSight with a 50-cluster algorithm program spanning 2001-2026, organized first by year and then by model family, while preserving the current local/global registry architecture and testing discipline.

**Architecture:** Treat the 50-cluster program as three product surfaces rather than one monolith: API-compatible local/global forecasting families that fit the existing lag-window and panel interfaces, new data-contract families that require graph or richer multivariate inputs, and external-checkpoint families that need inference wrappers instead of native training loops. Use the roadmap in `docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md` as the single taxonomy source of truth, then execute the work in waves that keep year labels and family labels visible in docs, tests, and registry metadata.

**Tech Stack:** Python 3.10+, NumPy, Pandas, PyTorch, optional graph libraries as needed, pytest, ruff, MkDocs

---

## Scope Guardrails

- Keep the **50 tracked clusters** in the roadmap as the canonical backlog.
- Preserve both axes in docs:
  - year-first: 2001-2026
  - family-first: recurrent, state-space, graph, transformer/lightweight, probabilistic/foundation
- Do not try to land all 50 families in one branch.
- For each family, choose one of three implementation modes:
  - native local/direct
  - native global/panel
  - wrapper / inference-only integration
- Do not add fake support for graph, exogenous, or checkpoint-heavy models before the required data contracts exist.

---

### Task 1: Freeze The 50-Cluster Taxonomy And Status Surface

**Files:**
- Modify: `docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md`
- Modify: `README.md`
- Modify: `docs/models.md`

**Step 1: Keep the roadmap as the single backlog source**

Ensure the roadmap tracks all 50 clusters with:
- year
- family
- status
- rationale

**Step 2: Keep statuses current**

When a family lands in code, immediately update its roadmap status from `missing` or `partial` to `implemented`.

**Step 3: Keep docs discoverable**

Reflect new families in:
- model zoo tables in `README.md`
- generated registry capability docs in `docs/models.md`

**Step 4: Verify**

Run:

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src python tools/check_capability_docs.py
```

Expected: generated docs stay in sync with the registry.

---

### Task 2: Finish The API-Compatible Sequence Families

**Files:**
- Modify: `src/foresight/models/torch_nn.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_optional_deps_torch.py`
- Test: `tests/test_models_torch_xformer_seq2seq_smoke.py`

**Target clusters:**
- #32 FiLM
- #34 MICN
- #35 Koopa
- #38 TimeXer
- #39 SAMformer

**Step 1: Add failing tests one family at a time**

For each family:
- add a registry assertion
- add optional-dependency coverage
- add a CPU smoke case with tiny settings

**Step 2: Implement only the minimum honest lite variant**

Each family should:
- fit the existing `make_forecaster(...)(train, horizon)` contract
- train on lag windows
- use the current `_train_loop()` path or a close neighbor
- avoid claiming exogenous support unless the model really uses it

**Step 3: Verify each family before moving on**

Run targeted tests after every family:

```bash
PYTHONPATH=src pytest -q tests/test_models_registry.py -k "<family>"
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py -k "<family>"
PYTHONPATH=src pytest -q tests/test_models_torch_xformer_seq2seq_smoke.py -k "<family>"
```

**Step 4: Commit by family**

Use one commit per family or per tightly related pair.

---

### Task 3: Add A Graph Forecasting Product Surface

**Files:**
- Create: `src/foresight/models/torch_graph.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/data/prep.py`
- Modify: `src/foresight/eval_forecast.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_eval_panel.py`
- Test: `tests/test_models_graph_smoke.py`

**Target clusters:**
- #14 STGCN
- #15 DCRNN
- #16 graph-attention forecasters
- #17 Graph WaveNet
- #18 graph-structure-learning forecasters
- #19 AGCRN
- #20 MTGNN
- #21 StemGNN
- #22 STEP-style pretrained STGNNs
- #23 STID
- #24 FourierGNN

**Step 1: Introduce the graph data contract first**

Define the minimum graph forecasting payload needed by the first wave:
- target matrix or long panel
- node ids
- optional adjacency or learned-graph mode

**Step 2: Start with the smallest native graph batch**

Land only the first coherent wave:
- STGCN
- Graph WaveNet
- STID

**Step 3: Add evaluation and smoke coverage**

Smoke tests must verify:
- node-major prediction shape
- deterministic small CPU runs
- clear errors when graph metadata is missing

**Step 4: Expand only after the contract is stable**

Do not add the remaining graph families until the first three run cleanly through forecast and eval.

---

### Task 4: Add Continuous-Time And State-Space Waves

**Files:**
- Create: `src/foresight/models/torch_ssm.py`
- Create: `src/foresight/models/torch_ct_rnn.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_torch_ssm_smoke.py`

**Target clusters:**
- #4 LMU
- #5 LTC
- #6 CfC
- #7 S4
- #8 DSS / S4D
- #9 S5
- #11 Mamba-2 / SSD refinements
- #12 xLSTM
- #13 Griffin / Hawk

**Step 1: Split the work by mechanics**

Use two modules:
- `torch_ct_rnn.py` for LMU / LTC / CfC / xLSTM / Griffin-Hawk
- `torch_ssm.py` for S4 / DSS-S4D / S5 / Mamba-2

**Step 2: Start with low-risk lite representatives**

Recommended first landing order:
1. LMU
2. DSS / S4D
3. xLSTM
4. Mamba-2-style refinement

**Step 3: Keep names honest**

If an implementation is a lite approximation, say so in the registry description.

**Step 4: Verify per family**

Add small CPU smoke tests and registry assertions before expanding the wave.

---

### Task 5: Add Probabilistic And Foundation Wrappers

**Files:**
- Create: `src/foresight/models/foundation.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/io.py`
- Test: `tests/test_models_registry.py`
- Test: `tests/test_models_foundation_smoke.py`
- Test: `tests/test_forecast_api.py`

**Target clusters:**
- #41 TimeGrad
- #42 TACTiS
- #43 Lag-Llama
- #44 Time-LLM
- #45 TimesFM
- #46 Chronos / Chronos-Bolt
- #47 Moirai / Moirai-MoE
- #48 MOMENT
- #49 Time-MoE
- #50 Timer-S1

**Step 1: Separate native-training vs wrapper families**

- Native probabilistic candidates: TimeGrad, TACTiS
- Wrapper / inference-first candidates: Lag-Llama, TimesFM, Chronos, Moirai, MOMENT, Time-MoE, Timer-S1

**Step 2: Standardize checkpoint-facing registry entries**

Every wrapper family should declare:
- required backend
- checkpoint source
- offline vs online loading constraints
- supported inference mode

**Step 3: Add smoke tests that do not require big downloads by default**

Use:
- mocked or tiny local fixtures for wrapper plumbing
- explicit skip behavior when checkpoints are unavailable

**Step 4: Keep the public API strict**

Do not expose foundation families through the same training-first semantics unless they genuinely support fitting in ForeSight.

---

### Task 6: Keep Year Labels And Family Labels Visible In The Product

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/models.md`
- Modify: `src/foresight/cli.py`
- Test: `tests/test_cli_models.py`
- Test: `tests/test_cli_models_list_extended.py`

**Step 1: Expose taxonomy metadata**

Surface for each model family where useful:
- `year`
- `family`
- `interface`
- dependency requirements

**Step 2: Add CLI affordances**

At minimum support listing or filtering models by:
- family
- optional dependency
- interface

If feasible, add year filtering too.

**Step 3: Keep docs generated**

Avoid hand-maintained family/year tables drifting from the registry.

---

### Task 7: Verification And Release Discipline

**Files:**
- Verify only

**Step 1: Run focused checks for each wave**

```bash
ruff check src tests
PYTHONPATH=src pytest -q tests/test_models_registry.py
PYTHONPATH=src pytest -q tests/test_models_optional_deps_torch.py
```

**Step 2: Run per-wave smoke suites**

Create or extend targeted smoke files for:
- sequence families
- graph families
- state-space / recurrent revival
- foundation wrappers

**Step 3: Keep documentation in sync**

```bash
PYTHONPATH=src python tools/generate_model_capability_docs.py
PYTHONPATH=src python tools/check_capability_docs.py
```

**Step 4: Commit at wave boundaries**

Do not mix graph, state-space, and foundation work in the same commit series.

---

## Execution Order Summary

1. Freeze taxonomy and docs.
2. Finish API-compatible local/global sequence families.
3. Build the graph data contract and first graph wave.
4. Add continuous-time and state-space waves.
5. Add probabilistic and foundation wrappers.
6. Expose family/year metadata in docs and CLI.
7. Keep verification and generated docs green after every wave.
