# Time-Series Wave 1 Parallel Worktrees Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up an eight-lane parallel worktree program for missing time-series algorithm families, land honest lite implementations or wrappers per lane, and converge everything through a single integration branch with green tests and synced docs.

**Architecture:** Use one integration branch with isolated project-local worktrees for each feature lane. Reduce merge pressure by creating shared scaffolding first, then let each lane own a dedicated module and smoke-test surface. Hold all generated docs and product-surface updates until the end on the integration branch.

**Tech Stack:** Git worktrees, Python 3.10+, NumPy, Pandas, PyTorch optional extras, pytest, ruff, MkDocs

---

### Task 1: Create The Integration Baseline

**Files:**
- Create: `docs/plans/2026-03-10-ts-wave1-parallel-worktrees-design.md`
- Create: `docs/plans/2026-03-10-ts-wave1-parallel-worktrees-implementation.md`

**Step 1: Verify the main worktree is clean**

Run: `git status --short --branch`
Expected: `main` with no tracked file changes

**Step 2: Create the integration branch worktree**

Run: `git worktree add .worktrees/ts-wave1-integration -b feat/ts-wave1-integration`
Expected: worktree created at `.worktrees/ts-wave1-integration`

**Step 3: Run a lightweight baseline test in the integration worktree**

Run: `python -m pytest tests/test_root_import.py tests/test_models_registry.py -q`
Expected: PASS

**Step 4: Commit the design and plan docs**

```bash
git add docs/plans/2026-03-10-ts-wave1-parallel-worktrees-design.md docs/plans/2026-03-10-ts-wave1-parallel-worktrees-implementation.md
git commit -m "docs: add ts wave1 parallel worktree design and plan"
```

---

### Task 2: Create Shared Scaffolding Before Parallelization

**Files:**
- Create: `src/foresight/models/torch_reservoir.py`
- Create: `src/foresight/models/torch_structured_rnn.py`
- Create: `src/foresight/models/torch_graph_attention.py`
- Create: `src/foresight/models/torch_graph_structure.py`
- Create: `src/foresight/models/torch_graph_spectral.py`
- Create: `src/foresight/models/torch_probabilistic.py`
- Create: `src/foresight/models/foundation.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/models/registry.py`
- Create: `tests/test_models_torch_reservoir_smoke.py`
- Create: `tests/test_models_torch_structured_rnn_smoke.py`
- Create: `tests/test_models_graph_attention_smoke.py`
- Create: `tests/test_models_graph_structure_smoke.py`
- Create: `tests/test_models_graph_spectral_smoke.py`
- Create: `tests/test_models_probabilistic_smoke.py`
- Create: `tests/test_models_foundation_smoke.py`

**Step 1: Write one failing scaffold-presence test**

```python
def test_wave1_scaffold_modules_import() -> None:
    import foresight.models.foundation
    import foresight.models.torch_graph_attention
    import foresight.models.torch_graph_spectral
    import foresight.models.torch_graph_structure
    import foresight.models.torch_probabilistic
    import foresight.models.torch_reservoir
    import foresight.models.torch_structured_rnn
```

**Step 2: Run the scaffold test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_models_foundation_smoke.py -k scaffold`
Expected: FAIL with import errors

**Step 3: Add empty module shells with lane header comments**

Each new module should include:

```python
from __future__ import annotations
```

and a short header comment identifying the owning lane and allowed scope.

**Step 4: Add empty smoke files and registry anchor comments**

Expected anchor areas:
- reservoir
- structured recurrent
- graph attention
- graph structure learning
- graph spectral
- probabilistic native
- foundation wrappers A
- foundation wrappers B

**Step 5: Run the scaffold test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_models_foundation_smoke.py -k scaffold`
Expected: PASS

**Step 6: Commit the scaffolding**

```bash
git add src/foresight/models/__init__.py src/foresight/models/registry.py src/foresight/models/foundation.py src/foresight/models/torch_reservoir.py src/foresight/models/torch_structured_rnn.py src/foresight/models/torch_graph_attention.py src/foresight/models/torch_graph_structure.py src/foresight/models/torch_graph_spectral.py src/foresight/models/torch_probabilistic.py tests/test_models_torch_reservoir_smoke.py tests/test_models_torch_structured_rnn_smoke.py tests/test_models_graph_attention_smoke.py tests/test_models_graph_structure_smoke.py tests/test_models_graph_spectral_smoke.py tests/test_models_probabilistic_smoke.py tests/test_models_foundation_smoke.py
git commit -m "chore: add wave1 parallel scaffolding"
```

---

### Task 3: Create The Eight Feature Branches And Worktrees

**Files:**
- Verify only

**Step 1: Create branch and worktree for lane 01**

Run: `git worktree add .worktrees/ts-lane-01-reservoir-lite -b feat/ts-lane-01-reservoir-lite feat/ts-wave1-integration`
Expected: new worktree on the integration baseline

**Step 2: Repeat for the remaining seven lanes**

Use the exact branch names from the design doc.

**Step 3: Verify all worktrees**

Run: `git worktree list`
Expected: integration worktree plus eight lane worktrees

**Step 4: Smoke-test one lane worktree baseline**

Run: `python -m pytest tests/test_root_import.py -q`
Expected: PASS

**Step 5: Commit nothing in this task**

This task changes git metadata only.

---

### Task 4: Implement Lane 01 Reservoir Lite

**Files:**
- Modify: `src/foresight/models/torch_reservoir.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_torch_reservoir_smoke.py`

**Step 1: Write the failing registry test**

```python
def test_torch_reservoir_models_are_registered() -> None:
    for key in ("torch-esn-direct", "torch-liquid-state-direct"):
        spec = get_model_spec(key)
        assert spec.interface == "local"
        assert "torch" in spec.requires
```

**Step 2: Run it to confirm failure**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k reservoir`
Expected: FAIL with unknown model key

**Step 3: Add the smallest honest local implementations**

Required outcomes:
- simple reservoir state update
- deterministic tiny CPU path
- direct horizon forecasting
- clear description that this is a lite approximation

**Step 4: Add the smoke test**

Run: `PYTHONPATH=src pytest -q tests/test_models_torch_reservoir_smoke.py`
Expected: PASS

**Step 5: Run the lane suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k reservoir tests/test_models_optional_deps_torch.py -k reservoir tests/test_models_torch_reservoir_smoke.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/models/torch_reservoir.py src/foresight/models/registry.py src/foresight/models/__init__.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_torch_reservoir_smoke.py
git commit -m "feat: add reservoir lite forecasters"
```

---

### Task 5: Implement Lane 02 Structured Recurrent Lite

**Files:**
- Modify: `src/foresight/models/torch_structured_rnn.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_torch_structured_rnn_smoke.py`

**Step 1: Write the failing registry test**

```python
def test_torch_structured_rnn_models_are_registered() -> None:
    spec = get_model_spec("torch-multidim-rnn-direct")
    assert spec.interface == "local"
    assert "torch" in spec.requires
```

**Step 2: Run it to confirm failure**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k structured`
Expected: FAIL

**Step 3: Add the minimum honest implementation**

Required outcomes:
- lite structured recurrent block
- no false graph support claims
- deterministic CPU smoke path

**Step 4: Run the lane suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k structured tests/test_models_optional_deps_torch.py -k structured tests/test_models_torch_structured_rnn_smoke.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/models/torch_structured_rnn.py src/foresight/models/registry.py src/foresight/models/__init__.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_torch_structured_rnn_smoke.py
git commit -m "feat: add structured recurrent lite forecasters"
```

---

### Task 6: Implement Lanes 03-05 Graph Families

**Files:**
- Modify: `src/foresight/models/torch_graph_attention.py`
- Modify: `src/foresight/models/torch_graph_structure.py`
- Modify: `src/foresight/models/torch_graph_spectral.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_graph_attention_smoke.py`
- Modify: `tests/test_models_graph_structure_smoke.py`
- Modify: `tests/test_models_graph_spectral_smoke.py`

**Step 1: For each lane, add one failing registration test**

Suggested keys:
- `torch-astgcn-multivariate`
- `torch-gman-multivariate`
- `torch-agcrn-multivariate`
- `torch-mtgnn-multivariate`
- `torch-stemgnn-multivariate`
- `torch-fouriergnn-multivariate`

**Step 2: Run lane-specific failures first**

Example:

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k astgcn`
Expected: FAIL

**Step 3: Add honest lite multivariate implementations**

Requirements:
- accept wide multivariate input compatible with current multivariate path
- avoid claiming learned adjacency if the implementation uses a simplified fallback
- produce deterministic small-CPU smoke behavior

**Step 4: Run each lane suite separately**

Example:

Run: `PYTHONPATH=src pytest -q tests/test_models_graph_attention_smoke.py`
Expected: PASS

**Step 5: After all three graph lanes are done, run a graph batch suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_multivariate.py tests/test_eval_multivariate.py tests/test_models_graph_attention_smoke.py tests/test_models_graph_structure_smoke.py tests/test_models_graph_spectral_smoke.py`
Expected: PASS

**Step 6: Commit one commit per lane**

Example messages:
- `feat: add graph attention lite forecasters`
- `feat: add graph structure lite forecasters`
- `feat: add graph spectral lite forecasters`

---

### Task 7: Implement Lane 06 Probabilistic Native Lite

**Files:**
- Modify: `src/foresight/models/torch_probabilistic.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/forecast.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_optional_deps_torch.py`
- Modify: `tests/test_models_probabilistic_smoke.py`
- Modify: `tests/test_forecast_api.py`

**Step 1: Write the failing registry test**

```python
def test_probabilistic_native_models_are_registered() -> None:
    for key in ("torch-timegrad-direct", "torch-tactis-direct"):
        spec = get_model_spec(key)
        assert spec.interface == "local"
```

**Step 2: Run it to confirm failure**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k timegrad`
Expected: FAIL

**Step 3: Add the smallest honest native implementations**

Requirements:
- produce point forecast first
- expose probabilistic behavior only where actually implemented
- avoid pretending to support full research-model semantics

**Step 4: Add minimal forecast output plumbing**

Only add what the smoke tests require. Do not redesign the forecast API.

**Step 5: Run the lane suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k \"timegrad or tactis\" tests/test_models_optional_deps_torch.py -k \"timegrad or tactis\" tests/test_models_probabilistic_smoke.py tests/test_forecast_api.py -k probabilistic`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/models/torch_probabilistic.py src/foresight/models/registry.py src/foresight/models/__init__.py src/foresight/forecast.py tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_probabilistic_smoke.py tests/test_forecast_api.py
git commit -m "feat: add probabilistic native lite forecasters"
```

---

### Task 8: Implement Lanes 07-08 Foundation Wrappers

**Files:**
- Modify: `src/foresight/models/foundation.py`
- Modify: `src/foresight/models/registry.py`
- Modify: `src/foresight/models/__init__.py`
- Modify: `src/foresight/forecast.py`
- Modify: `src/foresight/io.py`
- Modify: `tests/test_models_registry.py`
- Modify: `tests/test_models_foundation_smoke.py`
- Modify: `tests/test_forecast_api.py`

**Step 1: Write failing registration tests for wrappers A**

```python
def test_foundation_wrappers_a_are_registered() -> None:
    for key in ("chronos", "chronos-bolt", "timesfm", "lag-llama"):
        assert "wrapper" in get_model_spec(key).description.lower()
```

**Step 2: Write failing registration tests for wrappers B**

```python
def test_foundation_wrappers_b_are_registered() -> None:
    for key in ("moirai", "moment", "time-moe", "timer-s1"):
        assert "wrapper" in get_model_spec(key).description.lower()
```

**Step 3: Run the tests to confirm failure**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k \"chronos or moirai\"`
Expected: FAIL

**Step 4: Add wrapper helper contracts**

Every wrapper should declare:
- backend requirement
- offline vs online behavior
- checkpoint or model identifier source
- skip or clear error behavior when assets are unavailable

**Step 5: Add smoke tests using mocked or tiny local fixtures**

The default test path must not require large downloads.

**Step 6: Run the wrapper suite**

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py -k \"chronos or moirai or timesfm or lag\" tests/test_models_foundation_smoke.py tests/test_forecast_api.py -k foundation`
Expected: PASS

**Step 7: Commit one commit per wrapper lane**

Example messages:
- `feat: add foundation wrapper lane a`
- `feat: add foundation wrapper lane b`

---

### Task 9: Merge Lanes Back Into The Integration Branch

**Files:**
- Verify only

**Step 1: Merge lane 01**

Run: `git merge --no-ff feat/ts-lane-01-reservoir-lite`
Expected: merge commit or fast clean merge

**Step 2: Repeat for lanes 02 through 08 in the approved order**

Use the merge order from the design doc.

**Step 3: After each merge, run the lane-specific suite**

Expected: PASS before moving to the next lane

**Step 4: If a shared helper conflict appears, fix it on the integration branch**

Do not push the same helper redesign back into multiple lane branches.

---

### Task 10: Regenerate Docs And Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/models.md`
- Modify: `docs/api.md`
- Modify: `docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md`

**Step 1: Update roadmap statuses to match landed code**

Only mark families implemented if the code and tests are already merged on the integration branch.

**Step 2: Refresh generated docs**

Run: `PYTHONPATH=src python tools/generate_model_capability_docs.py`
Expected: generated docs updated or confirmed current

**Step 3: Validate generated docs**

Run: `PYTHONPATH=src python tools/check_capability_docs.py`
Expected: PASS

**Step 4: Run the final verification batch**

Run: `ruff check src tests`
Expected: PASS

Run: `PYTHONPATH=src pytest -q tests/test_models_registry.py tests/test_models_optional_deps_torch.py tests/test_models_torch_reservoir_smoke.py tests/test_models_torch_structured_rnn_smoke.py tests/test_models_graph_attention_smoke.py tests/test_models_graph_structure_smoke.py tests/test_models_graph_spectral_smoke.py tests/test_models_probabilistic_smoke.py tests/test_models_foundation_smoke.py`
Expected: PASS

**Step 5: Commit the integration sweep**

```bash
git add README.md docs/index.md docs/models.md docs/api.md docs/plans/2026-03-09-algorithm-clusters-2001-2026-roadmap.md
git commit -m "docs: sync wave1 registry and roadmap"
```
