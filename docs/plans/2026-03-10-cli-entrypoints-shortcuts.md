# CLI Entrypoints And Shortcuts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document the library's source entry order and add root-level CLI discovery shortcuts such as `--list` without disrupting existing subcommand workflows.

**Architecture:** Keep the existing `argparse` command tree intact and layer a small root-level shortcut path in `foresight.cli.main()` so top-level flags can reuse the existing handlers. Document the runtime entry flow separately so the package import path, CLI path, registry path, and evaluation path are easy to follow.

**Tech Stack:** Python 3.10+, argparse, NumPy, Pandas, pytest, MkDocs docs

---

### Task 1: Lock Shortcut Behavior In Tests

**Files:**
- Modify: `tests/test_cli_models.py`
- Modify: `tests/test_cli_datasets.py`

**Step 1: Add a root `--list` test**

Require `python -m foresight --list` to succeed and emit model keys that already appear under `foresight models list`.

**Step 2: Add explicit root discovery flag tests**

Require top-level flags for model listing and dataset listing to route to the same outputs as the subcommands.

**Step 3: Run RED**

```bash
PYTHONPATH=src pytest -q tests/test_cli_models.py tests/test_cli_datasets.py
```

---

### Task 2: Implement Root-Level Discovery Shortcuts

**Files:**
- Modify: `src/foresight/cli.py`

**Step 1: Add root parser flags**

Add root-level arguments for:
- `--list` as a shortcut for model listing
- `--list-models` as an explicit alias
- `--list-datasets` as a dataset discovery shortcut

**Step 2: Route through existing handlers**

Do not duplicate listing logic. Convert the parsed root flags into calls to the existing `_cmd_models_list()` and `_cmd_datasets_list()` handlers.

**Step 3: Preserve existing CLI behavior**

Keep `--version`, help text, subcommands, and error handling stable.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/test_cli_models.py tests/test_cli_datasets.py
```

---

### Task 3: Document Source Entrypoints

**Files:**
- Create: `docs/SOURCE_ENTRYPOINTS.md`
- Modify: `README.md`

**Step 1: Write the entry-order document**

Document the main flows:
- CLI startup: `python -m foresight` -> `src/foresight/__main__.py` -> `src/foresight/cli.py`
- Python import path: `import foresight` -> lazy exports in `src/foresight/__init__.py`
- Forecast/eval path: public API -> registry -> forecaster objects/functions -> backtesting/eval modules

**Step 2: Link it from the README**

Add a lightweight pointer so contributors can find the document quickly.

**Step 3: Verify docs stay aligned with the current CLI shape**

Keep examples consistent with root discovery shortcuts and existing subcommands.

---

### Task 4: Verify Integrated Behavior

**Files:**
- Verify only

**Step 1: Run targeted CLI tests**

```bash
PYTHONPATH=src pytest -q tests/test_cli_models.py tests/test_cli_datasets.py tests/test_root_import.py
```

**Step 2: Run a direct CLI smoke check**

```bash
PYTHONPATH=src python -m foresight --list
PYTHONPATH=src python -m foresight --list-datasets
PYTHONPATH=src python -m foresight --version
```
