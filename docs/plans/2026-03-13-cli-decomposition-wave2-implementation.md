# CLI Decomposition Wave 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose `src/foresight/cli.py` into smaller CLI-owned modules while preserving every existing CLI command, argument, and output shape.

**Architecture:** Keep `cli.py` as the CLI entry facade, but extract shared formatting/parsing helpers and command-family registration/handlers into dedicated `cli_*` modules. This wave focuses on CLI composition boundaries, not on changing `services` or public behavior.

**Tech Stack:** Python 3.10+, argparse, pandas, pytest

---

## Constraints

- Preserve all current CLI commands and arguments.
- Preserve output formats and payload shapes covered by existing tests.
- Do not replace `argparse`.
- Do not move business logic out of `services` back into CLI helper code.
- Keep each task independently shippable with focused tests.

### Task 1: Extract Shared CLI Helpers

**Files:**
- Create: `src/foresight/cli_shared.py`
- Modify: `src/foresight/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing CLI structure test**

Add regression coverage that asserts `cli.py` no longer owns the shared formatting/parsing helpers after this task.

Candidate assertions:

- `cli.py` should not define `_emit_table`
- `cli.py` should not define `_format_table`
- `cli.py` should not define `_parse_model_params`
- `cli.py` should not define `_parse_grid_params`

**Step 2: Run the failing focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py`
Expected: FAIL because those helpers still live in `cli.py`

**Step 3: Implement `cli_shared.py`**

Move into the new module:

- `_coerce_model_param_value`
- `_parse_model_params`
- `_parse_grid_params`
- `_emit_text`
- `_emit_dataframe`
- `_emit`
- `_emit_table`
- `_format_payload`
- `_format_csv`
- `_format_markdown`
- `_format_table`
- small TSV/filter helpers that are shared broadly

**Step 4: Rewire `cli.py` to import those helpers**

Keep behavior identical. Do not change parser args or output shape.

**Step 5: Re-run targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/cli_shared.py src/foresight/cli.py tests/test_cli.py
git commit -m "refactor: extract shared cli helpers"
```

### Task 2: Extract Metadata And Catalog CLI Commands

**Files:**
- Create: `src/foresight/cli_catalog.py`
- Modify: `src/foresight/cli.py`
- Modify: `tests/test_cli_models.py`
- Modify: `tests/test_docs_rnn_generated.py`

**Step 1: Write the failing regression test**

Add focused coverage that asserts `cli.py` no longer defines:

- `_load_rnn_paper_metadata`
- `_cmd_models_list`
- `_cmd_models_info`
- `_cmd_models_search`
- `_cmd_papers_list`
- `_cmd_papers_info`
- `_cmd_papers_models`
- `_cmd_docs_rnn`

**Step 2: Run the targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_models.py tests/test_docs_rnn_generated.py`
Expected: FAIL before extraction

**Step 3: Implement `cli_catalog.py`**

Move:

- RNN metadata cache/load/payload helpers
- model/paper/docs handlers
- parser-registration helpers for `models`, `papers`, `docs`

`cli.py` should call registration helpers instead of building these subtrees inline.

**Step 4: Re-run targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_models.py tests/test_docs_rnn_generated.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli_catalog.py src/foresight/cli.py tests/test_cli.py tests/test_cli_models.py tests/test_docs_rnn_generated.py
git commit -m "refactor: extract catalog cli commands"
```

### Task 3: Extract Dataset And Data Utility CLI Commands

**Files:**
- Create: `src/foresight/cli_data.py`
- Modify: `src/foresight/cli.py`
- Modify: `tests/test_cli.py`
- Modify: dataset/data CLI tests if needed

**Step 1: Write the failing structure test**

Add assertions that `cli.py` no longer defines:

- `_cmd_datasets_list`
- `_cmd_datasets_preview`
- `_cmd_datasets_path`
- `_cmd_datasets_validate`
- `_cmd_data_to_long`
- `_cmd_data_prepare_long`
- `_cmd_data_infer_freq`
- `_cmd_data_splits_rolling_origin`

**Step 2: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: FAIL until extraction is complete

**Step 3: Implement `cli_data.py`**

Move the handlers plus parser-registration helpers for:

- `datasets`
- `data`

Re-use helpers from `cli_shared.py`; do not duplicate emit/format logic.

**Step 4: Re-run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/foresight/cli_data.py src/foresight/cli.py tests/test_cli.py
git commit -m "refactor: extract data cli commands"
```

### Task 4: Extract Leaderboard CLI Commands And Summary Logic

**Files:**
- Create: `src/foresight/cli_leaderboard.py`
- Modify: `src/foresight/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_leaderboard.py`
- Modify: `tests/test_cli_leaderboard_sweep.py`

**Step 1: Write the failing structure test**

Add assertions that `cli.py` no longer defines:

- `_cmd_leaderboard_naive`
- `_cmd_leaderboard_models`
- `_leaderboard_sweep_worker`
- `_cmd_leaderboard_sweep`
- `_cmd_leaderboard_summarize`
- `_summarize_leaderboard_rows`
- `_leaderboard_columns`
- `_leaderboard_summary_columns`
- `_run_parallel_tasks`

**Step 2: Run the targeted leaderboard suite**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_leaderboard.py tests/test_cli_leaderboard_sweep.py`
Expected: FAIL until extraction is complete

**Step 3: Implement `cli_leaderboard.py`**

Move:

- leaderboard handlers
- sweep cache/worker helpers
- summary aggregation
- leaderboard-specific column helpers
- parallel task runner used only by leaderboard sweep
- parser-registration helpers for `leaderboard`

Keep output ordering and compact-row behavior unchanged.

**Step 4: Reduce `cli.py` to composition**

After this task, `cli.py` should mainly own:

- root parser / root shortcut logic
- `main()`
- command-family registration calls

**Step 5: Re-run the targeted suite**

Run: `PYTHONPATH=src pytest -q tests/test_cli.py tests/test_cli_leaderboard.py tests/test_cli_leaderboard_sweep.py tests/test_cli_eval.py tests/test_cli_forecast.py`
Expected: PASS

**Step 6: Commit**

```bash
git add src/foresight/cli_leaderboard.py src/foresight/cli.py tests/test_cli.py tests/test_cli_leaderboard.py tests/test_cli_leaderboard_sweep.py
git commit -m "refactor: extract leaderboard cli commands"
```

### Final Verification

Run the CLI-focused regression suite:

```bash
PYTHONPATH=src pytest -q \
  tests/test_cli.py \
  tests/test_cli_eval.py \
  tests/test_cli_forecast.py \
  tests/test_cli_models.py \
  tests/test_cli_leaderboard.py \
  tests/test_cli_leaderboard_sweep.py \
  tests/test_docs_rnn_generated.py
```

Expected: PASS

Then run broad package verification:

```bash
PYTHONPATH=src pytest -q
```

Expected: PASS
