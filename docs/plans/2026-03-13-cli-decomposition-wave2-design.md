# CLI Decomposition Wave 2 Design

**Date:** 2026-03-13

**Status:** Drafted from validated direction in-session

**Goal:** Continue the package modularization effort by reducing `src/foresight/cli.py` to a thinner entrypoint facade, while preserving every existing CLI command, argument, and output shape.

---

## 1. Problem Statement

The previous package modularization wave removed the heaviest forecast/eval orchestration from `cli.py`, but the file is still one of the biggest and highest-coupling modules in the repository.

Current structural issues:

1. `build_parser()` owns the full command tree and has grown into a 1,200+ line function.
2. CLI-only support behavior is still mixed together:
   - output formatting
   - model/grid parameter parsing
   - RNN paper metadata lookups
   - leaderboard aggregation/sweep helpers
3. Command families that are already conceptually independent still share one module namespace:
   - model/paper/docs metadata commands
   - dataset/data utility commands
   - leaderboard and sweep commands

This makes the package feel less like one deliberate CLI surface and more like an accreting script. The problem is no longer missing service extraction; it is missing CLI-owned module boundaries.

## 2. Constraints

This wave must preserve:

- the `foresight` CLI entrypoint
- all current command names and argument names
- current exit behavior and formatting behavior
- current output schemas used by tests and scripts

This wave must not:

- replace `argparse`
- redesign command semantics
- move business logic back out of `services` into CLI helpers

## 3. Target Shape

After this wave, `src/foresight/cli.py` should mainly own:

- the root parser
- root-level shortcuts (`--list`, `--list-datasets`, `--version`)
- broken-pipe handling
- `main()`
- composition of command-family registration helpers

The rest should be decomposed into explicit CLI-owned modules.

### Proposed Modules

`src/foresight/cli_shared.py`

- output helpers like `_emit`, `_emit_text`, `_emit_dataframe`, `_emit_table`
- table/markdown/csv formatting helpers
- model/grid parameter parsing helpers
- small TSV / filter utilities used across command families

`src/foresight/cli_catalog.py`

- RNN paper metadata loading and payload helpers
- `models list/info/search`
- `papers list/info/models`
- `docs rnn`
- parser registration for the metadata/catalog command family

`src/foresight/cli_data.py`

- `datasets *`
- `data *`
- parser registration for dataset/data utility commands

`src/foresight/cli_leaderboard.py`

- `leaderboard naive/models/sweep/summarize`
- sweep worker/cache helpers
- leaderboard summary calculation
- registration for the leaderboard command family

This keeps CLI-specific concerns in CLI modules rather than incorrectly pushing them into `services`.

## 4. Dependency Rules For This Wave

These are the intended rules after extraction:

- `cli.py` may import the new `cli_*` helper modules.
- `cli_*` modules may import `services`, `contracts`, datasets, model registry, and eval helpers as needed.
- `services` must still not import `cli.py` or the new `cli_*` modules.
- output formatting shared across CLI commands should live in `cli_shared.py`, not be recopied into each module.

The point is not to create more layers than necessary. The point is to make command-family ownership obvious.

## 5. Extraction Order

The safest order is:

1. extract shared CLI formatting/parsing helpers
2. extract metadata/catalog command family (`models`, `papers`, `docs`)
3. extract dataset/data command family
4. extract leaderboard family and sweep/summarization helpers
5. shrink `cli.py` to composition

This order minimizes merge risk because:

- shared helpers remove duplication first
- metadata and data commands are lower-risk than leaderboard concurrency
- leaderboard extraction can then reuse the shared helpers instead of inventing another mini-framework

## 6. Acceptance Criteria

This wave is successful when:

- `cli.py` is materially smaller and easier to scan
- `build_parser()` delegates to command-family registration helpers
- leaderboard sweep/summarization no longer lives in `cli.py`
- CLI support helpers have one canonical home
- all current CLI regression tests keep passing without argument changes

## 7. Non-Goals

This wave will not:

- split `cli.py` into a package named `foresight.cli.*`
- convert to Click/Typer
- redesign output schemas
- change service contracts

The work is intentionally a structural refactor, not a user-facing CLI rewrite.
