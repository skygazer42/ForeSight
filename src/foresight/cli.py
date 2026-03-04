from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

_RNN_PAPER_METADATA_CACHE: dict[str, dict[str, Any]] | None = None


def _load_rnn_paper_metadata() -> dict[str, dict[str, Any]]:
    """
    Best-effort loader for `docs/rnn_paper_metadata.json` (used by the RNN Paper Zoo docs).

    This is intentionally dependency-free and safe to call when optional ML deps are missing.
    """

    global _RNN_PAPER_METADATA_CACHE
    if _RNN_PAPER_METADATA_CACHE is not None:
        return _RNN_PAPER_METADATA_CACHE

    env_path = str(os.environ.get("FORESIGHT_RNN_PAPER_METADATA", "")).strip()

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    # Installed-package fallback: ship the same JSON under `foresight/data/`.
    try:
        candidates.append(Path(__file__).resolve().parent / "data" / "rnn_paper_metadata.json")
    except Exception:  # noqa: BLE001
        pass
    # Common dev path: run CLI from repo root.
    candidates.append(Path.cwd() / "docs" / "rnn_paper_metadata.json")
    # When running from source, `src/foresight/cli.py` -> repo root is `parents[2]`.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "docs" / "rnn_paper_metadata.json")
    except Exception:  # noqa: BLE001
        pass

    raw: object | None = None
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                break
        except Exception:  # noqa: BLE001
            continue

    if not isinstance(raw, dict):
        _RNN_PAPER_METADATA_CACHE = {}
        return _RNN_PAPER_METADATA_CACHE

    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v

    _RNN_PAPER_METADATA_CACHE = out
    return out


def _rnnpaper_id_from_model_key(key: str) -> str | None:
    """
    Parse: torch-rnnpaper-<paper_id>-direct  ->  <paper_id>
    """

    k = str(key).strip()
    prefix = "torch-rnnpaper-"
    suffix = "-direct"
    if not (k.startswith(prefix) and k.endswith(suffix)):
        return None
    pid = k[len(prefix) : -len(suffix)]
    pid = pid.strip("-").strip()
    return pid or None


def _rnnzoo_base_from_model_key(key: str) -> str | None:
    """
    Parse:
      torch-rnnzoo-<base>-direct
      torch-rnnzoo-<base>-<variant>-direct
    """

    base, _variant = _rnnzoo_base_and_variant_from_model_key(key)
    return base


def _rnnzoo_base_and_variant_from_model_key(key: str) -> tuple[str | None, str | None]:
    """
    Parse:
      torch-rnnzoo-<base>-direct
      torch-rnnzoo-<base>-<variant>-direct
    Returns: (base, variant) where variant is one of: direct, bidir, ln, attn, proj
    """

    k = str(key).strip()
    prefix = "torch-rnnzoo-"
    suffix = "-direct"
    if not (k.startswith(prefix) and k.endswith(suffix)):
        return (None, None)
    body = k[len(prefix) : -len(suffix)]
    body = body.strip("-").strip()
    if not body:
        return (None, None)

    parts = body.split("-")
    if parts[-1] in {"bidir", "ln", "attn", "proj"} and len(parts) >= 2:
        base = "-".join(parts[:-1]).strip("-").strip()
        return (base or None, parts[-1])
    return (body, "direct")


def _rnnzoo_paper_id_from_model_key(key: str) -> str | None:
    base = _rnnzoo_base_from_model_key(key)
    if not base:
        return None
    base_s = str(base).strip().lower()

    # RNN Zoo base ids are not 1:1 with the RNN Paper Zoo `paper_id` set.
    # Map known aliases back to the canonical ids used in `docs/rnn_paper_metadata.json`.
    base_to_paper_id = {
        "elman": "elman-srn",
        "clockwork": "clockwork-rnn",
        "fastrnn": "fast-rnn",
        "fastgrnn": "fast-grnn",
    }
    return base_to_paper_id.get(base_s, base_s)


def _rnnzoo_wrapper_paper_id_from_model_key(key: str) -> str | None:
    _base, variant = _rnnzoo_base_and_variant_from_model_key(key)
    if not variant or variant == "direct":
        return None
    variant_to_paper_id = {
        "bidir": "bidirectional-rnn",
        "ln": "layer-normalization",
        "attn": "bahdanau-attention",
        "proj": "lstm-projection",
    }
    return variant_to_paper_id.get(str(variant).strip().lower(), None)


def _paper_payload_for_paper_id(paper_id: str) -> dict[str, Any] | None:
    pid = str(paper_id).strip()
    if not pid:
        return None
    meta = _load_rnn_paper_metadata()
    entry = meta.get(pid, {})
    if not isinstance(entry, dict):
        entry = {}

    # Keep payload compact and stable for the CLI.
    title = str(entry.get("title", "")).strip()
    doi = str(entry.get("doi", "")).strip()
    arxiv_id = str(entry.get("arxiv_id", "")).strip()
    url = str(entry.get("url", "")).strip()

    year_raw = entry.get("year", None)
    year = int(year_raw) if isinstance(year_raw, int) else None
    if not (title or doi or arxiv_id or url or year is not None):
        return None

    return {
        "paper_id": pid,
        "title": title,
        "year": year,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": url,
    }


def _paper_payload_for_model_key(key: str) -> dict[str, Any] | None:
    paper_id = _rnnpaper_id_from_model_key(key)
    if paper_id:
        return _paper_payload_for_paper_id(paper_id)

    paper_id = _rnnzoo_paper_id_from_model_key(key)
    if paper_id:
        return _paper_payload_for_paper_id(paper_id)

    return None


def _sanitize_tsv_cell(value: object) -> str:
    """
    TSV output is used for CLI piping. Keep it single-line and tab-safe.
    """

    s = str(value) if value is not None else ""
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


_CORE_REQUIRES_ALIASES = {"core", "none", "empty", "no", "norequires", "no-requires"}


def _parse_requires_filter(raw: str) -> tuple[set[str], bool]:
    """
    Parse `--requires` / `--exclude-requires` values.

    - Tokens are comma-separated.
    - Special tokens like "core"/"none" refer to models with no optional requires.
    Returns: (requires_set, include_core_flag)
    """

    items = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    tokens = set(items)
    include_core = bool(tokens.intersection(_CORE_REQUIRES_ALIASES))
    tokens.difference_update(_CORE_REQUIRES_ALIASES)
    return (tokens, include_core)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foresight",
        description="ForeSight: time-series forecasting models and utilities.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks on errors.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Base directory to resolve dataset files (overrides FORESIGHT_DATA_DIR).",
    )

    sub = p.add_subparsers(dest="command")

    models = sub.add_parser("models", help="Model registry utilities")
    models_sub = models.add_subparsers(dest="models_command", required=True)

    models_list = models_sub.add_parser("list", help="List available models")
    models_list.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    models_list.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    models_list.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional model key prefix filter (e.g. torch-rnnpaper)",
    )
    models_list.add_argument(
        "--interface",
        choices=["any", "local", "global"],
        default="any",
        help="Filter by interface (default: any)",
    )
    models_list.add_argument(
        "--requires",
        type=str,
        default="",
        help="Filter by requires (comma-separated). Example: torch or core,torch",
    )
    models_list.add_argument(
        "--exclude-requires",
        type=str,
        default="",
        help="Exclude requires (comma-separated). Example: stats or core",
    )
    models_list.add_argument(
        "--columns",
        type=str,
        default="",
        help="Optional TSV columns (comma-separated). Example: key,requires,paper_id,paper_year",
    )
    models_list.add_argument(
        "--header",
        action="store_true",
        help="Include a header row for TSV output",
    )
    models_list.add_argument(
        "--sort",
        type=str,
        default="key",
        help="Sort key. Example: key, paper_year",
    )
    models_list.add_argument(
        "--desc",
        action="store_true",
        help="Sort descending (equivalent to --sort=-<key>)",
    )
    models_list.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of results (0 means no limit)",
    )
    models_list.set_defaults(_handler=_cmd_models_list)

    models_info = models_sub.add_parser("info", help="Show details about a model")
    models_info.add_argument("key", help="Model key (see: `foresight models list`)")
    models_info.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
    )
    models_info.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    models_info.set_defaults(_handler=_cmd_models_info)

    models_search = models_sub.add_parser(
        "search", help="Search models by key/description/paper metadata"
    )
    models_search.add_argument("query", help="Search query (space-separated tokens)")
    models_search.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    models_search.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    models_search.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional model key prefix filter (e.g. torch-rnnpaper)",
    )
    models_search.add_argument(
        "--interface",
        choices=["any", "local", "global"],
        default="any",
        help="Filter by interface (default: any)",
    )
    models_search.add_argument(
        "--requires",
        type=str,
        default="",
        help="Filter by requires (comma-separated). Example: torch or core,torch",
    )
    models_search.add_argument(
        "--exclude-requires",
        type=str,
        default="",
        help="Exclude requires (comma-separated). Example: stats or core",
    )
    models_search.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of results (default: 20)",
    )
    models_search.add_argument(
        "--any",
        action="store_true",
        help="Match any token (OR) instead of all tokens (AND)",
    )
    models_search.set_defaults(_handler=_cmd_models_search)

    papers = sub.add_parser("papers", help="Paper metadata utilities")
    papers_sub = papers.add_subparsers(dest="papers_command", required=True)

    papers_list = papers_sub.add_parser("list", help="List known paper metadata entries")
    papers_list.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    papers_list.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    papers_list.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional substring filter (paper_id/title)",
    )
    papers_list.set_defaults(_handler=_cmd_papers_list)

    papers_info = papers_sub.add_parser("info", help="Show metadata about a paper_id")
    papers_info.add_argument("paper_id", help="Paper id (see: `foresight papers list`)")
    papers_info.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
    )
    papers_info.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    papers_info.set_defaults(_handler=_cmd_papers_info)

    papers_models = papers_sub.add_parser(
        "models", help="List models that reference a given paper_id"
    )
    papers_models.add_argument("paper_id", help="Paper id (see: `foresight papers list`)")
    papers_models.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    papers_models.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    papers_models.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional model key prefix filter (e.g. torch-rnn)",
    )
    papers_models.add_argument(
        "--role",
        choices=["any", "base", "wrapper"],
        default="any",
        help="Match role: base architecture, wrapper, or both (default: any)",
    )
    papers_models.set_defaults(_handler=_cmd_papers_models)

    docs = sub.add_parser("docs", help="Documentation utilities")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)

    docs_rnn = docs_sub.add_parser("rnn", help="Generate RNN zoo docs (paper zoo + rnn zoo)")
    docs_rnn.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Output directory (default: ./docs)",
    )
    docs_rnn.add_argument(
        "--check",
        action="store_true",
        help="Check that docs are up to date instead of writing files",
    )
    docs_rnn.set_defaults(_handler=_cmd_docs_rnn)

    cv = sub.add_parser("cv", help="Cross-validation utilities")
    cv_sub = cv.add_subparsers(dest="cv_command", required=True)

    cv_run = cv_sub.add_parser("run", help="Run rolling-origin CV and output predictions")
    cv_run.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    cv_run.add_argument("--dataset", required=True, help="Dataset key")
    cv_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    cv_run.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    cv_run.add_argument("--step-size", type=int, default=1, help="CV step size (default: 1)")
    cv_run.add_argument("--min-train-size", type=int, required=True, help="Minimum train size")
    cv_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
    )
    cv_run.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help="Optional limit to the last N CV windows.",
    )
    cv_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    cv_run.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write output",
    )
    cv_run.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for predictions (default: csv)",
    )
    cv_run.set_defaults(_handler=_cmd_cv_run)

    datasets = sub.add_parser("datasets", help="Dataset utilities")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)

    datasets_list = datasets_sub.add_parser("list", help="List available datasets")
    datasets_list.add_argument(
        "--with-path",
        action="store_true",
        help="Include resolved local file paths in the output.",
    )
    datasets_list.set_defaults(_handler=_cmd_datasets_list)

    datasets_preview = datasets_sub.add_parser("preview", help="Preview a dataset (head)")
    datasets_preview.add_argument("key", help="Dataset key (see: `foresight datasets list`)")
    datasets_preview.add_argument("--nrows", type=int, default=20, help="Number of rows to preview")
    datasets_preview.set_defaults(_handler=_cmd_datasets_preview)

    datasets_path = datasets_sub.add_parser("path", help="Print the local path for a dataset")
    datasets_path.add_argument("key", help="Dataset key (see: `foresight datasets list`)")
    datasets_path.set_defaults(_handler=_cmd_datasets_path)

    datasets_validate = datasets_sub.add_parser(
        "validate", help="Smoke-check local dataset files and basic schemas"
    )
    datasets_validate.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset key to validate (default: validate all).",
    )
    datasets_validate.add_argument(
        "--nrows",
        type=int,
        default=5,
        help="Number of rows to load for large datasets (default: 5)",
    )
    datasets_validate.add_argument(
        "--check-time",
        action="store_true",
        help="Check per-series ds ordering and duplicates (uses dataset spec).",
    )
    datasets_validate.set_defaults(_handler=_cmd_datasets_validate)

    eval_p = sub.add_parser("eval", help="Evaluation utilities")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)

    eval_naive_last = eval_sub.add_parser("naive-last", help="Evaluate naive-last baseline")
    eval_naive_last.add_argument("--dataset", required=True, help="Dataset key")
    eval_naive_last.add_argument("--y-col", required=True, help="Target column name")
    eval_naive_last.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_naive_last.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_naive_last.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_naive_last.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_naive_last.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_naive_last.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_naive_last.set_defaults(_handler=_cmd_eval_naive_last)

    eval_seasonal_naive = eval_sub.add_parser(
        "seasonal-naive", help="Evaluate seasonal naive baseline"
    )
    eval_seasonal_naive.add_argument("--dataset", required=True, help="Dataset key")
    eval_seasonal_naive.add_argument("--y-col", required=True, help="Target column name")
    eval_seasonal_naive.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_seasonal_naive.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_seasonal_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_seasonal_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_seasonal_naive.add_argument(
        "--season-length",
        type=int,
        required=True,
        help="Season length for seasonal naive",
    )
    eval_seasonal_naive.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_seasonal_naive.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_seasonal_naive.set_defaults(_handler=_cmd_eval_seasonal_naive)

    eval_run = eval_sub.add_parser("run", help="Evaluate any registered model")
    eval_run.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    eval_run.add_argument("--dataset", required=True, help="Dataset key")
    eval_run.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    eval_run.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_run.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_run.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_run.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
    )
    eval_run.add_argument(
        "--conformal-levels",
        type=str,
        default="",
        help="Optional conformal levels as percentages or floats, e.g. 80,90 or 0.8,0.9",
    )
    eval_run.add_argument(
        "--conformal-pooled",
        action="store_true",
        help="Pool residuals across steps instead of per-step conformal radii.",
    )
    eval_run.add_argument(
        "--model-param",
        action="append",
        default=[],
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    eval_run.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_run.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_run.set_defaults(_handler=_cmd_eval_run)

    eval_csv = eval_sub.add_parser("csv", help="Evaluate a model on an arbitrary CSV file")
    eval_csv.add_argument("--model", required=True, help="Model key (see: `foresight models list`)")
    eval_csv.add_argument("--path", required=True, help="Path to a CSV file")
    eval_csv.add_argument("--time-col", required=True, help="Time column name")
    eval_csv.add_argument("--y-col", required=True, help="Target column name")
    eval_csv.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Optional comma-separated ID columns for panel data (e.g. store,dept)",
    )
    eval_csv.add_argument(
        "--parse-dates",
        action="store_true",
        help="Parse the time column as datetime.",
    )
    eval_csv.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    eval_csv.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    eval_csv.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    eval_csv.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    eval_csv.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Optional rolling train window size (default: expanding window).",
    )
    eval_csv.add_argument(
        "--conformal-levels",
        type=str,
        default="",
        help="Optional conformal levels as percentages or floats, e.g. 80,90 or 0.8,0.9",
    )
    eval_csv.add_argument(
        "--conformal-pooled",
        action="store_true",
        help="Pool residuals across steps instead of per-step conformal radii.",
    )
    eval_csv.add_argument(
        "--model-param",
        action="append",
        default=[],
        help="Model parameter as key=value (repeatable). Example: --model-param season_length=12",
    )
    eval_csv.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics output",
    )
    eval_csv.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    eval_csv.set_defaults(_handler=_cmd_eval_csv)

    leaderboard = sub.add_parser("leaderboard", help="Run a small builtin leaderboard")
    leaderboard_sub = leaderboard.add_subparsers(dest="leaderboard_command", required=True)

    leaderboard_naive = leaderboard_sub.add_parser("naive", help="Run naive baselines leaderboard")
    leaderboard_naive.add_argument("--dataset", required=True, help="Dataset key")
    leaderboard_naive.add_argument("--y-col", required=True, help="Target column name")
    leaderboard_naive.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    leaderboard_naive.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    leaderboard_naive.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    leaderboard_naive.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    leaderboard_naive.add_argument(
        "--season-length",
        type=int,
        required=True,
        help="Season length for seasonal naive",
    )
    leaderboard_naive.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON leaderboard",
    )
    leaderboard_naive.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    leaderboard_naive.set_defaults(_handler=_cmd_leaderboard_naive)

    leaderboard_models = leaderboard_sub.add_parser(
        "models", help="Run a leaderboard across registered models"
    )
    leaderboard_models.add_argument("--dataset", required=True, help="Dataset key")
    leaderboard_models.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y).",
    )
    leaderboard_models.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    leaderboard_models.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    leaderboard_models.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    leaderboard_models.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    leaderboard_models.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model keys (default: all core models).",
    )
    leaderboard_models.add_argument(
        "--include-optional",
        action="store_true",
        help="Include models requiring optional extras (e.g. statsmodels).",
    )
    leaderboard_models.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write leaderboard output",
    )
    leaderboard_models.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    leaderboard_models.set_defaults(_handler=_cmd_leaderboard_models)

    leaderboard_sweep = leaderboard_sub.add_parser(
        "sweep", help="Run a leaderboard across multiple datasets and models"
    )
    leaderboard_sweep.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset keys (e.g. catfish,ice_cream_interest)",
    )
    leaderboard_sweep.add_argument(
        "--y-col",
        type=str,
        default="",
        help="Optional target column name (default: use dataset spec default_y per dataset).",
    )
    leaderboard_sweep.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    leaderboard_sweep.add_argument("--step", type=int, default=1, help="Walk-forward step size")
    leaderboard_sweep.add_argument(
        "--min-train-size",
        type=int,
        required=True,
        help="Minimum training size for first window",
    )
    leaderboard_sweep.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit on the number of walk-forward windows",
    )
    leaderboard_sweep.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model keys (default: all core models).",
    )
    leaderboard_sweep.add_argument(
        "--include-optional",
        action="store_true",
        help="Include models requiring optional extras (e.g. statsmodels).",
    )
    leaderboard_sweep.add_argument(
        "--model-param",
        action="append",
        default=[],
        help=(
            "Model parameter as key=value (repeatable). Example: --model-param window=7. "
            "Applied to every selected model; models that don't accept the parameter may SKIP "
            "unless --strict is set."
        ),
    )
    leaderboard_sweep.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Optional path to an existing sweep output (.json/.csv) to resume from. "
            "Rows matching the current datasets/models and evaluation settings are kept, and "
            "their (dataset, model) pairs are skipped."
        ),
    )
    leaderboard_sweep.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional path to write an aggregated model summary table (across datasets).",
    )
    leaderboard_sweep.add_argument(
        "--summary-format",
        choices=["json", "csv", "md"],
        default="json",
        help="Summary output format (default: json).",
    )
    leaderboard_sweep.add_argument(
        "--summary-sort",
        type=str,
        default="mae_rank_mean",
        help="Summary sort column (default: mae_rank_mean). Use --summary-sort=-col for descending.",
    )
    leaderboard_sweep.add_argument(
        "--summary-limit",
        type=int,
        default=0,
        help="Optional limit for summary rows (default: 0 = no limit).",
    )
    leaderboard_sweep.add_argument(
        "--summary-min-datasets",
        type=int,
        default=0,
        help="Optional minimum dataset coverage for a model in the summary (default: 0 = no filter).",
    )
    leaderboard_sweep.add_argument(
        "--failures-output",
        type=str,
        default="",
        help="Optional path to write SKIP/failure lines (one per line).",
    )
    leaderboard_sweep.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write leaderboard output",
    )
    leaderboard_sweep.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    leaderboard_sweep.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    leaderboard_sweep.add_argument(
        "--backend",
        choices=["process", "thread"],
        default="process",
        help="Parallel backend when --jobs>1 (default: process)",
    )
    leaderboard_sweep.add_argument(
        "--progress",
        action="store_true",
        help="Print progress to stderr (safe for piping JSON/CSV/MD on stdout)",
    )
    leaderboard_sweep.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help=(
            "Number of models to evaluate per task, per dataset (default: 1). "
            "Use 0 to run all models for a dataset in a single task (better dataset reuse, "
            "less parallelism when there are few datasets)."
        ),
    )
    leaderboard_sweep.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first error instead of skipping failed (dataset, model) pairs.",
    )
    leaderboard_sweep.set_defaults(_handler=_cmd_leaderboard_sweep)

    leaderboard_summarize = leaderboard_sub.add_parser(
        "summarize", help="Aggregate leaderboard rows across datasets"
    )
    leaderboard_summarize.add_argument(
        "--input",
        type=str,
        default="-",
        help="Input path (.json/.csv) or '-' for stdin (default: -).",
    )
    leaderboard_summarize.add_argument(
        "--input-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Input format (default: auto).",
    )
    leaderboard_summarize.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write summary output",
    )
    leaderboard_summarize.add_argument(
        "--format",
        choices=["json", "csv", "md"],
        default="json",
        help="Output format (default: json)",
    )
    leaderboard_summarize.add_argument(
        "--sort",
        type=str,
        default="mae_rank_mean",
        help="Sort column (default: mae_rank_mean). Use --sort=-col for descending.",
    )
    leaderboard_summarize.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of models in the summary (default: 0 = no limit).",
    )
    leaderboard_summarize.add_argument(
        "--min-datasets",
        type=int,
        default=0,
        help="Optional minimum dataset coverage for a model (default: 0 = no filter).",
    )
    leaderboard_summarize.set_defaults(_handler=_cmd_leaderboard_summarize)

    return p


def _silence_stdout_broken_pipe() -> None:
    """
    Best-effort: redirect stdout to /dev/null so Python's final flush/close
    does not emit "Exception ignored ... BrokenPipeError" noise when stdout was
    closed early by a downstream consumer (e.g. `| head -n 1`).
    """

    try:
        stdout_fd = sys.stdout.fileno()
    except Exception:  # noqa: BLE001
        return

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except Exception:  # noqa: BLE001
        return

    try:
        os.dup2(devnull_fd, stdout_fd)
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            os.close(devnull_fd)
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        # Avoid importing package metadata; keep it simple and dependency-free.
        from . import __version__

        try:
            print(f"ForeSight {__version__}")
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
        return 0

    handler = getattr(args, "_handler", None)
    if handler is None:
        try:
            parser.print_help()
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
        return 0

    try:
        code = int(handler(args))
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout_broken_pipe()
            return 0
        return code
    except BrokenPipeError:
        _silence_stdout_broken_pipe()
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def _coerce_model_param_value(raw: str) -> Any:
    s = str(raw).strip()
    lower = s.lower()

    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return tuple(_coerce_model_param_value(p) for p in parts)

    try:
        return int(s)
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        pass
    return s


def _parse_model_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"--model-param must be key=value, got: {item!r}")
        key, value = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--model-param must be key=value, got: {item!r}")
        params[key] = _coerce_model_param_value(value)
    return params


def _emit_text(text: str, *, output: str) -> None:
    print(text)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _emit_dataframe(df: Any, *, output: str, fmt: str) -> None:
    if fmt == "csv":
        text = df.to_csv(index=False).rstrip("\n")
        _emit_text(text, output=output)
        return
    if fmt == "json":
        text = df.to_json(orient="records", date_format="iso")
        _emit_text(text, output=output)
        return
    raise ValueError(f"Unknown dataframe format: {fmt!r}")


def _cmd_models_list(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    interface_filter = str(getattr(args, "interface", "any")).strip().lower()
    if interface_filter not in {"any", "local", "global"}:
        raise ValueError("--interface must be one of: any, local, global")

    req_set, include_core = _parse_requires_filter(str(getattr(args, "requires", "")))
    exc_set, exclude_core = _parse_requires_filter(str(getattr(args, "exclude_requires", "")))

    rows: list[dict[str, Any]] = []
    prefix = str(args.prefix).strip()
    for key in list_models():
        if prefix and not str(key).startswith(prefix):
            continue
        spec = get_model_spec(key)

        if interface_filter != "any" and str(spec.interface) != interface_filter:
            continue

        reqs = set(spec.requires)
        if req_set or include_core:
            ok = False
            if include_core and not reqs:
                ok = True
            if req_set and reqs.intersection(req_set):
                ok = True
            if not ok:
                continue

        if exc_set or exclude_core:
            if exclude_core and not reqs:
                continue
            if exc_set and reqs.intersection(exc_set):
                continue

        row: dict[str, Any] = {
            "key": spec.key,
            "interface": str(spec.interface),
            "requires": ",".join(spec.requires),
            "description": spec.description,
            "default_params": dict(spec.default_params),
        }
        paper = _paper_payload_for_model_key(spec.key)
        if paper:
            row["paper"] = paper
        wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
        if wrapper_pid:
            wrapper = _paper_payload_for_paper_id(wrapper_pid)
            if wrapper:
                row["wrapper_paper"] = wrapper
        rows.append(row)

    sort = str(args.sort).strip() or "key"
    descending = bool(getattr(args, "desc", False))
    if sort.startswith("-"):
        descending = True
        sort = sort[1:]
    sort_key = sort.strip() or "key"

    def _sort_value(r: dict[str, Any]) -> object | None:
        paper = r.get("paper") if isinstance(r.get("paper"), dict) else {}
        wrapper = r.get("wrapper_paper") if isinstance(r.get("wrapper_paper"), dict) else {}

        if sort_key == "key":
            return str(r.get("key", "")).lower()
        if sort_key == "interface":
            return str(r.get("interface", "")).lower()
        if sort_key == "requires":
            return str(r.get("requires", "")).lower()
        if sort_key == "description":
            return str(r.get("description", "")).lower()
        if sort_key == "paper_id":
            return str(paper.get("paper_id", "")).lower()
        if sort_key == "paper_year":
            year = paper.get("year", None)
            return int(year) if isinstance(year, int) else None
        if sort_key == "wrapper_paper_id":
            return str(wrapper.get("paper_id", "")).lower()
        if sort_key == "wrapper_year":
            year = wrapper.get("year", None)
            return int(year) if isinstance(year, int) else None
        raise ValueError(
            "--sort must be one of: key, interface, requires, description, paper_id, paper_year, wrapper_paper_id, wrapper_year"
        )

    if sort_key in {"paper_year", "wrapper_year"}:
        present = []
        missing = []
        for r in rows:
            if _sort_value(r) is None:
                missing.append(r)
            else:
                present.append(r)
        present.sort(key=lambda r: int(_sort_value(r)), reverse=descending)  # type: ignore[arg-type]
        rows = present + missing
    else:
        rows.sort(key=lambda r: str(_sort_value(r)), reverse=descending)  # type: ignore[arg-type]

    limit = int(args.limit)
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if limit:
        rows = rows[: min(limit, 100000)]

    fmt = str(args.format)
    if fmt == "json":
        _emit(rows, output=str(args.output), fmt="json")
        return 0

    columns_raw = str(getattr(args, "columns", "")).strip()
    if columns_raw:
        columns = [c.strip() for c in columns_raw.split(",") if c.strip()]
    else:
        columns = ["key", "requires", "description"]

    def _col_value(r: dict[str, Any], col: str) -> object:
        paper = r.get("paper") if isinstance(r.get("paper"), dict) else {}
        wrapper = r.get("wrapper_paper") if isinstance(r.get("wrapper_paper"), dict) else {}

        if col == "key":
            return r.get("key", "")
        if col == "interface":
            return r.get("interface", "")
        if col == "requires":
            return r.get("requires", "")
        if col == "description":
            return r.get("description", "")
        if col == "paper_id":
            return paper.get("paper_id", "")
        if col == "paper_year":
            return paper.get("year", "")
        if col == "paper_title":
            return paper.get("title", "")
        if col == "wrapper_paper_id":
            return wrapper.get("paper_id", "")
        if col == "wrapper_year":
            return wrapper.get("year", "")
        if col == "wrapper_title":
            return wrapper.get("title", "")
        raise ValueError(
            "--columns must be a comma-separated list of: key, interface, requires, description, paper_id, paper_year, paper_title, wrapper_paper_id, wrapper_year, wrapper_title"
        )

    lines: list[str] = []
    if bool(getattr(args, "header", False)):
        lines.append("\t".join(columns))
    for r in rows:
        lines.append("\t".join(_sanitize_tsv_cell(_col_value(r, c)) for c in columns))
    _emit_text("\n".join(lines), output=str(args.output))
    return 0


def _cmd_models_info(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec

    spec = get_model_spec(str(args.key))
    payload = {
        "key": spec.key,
        "interface": str(spec.interface),
        "description": spec.description,
        "requires": list(spec.requires),
        "default_params": dict(spec.default_params),
        "param_help": dict(spec.param_help),
    }
    paper = _paper_payload_for_model_key(spec.key)
    if paper:
        payload["paper"] = paper
    wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
    if wrapper_pid:
        wrapper = _paper_payload_for_paper_id(wrapper_pid)
        if wrapper:
            payload["wrapper_paper"] = wrapper
    _emit(payload, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_models_search(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    q = str(args.query).strip()
    if not q:
        raise ValueError("query must be non-empty")

    tokens = [t.strip().lower() for t in q.split() if t.strip()]
    any_mode = bool(args.any)
    prefix = str(args.prefix).strip()

    interface_filter = str(getattr(args, "interface", "any")).strip().lower()
    if interface_filter not in {"any", "local", "global"}:
        raise ValueError("--interface must be one of: any, local, global")

    req_set, include_core = _parse_requires_filter(str(getattr(args, "requires", "")))
    exc_set, exclude_core = _parse_requires_filter(str(getattr(args, "exclude_requires", "")))

    limit = int(args.limit)
    if limit <= 0:
        raise ValueError("--limit must be >= 1")
    # Avoid pathological output sizes when called from a script by accident.
    limit = min(limit, 1000)

    def _score_token(
        token: str,
        *,
        key_l: str,
        desc_l: str,
        requires_l: str,
        paper_id_l: str,
        paper_title_l: str,
        paper_year: int | None,
        wrapper_id_l: str,
        wrapper_title_l: str,
        wrapper_year: int | None,
    ) -> int:
        best = 0
        if token in key_l:
            best = max(best, 10)
        if token in desc_l:
            best = max(best, 4)
        if token in requires_l:
            best = max(best, 2)
        if token in paper_id_l:
            best = max(best, 8)
        if token in paper_title_l:
            best = max(best, 6)
        if token in wrapper_id_l:
            best = max(best, 7)
        if token in wrapper_title_l:
            best = max(best, 5)
        if token.isdigit():
            year = int(token)
            if paper_year is not None and int(paper_year) == year:
                best = max(best, 6)
            if wrapper_year is not None and int(wrapper_year) == year:
                best = max(best, 5)
        return best

    rows: list[dict[str, Any]] = []
    for key in list_models():
        if prefix and not str(key).startswith(prefix):
            continue
        spec = get_model_spec(key)

        if interface_filter != "any" and str(spec.interface) != interface_filter:
            continue

        reqs = set(spec.requires)
        if req_set or include_core:
            ok = False
            if include_core and not reqs:
                ok = True
            if req_set and reqs.intersection(req_set):
                ok = True
            if not ok:
                continue

        if exc_set or exclude_core:
            if exclude_core and not reqs:
                continue
            if exc_set and reqs.intersection(exc_set):
                continue

        paper = _paper_payload_for_model_key(spec.key) or {}
        wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
        wrapper = _paper_payload_for_paper_id(wrapper_pid) if wrapper_pid else None
        wrapper = wrapper or {}

        key_l = str(spec.key).lower()
        desc_l = str(spec.description).lower()
        requires_l = ",".join(spec.requires).lower()

        paper_id_l = str(paper.get("paper_id", "")).lower()
        paper_title_l = str(paper.get("title", "")).lower()
        paper_year = paper.get("year", None)
        paper_year_i = int(paper_year) if isinstance(paper_year, int) else None

        wrapper_id_l = str(wrapper.get("paper_id", "")).lower()
        wrapper_title_l = str(wrapper.get("title", "")).lower()
        wrapper_year = wrapper.get("year", None)
        wrapper_year_i = int(wrapper_year) if isinstance(wrapper_year, int) else None

        token_scores = [
            _score_token(
                t,
                key_l=key_l,
                desc_l=desc_l,
                requires_l=requires_l,
                paper_id_l=paper_id_l,
                paper_title_l=paper_title_l,
                paper_year=paper_year_i,
                wrapper_id_l=wrapper_id_l,
                wrapper_title_l=wrapper_title_l,
                wrapper_year=wrapper_year_i,
            )
            for t in tokens
        ]

        matched = any(s > 0 for s in token_scores) if any_mode else all(s > 0 for s in token_scores)
        if not matched:
            continue

        score = int(sum(token_scores))
        row: dict[str, Any] = {
            "key": spec.key,
            "score": score,
            "requires": ",".join(spec.requires),
            "description": spec.description,
        }
        if paper:
            row["paper"] = paper
        if wrapper:
            # Only include if it looks like a real wrapper paper.
            if str(wrapper.get("paper_id", "")).strip():
                row["wrapper_paper"] = wrapper
        rows.append(row)

    rows.sort(key=lambda r: (-int(r.get("score", 0)), str(r.get("key", ""))))
    rows = rows[:limit]

    fmt = str(args.format)
    if fmt == "json":
        _emit(rows, output=str(args.output), fmt="json")
        return 0

    lines: list[str] = []
    for r in rows:
        paper = r.get("paper") if isinstance(r.get("paper"), dict) else {}
        pid = str(paper.get("paper_id", "")).strip()
        year = paper.get("year", None)
        year_s = str(year) if isinstance(year, int) else ""

        lines.append(
            "\t".join(
                [
                    _sanitize_tsv_cell(r.get("key", "")),
                    _sanitize_tsv_cell(r.get("score", "")),
                    _sanitize_tsv_cell(pid),
                    _sanitize_tsv_cell(year_s),
                    _sanitize_tsv_cell(r.get("requires", "")),
                    _sanitize_tsv_cell(r.get("description", "")),
                ]
            )
        )
    _emit_text("\n".join(lines), output=str(args.output))
    return 0


def _cmd_papers_list(args: argparse.Namespace) -> int:
    meta = _load_rnn_paper_metadata()
    query = str(args.query).strip().lower()

    rows: list[dict[str, Any]] = []
    for paper_id in sorted(meta):
        entry = meta.get(paper_id, {}) if isinstance(meta.get(paper_id, {}), dict) else {}
        title = str(entry.get("title", "")).strip()
        if query and query not in str(paper_id).lower() and query not in title.lower():
            continue
        year_raw = entry.get("year", None)
        year = int(year_raw) if isinstance(year_raw, int) else None
        rows.append(
            {
                "paper_id": str(paper_id),
                "title": title,
                "year": year,
                "doi": str(entry.get("doi", "")).strip(),
                "arxiv_id": str(entry.get("arxiv_id", "")).strip(),
                "url": str(entry.get("url", "")).strip(),
            }
        )

    fmt = str(args.format)
    if fmt == "json":
        _emit(rows, output=str(args.output), fmt="json")
        return 0

    lines = [
        "\t".join(
            [
                _sanitize_tsv_cell(r.get("paper_id", "")),
                _sanitize_tsv_cell(r.get("year", "")),
                _sanitize_tsv_cell(r.get("title", "")),
                _sanitize_tsv_cell(r.get("url", "")),
            ]
        )
        for r in rows
    ]
    _emit_text("\n".join(lines), output=str(args.output))
    return 0


def _cmd_papers_info(args: argparse.Namespace) -> int:
    pid = str(args.paper_id).strip()
    if not pid:
        raise ValueError("paper_id must be non-empty")

    meta = _load_rnn_paper_metadata()
    entry = meta.get(pid, None)
    if not isinstance(entry, dict):
        raise ValueError(f"Unknown paper_id: {pid!r}")

    title = str(entry.get("title", "")).strip()
    year_raw = entry.get("year", None)
    year = int(year_raw) if isinstance(year_raw, int) else None
    payload = {
        "paper_id": pid,
        "title": title,
        "year": year,
        "doi": str(entry.get("doi", "")).strip(),
        "arxiv_id": str(entry.get("arxiv_id", "")).strip(),
        "url": str(entry.get("url", "")).strip(),
    }
    _emit(payload, output=str(args.output), fmt="json")
    return 0


def _cmd_papers_models(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    pid = str(args.paper_id).strip()
    if not pid:
        raise ValueError("paper_id must be non-empty")

    meta = _load_rnn_paper_metadata()
    if pid not in meta:
        raise ValueError(f"Unknown paper_id: {pid!r}")

    prefix = str(args.prefix).strip()
    role = str(args.role).strip().lower()
    if role not in {"any", "base", "wrapper"}:
        raise ValueError("--role must be one of: any, base, wrapper")

    rows: list[dict[str, Any]] = []
    for key in list_models():
        if prefix and not str(key).startswith(prefix):
            continue
        spec = get_model_spec(key)

        base = _paper_payload_for_model_key(spec.key) or {}
        base_pid = str(base.get("paper_id", "")).strip()

        wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
        wrapper = _paper_payload_for_paper_id(wrapper_pid) if wrapper_pid else None
        wrapper = wrapper or {}
        wrap_pid = str(wrapper.get("paper_id", "")).strip()

        hit_base = base_pid == pid
        hit_wrap = wrap_pid == pid
        if role == "base":
            if not hit_base:
                continue
        elif role == "wrapper":
            if not hit_wrap:
                continue
        else:
            if not (hit_base or hit_wrap):
                continue

        if hit_base:
            rows.append(
                {
                    "key": spec.key,
                    "role": "base",
                    "requires": ",".join(spec.requires),
                    "description": spec.description,
                }
            )
        if hit_wrap:
            rows.append(
                {
                    "key": spec.key,
                    "role": "wrapper",
                    "requires": ",".join(spec.requires),
                    "description": spec.description,
                }
            )

    rows.sort(key=lambda r: (str(r.get("role", "")), str(r.get("key", ""))))

    fmt = str(args.format)
    if fmt == "json":
        _emit(rows, output=str(args.output), fmt="json")
        return 0

    lines = [
        "\t".join(
            [
                _sanitize_tsv_cell(r.get("key", "")),
                _sanitize_tsv_cell(r.get("role", "")),
                _sanitize_tsv_cell(r.get("requires", "")),
                _sanitize_tsv_cell(r.get("description", "")),
            ]
        )
        for r in rows
    ]
    _emit_text("\n".join(lines), output=str(args.output))
    return 0


def _cmd_docs_rnn(args: argparse.Namespace) -> int:
    from .docsgen.rnn import render_rnn_paper_zoo_doc, render_rnn_zoo_doc, write_rnn_docs

    out_dir = Path(str(args.output_dir)).expanduser().resolve()

    if bool(args.check):
        expected_paper = render_rnn_paper_zoo_doc()
        expected_zoo = render_rnn_zoo_doc()
        paper_path = out_dir / "rnn_paper_zoo.md"
        zoo_path = out_dir / "rnn_zoo.md"

        failures: list[str] = []
        if not paper_path.exists():
            failures.append(str(paper_path))
        else:
            actual = paper_path.read_text(encoding="utf-8")
            if actual != expected_paper:
                failures.append(str(paper_path))

        if not zoo_path.exists():
            failures.append(str(zoo_path))
        else:
            actual = zoo_path.read_text(encoding="utf-8")
            if actual != expected_zoo:
                failures.append(str(zoo_path))

        if failures:
            print("Docs out of date (or missing):", file=sys.stderr)
            for p in failures:
                print(f"- {p}", file=sys.stderr)
            return 1

        print("OK: RNN docs are up to date.")
        return 0

    write_rnn_docs(output_dir=out_dir)
    print("Wrote:")
    print(f"- {(out_dir / 'rnn_paper_zoo.md').as_posix()}")
    print(f"- {(out_dir / 'rnn_zoo.md').as_posix()}")
    return 0


def _cmd_cv_run(args: argparse.Namespace) -> int:
    from .cv import cross_validation_predictions

    model_params = _parse_model_params(list(args.model_param))
    y_col = str(args.y_col).strip() or None
    df = cross_validation_predictions(
        model=str(args.model),
        dataset=str(args.dataset),
        y_col=y_col,
        horizon=int(args.horizon),
        step_size=int(args.step_size),
        min_train_size=int(args.min_train_size),
        max_train_size=args.max_train_size,
        n_windows=args.n_windows,
        model_params=model_params,
        data_dir=str(args.data_dir),
    )
    _emit_dataframe(df, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_datasets_list(args: argparse.Namespace) -> int:
    from .datasets.registry import (
        describe_dataset,
        get_dataset_spec,
        list_datasets,
        resolve_dataset_path,
    )

    for key in list_datasets():
        if args.with_path:
            try:
                path = resolve_dataset_path(key, data_dir=str(args.data_dir))
                path_s = path.as_posix()
            except FileNotFoundError:
                # For non-bundled datasets, show the expected relative path instead
                # of failing the whole listing.
                path_s = get_dataset_spec(key).rel_path.as_posix()
            print(f"{key}\t{path_s}\t{describe_dataset(key)}")
        else:
            print(f"{key}\t{describe_dataset(key)}")
    return 0


def _cmd_datasets_preview(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset

    df = load_dataset(args.key, nrows=int(args.nrows), data_dir=str(args.data_dir))
    print(df.head(int(args.nrows)).to_string(index=False))
    return 0


def _cmd_datasets_path(args: argparse.Namespace) -> int:
    from .datasets.registry import resolve_dataset_path

    path = resolve_dataset_path(str(args.key), data_dir=str(args.data_dir))
    print(path.as_posix())
    return 0


def _cmd_datasets_validate(args: argparse.Namespace) -> int:
    from .datasets.loaders import load_dataset
    from .datasets.registry import get_dataset_spec, list_datasets

    nrows = int(args.nrows)
    keys = [str(args.dataset)] if str(args.dataset) else list_datasets()
    failures = 0
    for key in keys:
        try:
            spec = get_dataset_spec(key)
            df = load_dataset(key, nrows=nrows, data_dir=str(args.data_dir))
            if len(df) <= 0:
                raise ValueError("loaded 0 rows")
            missing = sorted(spec.expected_columns.difference(df.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")

            if spec.parse_dates:
                import pandas as pd

                for col in spec.parse_dates:
                    if col not in df.columns:
                        raise ValueError(f"missing parse_dates column: {col!r}")
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        raise ValueError(f"parse_dates column is not datetime: {col!r}")
                    if df[col].isna().any():
                        raise ValueError(f"parse_dates column contains NaT/NA values: {col!r}")

            if bool(args.check_time):
                from .data.format import to_long, validate_long_df

                long_df = to_long(
                    df,
                    time_col=spec.time_col,
                    y_col=spec.default_y,
                    id_cols=tuple(spec.group_cols),
                    dropna=True,
                )
                long_df = long_df.sort_values(["unique_id", "ds"], kind="mergesort")
                validate_long_df(long_df, require_sorted=True, require_unique_ds=True)
            print(f"OK {key} rows={len(df)} cols={len(df.columns)}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {key}: {type(e).__name__}: {e}", file=sys.stderr)

    return 1 if failures else 0


def _cmd_eval_naive_last(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last

    payload = eval_naive_last(
        dataset=str(args.dataset),
        y_col=str(args.y_col),
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
        max_windows=args.max_windows,
        data_dir=str(args.data_dir),
    )
    _emit(payload, output=args.output, fmt=str(args.format))

    return 0


def _cmd_eval_seasonal_naive(args: argparse.Namespace) -> int:
    from .eval import eval_seasonal_naive

    payload = eval_seasonal_naive(
        dataset=str(args.dataset),
        y_col=str(args.y_col),
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
        season_length=int(args.season_length),
        max_windows=args.max_windows,
        data_dir=str(args.data_dir),
    )
    _emit(payload, output=args.output, fmt=str(args.format))
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model

    model_params = _parse_model_params(list(args.model_param))
    y_col = str(args.y_col).strip() or None
    payload = eval_model(
        model=str(args.model),
        dataset=str(args.dataset),
        y_col=y_col,
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
        max_windows=args.max_windows,
        max_train_size=args.max_train_size,
        conformal_levels=str(args.conformal_levels).strip() or None,
        conformal_per_step=(not bool(args.conformal_pooled)),
        model_params=model_params,
        data_dir=str(args.data_dir),
    )
    _emit(payload, output=args.output, fmt=str(args.format))
    return 0


def _cmd_eval_csv(args: argparse.Namespace) -> int:
    from .data.format import to_long
    from .eval_forecast import eval_model_long_df
    from .io import ensure_datetime, load_csv, parse_id_cols
    from .models.registry import get_model_spec

    model_params = _parse_model_params(list(args.model_param))
    id_cols = parse_id_cols(str(args.id_cols))

    df = load_csv(str(args.path))
    if bool(args.parse_dates):
        ensure_datetime(df, str(args.time_col))

    model_spec = get_model_spec(str(args.model))
    x_cols: tuple[str, ...] = ()
    if model_spec.interface == "global" and "x_cols" in model_params:
        raw = model_params.get("x_cols")
        if raw is not None:
            if isinstance(raw, str):
                x_cols = tuple([p.strip() for p in raw.split(",") if p.strip()])
            elif isinstance(raw, list | tuple):
                x_cols = tuple([str(p).strip() for p in raw if str(p).strip()])
            else:
                s = str(raw).strip()
                x_cols = (s,) if s else ()

    long_df = to_long(
        df,
        time_col=str(args.time_col),
        y_col=str(args.y_col),
        id_cols=id_cols,
        x_cols=x_cols,
        dropna=True,
    )
    payload = eval_model_long_df(
        model=str(args.model),
        long_df=long_df,
        horizon=int(args.horizon),
        step=int(args.step),
        min_train_size=int(args.min_train_size),
        max_windows=args.max_windows,
        max_train_size=args.max_train_size,
        conformal_levels=str(args.conformal_levels).strip() or None,
        conformal_per_step=(not bool(args.conformal_pooled)),
        model_params=model_params,
    )
    payload.update(
        {
            "dataset": str(args.path),
            "time_col": str(args.time_col),
            "y_col": str(args.y_col),
            "id_cols": list(id_cols),
        }
    )
    _emit(payload, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_leaderboard_naive(args: argparse.Namespace) -> int:
    from .eval import eval_naive_last, eval_seasonal_naive

    rows = [
        eval_naive_last(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        ),
        eval_seasonal_naive(
            dataset=str(args.dataset),
            y_col=str(args.y_col),
            horizon=int(args.horizon),
            step=int(args.step),
            min_train_size=int(args.min_train_size),
            season_length=int(args.season_length),
            max_windows=args.max_windows,
            data_dir=str(args.data_dir),
        ),
    ]
    _emit(rows, output=args.output, fmt=str(args.format))
    return 0


def _cmd_leaderboard_models(args: argparse.Namespace) -> int:
    from .eval_forecast import eval_model
    from .models.registry import get_model_spec, list_models

    y_col = str(args.y_col).strip() or None

    if str(args.models).strip():
        keys = [k.strip() for k in str(args.models).split(",") if k.strip()]
    else:
        keys = [k for k in list_models() if args.include_optional or not get_model_spec(k).requires]

    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            payload = eval_model(
                model=str(key),
                dataset=str(args.dataset),
                y_col=y_col,
                horizon=int(args.horizon),
                step=int(args.step),
                min_train_size=int(args.min_train_size),
                max_windows=args.max_windows,
                data_dir=str(args.data_dir),
            )
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {key}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        slim = {k: v for k, v in payload.items() if not str(k).endswith("_by_step")}
        rows.append(slim)

    rows.sort(key=lambda r: float(r.get("mae", float("inf"))))
    _emit(rows, output=str(args.output), fmt=str(args.format))
    return 0


_SWEEP_LONG_DF_CACHE: dict[tuple[str, str, str], Any] = {}


def _leaderboard_sweep_worker(
    dataset: str,
    model_keys: tuple[str, ...],
    y_col: str | None,
    horizon: int,
    step: int,
    min_train_size: int,
    max_windows: int | None,
    data_dir: str,
    strict: bool,
    model_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    import threading

    from .data.format import to_long
    from .datasets.loaders import load_dataset
    from .datasets.registry import get_dataset_spec
    from .eval_forecast import eval_model_long_df

    data_dir_s = str(data_dir).strip()
    data_dir_arg = data_dir_s or None

    dataset_key = str(dataset)
    spec = get_dataset_spec(dataset_key)
    y_col_final = (
        str(y_col).strip() if (y_col is not None and str(y_col).strip()) else spec.default_y
    )

    cache_key = (dataset_key, y_col_final, data_dir_s)
    lock = getattr(_leaderboard_sweep_worker, "_cache_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(_leaderboard_sweep_worker, "_cache_lock", lock)

    with lock:
        long_df = _SWEEP_LONG_DF_CACHE.get(cache_key)

    if long_df is None:
        df = load_dataset(dataset_key, data_dir=data_dir_arg)
        long_df = to_long(
            df,
            time_col=spec.time_col,
            y_col=y_col_final,
            id_cols=tuple(spec.group_cols),
            dropna=True,
        )
        if long_df.empty:
            raise ValueError("Loaded 0 rows after to_long(dropna=True). Check dataset and y_col.")

        with lock:
            _SWEEP_LONG_DF_CACHE[cache_key] = long_df

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for model_key in model_keys:
        try:
            payload = eval_model_long_df(
                model=str(model_key),
                long_df=long_df,
                horizon=int(horizon),
                step=int(step),
                min_train_size=int(min_train_size),
                max_windows=max_windows,
                model_params=dict(model_params),
            )
        except Exception as e:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"{dataset_key}/{model_key}: {type(e).__name__}: {e}") from e
            errors.append(f"SKIP {dataset_key}/{model_key}: {type(e).__name__}: {e}")
            continue

        payload["dataset"] = dataset_key
        payload["y_col"] = y_col_final
        payload["data_dir"] = data_dir_s
        payload["model_params"] = dict(model_params)

        # Keep output compact (same as other leaderboard commands).
        rows.append({k: v for k, v in payload.items() if not str(k).endswith("_by_step")})

    return (rows, errors)


def _cmd_leaderboard_sweep(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset key")

    y_col = str(args.y_col).strip() or None

    if str(args.models).strip():
        keys = [k.strip() for k in str(args.models).split(",") if k.strip()]
    else:
        keys = [k for k in list_models() if args.include_optional or not get_model_spec(k).requires]

    jobs = int(args.jobs)
    if jobs <= 0:
        raise ValueError("--jobs must be >= 1")

    chunk_size = int(getattr(args, "chunk_size", 1))
    if chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    strict = bool(getattr(args, "strict", False))
    model_params = _parse_model_params(list(getattr(args, "model_param", [])))

    resume_path = str(getattr(args, "resume", "")).strip()
    resume_rows: list[dict[str, Any]] = []
    done_pairs: set[tuple[str, str]] = set()
    if resume_path:
        from .datasets.registry import get_dataset_spec

        path = Path(resume_path)
        if not path.exists():
            raise FileNotFoundError(f"--resume path not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        lower = path.name.lower()
        fmt = "csv" if lower.endswith(".csv") else "json"
        if not (lower.endswith(".csv") or lower.endswith(".json")):
            try:
                json.loads(raw_text)
                fmt = "json"
            except Exception:  # noqa: BLE001
                fmt = "csv"

        rows_raw: list[dict[str, Any]] = []
        if fmt == "json":
            parsed = json.loads(raw_text) if str(raw_text).strip() else []
            if isinstance(parsed, dict):
                rows_raw = [parsed]
            elif isinstance(parsed, list):
                rows_raw = [r for r in parsed if isinstance(r, dict)]
            else:
                raise TypeError("--resume json must be a dict row or a list of dict rows")
        else:
            reader = csv.DictReader(io.StringIO(raw_text))
            rows_raw = [dict(r) for r in reader]

        datasets_set = set(datasets)
        keys_set = set(keys)

        dataset_y_cols: dict[str, str] = {}
        for d in datasets:
            spec = get_dataset_spec(str(d))
            dataset_y_cols[str(d)] = (
                str(y_col).strip() if y_col is not None else str(spec.default_y)
            )

        horizon_i = int(args.horizon)
        step_i = int(args.step)
        min_train_size_i = int(args.min_train_size)
        max_windows_i = None if args.max_windows is None else int(args.max_windows)
        expected_data_dir = str(args.data_dir).strip()
        expected_params_norm = json.loads(
            json.dumps(model_params, ensure_ascii=False, sort_keys=True)
        )

        def _as_int(v: object) -> int | None:
            if v is None:
                return None
            if isinstance(v, bool):
                return None
            if isinstance(v, int):
                return int(v)
            if isinstance(v, float) and float(v).is_integer():
                return int(v)
            s = str(v).strip()
            if not s:
                return None
            lower_s = s.lower()
            if lower_s in {"none", "null"}:
                return None
            try:
                return int(float(s))
            except Exception:  # noqa: BLE001
                return None

        def _as_str(v: object) -> str:
            if v is None:
                return ""
            return str(v).strip()

        def _as_params_dict(v: object) -> dict[str, Any] | None:
            if v is None:
                return None
            if isinstance(v, dict):
                out: dict[str, Any] = {}
                for k, val in v.items():
                    if isinstance(k, str):
                        out[k] = val
                    else:
                        out[str(k)] = val
                return out
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                try:
                    parsed = json.loads(s)
                except Exception:  # noqa: BLE001
                    return None
                if isinstance(parsed, dict):
                    return {str(k): val for k, val in parsed.items()}
                return None
            return None

        for r in rows_raw:
            ds = _as_str(r.get("dataset"))
            model = _as_str(r.get("model"))
            if not ds or not model:
                continue
            if ds not in datasets_set or model not in keys_set:
                continue

            if _as_str(r.get("y_col")) != dataset_y_cols.get(ds, ""):
                continue
            if _as_int(r.get("horizon")) != horizon_i:
                continue
            if _as_int(r.get("step")) != step_i:
                continue
            if _as_int(r.get("min_train_size")) != min_train_size_i:
                continue
            if _as_int(r.get("max_windows")) != max_windows_i:
                continue
            if _as_str(r.get("data_dir")) != expected_data_dir:
                continue

            row_params = _as_params_dict(r.get("model_params"))
            if row_params is None:
                if expected_params_norm:
                    continue
            else:
                row_params_norm = json.loads(
                    json.dumps(row_params, ensure_ascii=False, sort_keys=True)
                )
                if row_params_norm != expected_params_norm:
                    continue

            resume_rows.append(r)
            done_pairs.add((ds, model))

        if args.progress:
            print(
                f"RESUME keep={len(resume_rows)} skip={len(done_pairs)}",
                file=sys.stderr,
            )

    tasks: list[tuple[str, tuple[str, ...]]] = []
    for dataset in datasets:
        remaining = [k for k in keys if (dataset, k) not in done_pairs]
        if not remaining:
            continue
        if chunk_size == 0:
            tasks.append((dataset, tuple(remaining)))
            continue
        for i in range(0, len(remaining), chunk_size):
            chunk = tuple(remaining[i : i + chunk_size])
            if chunk:
                tasks.append((dataset, chunk))

    failure_lines: list[str] = []
    rows, _failures = _run_parallel_tasks(
        tasks,
        jobs=jobs,
        backend=str(args.backend),
        progress=bool(args.progress),
        strict=strict,
        worker=_leaderboard_sweep_worker,
        worker_args=(
            y_col,
            int(args.horizon),
            int(args.step),
            int(args.min_train_size),
            args.max_windows,
            str(args.data_dir),
            strict,
            model_params,
        ),
        errors_out=failure_lines,
    )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for r in resume_rows:
        ds = str(r.get("dataset", "")).strip()
        model = str(r.get("model", "")).strip()
        if ds and model:
            merged[(ds, model)] = r
    for r in rows:
        ds = str(r.get("dataset", "")).strip()
        model = str(r.get("model", "")).strip()
        if ds and model:
            merged[(ds, model)] = r

    final_rows = list(merged.values()) if merged else rows

    def _mae_key(v: object) -> float:
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return float("inf")

    final_rows.sort(key=lambda r: (str(r.get("dataset", "")), _mae_key(r.get("mae", float("inf")))))

    failures_output = str(getattr(args, "failures_output", "")).strip()
    if failures_output:
        out_path = Path(failures_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if failure_lines:
            out_path.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
        else:
            out_path.write_text("", encoding="utf-8")

    summary_output = str(getattr(args, "summary_output", "")).strip()
    if summary_output:
        summary_format = str(getattr(args, "summary_format", "json")).strip()
        summary_sort = str(getattr(args, "summary_sort", "mae_rank_mean"))
        summary_limit = int(getattr(args, "summary_limit", 0))
        summary_min_datasets = int(getattr(args, "summary_min_datasets", 0))

        summary_rows = (
            []
            if not final_rows
            else _summarize_leaderboard_rows(
                final_rows,
                sort=summary_sort,
                limit=summary_limit,
                min_datasets=summary_min_datasets,
            )
        )
        text = _format_table(
            summary_rows,
            columns=_leaderboard_summary_columns(),
            fmt=summary_format,
        )
        out_path = Path(summary_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    _emit(final_rows, output=str(args.output), fmt=str(args.format))
    return 0


def _cmd_leaderboard_summarize(args: argparse.Namespace) -> int:
    input_path = str(getattr(args, "input", "-")).strip() or "-"
    input_fmt = str(getattr(args, "input_format", "auto")).strip().lower()

    if input_path == "-":
        raw_text = sys.stdin.read()
    else:
        raw_text = Path(input_path).read_text(encoding="utf-8")

    fmt = input_fmt
    if fmt == "auto":
        lower = input_path.lower()
        if input_path != "-" and lower.endswith(".csv"):
            fmt = "csv"
        elif input_path != "-" and lower.endswith(".json"):
            fmt = "json"
        else:
            try:
                json.loads(raw_text)
                fmt = "json"
            except Exception:  # noqa: BLE001
                fmt = "csv"

    rows_raw: list[dict[str, Any]] = []
    if fmt == "json":
        parsed = json.loads(raw_text) if str(raw_text).strip() else []
        if isinstance(parsed, dict):
            rows_raw = [parsed]
        elif isinstance(parsed, list):
            rows_raw = [r for r in parsed if isinstance(r, dict)]
        else:
            raise TypeError("JSON input must be a dict row or a list of dict rows")
    elif fmt == "csv":
        reader = csv.DictReader(io.StringIO(raw_text))
        rows_raw = [dict(r) for r in reader]
    else:
        raise ValueError("--input-format must be one of: auto, json, csv")

    if not rows_raw:
        raise ValueError("No rows found in input")

    summary = _summarize_leaderboard_rows(
        rows_raw,
        sort=str(getattr(args, "sort", "mae_mean")),
        limit=int(getattr(args, "limit", 0)),
        min_datasets=int(getattr(args, "min_datasets", 0)),
    )
    _emit_table(
        summary,
        columns=_leaderboard_summary_columns(),
        output=str(args.output),
        fmt=str(args.format),
    )
    return 0


def _summarize_leaderboard_rows(
    rows_raw: list[dict[str, Any]],
    *,
    sort: str,
    limit: int,
    min_datasets: int = 0,
) -> list[dict[str, Any]]:
    import statistics

    def _as_float(v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int | float):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:  # noqa: BLE001
            return None

    def _as_int(v: object) -> int | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:  # noqa: BLE001
            return None

    cleaned: list[dict[str, Any]] = []
    bad = 0
    for r in rows_raw:
        model = str(r.get("model", "")).strip()
        dataset = str(r.get("dataset", "")).strip()
        if not model or not dataset:
            bad += 1
            continue

        cleaned.append(
            {
                "model": model,
                "dataset": dataset,
                "mae": _as_float(r.get("mae")),
                "rmse": _as_float(r.get("rmse")),
                "mape": _as_float(r.get("mape")),
                "smape": _as_float(r.get("smape")),
                "n_points": _as_int(r.get("n_points")),
            }
        )

    if not cleaned:
        raise ValueError(f"No valid rows found (bad_rows={bad})")

    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in cleaned:
        by_model.setdefault(str(r["model"]), []).append(r)

    n_datasets_total = int(len({str(r["dataset"]) for r in cleaned}))

    metrics = ["mae", "rmse", "mape", "smape"]
    best_by_dataset_metric: dict[tuple[str, str], float] = {}
    for dataset in sorted({str(r["dataset"]) for r in cleaned}):
        rows_ds = [r for r in cleaned if str(r["dataset"]) == dataset]
        for m in metrics:
            values = [float(r[m]) for r in rows_ds if r.get(m) is not None]
            if not values:
                continue
            best_by_dataset_metric[(dataset, m)] = float(min(values))

    rank_by_dataset_metric_model: dict[tuple[str, str, str], float] = {}
    for dataset in sorted({str(r["dataset"]) for r in cleaned}):
        rows_ds = [r for r in cleaned if str(r["dataset"]) == dataset]
        for m in metrics:
            vals = [(str(r["model"]), r.get(m)) for r in rows_ds if r.get(m) is not None]
            if not vals:
                continue
            # Dense ranks: 1,2,2,3...
            vals_sorted = sorted(vals, key=lambda t: float(t[1]))  # type: ignore[arg-type]
            rank = 1
            prev: float | None = None
            for model, v_obj in vals_sorted:
                v = float(v_obj)  # type: ignore[arg-type]
                if prev is None:
                    prev = v
                elif v != prev:
                    rank += 1
                    prev = v
                rank_by_dataset_metric_model[(dataset, m, model)] = float(rank)

    out: list[dict[str, Any]] = []
    for model, items in by_model.items():
        datasets = {str(it["dataset"]) for it in items}
        row: dict[str, Any] = {
            "model": model,
            "n_datasets": int(len(datasets)),
            "n_datasets_total": int(n_datasets_total),
            "dataset_coverage": (
                None
                if n_datasets_total <= 0
                else float(int(len(datasets)) / float(int(n_datasets_total)))
            ),
            "n_rows": int(len(items)),
        }

        for m in metrics:
            vals = [float(it[m]) for it in items if it.get(m) is not None]
            row[f"{m}_mean"] = None if not vals else float(sum(vals) / float(len(vals)))
            row[f"{m}_median"] = None if not vals else float(statistics.median(vals))

            pairs = [
                (float(it[m]), int(it["n_points"]))
                for it in items
                if it.get(m) is not None
                and it.get("n_points") is not None
                and int(it["n_points"]) > 0
            ]
            w_sum = float(sum(w for _v, w in pairs))
            row[f"{m}_wmean"] = (
                None if w_sum <= 0 else float(sum(v * float(w) for v, w in pairs) / w_sum)
            )

            rels: list[float] = []
            for it in items:
                v_obj = it.get(m)
                if v_obj is None:
                    continue
                best = best_by_dataset_metric.get((str(it["dataset"]), m))
                if best is None:
                    continue
                v = float(v_obj)
                if best == 0.0:
                    rel = 1.0 if v == 0.0 else float("inf")
                else:
                    rel = float(v / best)
                rels.append(rel)

            row[f"{m}_rel_mean"] = None if not rels else float(sum(rels) / float(len(rels)))
            row[f"{m}_rel_median"] = None if not rels else float(statistics.median(rels))

            rel_pairs = []
            for it in items:
                v_obj = it.get(m)
                if v_obj is None:
                    continue
                w = it.get("n_points")
                if w is None or int(w) <= 0:
                    continue
                best = best_by_dataset_metric.get((str(it["dataset"]), m))
                if best is None:
                    continue
                v = float(v_obj)
                if best == 0.0:
                    rel = 1.0 if v == 0.0 else float("inf")
                else:
                    rel = float(v / best)
                rel_pairs.append((rel, int(w)))

            rel_wsum = float(sum(w for _r, w in rel_pairs))
            row[f"{m}_rel_wmean"] = (
                None if rel_wsum <= 0 else float(sum(r * float(w) for r, w in rel_pairs) / rel_wsum)
            )

            ranks = [
                rank_by_dataset_metric_model.get((str(it["dataset"]), m, model))
                for it in items
                if rank_by_dataset_metric_model.get((str(it["dataset"]), m, model)) is not None
            ]
            row[f"{m}_rank_mean"] = None if not ranks else float(sum(ranks) / float(len(ranks)))

            rank_pairs = [
                (
                    float(rank_by_dataset_metric_model[(str(it["dataset"]), m, model)]),
                    int(it["n_points"]),
                )
                for it in items
                if (str(it["dataset"]), m, model) in rank_by_dataset_metric_model
                and it.get("n_points") is not None
                and int(it["n_points"]) > 0
            ]
            rank_wsum = float(sum(w for _r, w in rank_pairs))
            row[f"{m}_rank_wmean"] = (
                None
                if rank_wsum <= 0
                else float(sum(r * float(w) for r, w in rank_pairs) / rank_wsum)
            )

        score_rank_items = [row.get(f"{m}_rank_mean") for m in metrics]
        score_rank_vals = [float(v) for v in score_rank_items if v is not None]
        row["score_rank_mean"] = (
            None if len(score_rank_vals) != len(metrics) else float(sum(score_rank_vals) / 4.0)
        )

        score_rank_w_items = [row.get(f"{m}_rank_wmean") for m in metrics]
        score_rank_w_vals = [float(v) for v in score_rank_w_items if v is not None]
        row["score_rank_wmean"] = (
            None if len(score_rank_w_vals) != len(metrics) else float(sum(score_rank_w_vals) / 4.0)
        )

        score_rel_items = [row.get(f"{m}_rel_mean") for m in metrics]
        score_rel_vals = [float(v) for v in score_rel_items if v is not None]
        row["score_rel_mean"] = (
            None if len(score_rel_vals) != len(metrics) else float(sum(score_rel_vals) / 4.0)
        )

        score_rel_w_items = [row.get(f"{m}_rel_wmean") for m in metrics]
        score_rel_w_vals = [float(v) for v in score_rel_w_items if v is not None]
        row["score_rel_wmean"] = (
            None if len(score_rel_w_vals) != len(metrics) else float(sum(score_rel_w_vals) / 4.0)
        )

        out.append(row)

    sort_s = str(sort).strip() or "mae_rank_mean"
    descending = False
    if sort_s.startswith("-"):
        descending = True
        sort_s = sort_s[1:].strip()
    if sort_s not in out[0]:
        raise ValueError(f"--sort must be a known summary column, got: {sort_s!r}")

    # Deterministic ordering:
    # - Missing values always last (even for descending sorts)
    # - Tie-break by mae_mean (if present), then model key.
    secondary = "mae_mean" if "mae_mean" in out[0] else ""

    def _num(v: object) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    def _sort_key(r: dict[str, Any]) -> tuple[int, float, int, float, str]:
        v = _num(r.get(sort_s))
        missing = 1 if v is None else 0
        val_key = 0.0 if v is None else (float(-v) if descending else float(v))

        sv = _num(r.get(secondary)) if secondary else None
        smissing = 1 if sv is None else 0
        sval_key = 0.0 if sv is None else (float(-sv) if descending else float(sv))

        return (missing, val_key, smissing, sval_key, str(r.get("model", "")))

    out.sort(key=_sort_key)

    min_datasets_i = int(min_datasets)
    if min_datasets_i > 0:
        out = [r for r in out if int(r.get("n_datasets", 0) or 0) >= min_datasets_i]

    limit_i = int(limit)
    if limit_i > 0:
        out = out[:limit_i]

    return out


def _emit(payload: object, *, output: str, fmt: str) -> None:
    text = _format_payload(payload, fmt=fmt)
    print(text)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _format_payload(payload: object, *, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if fmt == "csv":
        rows: list[dict]
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise TypeError("csv format expects a dict row or list of dict rows")
        return _format_csv(rows)
    if fmt == "md":
        rows_md: list[dict]
        if isinstance(payload, dict):
            rows_md = [payload]
        elif isinstance(payload, list):
            rows_md = payload
        else:
            raise TypeError("md format expects a dict row or list of dict rows")
        return _format_markdown(rows_md)
    raise ValueError(f"Unknown format: {fmt!r}")


def _leaderboard_columns() -> list[str]:
    # Stable output makes diffs and automation easier.
    return [
        "model",
        "dataset",
        "y_col",
        "horizon",
        "step",
        "min_train_size",
        "max_windows",
        "season_length",
        "n_series",
        "n_series_skipped",
        "n_windows",
        "n_points",
        "mae",
        "rmse",
        "mape",
        "smape",
    ]


def _format_csv(rows: list[dict]) -> str:
    cols = _leaderboard_columns()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in cols})
    return buf.getvalue().rstrip("\n")


def _format_markdown(rows: list[dict]) -> str:
    cols = _leaderboard_columns()

    def _fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(_fmt(row.get(k, "")) for k in cols) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _leaderboard_summary_columns() -> list[str]:
    return [
        "model",
        "n_datasets",
        "n_datasets_total",
        "dataset_coverage",
        "n_rows",
        "score_rank_mean",
        "score_rank_wmean",
        "score_rel_mean",
        "score_rel_wmean",
        "mae_mean",
        "mae_median",
        "mae_wmean",
        "mae_rel_mean",
        "mae_rel_median",
        "mae_rel_wmean",
        "rmse_mean",
        "rmse_median",
        "rmse_wmean",
        "rmse_rel_mean",
        "rmse_rel_median",
        "rmse_rel_wmean",
        "mape_mean",
        "mape_median",
        "mape_wmean",
        "mape_rel_mean",
        "mape_rel_median",
        "mape_rel_wmean",
        "smape_mean",
        "smape_median",
        "smape_wmean",
        "smape_rel_mean",
        "smape_rel_median",
        "smape_rel_wmean",
        "mae_rank_mean",
        "mae_rank_wmean",
        "rmse_rank_mean",
        "rmse_rank_wmean",
        "mape_rank_mean",
        "mape_rank_wmean",
        "smape_rank_mean",
        "smape_rank_wmean",
    ]


def _emit_table(rows: list[dict[str, Any]], *, columns: list[str], output: str, fmt: str) -> None:
    text = _format_table(rows, columns=columns, fmt=fmt)
    print(text)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _format_table(rows: list[dict[str, Any]], *, columns: list[str], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(rows, ensure_ascii=False, sort_keys=True)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
        return buf.getvalue().rstrip("\n")
    if fmt == "md":
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"

        def _fmt(v: object) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        body = ["| " + " | ".join(_fmt(row.get(k, "")) for k in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])
    raise ValueError(f"Unknown format: {fmt!r}")


def _run_parallel_tasks(
    tasks: list[tuple[str, tuple[str, ...]]],
    *,
    jobs: int,
    backend: str,
    progress: bool,
    strict: bool,
    worker: Any,
    worker_args: tuple[Any, ...] = (),
    errors_out: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Run (dataset, model_keys_chunk) tasks in parallel, returning: (results, failures).

    `worker(dataset, model_keys)` must return: (rows, errors).
    - rows: list of row dict payloads
    - errors: list of "SKIP ..." lines to print on stderr
    """

    import concurrent.futures
    import multiprocessing

    n = len(tasks)
    if n <= 0:
        return ([], 0)

    def _task_label(dataset: str, model_keys: tuple[str, ...]) -> str:
        if len(model_keys) == 1:
            return f"{dataset}/{model_keys[0]}"
        return f"{dataset}/[{len(model_keys)} models]"

    if jobs <= 1:
        out: list[dict[str, Any]] = []
        failures = 0
        for i, (dataset, model_keys) in enumerate(tasks, start=1):
            label = _task_label(dataset, model_keys)
            try:
                rows, errors = worker(dataset, model_keys, *worker_args)
                out.extend(rows)
                for err in errors:
                    failures += 1
                    print(err, file=sys.stderr)
                    if errors_out is not None:
                        errors_out.append(str(err))
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                failures += 1
                line = f"SKIP {label}: {type(e).__name__}: {e}"
                print(line, file=sys.stderr)
                if errors_out is not None:
                    errors_out.append(line)
            if progress:
                print(f"DONE {i}/{n} {label}", file=sys.stderr)
        return (out, failures)

    max_workers = min(int(jobs), max(1, n))
    backend_s = str(backend).strip().lower()
    if backend_s not in {"thread", "process"}:
        raise ValueError("--backend must be one of: thread, process")

    if backend_s == "process":
        ctx = multiprocessing.get_context("spawn")
        executor: Any = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx
        )
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    done = 0
    failures = 0
    out: list[dict[str, Any]] = []
    try:
        fut_to_task = {
            executor.submit(worker, dataset, model_keys, *worker_args): (dataset, model_keys)
            for dataset, model_keys in tasks
        }
        for fut in concurrent.futures.as_completed(fut_to_task):
            dataset, model_keys = fut_to_task[fut]
            label = _task_label(dataset, model_keys)
            try:
                rows, errors = fut.result()
                out.extend(rows)
                for err in errors:
                    failures += 1
                    print(err, file=sys.stderr)
                    if errors_out is not None:
                        errors_out.append(str(err))
            except Exception as e:  # noqa: BLE001
                if strict:
                    for other in fut_to_task:
                        if other is not fut:
                            other.cancel()
                    raise RuntimeError(f"{label}: {type(e).__name__}: {e}") from e
                failures += 1
                line = f"SKIP {label}: {type(e).__name__}: {e}"
                print(line, file=sys.stderr)
                if errors_out is not None:
                    errors_out.append(line)
            done += 1
            if progress:
                print(f"DONE {done}/{n} {label}", file=sys.stderr)
    finally:
        executor.shutdown(cancel_futures=True)

    return (out, failures)
