from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import cli_shared as _cli_shared
from .optional_deps import editable_install_command, package_install_command

_RNN_PAPER_METADATA_CACHE: dict[str, dict[str, Any]] | None = None
_OUTPUT_FORMAT_DEFAULT_TSV_HELP = "Output format (default: tsv)"
_OPTIONAL_OUTPUT_PATH_HELP = "Optional path to write output"
_RNN_PAPER_METADATA_FILENAME = "rnn_paper_metadata.json"


def register_catalog_subparsers(sub: Any) -> None:
    _register_models_parser(sub)
    _register_papers_parser(sub)
    _register_docs_parser(sub)


def _register_models_parser(sub: Any) -> None:
    models = sub.add_parser("models", help="Model registry utilities")
    models_sub = models.add_subparsers(dest="models_command", required=True)

    models_list = models_sub.add_parser("list", help="List available models")
    models_list.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help=_OUTPUT_FORMAT_DEFAULT_TSV_HELP,
    )
    models_list.add_argument(
        "--output",
        type=str,
        default="",
        help=_OPTIONAL_OUTPUT_PATH_HELP,
    )
    models_list.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional model key prefix filter (e.g. torch-rnnpaper)",
    )
    models_list.add_argument(
        "--interface",
        choices=["any", "local", "global", "multivariate"],
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
        "--extra",
        dest="requires",
        type=str,
        help="Alias for --requires.",
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
    models_list.add_argument(
        "--stability",
        choices=["any", "stable", "beta", "experimental"],
        default="any",
        help="Filter by stability level (default: any)",
    )
    models_list.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Capability filter as name=true/false. Repeatable. Example: --capability supports_x_cols=true",
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
        help=_OPTIONAL_OUTPUT_PATH_HELP,
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
        help=_OUTPUT_FORMAT_DEFAULT_TSV_HELP,
    )
    models_search.add_argument(
        "--output",
        type=str,
        default="",
        help=_OPTIONAL_OUTPUT_PATH_HELP,
    )
    models_search.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional model key prefix filter (e.g. torch-rnnpaper)",
    )
    models_search.add_argument(
        "--interface",
        choices=["any", "local", "global", "multivariate"],
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
        "--extra",
        dest="requires",
        type=str,
        help="Alias for --requires.",
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
        "--stability",
        choices=["any", "stable", "beta", "experimental"],
        default="any",
        help="Filter by stability level (default: any)",
    )
    models_search.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Capability filter as name=true/false. Repeatable. Example: --capability supports_x_cols=true",
    )
    models_search.add_argument(
        "--any",
        action="store_true",
        help="Match any token (OR) instead of all tokens (AND)",
    )
    models_search.set_defaults(_handler=_cmd_models_search)


def _register_papers_parser(sub: Any) -> None:
    papers = sub.add_parser("papers", help="Paper metadata utilities")
    papers_sub = papers.add_subparsers(dest="papers_command", required=True)

    papers_list = papers_sub.add_parser("list", help="List known paper metadata entries")
    papers_list.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help=_OUTPUT_FORMAT_DEFAULT_TSV_HELP,
    )
    papers_list.add_argument(
        "--output",
        type=str,
        default="",
        help=_OPTIONAL_OUTPUT_PATH_HELP,
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
        help=_OPTIONAL_OUTPUT_PATH_HELP,
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
        help=_OUTPUT_FORMAT_DEFAULT_TSV_HELP,
    )
    papers_models.add_argument(
        "--output",
        type=str,
        default="",
        help=_OPTIONAL_OUTPUT_PATH_HELP,
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


def _register_docs_parser(sub: Any) -> None:
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
    try:
        candidates.append(Path(__file__).resolve().parent / "data" / _RNN_PAPER_METADATA_FILENAME)
    except Exception:  # noqa: BLE001
        pass
    candidates.append(Path.cwd() / "docs" / _RNN_PAPER_METADATA_FILENAME)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "docs" / _RNN_PAPER_METADATA_FILENAME)
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
    k = str(key).strip()
    prefix = "torch-rnnpaper-"
    suffix = "-direct"
    if not (k.startswith(prefix) and k.endswith(suffix)):
        return None
    pid = k[len(prefix) : -len(suffix)]
    pid = pid.strip("-").strip()
    return pid or None


def _rnnzoo_base_from_model_key(key: str) -> str | None:
    base, _variant = _rnnzoo_base_and_variant_from_model_key(key)
    return base


def _rnnzoo_base_and_variant_from_model_key(key: str) -> tuple[str | None, str | None]:
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


def _validated_model_interface_filter(raw: object) -> str:
    interface_filter = str(raw).strip().lower()
    if interface_filter not in {"any", "local", "global", "multivariate"}:
        raise ValueError("--interface must be one of: any, local, global, multivariate")
    return interface_filter


def _matches_required_model_requires(
    reqs: set[str],
    *,
    req_set: set[str],
    include_core: bool,
) -> bool:
    if not (req_set or include_core):
        return True
    if include_core and not reqs:
        return True
    return bool(req_set and reqs.intersection(req_set))


def _matches_excluded_model_requires(
    reqs: set[str],
    *,
    exc_set: set[str],
    exclude_core: bool,
) -> bool:
    if exclude_core and not reqs:
        return False
    if exc_set and reqs.intersection(exc_set):
        return False
    return True


def _model_spec_matches_filters(
    spec: Any,
    *,
    prefix: str,
    interface_filter: str,
    stability_filter: str,
    capability_filters: dict[str, bool],
    req_set: set[str],
    include_core: bool,
    exc_set: set[str],
    exclude_core: bool,
) -> bool:
    if prefix and not str(spec.key).startswith(prefix):
        return False
    if interface_filter != "any" and str(spec.interface) != interface_filter:
        return False
    if stability_filter != "any" and str(spec.stability_level) != stability_filter:
        return False

    reqs = set(spec.requires)
    if not _matches_required_model_requires(
        reqs,
        req_set=req_set,
        include_core=include_core,
    ):
        return False
    if not _matches_excluded_model_requires(
        reqs,
        exc_set=exc_set,
        exclude_core=exclude_core,
    ):
        return False

    capabilities = dict(spec.capabilities)
    for name, expected in capability_filters.items():
        if bool(capabilities.get(name, False)) is not expected:
            return False

    return True


def _catalog_model_row_from_spec(spec: Any) -> dict[str, Any]:
    required_extra = str(spec.required_extra)
    row: dict[str, Any] = {
        "key": spec.key,
        "interface": str(spec.interface),
        "requires": ",".join(spec.requires),
        "required_extra": required_extra,
        "package_install_command": package_install_command(required_extra),
        "editable_install_command": editable_install_command(required_extra),
        "stability": str(spec.stability_level),
        "description": spec.description,
        "default_params": dict(spec.default_params),
        "capabilities": dict(spec.capabilities),
    }
    paper = _paper_payload_for_model_key(spec.key)
    if paper:
        row["paper"] = paper
    wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
    if wrapper_pid:
        wrapper = _paper_payload_for_paper_id(wrapper_pid)
        if wrapper:
            row["wrapper_paper"] = wrapper
    return row


def _models_list_sort_value(row: dict[str, Any], sort_key: str) -> object | None:
    paper_value = row.get("paper")
    wrapper_value = row.get("wrapper_paper")
    paper: dict[str, Any] = paper_value if isinstance(paper_value, dict) else {}
    wrapper: dict[str, Any] = wrapper_value if isinstance(wrapper_value, dict) else {}

    if sort_key == "key":
        return str(row.get("key", "")).lower()
    if sort_key == "interface":
        return str(row.get("interface", "")).lower()
    if sort_key == "requires":
        return str(row.get("requires", "")).lower()
    if sort_key == "required_extra":
        return str(row.get("required_extra", "")).lower()
    if sort_key == "stability":
        return str(row.get("stability", "")).lower()
    if sort_key == "description":
        return str(row.get("description", "")).lower()
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
        "--sort must be one of: key, interface, requires, required_extra, stability, description, paper_id, paper_year, wrapper_paper_id, wrapper_year"
    )


def _sort_catalog_model_rows(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    descending: bool,
) -> list[dict[str, Any]]:
    if sort_key in {"paper_year", "wrapper_year"}:
        present: list[tuple[int, dict[str, Any]]] = []
        missing = []
        for row in rows:
            sort_value = _models_list_sort_value(row, sort_key)
            if not isinstance(sort_value, int):
                missing.append(row)
            else:
                present.append((sort_value, row))
        present.sort(key=lambda item: item[0], reverse=descending)
        return [row for _, row in present] + missing

    rows.sort(
        key=lambda row: str(_models_list_sort_value(row, sort_key)),
        reverse=descending,
    )
    return rows


def _models_list_column_value(row: dict[str, Any], col: str) -> object:
    paper_value = row.get("paper")
    wrapper_value = row.get("wrapper_paper")
    paper: dict[str, Any] = paper_value if isinstance(paper_value, dict) else {}
    wrapper: dict[str, Any] = wrapper_value if isinstance(wrapper_value, dict) else {}

    if col == "key":
        return row.get("key", "")
    if col == "interface":
        return row.get("interface", "")
    if col == "requires":
        return row.get("requires", "")
    if col == "required_extra":
        return row.get("required_extra", "")
    if col == "package_install_command":
        return row.get("package_install_command", "")
    if col == "editable_install_command":
        return row.get("editable_install_command", "")
    if col == "stability":
        return row.get("stability", "")
    if col == "description":
        return row.get("description", "")
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
        "--columns must be a comma-separated list of: key, interface, requires, required_extra, package_install_command, editable_install_command, stability, description, paper_id, paper_year, paper_title, wrapper_paper_id, wrapper_year, wrapper_title"
    )


def _models_list_tsv_lines(
    rows: list[dict[str, Any]],
    *,
    columns: list[str],
    include_header: bool,
) -> list[str]:
    lines: list[str] = []
    if include_header:
        lines.append("\t".join(columns))
    for row in rows:
        lines.append(
            "\t".join(
                _cli_shared._sanitize_tsv_cell(_models_list_column_value(row, col))
                for col in columns
            )
        )
    return lines


def _parse_capability_filters(raw_filters: list[str]) -> dict[str, bool]:
    parsed: dict[str, bool] = {}
    for raw in raw_filters:
        item = str(raw).strip()
        if not item:
            continue
        if "=" in item:
            name, raw_value = item.split("=", 1)
        else:
            name, raw_value = item, "true"
        key = str(name).strip()
        value_s = str(raw_value).strip().lower()
        if not key:
            raise ValueError("--capability entries must include a capability name")
        if value_s not in {"true", "false"}:
            raise ValueError("--capability values must be true or false")
        parsed[key] = value_s == "true"
    return parsed


def _score_model_search_token(
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


def _model_search_metadata(spec: Any) -> dict[str, Any]:
    paper = _paper_payload_for_model_key(spec.key) or {}
    wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
    wrapper = _paper_payload_for_paper_id(wrapper_pid) if wrapper_pid else None
    wrapper = wrapper or {}

    paper_year = paper.get("year", None)
    wrapper_year = wrapper.get("year", None)

    return {
        "paper": paper,
        "wrapper": wrapper,
        "key_l": str(spec.key).lower(),
        "desc_l": str(spec.description).lower(),
        "requires_l": ",".join(spec.requires).lower(),
        "paper_id_l": str(paper.get("paper_id", "")).lower(),
        "paper_title_l": str(paper.get("title", "")).lower(),
        "paper_year": int(paper_year) if isinstance(paper_year, int) else None,
        "wrapper_id_l": str(wrapper.get("paper_id", "")).lower(),
        "wrapper_title_l": str(wrapper.get("title", "")).lower(),
        "wrapper_year": int(wrapper_year) if isinstance(wrapper_year, int) else None,
    }


def _model_search_row(
    spec: Any,
    *,
    score: int,
    paper: dict[str, Any],
    wrapper: dict[str, Any],
) -> dict[str, Any]:
    required_extra = str(spec.required_extra)
    row: dict[str, Any] = {
        "key": spec.key,
        "score": score,
        "interface": str(spec.interface),
        "requires": ",".join(spec.requires),
        "required_extra": required_extra,
        "package_install_command": package_install_command(required_extra),
        "editable_install_command": editable_install_command(required_extra),
        "stability": str(spec.stability_level),
        "description": spec.description,
        "capabilities": dict(spec.capabilities),
    }
    if paper:
        row["paper"] = paper
    if wrapper and str(wrapper.get("paper_id", "")).strip():
        row["wrapper_paper"] = wrapper
    return row


def _models_search_tsv_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        paper_value = row.get("paper")
        paper: dict[str, Any] = paper_value if isinstance(paper_value, dict) else {}
        pid = str(paper.get("paper_id", "")).strip()
        year = paper.get("year", None)
        year_s = str(year) if isinstance(year, int) else ""

        lines.append(
            "\t".join(
                [
                    _cli_shared._sanitize_tsv_cell(row.get("key", "")),
                    _cli_shared._sanitize_tsv_cell(row.get("score", "")),
                    _cli_shared._sanitize_tsv_cell(pid),
                    _cli_shared._sanitize_tsv_cell(year_s),
                    _cli_shared._sanitize_tsv_cell(row.get("requires", "")),
                    _cli_shared._sanitize_tsv_cell(row.get("description", "")),
                ]
            )
        )
    return lines


def _cmd_models_list(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    interface_filter = _validated_model_interface_filter(
        _cli_shared._string_arg_value(args, "interface", default="any")
    )
    stability_filter = (
        _cli_shared._stripped_arg_value(args, "stability", default="any").lower() or "any"
    )
    capability_filters = _parse_capability_filters(_cli_shared._list_arg_values(args, "capability"))
    req_set, include_core = _cli_shared._parse_requires_arg(args, "requires")
    exc_set, exclude_core = _cli_shared._parse_requires_arg(args, "exclude_requires")

    rows: list[dict[str, Any]] = []
    prefix = _cli_shared._stripped_arg_value(args, "prefix")
    for key in list_models():
        spec = get_model_spec(key)
        if _model_spec_matches_filters(
            spec,
            prefix=prefix,
            interface_filter=interface_filter,
            stability_filter=stability_filter,
            capability_filters=capability_filters,
            req_set=req_set,
            include_core=include_core,
            exc_set=exc_set,
            exclude_core=exclude_core,
        ):
            rows.append(_catalog_model_row_from_spec(spec))

    sort = _cli_shared._stripped_arg_value(args, "sort") or "key"
    descending = _cli_shared._bool_arg_value(args, "desc")
    if sort.startswith("-"):
        descending = True
        sort = sort[1:]
    sort_key = sort.strip() or "key"

    rows = _sort_catalog_model_rows(rows, sort_key=sort_key, descending=descending)

    limit = int(args.limit)
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if limit:
        rows = rows[: min(limit, 100000)]

    fmt = _cli_shared._format_arg_value(args)
    if fmt == "json":
        _cli_shared._emit(rows, output=_cli_shared._output_arg_value(args), fmt="json")
        return 0

    columns_raw = _cli_shared._stripped_arg_value(args, "columns")
    if columns_raw:
        columns = _cli_shared._split_csv_items(columns_raw)
    else:
        columns = ["key", "requires", "description"]

    lines = _models_list_tsv_lines(
        rows,
        columns=columns,
        include_header=_cli_shared._bool_arg_value(args, "header"),
    )
    _cli_shared._emit_text("\n".join(lines), output=_cli_shared._output_arg_value(args))
    return 0


def _cmd_models_info(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec

    spec = get_model_spec(_cli_shared._string_arg_value(args, "key"))
    payload = {
        "key": spec.key,
        "interface": str(spec.interface),
        "description": spec.description,
        "requires": list(spec.requires),
        "required_extra": str(spec.required_extra),
        "package_install_command": package_install_command(str(spec.required_extra)),
        "editable_install_command": editable_install_command(str(spec.required_extra)),
        "stability": str(spec.stability_level),
        "default_params": dict(spec.default_params),
        "param_help": dict(spec.param_help),
        "capabilities": dict(spec.capabilities),
    }
    paper = _paper_payload_for_model_key(spec.key)
    if paper:
        payload["paper"] = paper
    wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
    if wrapper_pid:
        wrapper = _paper_payload_for_paper_id(wrapper_pid)
        if wrapper:
            payload["wrapper_paper"] = wrapper
    _cli_shared._emit(
        payload,
        output=_cli_shared._output_arg_value(args),
        fmt=_cli_shared._format_arg_value(args),
    )
    return 0


def _cmd_models_search(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    q = _cli_shared._stripped_arg_value(args, "query")
    if not q:
        raise ValueError("query must be non-empty")

    tokens = [t.strip().lower() for t in q.split() if t.strip()]
    any_mode = bool(args.any)
    prefix = _cli_shared._stripped_arg_value(args, "prefix")

    interface_filter = _validated_model_interface_filter(
        _cli_shared._string_arg_value(args, "interface", default="any")
    )
    stability_filter = (
        _cli_shared._stripped_arg_value(args, "stability", default="any").lower() or "any"
    )
    capability_filters = _parse_capability_filters(_cli_shared._list_arg_values(args, "capability"))
    req_set, include_core = _cli_shared._parse_requires_arg(args, "requires")
    exc_set, exclude_core = _cli_shared._parse_requires_arg(args, "exclude_requires")

    limit = int(args.limit)
    if limit <= 0:
        raise ValueError("--limit must be >= 1")
    limit = min(limit, 1000)

    rows: list[dict[str, Any]] = []
    for key in list_models():
        spec = get_model_spec(key)
        if not _model_spec_matches_filters(
            spec,
            prefix=prefix,
            interface_filter=interface_filter,
            stability_filter=stability_filter,
            capability_filters=capability_filters,
            req_set=req_set,
            include_core=include_core,
            exc_set=exc_set,
            exclude_core=exclude_core,
        ):
            continue

        metadata = _model_search_metadata(spec)

        token_scores = [
            _score_model_search_token(
                t,
                key_l=str(metadata["key_l"]),
                desc_l=str(metadata["desc_l"]),
                requires_l=str(metadata["requires_l"]),
                paper_id_l=str(metadata["paper_id_l"]),
                paper_title_l=str(metadata["paper_title_l"]),
                paper_year=metadata["paper_year"],
                wrapper_id_l=str(metadata["wrapper_id_l"]),
                wrapper_title_l=str(metadata["wrapper_title_l"]),
                wrapper_year=metadata["wrapper_year"],
            )
            for t in tokens
        ]

        matched = any(s > 0 for s in token_scores) if any_mode else all(s > 0 for s in token_scores)
        if not matched:
            continue

        score = int(sum(token_scores))
        rows.append(
            _model_search_row(
                spec,
                score=score,
                paper=metadata["paper"],
                wrapper=metadata["wrapper"],
            )
        )

    rows.sort(key=lambda r: (-int(r.get("score", 0)), str(r.get("key", ""))))
    rows = rows[:limit]

    fmt = _cli_shared._format_arg_value(args)
    if fmt == "json":
        _cli_shared._emit(rows, output=_cli_shared._output_arg_value(args), fmt="json")
        return 0

    lines = _models_search_tsv_lines(rows)
    _cli_shared._emit_text("\n".join(lines), output=_cli_shared._output_arg_value(args))
    return 0


def _cmd_papers_list(args: argparse.Namespace) -> int:
    meta = _load_rnn_paper_metadata()
    query = _cli_shared._stripped_arg_value(args, "query").lower()

    rows: list[dict[str, Any]] = []
    for paper_id in sorted(meta):
        payload = _paper_payload_for_paper_id(str(paper_id))
        title = str(payload.get("title", "")).strip() if payload is not None else ""
        if query and query not in str(paper_id).lower() and query not in title.lower():
            continue
        if payload is not None:
            rows.append(payload)

    fmt = _cli_shared._format_arg_value(args)
    if fmt == "json":
        _cli_shared._emit(rows, output=_cli_shared._output_arg_value(args), fmt="json")
        return 0

    lines = [
        "\t".join(
            [
                _cli_shared._sanitize_tsv_cell(r.get("paper_id", "")),
                _cli_shared._sanitize_tsv_cell(r.get("year", "")),
                _cli_shared._sanitize_tsv_cell(r.get("title", "")),
                _cli_shared._sanitize_tsv_cell(r.get("url", "")),
            ]
        )
        for r in rows
    ]
    _cli_shared._emit_text("\n".join(lines), output=_cli_shared._output_arg_value(args))
    return 0


def _cmd_papers_info(args: argparse.Namespace) -> int:
    pid = _cli_shared._stripped_arg_value(args, "paper_id")
    if not pid:
        raise ValueError("paper_id must be non-empty")

    meta = _load_rnn_paper_metadata()
    if pid not in meta:
        raise ValueError(f"Unknown paper_id: {pid!r}")

    payload = _paper_payload_for_paper_id(pid)
    if payload is None:
        raise ValueError(f"Unknown paper_id: {pid!r}")
    _cli_shared._emit(payload, output=_cli_shared._output_arg_value(args), fmt="json")
    return 0


def _cmd_papers_models(args: argparse.Namespace) -> int:
    from .models.registry import get_model_spec, list_models

    pid = _cli_shared._stripped_arg_value(args, "paper_id")
    if not pid:
        raise ValueError("paper_id must be non-empty")

    meta = _load_rnn_paper_metadata()
    if pid not in meta:
        raise ValueError(f"Unknown paper_id: {pid!r}")

    prefix = _cli_shared._stripped_arg_value(args, "prefix")
    role = _cli_shared._stripped_arg_value(args, "role").lower()
    if role not in {"any", "base", "wrapper"}:
        raise ValueError("--role must be one of: any, base, wrapper")

    rows: list[dict[str, Any]] = []
    for key in list_models():
        rows.extend(
            _paper_matching_rows_for_spec(
                get_model_spec(key),
                pid=pid,
                prefix=prefix,
                role=role,
            )
        )

    rows.sort(key=lambda r: (str(r.get("role", "")), str(r.get("key", ""))))

    fmt = _cli_shared._format_arg_value(args)
    if fmt == "json":
        _cli_shared._emit(rows, output=_cli_shared._output_arg_value(args), fmt="json")
        return 0

    lines = [
        "\t".join(
            [
                _cli_shared._sanitize_tsv_cell(r.get("key", "")),
                _cli_shared._sanitize_tsv_cell(r.get("role", "")),
                _cli_shared._sanitize_tsv_cell(r.get("requires", "")),
                _cli_shared._sanitize_tsv_cell(r.get("description", "")),
            ]
        )
        for r in rows
    ]
    _cli_shared._emit_text("\n".join(lines), output=_cli_shared._output_arg_value(args))
    return 0


def _cmd_docs_rnn(args: argparse.Namespace) -> int:
    from .docsgen.rnn import render_rnn_paper_zoo_doc, render_rnn_zoo_doc, write_rnn_docs

    out_dir = Path(_cli_shared._string_arg_value(args, "output_dir")).expanduser().resolve()

    if bool(args.check):
        expected_paper = render_rnn_paper_zoo_doc()
        expected_zoo = render_rnn_zoo_doc()
        failures = _rnn_doc_check_failures(
            out_dir,
            expected_paper=expected_paper,
            expected_zoo=expected_zoo,
        )

        if failures:
            print("Docs out of date (or missing):", file=sys.stderr)
            for path in failures:
                print(f"- {path}", file=sys.stderr)
            return 1

        print("OK: RNN docs are up to date.")
        return 0

    write_rnn_docs(output_dir=out_dir)
    print("Wrote:")
    print(f"- {(out_dir / 'rnn_paper_zoo.md').as_posix()}")
    print(f"- {(out_dir / 'rnn_zoo.md').as_posix()}")
    return 0


def _paper_model_matches_role(*, role: str, hit_base: bool, hit_wrap: bool) -> bool:
    if role == "base":
        return hit_base
    if role == "wrapper":
        return hit_wrap
    return hit_base or hit_wrap


def _paper_model_rows_for_spec(
    spec: Any,
    *,
    hit_base: bool,
    hit_wrap: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    return rows


def _paper_matching_rows_for_spec(
    spec: Any,
    *,
    pid: str,
    prefix: str,
    role: str,
) -> list[dict[str, Any]]:
    if prefix and not str(spec.key).startswith(prefix):
        return []

    base = _paper_payload_for_model_key(spec.key) or {}
    base_pid = str(base.get("paper_id", "")).strip()

    wrapper_pid = _rnnzoo_wrapper_paper_id_from_model_key(spec.key)
    wrapper = _paper_payload_for_paper_id(wrapper_pid) if wrapper_pid else None
    wrapper = wrapper or {}
    wrap_pid = str(wrapper.get("paper_id", "")).strip()

    hit_base = base_pid == pid
    hit_wrap = wrap_pid == pid
    if not _paper_model_matches_role(role=role, hit_base=hit_base, hit_wrap=hit_wrap):
        return []
    return _paper_model_rows_for_spec(spec, hit_base=hit_base, hit_wrap=hit_wrap)


def _rnn_doc_check_failures(
    out_dir: Path,
    *,
    expected_paper: str,
    expected_zoo: str,
) -> list[str]:
    doc_expectations = {
        out_dir / "rnn_paper_zoo.md": expected_paper,
        out_dir / "rnn_zoo.md": expected_zoo,
    }
    failures: list[str] = []
    for path, expected_text in doc_expectations.items():
        if not path.exists():
            failures.append(str(path))
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected_text:
            failures.append(str(path))
    return failures
