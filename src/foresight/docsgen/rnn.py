from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path

RNN_PAPER_METADATA_FILENAME = "rnn_paper_metadata.json"
RNN_DOC_TABLE_RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _find_repo_root_from(start: Path) -> Path | None:
    """
    Best-effort repo root detector for source checkouts.

    We want a stable root for:
      - docs metadata JSON
      - `src/foresight/models/*.py` anchor lookup
    """

    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "foresight").exists():
            return p
    return None


def _repo_root() -> Path | None:
    # Prefer locating relative to this file (source checkout), otherwise fall back to CWD.
    here = Path(__file__).resolve()
    return _find_repo_root_from(here) or _find_repo_root_from(Path.cwd())


def _package_root() -> Path:
    # src/foresight/docsgen/rnn.py -> src/foresight
    return Path(__file__).resolve().parents[1]


def _resolve_text_path(relative_repo_path: str) -> Path | None:
    rel = str(relative_repo_path).replace("\\", "/").lstrip("/")

    root = _repo_root()
    if root is not None:
        p = (root / rel).resolve()
        if p.exists() and p.is_file():
            return p

    # Installed-package fallback: map `src/foresight/...` to `<site-packages>/foresight/...`.
    prefix = "src/foresight/"
    if rel.startswith(prefix):
        mapped = (_package_root() / rel[len(prefix) :]).resolve()
        if mapped.exists() and mapped.is_file():
            return mapped

    # CWD-relative fallback (e.g., running from repo root).
    p2 = (Path.cwd() / rel).resolve()
    if p2.exists() and p2.is_file():
        return p2

    return None


def _semanticscholar_search_url(query: str) -> str:
    q = str(query).strip()
    if not q:
        return "-"
    return "https://www.semanticscholar.org/search?" + urllib.parse.urlencode({"q": q})


def _arxiv_search_url(query: str) -> str:
    q = str(query).strip()
    if not q:
        return "-"
    return "https://arxiv.org/search/?" + urllib.parse.urlencode(
        {"query": q, "searchtype": "all", "source": "header"}
    )


def _crossref_search_url(query: str) -> str:
    q = str(query).strip()
    if not q:
        return "-"
    return "https://search.crossref.org/?" + urllib.parse.urlencode({"q": q})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _github_file_anchor(relative_path: str, line_number: int) -> str:
    p = str(relative_path).replace("\\", "/").lstrip("/")
    ln = int(line_number)
    if ln <= 0:
        return "-"
    return f"{p}#L{ln}"


def _find_first_line_anchor(
    relative_path: str,
    pattern: str,
    *,
    start_after: str | None = None,
) -> str:
    path = _resolve_text_path(relative_path)
    if path is None:
        return "-"

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return "-"

    lines = text.splitlines()
    start_idx = 0
    if start_after:
        for i, line in enumerate(lines):
            if start_after in line:
                start_idx = i + 1
                break

    regex = re.compile(pattern)
    for i in range(start_idx, len(lines)):
        if regex.search(lines[i]):
            return _github_file_anchor(relative_path, i + 1)
    return "-"


def _paper_impl_anchor(paper_id: str) -> str:
    pid = str(paper_id).strip()
    if not pid:
        return "-"
    return _find_first_line_anchor(
        "src/foresight/models/torch_rnn_paper_zoo.py",
        pattern=rf'\bpaper_id\b.*["\']{re.escape(pid)}["\']',
        start_after="# ---- Encoder selection ----",
    )


def _rnn_paper_metadata_candidate_paths() -> list[Path]:
    env_path = str(os.environ.get("FORESIGHT_RNN_PAPER_METADATA", "")).strip()

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    root = _repo_root()
    if root is not None:
        candidates.append(root / "docs" / RNN_PAPER_METADATA_FILENAME)

    candidates.append(Path.cwd() / "docs" / RNN_PAPER_METADATA_FILENAME)
    candidates.append(_package_root() / "data" / RNN_PAPER_METADATA_FILENAME)
    return candidates


def _normalize_rnn_paper_metadata(raw: object) -> dict[str, dict[str, object]] | None:
    if not isinstance(raw, dict):
        return None

    out: dict[str, dict[str, object]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _read_rnn_paper_metadata() -> dict[str, dict[str, object]]:
    for path in _rnn_paper_metadata_candidate_paths():
        if not (path.exists() and path.is_file()):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        normalized = _normalize_rnn_paper_metadata(raw)
        if normalized is None:
            continue
        return normalized

    return {}


def _doi_url(doi: str) -> str:
    d = str(doi).strip()
    if not d:
        return "-"
    return "https://doi.org/" + d


def _arxiv_abs_url(arxiv_id: str) -> str:
    a = str(arxiv_id).strip()
    if not a:
        return "-"
    return "https://arxiv.org/abs/" + a


def _metadata_primary_url(url: str, doi: str, arxiv_id: str) -> str:
    direct_url = str(url).strip()
    if direct_url:
        return direct_url

    doi_value = str(doi).strip()
    if doi_value:
        return _doi_url(doi_value)

    arxiv_value = str(arxiv_id).strip()
    if arxiv_value:
        return _arxiv_abs_url(arxiv_value)

    return "-"


def _metadata_entry_fields(meta_entry: object) -> tuple[str, str, str, str, str]:
    if not isinstance(meta_entry, dict):
        return "", "", "", "", ""
    return (
        str(meta_entry.get("title", "")).strip(),
        str(meta_entry.get("year", "")).strip(),
        str(meta_entry.get("doi", "")).strip(),
        str(meta_entry.get("arxiv_id", "")).strip(),
        str(meta_entry.get("url", "")).strip(),
    )


def _safe_doc_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _rnnzoo_base_impl_anchor(base: str) -> str:
    b = str(base).strip()
    if not b:
        return "-"
    return _find_first_line_anchor(
        "src/foresight/models/torch_rnn_zoo.py",
        pattern=rf'^\s*if base_s == ["\']{re.escape(b)}["\']\s*:',
        start_after="def _make_base_encoder",
    )


def _rnnzoo_base_to_paper_id(base: str) -> str:
    """
    Map RNN Zoo base ids to RNN Paper Zoo `paper_id` keys (metadata JSON).
    """

    b = str(base).strip().lower()
    if not b:
        return b
    base_to_paper_id = {
        "elman": "elman-srn",
        "clockwork": "clockwork-rnn",
        "fastrnn": "fast-rnn",
        "fastgrnn": "fast-grnn",
    }
    return base_to_paper_id.get(b, b)


def _rnnzoo_variant_impl_anchor(variant: str) -> str:
    v = str(variant).strip()
    if not v or v == "direct":
        return "-"
    return _find_first_line_anchor(
        "src/foresight/models/torch_rnn_zoo.py",
        pattern=rf'["\']{re.escape(v)}["\']',
        start_after="class _RNNZooNet",
    )


def _rnn_paper_zoo_index_row(
    paper_id: str,
    desc: str,
    meta_entry: object,
) -> str:
    title, year, doi, arxiv_id, url = _metadata_entry_fields(meta_entry)
    key = f"torch-rnnpaper-{paper_id}-direct"
    impl = _paper_impl_anchor(paper_id)
    doi_link = _doi_url(doi)
    ax = _arxiv_abs_url(arxiv_id) if arxiv_id else _arxiv_search_url(desc)
    primary_url = _metadata_primary_url(url, doi, arxiv_id)
    return (
        f"| `{paper_id}` | `{key}` | {_safe_doc_cell(desc)} | {_safe_doc_cell(title) if title else ''} | "
        f"{year} | {doi_link} | {ax} | {primary_url} | {impl} | "
        f"{_semanticscholar_search_url(desc)} | {_crossref_search_url(desc)} |"
    )


def _rnnzoo_base_index_row(
    base: str,
    desc: str,
    meta_entry: object,
) -> str:
    title, year, doi, arxiv_id, url = _metadata_entry_fields(meta_entry)
    paper_id = _rnnzoo_base_to_paper_id(base)
    impl = _rnnzoo_base_impl_anchor(base)
    doi_link = _doi_url(doi)
    ax = _arxiv_abs_url(arxiv_id) if arxiv_id else _arxiv_search_url(desc)
    primary_url = _metadata_primary_url(url, doi, arxiv_id)
    return (
        f"| `{base}` | {_safe_doc_cell(desc)} | `{paper_id}` | "
        f"{_safe_doc_cell(title) if title else ''} | {year} | {doi_link} | {ax} | "
        f"{primary_url} | {impl} | {_semanticscholar_search_url(desc)} | {_crossref_search_url(desc)} |"
    )


def _rnnzoo_variant_index_row(
    variant: str,
    desc: str,
    *,
    variant_to_paper_id: dict[str, str],
    meta_entry: object,
) -> str:
    impl = _rnnzoo_variant_impl_anchor(variant)
    if variant == "direct":
        return f"| `{variant}` | {_safe_doc_cell(desc)} | - |  |  | - | - | - | {impl} | - | - |"

    title, year, doi, arxiv_id, url = _metadata_entry_fields(meta_entry)
    paper_id = variant_to_paper_id.get(variant, "")
    doi_link = _doi_url(doi)
    ax = _arxiv_abs_url(arxiv_id) if arxiv_id else _arxiv_search_url(desc)
    primary_url = _metadata_primary_url(url, doi, arxiv_id)
    pid_cell = f"`{paper_id}`" if paper_id else "-"
    return (
        f"| `{variant}` | {_safe_doc_cell(desc)} | {pid_cell} | "
        f"{_safe_doc_cell(title) if title else ''} | {year} | {doi_link} | {ax} | "
        f"{primary_url} | {impl} | {_semanticscholar_search_url(desc)} | {_crossref_search_url(desc)} |"
    )


def _rnnzoo_model_index_row(
    spec: object,
    *,
    base_descriptions: dict[str, str],
    variant_descriptions: dict[str, str],
    variant_to_paper_id: dict[str, str],
    meta: dict[str, dict[str, object]],
) -> str:
    base_desc = base_descriptions[spec.base]
    base_pid = _rnnzoo_base_to_paper_id(spec.base)
    _, _, _, _, base_url = _metadata_entry_fields(meta.get(base_pid, {}))
    if not base_url:
        base_url = _semanticscholar_search_url(base_desc)

    if spec.variant == "direct":
        wrapper_url = "-"
    else:
        wrapper_pid = variant_to_paper_id.get(str(spec.variant), "")
        _, _, _, _, wrapper_url = _metadata_entry_fields(meta.get(wrapper_pid, {}))
        if not wrapper_url:
            wrapper_url = _semanticscholar_search_url(variant_descriptions[spec.variant])

    return (
        f"| `{spec.key}` | `{spec.base}` | `{spec.variant}` | {base_url} | {wrapper_url} |"
    )


def render_rnn_paper_zoo_doc() -> str:
    from foresight.models.torch_rnn_paper_zoo import _PAPER_DEFS

    meta = _read_rnn_paper_metadata()

    lines: list[str] = []
    lines.append("# RNN Paper Zoo (100)")
    lines.append("")
    lines.append(
        "This document enumerates the 100 **paper-named** recurrent architectures registered under"
    )
    lines.append("`torch-rnnpaper-*-direct`.")
    lines.append("")
    lines.append("- Implementation: `src/foresight/models/torch_rnn_paper_zoo.py`")
    lines.append("- Registry wiring: `src/foresight/models/registry.py`")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("List models:")
    lines.append("")
    lines.append("```bash")
    lines.append("foresight models list --prefix torch-rnnpaper")
    lines.append("```")
    lines.append("")
    lines.append("Use from Python:")
    lines.append("")
    lines.append("```python")
    lines.append("from foresight.models.registry import make_forecaster")
    lines.append("")
    lines.append("f = make_forecaster(")
    lines.append('    "torch-rnnpaper-elman-srn-direct",')
    lines.append("    lags=48,")
    lines.append("    hidden_size=32,")
    lines.append("    epochs=10,")
    lines.append("    batch_size=32,")
    lines.append("    lr=1e-3,")
    lines.append('    device="cpu",')
    lines.append(")")
    lines.append("")
    lines.append("yhat = f(train_1d, horizon=14)")
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- These implementations are designed as **lite baselines** under a unified *direct multi-horizon forecasting*"
    )
    lines.append("  interface.")
    lines.append(
        "- The repo enforces a **no built-in recurrent modules** rule (PyTorch RNN/GRU/LSTM and their Cell variants);"
    )
    lines.append("  all recurrent cores are implemented via")
    lines.append(
        "  manual scan/unroll (Linear + gates) so the structure is explicit and comparable."
    )
    lines.append(
        "- Reference links below include **DOI / arXiv / URL** when available, plus stable search links (Semantic Scholar /"
    )
    lines.append("  Crossref) for quick verification.")
    lines.append(
        "- The `implementation` column points to the corresponding selection branch in the source file for quick navigation."
    )
    lines.append("")
    lines.append("## Paper / Architecture Index")
    lines.append("")
    lines.append(
        "| paper_id | model_key | architecture (as in code) | paper_title | year | DOI | arXiv | URL | implementation | Semantic Scholar | Crossref |"
    )
    lines.append(RNN_DOC_TABLE_RULE)
    for paper_id, desc in _PAPER_DEFS:
        lines.append(_rnn_paper_zoo_index_row(paper_id, desc, meta.get(paper_id, {})))
    return "\n".join(lines) + "\n"


def render_rnn_zoo_doc() -> str:
    from foresight.models.torch_rnn_zoo import (
        _BASE_DESCRIPTIONS,
        _VARIANT_DESCRIPTIONS,
        list_rnnzoo_specs,
    )

    meta = _read_rnn_paper_metadata()
    variant_to_paper_id = {
        "bidir": "bidirectional-rnn",
        "ln": "layer-normalization",
        "attn": "bahdanau-attention",
        "proj": "lstm-projection",
    }

    lines: list[str] = []
    lines.append("# RNN Zoo (100)")
    lines.append("")
    lines.append(
        "This document enumerates the 100 **RNN Zoo** models registered under `torch-rnnzoo-*-direct`."
    )
    lines.append("")
    lines.append("RNN Zoo is a compact combinatorial family:")
    lines.append("")
    lines.append("- **20 bases** (paper-named recurrent cores)")
    lines.append("- **5 wrappers/variants** (`direct`, `bidir`, `ln`, `attn`, `proj`)")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("List models:")
    lines.append("")
    lines.append("```bash")
    lines.append("foresight models list --prefix torch-rnnzoo")
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- These implementations are **lite baselines** under a unified *direct multi-horizon forecasting* interface."
    )
    lines.append(
        "- The repo enforces a **no built-in recurrent modules** rule (PyTorch RNN/GRU/LSTM and their Cell variants);"
    )
    lines.append("  all recurrent cores are manual scan/unroll.")
    lines.append(
        "- Reference links below include **DOI / arXiv / URL** when available, plus search links for quick verification."
    )
    lines.append("- `implementation` links point into this repo’s source for fast code navigation.")
    lines.append("")
    lines.append("## Base Index (20)")
    lines.append("")
    lines.append(
        "| base | description | paper_id | paper_title | year | DOI | arXiv | URL | implementation | Semantic Scholar | Crossref |"
    )
    lines.append(RNN_DOC_TABLE_RULE)
    for base, desc in _BASE_DESCRIPTIONS.items():
        lines.append(_rnnzoo_base_index_row(base, desc, meta.get(_rnnzoo_base_to_paper_id(base), {})))
    lines.append("")
    lines.append("## Variant Index (5)")
    lines.append("")
    lines.append(
        "| variant | description | paper_id | paper_title | year | DOI | arXiv | URL | implementation | Semantic Scholar | Crossref |"
    )
    lines.append(RNN_DOC_TABLE_RULE)
    for v, desc in _VARIANT_DESCRIPTIONS.items():
        lines.append(
            _rnnzoo_variant_index_row(
                v,
                desc,
                variant_to_paper_id=variant_to_paper_id,
                meta_entry=meta.get(variant_to_paper_id.get(v, ""), {}),
            )
        )
    lines.append("")
    lines.append("## Model Index (100)")
    lines.append("")
    lines.append("| model_key | base | variant | base ref | wrapper ref |")
    lines.append("| --- | --- | --- | --- | --- |")
    for spec in list_rnnzoo_specs():
        lines.append(
            _rnnzoo_model_index_row(
                spec,
                base_descriptions=_BASE_DESCRIPTIONS,
                variant_descriptions=_VARIANT_DESCRIPTIONS,
                variant_to_paper_id=variant_to_paper_id,
                meta=meta,
            )
        )
    return "\n".join(lines) + "\n"


def write_rnn_docs(*, output_dir: str | Path) -> None:
    out = Path(output_dir)
    _write_text(out / "rnn_paper_zoo.md", render_rnn_paper_zoo_doc())
    _write_text(out / "rnn_zoo.md", render_rnn_zoo_doc())
