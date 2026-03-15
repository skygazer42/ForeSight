#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METADATA_USER_AGENT = "ForeSight-metadata/0.1"
SLUG_SPLIT_REGEX = r"[^a-z0-9]+"
JOZEFOWICZ_MUTATION_TITLE = "An Empirical Exploration of Recurrent Network Architectures"
JOZEFOWICZ_MUTATION_URL = "https://proceedings.mlr.press/v37/jozefowicz15.html"


def _repo_root() -> Path:
    # tools/fetch_rnn_paper_metadata.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResolvedDocsJsonPath:
    path: Path

    def __post_init__(self) -> None:
        docs_dir = (_repo_root().resolve(strict=False) / "docs").resolve(strict=False)
        resolved = self.path.resolve(strict=False)
        if resolved.parent != docs_dir or resolved.suffix != ".json":
            raise ValueError("resolved docs JSON path must stay inside docs/")
        object.__setattr__(self, "path", resolved)

    def __str__(self) -> str:
        return str(self.path)


def _ensure_src_on_path(root: Path) -> None:
    import sys

    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _read_json(path: ResolvedDocsJsonPath) -> dict[str, Any]:
    if not isinstance(path, ResolvedDocsJsonPath):
        raise TypeError("expected resolved docs JSON path")
    resolved = path.path
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: ResolvedDocsJsonPath, payload: dict[str, Any]) -> None:
    if not isinstance(path, ResolvedDocsJsonPath):
        raise TypeError("expected resolved docs JSON path")
    resolved = path.path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_output_path(output: str | Path) -> ResolvedDocsJsonPath:
    name = str(output).strip()
    candidate = Path(name)
    if (
        not name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != name
        or re.fullmatch(r"[A-Za-z0-9._-]+\.json", name) is None
    ):
        raise ValueError("output must be a JSON filename inside docs/")
    return ResolvedDocsJsonPath(
        (_repo_root().resolve(strict=False) / "docs" / candidate.name).resolve(strict=False)
    )


def _parse_desc(desc: str) -> tuple[str, list[str], int | None]:
    """
    Parse description strings like:
      "Skip RNN (Campos et al., 2017)"
      "Bidirectional RNN (Schuster & Paliwal, 1997)"
    Returns: (name_part, author_last_names, year)
    """

    text = _normalize_spaces(desc)
    m = re.match(r"^(.*?)\s*\((.*?)\s*,\s*(\d{4})\)\s*$", text)
    if not m:
        return text, [], None

    name_part = _normalize_spaces(m.group(1))
    author_part = _normalize_spaces(m.group(2))
    year = int(m.group(3))

    author_part = author_part.replace("et al.", "").replace("et al", "")
    author_part = author_part.replace("&", ",").replace(" and ", ",")
    last_names: list[str] = []
    for chunk in author_part.split(","):
        c = chunk.strip()
        if not c:
            continue
        parts = c.split()
        if not parts:
            continue
        last = parts[-1]
        # Preserve common multi-token last names when written as "von Brecht".
        if last.lower() == "brecht" and len(parts) >= 2 and parts[-2].lower() == "von":
            last = "von Brecht"
        last_names.append(last)
    return name_part, last_names, year


def _expand_name_for_title_search(name: str) -> list[str]:
    """
    Generate title-search candidates for arXiv `ti:"..."`.
    """

    n = _normalize_spaces(name)
    out: list[str] = [n]

    expansions = [
        (r"\bRNNs\b", "Recurrent Neural Networks"),
        (r"\bRNN\b", "Recurrent Neural Network"),
        (r"\bLSTM\b", "Long Short-Term Memory"),
        (r"\bGRU\b", "Gated Recurrent Unit"),
        (r"\bQRNN\b", "Quasi-Recurrent Neural Network"),
        (r"\bSRU\b", "Simple Recurrent Unit"),
        (r"\bESN\b", "Echo State Network"),
        (r"\bDNC\b", "Differentiable Neural Computer"),
        (r"\bNTM\b", "Neural Turing Machine"),
    ]
    n2 = n
    for pat, rep in expansions:
        n2 = re.sub(pat, rep, n2)
    n2 = _normalize_spaces(n2)
    if n2 != n:
        out.append(n2)

    # Deduplicate, preserve order.
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s and s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": METADATA_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _http_get_text(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": METADATA_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _doi_url(doi: str) -> str:
    d = _normalize_spaces(doi)
    if not d:
        return ""
    return "https://doi.org/" + d


def _arxiv_abs_url(arxiv_id: str) -> str:
    a = _normalize_spaces(arxiv_id)
    if not a:
        return ""
    return "https://arxiv.org/abs/" + a


def _best_reference_url(
    *,
    paper_id: str,
    doi: str,
    arxiv_id: str,
    overrides: dict[str, str],
) -> tuple[str, str]:
    """
    Returns: (url, source_url)
    """

    d = _normalize_spaces(doi)
    if d:
        return (_doi_url(d), "doi")

    a = _normalize_spaces(arxiv_id)
    if a:
        return (_arxiv_abs_url(a), "arxiv")

    if paper_id in overrides:
        u = _normalize_spaces(overrides[paper_id])
        if u:
            return (u, "override")

    return ("", "")


@dataclass(frozen=True)
class ArxivHit:
    arxiv_id: str
    title: str
    year: int | None
    doi: str | None
    authors: tuple[str, ...]


def _normalize_arxiv_id(raw_id: str) -> str:
    arxiv_id = raw_id.split("/")[-1] if raw_id else ""
    if "v" in arxiv_id:
        arxiv_id = arxiv_id.split("v", 1)[0]
    return arxiv_id


def _published_year(published: str) -> int | None:
    if len(published) >= 4 and published[:4].isdigit():
        return int(published[:4])
    return None


def _arxiv_entry_authors(entry: ET.Element, ns: dict[str, str]) -> tuple[str, ...]:
    authors: list[str] = []
    for author in entry.findall("a:author", ns):
        name = _normalize_spaces(author.findtext("a:name", default="", namespaces=ns))
        if name:
            authors.append(name)
    return tuple(authors)


def _parse_arxiv_entry(entry: ET.Element, ns: dict[str, str]) -> ArxivHit | None:
    raw_id = _normalize_spaces(entry.findtext("a:id", default="", namespaces=ns))
    arxiv_id = _normalize_arxiv_id(raw_id)
    title = _normalize_spaces(entry.findtext("a:title", default="", namespaces=ns))
    if not arxiv_id or not title:
        return None

    published = _normalize_spaces(entry.findtext("a:published", default="", namespaces=ns))
    doi = _normalize_spaces(entry.findtext("arxiv:doi", default="", namespaces=ns)) or None
    return ArxivHit(
        arxiv_id=arxiv_id,
        title=title,
        year=_published_year(published),
        doi=doi,
        authors=_arxiv_entry_authors(entry, ns),
    )


def _arxiv_query(search_query: str, *, max_results: int = 5) -> list[ArxivHit]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": search_query, "start": 0, "max_results": int(max_results)}
    )
    xml = _http_get_text(url, timeout=30)

    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml)
    hits: list[ArxivHit] = []
    for entry in root.findall("a:entry", ns):
        hit = _parse_arxiv_entry(entry, ns)
        if hit is not None:
            hits.append(hit)
    return hits


def _score_year_delta(
    hit_year: int | None,
    expected_year: int | None,
    *,
    exact: float,
    near_one: float,
    near_two: float,
) -> float:
    if hit_year is None or expected_year is None:
        return 0.0

    delta = abs(int(hit_year) - int(expected_year))
    if delta == 0:
        return exact
    if delta == 1:
        return near_one
    if delta == 2:
        return near_two
    return 0.0


def _count_author_last_name_matches(author_last_names: list[str], author_blob: str) -> int:
    return sum(1 for ln in author_last_names if ln.lower() in author_blob)


def _arxiv_hint_title_score(
    hit: ArxivHit,
    *,
    author_last_names: list[str],
    year: int | None,
    hint_title: str | None,
    author_blob: str,
) -> float | None:
    if not hint_title:
        return 0.0

    sim = _jaccard(_title_token_set(hit.title), _title_token_set(hint_title))
    if sim < 0.40:
        return None
    if year is not None and hit.year is not None:
        if abs(int(hit.year) - int(year)) > 2 and sim < 0.60:
            return None
    elif sim < 0.60 and _count_author_last_name_matches(author_last_names, author_blob) == 0:
        return None
    return 4.0 * float(sim)


def _score_arxiv_hit(
    hit: ArxivHit,
    *,
    name: str,
    author_last_names: list[str],
    year: int | None,
    hint_title: str | None = None,
) -> float:
    score = 0.0
    title_l = hit.title.lower()
    author_blob = " ".join(hit.authors).lower()

    # If we have a strong expected title (via hint), require a close title match to avoid
    # false positives from broad `all:` searches (e.g. generic "cell" matches).
    #
    # Intentionally strict: if arXiv doesn't have the paper, we'd rather return None than
    # attach an unrelated arXiv id/DOI.
    hint_score = _arxiv_hint_title_score(
        hit,
        author_last_names=author_last_names,
        year=year,
        hint_title=hint_title,
        author_blob=author_blob,
    )
    if hint_score is None:
        return -1e9
    score += hint_score

    # Year match.
    score += _score_year_delta(hit.year, year, exact=3.0, near_one=1.5, near_two=0.5)

    # Author match (substring in authors list).
    score += float(_count_author_last_name_matches(author_last_names, author_blob))

    # Token overlap.
    name_tokens = [t for t in re.split(SLUG_SPLIT_REGEX, name.lower()) if t and len(t) >= 3]
    if name_tokens:
        overlap = sum(1 for t in set(name_tokens) if t in title_l)
        score += 0.4 * float(overlap)

    if "recurrent" in title_l:
        score += 0.2
    return score


def _best_arxiv_match(
    *,
    paper_id: str,
    desc: str,
    title_hints: dict[str, str],
    sleep_s: float,
) -> ArxivHit | None:
    name, author_last_names, year = _parse_desc(desc)
    hint = title_hints.get(paper_id)
    queries = _iter_arxiv_queries(
        name=name,
        author_last_names=author_last_names,
        hint_title=hint,
    )
    best = _search_best_arxiv_hit(
        queries,
        name=name,
        author_last_names=author_last_names,
        year=year,
        hint_title=hint,
        sleep_s=sleep_s,
    )
    if best is None or best[0] < 2.0:
        return None
    return best[1]


@dataclass(frozen=True)
class CrossrefHit:
    title: str
    year: int | None
    doi: str | None
    url: str | None
    authors: tuple[str, ...]


def _crossref_year(issued: Any) -> int | None:
    if (
        issued
        and isinstance(issued, list)
        and issued
        and isinstance(issued[0], list)
        and issued[0]
        and isinstance(issued[0][0], int)
    ):
        return int(issued[0][0])
    return None


def _crossref_authors(items: Any) -> tuple[str, ...]:
    authors: list[str] = []
    for item in items or []:
        family = _normalize_spaces(item.get("family") or "")
        given = _normalize_spaces(item.get("given") or "")
        if family and given:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)
    return tuple(authors)


def _parse_crossref_hit(item: dict[str, Any]) -> CrossrefHit | None:
    title_raw = item.get("title") or []
    title = _normalize_spaces(title_raw[0]) if isinstance(title_raw, list) and title_raw else ""
    if not title:
        return None

    doi = _normalize_spaces(item.get("DOI") or "") or None
    url = _normalize_spaces(item.get("URL") or "") or None
    return CrossrefHit(
        title=title,
        year=_crossref_year(item.get("issued", {}).get("date-parts")),
        doi=doi,
        url=url,
        authors=_crossref_authors(item.get("author")),
    )


def _crossref_query(query: str, *, year: int | None, rows: int = 5) -> list[CrossrefHit]:
    params: dict[str, str] = {"query.bibliographic": query, "rows": str(int(rows))}
    if year is not None:
        params["filter"] = f"from-pub-date:{int(year)}-01-01,until-pub-date:{int(year)}-12-31"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, headers={"User-Agent": METADATA_USER_AGENT}, timeout=30)

    out: list[CrossrefHit] = []
    for it in data.get("message", {}).get("items", []):
        hit = _parse_crossref_hit(it)
        if hit is not None:
            out.append(hit)
    return out


def _score_crossref_hit(
    hit: CrossrefHit,
    *,
    expected_title: str | None,
    author_last_names: list[str],
    year: int | None,
) -> float:
    score = 0.0
    title_l = hit.title.lower()

    score += _score_year_delta(hit.year, year, exact=2.0, near_one=1.0, near_two=0.2)

    author_blob = " ".join(hit.authors).lower()
    score += 0.8 * float(_count_author_last_name_matches(author_last_names, author_blob))

    if expected_title:
        et = expected_title.lower()
        etoks = {t for t in re.split(SLUG_SPLIT_REGEX, et) if t and len(t) >= 3}
        overlap = sum(1 for t in etoks if t in title_l)
        score += 0.3 * float(overlap)
    else:
        if "recurrent" in title_l:
            score += 0.3
        if "neural" in title_l:
            score += 0.1

    if hit.doi:
        score += 0.1
    return score


def _title_token_set(title: str) -> set[str]:
    return {t for t in re.split(SLUG_SPLIT_REGEX, str(title).lower()) if t and len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


def _iter_arxiv_queries(
    *,
    name: str,
    author_last_names: list[str],
    hint_title: str | None,
) -> list[str]:
    queries: list[str] = []
    if hint_title:
        queries.append(f'ti:"{hint_title}"')
    for candidate in _expand_name_for_title_search(name):
        queries.append(f'ti:"{candidate}"')
    if author_last_names:
        queries.append("all:" + " ".join([name] + author_last_names[:2]))
    else:
        queries.append("all:" + name)
    return queries


def _sleep_between_requests(sleep_s: float) -> None:
    if sleep_s > 0:
        time.sleep(float(sleep_s))


def _search_best_arxiv_hit(
    queries: list[str],
    *,
    name: str,
    author_last_names: list[str],
    year: int | None,
    hint_title: str | None,
    sleep_s: float,
) -> tuple[float, ArxivHit] | None:
    best: tuple[float, ArxivHit] | None = None
    for query in queries:
        try:
            hits = _arxiv_query(query, max_results=5)
        except Exception:
            hits = []

        _sleep_between_requests(sleep_s)

        for hit in hits:
            score = _score_arxiv_hit(
                hit,
                name=name,
                author_last_names=author_last_names,
                year=year,
                hint_title=hint_title,
            )
            if best is None or score > best[0]:
                best = (score, hit)

        if best is not None and best[0] >= 3.0 and query.startswith("ti:"):
            break
    return best


def _iter_crossref_queries(
    *,
    name: str,
    author_last_names: list[str],
    year: int | None,
    expected_title: str | None,
) -> list[str]:
    queries: list[str] = []
    if expected_title:
        queries.append(expected_title)
    queries.append(f"{name} {' '.join(author_last_names[:2])} {year or ''}".strip())
    queries.append(name)
    return queries


def _crossref_title_similarity(
    hit: CrossrefHit,
    *,
    expected_title: str | None,
    author_last_names: list[str],
    strict_title_match: bool,
) -> float | None:
    if not expected_title:
        return None

    title_sim = _jaccard(_title_token_set(hit.title), _title_token_set(expected_title))
    author_blob = " ".join(hit.authors).lower()
    has_author_match = any(last_name.lower() in author_blob for last_name in author_last_names)
    if strict_title_match:
        min_sim = 0.50
    elif author_last_names:
        min_sim = 0.45 if has_author_match else 0.65
    else:
        min_sim = 0.55
    if title_sim < min_sim:
        return None
    return title_sim


def _crossref_hits_for_query(
    query: str,
    *,
    year: int | None,
    sleep_s: float,
) -> list[CrossrefHit]:
    try:
        hits = _crossref_query(query, year=year, rows=5)
    except Exception:
        hits = []

    _sleep_between_requests(sleep_s)
    return hits


def _crossref_hit_score(
    hit: CrossrefHit,
    *,
    expected_title: str | None,
    author_last_names: list[str],
    year: int | None,
    strict_title_match: bool,
) -> float | None:
    title_sim = _crossref_title_similarity(
        hit,
        expected_title=expected_title,
        author_last_names=author_last_names,
        strict_title_match=strict_title_match,
    )
    if expected_title and title_sim is None:
        return None

    score = _score_crossref_hit(
        hit,
        expected_title=expected_title,
        author_last_names=author_last_names,
        year=year,
    )
    if title_sim is not None:
        score += 4.0 * float(title_sim)
    return score


def _search_best_crossref_hit(
    queries: list[str],
    *,
    expected_title: str | None,
    author_last_names: list[str],
    year: int | None,
    strict_title_match: bool,
    sleep_s: float,
) -> tuple[float, CrossrefHit] | None:
    best: tuple[float, CrossrefHit] | None = None
    for query in queries:
        hits = _crossref_hits_for_query(query, year=year, sleep_s=sleep_s)

        for hit in hits:
            score = _crossref_hit_score(
                hit,
                expected_title=expected_title,
                author_last_names=author_last_names,
                year=year,
                strict_title_match=strict_title_match,
            )
            if score is None:
                continue

            if best is None or score > best[0]:
                best = (score, hit)

        if best is not None and best[0] >= 2.5:
            break
    return best


def _best_crossref_match(
    *,
    desc: str,
    expected_title: str | None,
    strict_title_match: bool,
    sleep_s: float,
) -> CrossrefHit | None:
    name, author_last_names, year = _parse_desc(desc)
    queries = _iter_crossref_queries(
        name=name,
        author_last_names=author_last_names,
        year=year,
        expected_title=expected_title,
    )
    best = _search_best_crossref_hit(
        queries,
        expected_title=expected_title,
        author_last_names=author_last_names,
        year=year,
        strict_title_match=strict_title_match,
        sleep_s=sleep_s,
    )
    if best is None or best[0] < 1.6:
        return None
    return best[1]


def _metadata_seed_record(paper_id: str, desc: str, expected_year: int | None) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "desc": desc,
        "title": "",
        "year": int(expected_year) if expected_year is not None else None,
        "doi": "",
        "arxiv_id": "",
        "url": "",
        "source_title": "",
        "source_doi": "",
        "source_url": "",
    }


def _source_title_name(
    *,
    arxiv: ArxivHit | None,
    hint_title: str | None,
    crossref: CrossrefHit | None,
) -> str:
    if arxiv:
        return "arxiv"
    if hint_title:
        return "hint"
    if crossref:
        return "crossref"
    return ""


def _source_doi_name(*, arxiv: ArxivHit | None, crossref: CrossrefHit | None) -> str:
    if arxiv and arxiv.doi:
        return "arxiv"
    if crossref and crossref.doi:
        return "crossref"
    return ""


def _metadata_year(
    *,
    expected_year: int | None,
    arxiv: ArxivHit | None,
    crossref: CrossrefHit | None,
) -> int | None:
    if expected_year is not None:
        return expected_year
    if arxiv and arxiv.year is not None:
        return arxiv.year
    if crossref and crossref.year is not None:
        return crossref.year
    return None


def _metadata_doi(*, arxiv: ArxivHit | None, crossref: CrossrefHit | None) -> str:
    if arxiv and arxiv.doi:
        return arxiv.doi
    if crossref and crossref.doi:
        return crossref.doi
    return ""


def _resolve_entry_url_fields(
    *,
    paper_id: str,
    doi: str,
    arxiv_id: str,
    url: str,
    source_url: str,
    url_overrides: dict[str, str],
) -> tuple[str, str]:
    resolved_url = url
    resolved_source = source_url
    if not resolved_url:
        resolved_url, resolved_source = _best_reference_url(
            paper_id=paper_id,
            doi=doi,
            arxiv_id=arxiv_id,
            overrides=url_overrides,
        )
    if resolved_url and not resolved_source:
        if doi:
            resolved_source = "doi"
        elif arxiv_id:
            resolved_source = "arxiv"
        elif paper_id in url_overrides:
            resolved_source = "override"
        else:
            resolved_source = ""
    return (resolved_url, resolved_source)


def _expected_title_for_crossref(
    *,
    arxiv: ArxivHit | None,
    hint_title: str | None,
) -> str | None:
    if arxiv:
        return arxiv.title
    return hint_title or None


def _ensure_metadata_entry(
    *,
    paper_id: str,
    desc: str,
    expected_year: int | None,
    existing: dict[str, Any],
    updated: dict[str, Any],
    url_overrides: dict[str, str],
) -> dict[str, Any]:
    current = updated.get(paper_id)
    if isinstance(current, dict):
        current["paper_id"] = paper_id
        current["desc"] = desc
        if expected_year is not None:
            current["year"] = int(expected_year)
    else:
        current = _metadata_seed_record(paper_id, desc, expected_year)

    doi = _normalize_spaces(str(current.get("doi", "")))
    arxiv_id = _normalize_spaces(str(current.get("arxiv_id", "")))
    url = _normalize_spaces(str(current.get("url", "")))
    source_url = _normalize_spaces(str(current.get("source_url", "")))
    url, source_url = _resolve_entry_url_fields(
        paper_id=paper_id,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source_url=source_url,
        url_overrides=url_overrides,
    )

    current["url"] = url
    current["source_url"] = source_url
    updated[paper_id] = current
    return current


def _build_metadata_record(
    *,
    paper_id: str,
    desc: str,
    expected_year: int | None,
    title_hints: dict[str, str],
    url_overrides: dict[str, str],
    sleep_s: float,
) -> dict[str, Any]:
    arxiv = _best_arxiv_match(
        paper_id=paper_id,
        desc=desc,
        title_hints=title_hints,
        sleep_s=sleep_s,
    )
    hint_title = title_hints.get(paper_id)
    expected_title = _expected_title_for_crossref(arxiv=arxiv, hint_title=hint_title)
    crossref = _best_crossref_match(
        desc=desc,
        expected_title=expected_title,
        strict_title_match=bool(hint_title) and (arxiv is None),
        sleep_s=sleep_s,
    )

    doi = _metadata_doi(arxiv=arxiv, crossref=crossref)
    year = _metadata_year(
        expected_year=expected_year,
        arxiv=arxiv,
        crossref=crossref,
    )
    url, source_url = _best_reference_url(
        paper_id=paper_id,
        doi=doi,
        arxiv_id=(arxiv.arxiv_id if arxiv else ""),
        overrides=url_overrides,
    )
    return {
        "paper_id": paper_id,
        "desc": desc,
        "title": expected_title or (crossref.title if crossref else ""),
        "year": int(year) if year is not None else None,
        "doi": doi,
        "arxiv_id": arxiv.arxiv_id if arxiv else "",
        "url": url,
        "source_title": _source_title_name(
            arxiv=arxiv,
            hint_title=hint_title,
            crossref=crossref,
        ),
        "source_doi": _source_doi_name(arxiv=arxiv, crossref=crossref),
        "source_url": source_url,
    }


def _missing_metadata_ids(
    defs: list[tuple[str, str]],
    updated: dict[str, Any],
    *,
    predicate: Any,
) -> list[str]:
    missing: list[str] = []
    for paper_id, _desc in defs:
        record = updated.get(paper_id, {})
        if predicate(record):
            missing.append(paper_id)
    return missing


def _print_metadata_coverage(
    *,
    output_path: Path,
    updated: dict[str, Any],
    paper_defs: list[tuple[str, str]],
) -> None:
    missing_title = _missing_metadata_ids(
        paper_defs,
        updated,
        predicate=lambda record: not str(record.get("title", "")).strip(),
    )
    missing_any_ref = _missing_metadata_ids(
        paper_defs,
        updated,
        predicate=lambda record: (
            not str(record.get("doi", "")).strip()
            and not str(record.get("arxiv_id", "")).strip()
        ),
    )
    missing_url = _missing_metadata_ids(
        paper_defs,
        updated,
        predicate=lambda record: not str(record.get("url", "")).strip(),
    )

    print(f"\nWrote: {output_path}")
    print(
        f"Coverage: titles={100 - len(missing_title)}/100, doi_or_arxiv={100 - len(missing_any_ref)}/100"
    )
    print(f"Coverage: url={100 - len(missing_url)}/100")
    if missing_title:
        print("Missing titles:")
        for paper_id in missing_title:
            print(" -", paper_id)
    if missing_any_ref:
        print("Missing DOI+arXiv:")
        for paper_id in missing_any_ref:
            print(" -", paper_id)
    if missing_url:
        print("Missing URL:")
        for paper_id in missing_url:
            print(" -", paper_id)


def _resolve_wanted_paper_defs(
    *,
    paper_defs: list[tuple[str, str]],
    extra_defs: list[tuple[str, str]],
    only: str,
) -> list[tuple[str, str]]:
    wanted = list(paper_defs) + list(extra_defs)
    if only:
        wanted = [(pid, desc) for pid, desc in wanted if pid == only]
        if not wanted:
            raise SystemExit(f"Unknown paper_id: {only!r}")
    return wanted


def _should_rebuild_metadata_record(
    *,
    paper_id: str,
    existing: dict[str, Any],
    refresh: bool,
) -> bool:
    if refresh:
        return True
    if paper_id not in existing:
        return True
    return not str(existing.get(paper_id, {}).get("title", "")).strip()


def _print_metadata_progress_row(
    *,
    idx: int,
    total: int,
    paper_id: str,
    record: dict[str, Any],
) -> None:
    has_doi = bool(record["doi"])
    has_arxiv = bool(record["arxiv_id"])
    print(
        f"[{idx:03d}/{total:03d}] {paper_id:28s} year={str(record['year']):4s} "
        f"doi={'Y' if has_doi else '-'} arxiv={'Y' if has_arxiv else '-'} "
        f"title={str(record['title'])[:70]!r}",
        flush=True,
    )


def _refresh_metadata_record(
    *,
    updated: dict[str, Any],
    paper_id: str,
    desc: str,
    expected_year: int | None,
    title_hints: dict[str, str],
    url_overrides: dict[str, str],
    sleep_s: float,
    output_path: Path,
    idx: int,
    total: int,
) -> dict[str, Any]:
    updated[paper_id] = _build_metadata_record(
        paper_id=paper_id,
        desc=desc,
        expected_year=expected_year,
        title_hints=title_hints,
        url_overrides=url_overrides,
        sleep_s=sleep_s,
    )
    _print_metadata_progress_row(
        idx=idx,
        total=total,
        paper_id=paper_id,
        record=updated[paper_id],
    )
    # Persist progress so interruptions don't lose work.
    _write_json(output_path, updated)
    return updated[paper_id]


def fetch_all(*, output_path: Path, refresh: bool, sleep_s: float, only: str) -> None:
    root = _repo_root()
    _ensure_src_on_path(root)

    from foresight.models.torch_rnn_paper_zoo import _PAPER_DEFS

    existing: dict[str, Any] = _read_json(output_path)
    updated: dict[str, Any] = dict(existing)

    extra_defs: list[tuple[str, str]] = [
        # RNN Zoo-only base. (This is not part of the 100-paper zoo list, but the CLI uses the same metadata file.)
        ("janet", "JANET / Forget-gate LSTM (van der Westhuizen & Lasenby, 2018)"),
        # RNN Zoo wrapper paper (LayerNorm).
        ("layer-normalization", "Layer Normalization (Ba et al., 2016)"),
    ]

    # Hand-written title hints for cases where the architecture name doesn't appear in the paper title,
    # or where Crossref/arXiv matching is unreliable (older/obscure references).
    #
    # Keep this list relatively small; most items should be found by `ti:"<architecture name>"`.
    title_hints: dict[str, str] = {
        "elman-srn": "Finding Structure in Time",
        "jordan-srn": "Attractor dynamics and parallelism in a connectionist sequential machine",
        # Acronym-only architecture names.
        "lstm": "Long Short-Term Memory",
        "gru": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
        "rnn-encoder-decoder": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
        "seq2seq": "Sequence to Sequence Learning with Neural Networks",
        # Architecture nicknames that don't match the published title.
        "chrono-lstm": "Can recurrent neural networks warp time?",
        "peephole-lstm": "Learning Precise Timing with LSTM Recurrent Networks",
        "lstm-projection": "Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition",
        "tree-lstm": "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks",
        "on-lstm": "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks",
        "scrn": "Learning Longer Memory in Recurrent Neural Networks",
        "lstnet": "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks",
        "ran": "Recurrent Additive Networks",
        "fru": "Learning Long Term Dependencies via Fourier Recurrent Units",
        "star": "Gating Revisited: Deep Multi-layer RNNs That Can Be Trained",
        "brc": "A bio-inspired bistable recurrent cell allows for long-lasting memory",
        "nbrc": "A bio-inspired bistable recurrent cell allows for long-lasting memory",
        "residual-rnn": "Residual Recurrent Neural Networks for Learning Sequential Representations",
        "orthogonal-rnn": "Recurrent Orthogonal Networks and Long-Memory Tasks",
        "echo-state-network": 'The "echo state" approach to analysing and training recurrent neural networks',
        "liquid-state-machine": "Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations",
        "deep-ar": "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks",
        "esrnn": "A hybrid method of Exponential Smoothing and Recurrent Neural Networks for time series forecasting",
        "mqrnn": "A Multi-Horizon Quantile Recurrent Forecaster",
        "deepstate": "DeepState: Learning State Space Models for Time Series Forecasting",
        "janet": "The unreasonable effectiveness of the forget gate",
        "layer-normalization": "Layer Normalization",
        # Multiple named variants in a single source paper.
        "mut1": JOZEFOWICZ_MUTATION_TITLE,
        "mut2": JOZEFOWICZ_MUTATION_TITLE,
        "mut3": JOZEFOWICZ_MUTATION_TITLE,
        "neural-turing-machine": "Neural Turing Machines",
        "differentiable-neural-computer": "Hybrid computing using a neural network with dynamic external memory",
        "pointer-sentinel-mixture": "Pointer Sentinel Mixture Models",
        "rnn-transducer": "Sequence Transduction with Recurrent Neural Networks",
        "ode-rnn": "Latent ODEs for Irregularly-Sampled Time Series",
        "neural-cde": "Neural Controlled Differential Equations for Irregular Time Series",
        "dynamic-memory-networks": "Ask Me Anything: Dynamic Memory Networks for Natural Language Processing",
        "copynet": "Incorporating Copying Mechanism in Sequence-to-Sequence Learning",
        "bahdanau-attention": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "luong-attention": "Effective Approaches to Attention-based Neural Machine Translation",
        "neural-stack": "Learning to Transduce with Unbounded Memory",
        "neural-queue": "Learning to Transduce with Unbounded Memory",
        "convlstm": "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting",
        "convgru": "Delving Deeper into Convolutional Networks for Learning Video Representations",
        "trajgru": "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model",
        "predrnn": "PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs",
        "dcrnn": "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting",
    }

    # Stable landing pages for items without DOI/arXiv identifiers.
    url_overrides: dict[str, str] = {
        # Jordan (1986) technical report (stable repository page includes PDF download).
        "jordan-srn": "https://escholarship.org/uc/item/23d0t3cn",
        # Gers et al. (2002) is a JMLR paper (no DOI); use the canonical JMLR landing page.
        "peephole-lstm": "https://www.jmlr.org/papers/v3/gers02a.html",
        # Józefowicz et al. (2015) ICML / PMLR (no DOI); use proceedings landing page.
        "mut1": JOZEFOWICZ_MUTATION_URL,
        "mut2": JOZEFOWICZ_MUTATION_URL,
        "mut3": JOZEFOWICZ_MUTATION_URL,
        # Jaeger (2001) GMD report hosted on Fraunhofer's publica.
        "echo-state-network": "https://publica.fraunhofer.de/handle/publica/291207",
    }

    wanted = _resolve_wanted_paper_defs(
        paper_defs=list(_PAPER_DEFS),
        extra_defs=extra_defs,
        only=only,
    )

    total = len(wanted)
    for idx, (paper_id, desc) in enumerate(wanted, start=1):
        _, _, expected_year = _parse_desc(desc)
        _ensure_metadata_entry(
            paper_id=paper_id,
            desc=desc,
            expected_year=expected_year,
            existing=existing,
            updated=updated,
            url_overrides=url_overrides,
        )

        if not _should_rebuild_metadata_record(
            paper_id=paper_id,
            existing=existing,
            refresh=refresh,
        ):
            continue

        _refresh_metadata_record(
            updated=updated,
            paper_id=paper_id,
            desc=desc,
            expected_year=expected_year,
            title_hints=title_hints,
            url_overrides=url_overrides,
            sleep_s=sleep_s,
            output_path=output_path,
            idx=idx,
            total=total,
        )

    # Final write (ensures sort_keys/formatting is consistent).
    _write_json(output_path, updated)
    _print_metadata_coverage(output_path=output_path, updated=updated, paper_defs=list(_PAPER_DEFS))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch paper titles/DOIs/arXiv IDs for RNN Paper Zoo.")
    ap.add_argument(
        "--output",
        type=str,
        default="rnn_paper_metadata.json",
        help="Output JSON filename under docs/ (default: rnn_paper_metadata.json)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh entries even if already present.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between network requests (seconds).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Only fetch one paper_id (for debugging).",
    )
    args = ap.parse_args()

    fetch_all(
        output_path=_resolve_output_path(str(args.output)),
        refresh=bool(args.refresh),
        sleep_s=float(args.sleep),
        only=str(args.only).strip(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
