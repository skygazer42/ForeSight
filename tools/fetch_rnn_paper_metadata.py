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


def _ensure_src_on_path(root: Path) -> None:
    import sys

    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_output_path(output: str | Path) -> Path:
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
    return _repo_root().resolve(strict=False) / "docs" / candidate.name


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
        raw_id = _normalize_spaces(entry.findtext("a:id", default="", namespaces=ns))
        arxiv_id = raw_id.split("/")[-1] if raw_id else ""
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.split("v", 1)[0]

        title = _normalize_spaces(entry.findtext("a:title", default="", namespaces=ns))
        published = _normalize_spaces(entry.findtext("a:published", default="", namespaces=ns))
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None

        doi = _normalize_spaces(entry.findtext("arxiv:doi", default="", namespaces=ns))
        doi = doi or None

        authors: list[str] = []
        for a in entry.findall("a:author", ns):
            an = _normalize_spaces(a.findtext("a:name", default="", namespaces=ns))
            if an:
                authors.append(an)

        if arxiv_id and title:
            hits.append(
                ArxivHit(
                    arxiv_id=arxiv_id,
                    title=title,
                    year=year,
                    doi=doi,
                    authors=tuple(authors),
                )
            )
    return hits


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
    if hint_title:
        sim = _jaccard(_title_token_set(hit.title), _title_token_set(hint_title))
        if sim < 0.40:
            return -1e9
        if year is not None and hit.year is not None:
            # arXiv `published` year is upload year, which can differ from the paper year.
            # Enforce near-year matches unless the title match is very strong.
            if abs(int(hit.year) - int(year)) > 2 and sim < 0.60:
                return -1e9
        elif sim < 0.60:
            # If year is missing on either side, require an author last-name match unless
            # the title match is very strong.
            if not any(ln.lower() in author_blob for ln in author_last_names):
                return -1e9
        score += 4.0 * float(sim)

    # Year match.
    if year is not None and hit.year is not None:
        dy = abs(int(hit.year) - int(year))
        if dy == 0:
            score += 3.0
        elif dy == 1:
            score += 1.5
        elif dy == 2:
            score += 0.5

    # Author match (substring in authors list).
    for ln in author_last_names:
        if ln.lower() in author_blob:
            score += 1.0

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

    queries: list[str] = []
    hint = title_hints.get(paper_id)
    if hint:
        queries.append(f'ti:"{hint}"')
    for cand in _expand_name_for_title_search(name):
        queries.append(f'ti:"{cand}"')

    # Fallback keyword search.
    if author_last_names:
        queries.append("all:" + " ".join([name] + author_last_names[:2]))
    else:
        queries.append("all:" + name)

    best: tuple[float, ArxivHit] | None = None
    for q in queries:
        try:
            hits = _arxiv_query(q, max_results=5)
        except Exception:
            hits = []

        if sleep_s > 0:
            time.sleep(float(sleep_s))

        for h in hits:
            s = _score_arxiv_hit(
                h,
                name=name,
                author_last_names=author_last_names,
                year=year,
                hint_title=hint,
            )
            if best is None or s > best[0]:
                best = (s, h)

        # If we found a confident match from a title query, stop early.
        if best is not None and best[0] >= 3.0 and q.startswith("ti:"):
            break

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


def _crossref_query(query: str, *, year: int | None, rows: int = 5) -> list[CrossrefHit]:
    params: dict[str, str] = {"query.bibliographic": query, "rows": str(int(rows))}
    if year is not None:
        params["filter"] = f"from-pub-date:{int(year)}-01-01,until-pub-date:{int(year)}-12-31"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, headers={"User-Agent": METADATA_USER_AGENT}, timeout=30)

    out: list[CrossrefHit] = []
    for it in data.get("message", {}).get("items", []):
        title_raw = it.get("title") or []
        title = _normalize_spaces(title_raw[0]) if isinstance(title_raw, list) and title_raw else ""
        if not title:
            continue
        doi = _normalize_spaces(it.get("DOI") or "") or None
        url0 = _normalize_spaces(it.get("URL") or "") or None

        issued = it.get("issued", {}).get("date-parts")
        yr = None
        if (
            issued
            and isinstance(issued, list)
            and issued
            and isinstance(issued[0], list)
            and issued[0]
            and isinstance(issued[0][0], int)
        ):
            yr = int(issued[0][0])

        authors: list[str] = []
        for a in it.get("author") or []:
            fam = _normalize_spaces(a.get("family") or "")
            given = _normalize_spaces(a.get("given") or "")
            if fam and given:
                authors.append(f"{given} {fam}")
            elif fam:
                authors.append(fam)

        out.append(CrossrefHit(title=title, year=yr, doi=doi, url=url0, authors=tuple(authors)))
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

    if year is not None and hit.year is not None:
        dy = abs(int(hit.year) - int(year))
        if dy == 0:
            score += 2.0
        elif dy == 1:
            score += 1.0
        elif dy == 2:
            score += 0.2

    author_blob = " ".join(hit.authors).lower()
    for ln in author_last_names:
        if ln.lower() in author_blob:
            score += 0.8

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


def _best_crossref_match(
    *,
    desc: str,
    expected_title: str | None,
    strict_title_match: bool,
    sleep_s: float,
) -> CrossrefHit | None:
    name, author_last_names, year = _parse_desc(desc)

    queries: list[str] = []
    if expected_title:
        queries.append(expected_title)
    queries.append(f"{name} {' '.join(author_last_names[:2])} {year or ''}".strip())
    queries.append(name)

    best: tuple[float, CrossrefHit] | None = None
    for q in queries:
        try:
            hits = _crossref_query(q, year=year, rows=5)
        except Exception:
            hits = []

        if sleep_s > 0:
            time.sleep(float(sleep_s))

        for h in hits:
            title_sim = None
            if expected_title:
                title_sim = _jaccard(_title_token_set(h.title), _title_token_set(expected_title))
                author_blob = " ".join(h.authors).lower()
                has_author_match = any(ln.lower() in author_blob for ln in author_last_names)
                if strict_title_match:
                    min_sim = 0.50
                else:
                    if author_last_names:
                        # Without author evidence, be much more conservative: Crossref can return
                        # false positives from partial title overlap ("unreasonable effectiveness", etc.).
                        min_sim = 0.45 if has_author_match else 0.65
                    else:
                        min_sim = 0.55
                if title_sim < min_sim:
                    # When we have an expected title (usually from arXiv or a curated hint),
                    # reject Crossref hits that don't match it closely enough. This prevents
                    # false-positive DOIs from partial token overlap (e.g. "unreasonable effectiveness").
                    continue

            s = _score_crossref_hit(
                h,
                expected_title=expected_title,
                author_last_names=author_last_names,
                year=year,
            )
            if title_sim is not None:
                s += 4.0 * float(title_sim)
            if best is None or s > best[0]:
                best = (s, h)

        if best is not None and best[0] >= 2.5:
            break

    if best is None or best[0] < 1.6:
        return None
    return best[1]


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

    wanted = list(_PAPER_DEFS) + list(extra_defs)
    if only:
        wanted = [(pid, d) for pid, d in wanted if pid == only]
        if not wanted:
            raise SystemExit(f"Unknown paper_id: {only!r}")

    total = len(wanted)
    for idx, (paper_id, desc) in enumerate(wanted, start=1):
        # Always keep curated desc/year in sync, even when we skip network refresh.
        _, _, expected_year = _parse_desc(desc)
        if paper_id in updated and isinstance(updated.get(paper_id), dict):
            updated[paper_id]["paper_id"] = paper_id
            updated[paper_id]["desc"] = desc
            if expected_year is not None:
                updated[paper_id]["year"] = int(expected_year)
        else:
            updated[paper_id] = {
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

        # Even when we skip network refresh, we still want to backfill derived fields (e.g. `url`).
        current = updated.get(paper_id, {})
        if not isinstance(current, dict):
            current = {}
        doi0 = _normalize_spaces(str(current.get("doi", "")))
        arxiv0 = _normalize_spaces(str(current.get("arxiv_id", "")))
        url0 = _normalize_spaces(str(current.get("url", "")))
        source_url0 = _normalize_spaces(str(current.get("source_url", "")))
        if not url0:
            url0, source_url0 = _best_reference_url(
                paper_id=paper_id,
                doi=doi0,
                arxiv_id=arxiv0,
                overrides=url_overrides,
            )
        if url0 and not source_url0:
            # Backfill source based on which field is present.
            if doi0:
                source_url0 = "doi"
            elif arxiv0:
                source_url0 = "arxiv"
            elif paper_id in url_overrides:
                source_url0 = "override"
            else:
                source_url0 = ""

        current["url"] = url0
        current["source_url"] = source_url0
        updated[paper_id] = current

        if (
            not refresh
            and paper_id in existing
            and str(existing.get(paper_id, {}).get("title", "")).strip()
        ):
            continue

        arxiv = _best_arxiv_match(
            paper_id=paper_id,
            desc=desc,
            title_hints=title_hints,
            sleep_s=sleep_s,
        )
        hint_title = title_hints.get(paper_id)
        expected_title = arxiv.title if arxiv else (hint_title or None)
        crossref = _best_crossref_match(
            desc=desc,
            expected_title=expected_title,
            strict_title_match=bool(hint_title) and (arxiv is None),
            sleep_s=sleep_s,
        )

        title = expected_title or (crossref.title if crossref else "")
        # Prefer the year encoded in the curated description (paper year), which is stable and
        # consistent with the model registry. arXiv "published" dates can reflect upload time.
        year = expected_year
        if year is None:
            year = (
                (arxiv.year if (arxiv and arxiv.year) else None)
                or (crossref.year if (crossref and crossref.year) else None)
                or None
            )

        doi = None
        if arxiv and arxiv.doi:
            doi = arxiv.doi
        if not doi and crossref and crossref.doi:
            doi = crossref.doi

        url, source_url = _best_reference_url(
            paper_id=paper_id,
            doi=doi or "",
            arxiv_id=(arxiv.arxiv_id if arxiv else ""),
            overrides=url_overrides,
        )
        if arxiv:
            source_title = "arxiv"
        elif hint_title:
            source_title = "hint"
        elif crossref:
            source_title = "crossref"
        else:
            source_title = ""

        if arxiv and arxiv.doi:
            source_doi = "arxiv"
        elif crossref and crossref.doi:
            source_doi = "crossref"
        else:
            source_doi = ""

        updated[paper_id] = {
            "paper_id": paper_id,
            "desc": desc,
            "title": title,
            "year": int(year) if year is not None else None,
            "doi": doi or "",
            "arxiv_id": arxiv.arxiv_id if arxiv else "",
            "url": url,
            "source_title": source_title,
            "source_doi": source_doi,
            "source_url": source_url,
        }

        has_doi = bool(doi)
        has_arxiv = bool(arxiv and arxiv.arxiv_id)
        print(
            f"[{idx:03d}/{total:03d}] {paper_id:28s} year={str(updated[paper_id]['year']):4s} "
            f"doi={'Y' if has_doi else '-'} arxiv={'Y' if has_arxiv else '-'} title={title[:70]!r}",
            flush=True,
        )

        # Persist progress so interruptions don't lose work.
        _write_json(output_path, updated)

    # Final write (ensures sort_keys/formatting is consistent).
    _write_json(output_path, updated)

    missing_title = [
        pid for pid, _d in _PAPER_DEFS if not str(updated.get(pid, {}).get("title", "")).strip()
    ]
    missing_any_ref = [
        pid
        for pid, _d in _PAPER_DEFS
        if not str(updated.get(pid, {}).get("doi", "")).strip()
        and not str(updated.get(pid, {}).get("arxiv_id", "")).strip()
    ]
    missing_url = [
        pid for pid, _d in _PAPER_DEFS if not str(updated.get(pid, {}).get("url", "")).strip()
    ]

    print(f"\nWrote: {output_path}")
    print(
        f"Coverage: titles={100 - len(missing_title)}/100, doi_or_arxiv={100 - len(missing_any_ref)}/100"
    )
    print(f"Coverage: url={100 - len(missing_url)}/100")
    if missing_title:
        print("Missing titles:")
        for pid in missing_title:
            print(" -", pid)
    if missing_any_ref:
        print("Missing DOI+arXiv:")
        for pid in missing_any_ref:
            print(" -", pid)
    if missing_url:
        print("Missing URL:")
        for pid in missing_url:
            print(" -", pid)


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
