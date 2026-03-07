from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, env_extra: dict[str, str] | None = None) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    assert src_dir.exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout is not None
    return proc.stdout.strip()


def test_cli_models_info_includes_paper_metadata_for_rnnpaper() -> None:
    out = _run_cli("models", "info", "torch-rnnpaper-elman-srn-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnpaper-elman-srn-direct"

    paper = payload.get("paper")
    assert isinstance(paper, dict)
    assert paper.get("paper_id") == "elman-srn"
    assert str(paper.get("title", "")).strip()
    assert paper.get("year") == 1990
    assert str(paper.get("url", "")).startswith("https://")


def test_cli_models_info_omits_paper_metadata_for_non_rnnpaper() -> None:
    out = _run_cli("models", "info", "mean")
    payload = json.loads(out)
    assert payload["key"] == "mean"
    assert "paper" not in payload


def test_cli_models_info_includes_paper_metadata_for_rnnzoo_alias_bases() -> None:
    out = _run_cli("models", "info", "torch-rnnzoo-elman-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnzoo-elman-direct"
    paper = payload.get("paper")
    assert isinstance(paper, dict)
    assert paper.get("paper_id") == "elman-srn"
    assert paper.get("year") == 1990
    assert str(paper.get("url", "")).startswith("https://")


def test_cli_models_info_includes_paper_metadata_for_rnnzoo_direct_base() -> None:
    out = _run_cli("models", "info", "torch-rnnzoo-gru-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnzoo-gru-direct"
    paper = payload.get("paper")
    assert isinstance(paper, dict)
    assert paper.get("paper_id") == "gru"
    assert paper.get("year") == 2014
    assert str(paper.get("url", "")).startswith("https://")

    assert "wrapper_paper" not in payload


def test_cli_models_info_includes_wrapper_paper_metadata_for_rnnzoo_variant() -> None:
    out = _run_cli("models", "info", "torch-rnnzoo-clockwork-attn-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnzoo-clockwork-attn-direct"

    base = payload.get("paper")
    assert isinstance(base, dict)
    assert base.get("paper_id") == "clockwork-rnn"

    wrapper = payload.get("wrapper_paper")
    assert isinstance(wrapper, dict)
    assert wrapper.get("paper_id") == "bahdanau-attention"
    assert wrapper.get("year") == 2015
    assert str(wrapper.get("url", "")).startswith("https://")


def test_cli_models_info_includes_wrapper_paper_metadata_for_rnnzoo_ln_variant() -> None:
    out = _run_cli("models", "info", "torch-rnnzoo-gru-ln-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnzoo-gru-ln-direct"

    wrapper = payload.get("wrapper_paper")
    assert isinstance(wrapper, dict)
    assert wrapper.get("paper_id") == "layer-normalization"
    assert wrapper.get("year") == 2016
    assert str(wrapper.get("url", "")).startswith("https://")


def test_cli_models_info_includes_paper_metadata_for_rnnzoo_janet() -> None:
    out = _run_cli("models", "info", "torch-rnnzoo-janet-direct")
    payload = json.loads(out)
    assert payload["key"] == "torch-rnnzoo-janet-direct"
    paper = payload.get("paper")
    assert isinstance(paper, dict)
    assert paper.get("paper_id") == "janet"
    assert paper.get("year") == 2018
    assert str(paper.get("url", "")).startswith("https://")

    assert "wrapper_paper" not in payload


def test_cli_models_list_json_includes_paper_metadata_rows() -> None:
    out = _run_cli(
        "models",
        "list",
        "--format",
        "json",
        "--prefix",
        "torch-rnnpaper-elman-srn",
    )
    rows = json.loads(out)
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert row["key"] == "torch-rnnpaper-elman-srn-direct"
    assert isinstance(row.get("paper"), dict)


def test_project_metadata_points_to_published_docs_site() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert 'Documentation = "https://skygazer42.github.io/ForeSight/"' in pyproject
    assert "https://skygazer42.github.io/ForeSight/" in readme
