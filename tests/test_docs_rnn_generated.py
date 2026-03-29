from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generate_rnn_docs_module(repo_root: Path):
    path = repo_root / "tools" / "generate_rnn_docs.py"
    spec = importlib.util.spec_from_file_location("generate_rnn_docs", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_generate_model_capability_docs_module(repo_root: Path):
    path = repo_root / "tools" / "generate_model_capability_docs.py"
    spec = importlib.util.spec_from_file_location("generate_model_capability_docs", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rnn_docs_are_up_to_date() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"

    mod = _load_generate_rnn_docs_module(repo_root)

    expected_paper = mod._render_rnn_paper_zoo_doc()  # type: ignore[attr-defined]
    expected_zoo = mod._render_rnn_zoo_doc()  # type: ignore[attr-defined]

    actual_paper = (docs_dir / "rnn_paper_zoo.md").read_text(encoding="utf-8")
    actual_zoo = (docs_dir / "rnn_zoo.md").read_text(encoding="utf-8")

    assert actual_paper == expected_paper
    assert actual_zoo == expected_zoo


def test_model_capability_docs_are_up_to_date() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"

    mod = _load_generate_model_capability_docs_module(repo_root)

    expected_models = mod._render_models_doc()  # type: ignore[attr-defined]
    expected_api = mod._render_api_doc()  # type: ignore[attr-defined]

    actual_models = (docs_dir / "models.md").read_text(encoding="utf-8")
    actual_api = (docs_dir / "api.md").read_text(encoding="utf-8")

    assert actual_models == expected_models
    assert "`supports_interval_forecast`" in actual_models
    assert "| stability |" in actual_models
    assert "`xgb-step-lag-global`" in actual_models

    assert actual_api == expected_api
    assert "`forecast_model`" in actual_api
    assert "`eval_model_long_df`" in actual_api
    assert "`load_forecaster_artifact`" in actual_api


def test_docs_site_navigation_includes_generated_pages() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    mkdocs = (repo_root / "mkdocs.yml").read_text(encoding="utf-8")
    index_doc = (repo_root / "docs" / "index.md").read_text(encoding="utf-8")

    assert "site_name: ForeSight" in mkdocs
    assert "site_url: https://skygazer42.github.io/ForeSight/" in mkdocs
    assert "Home: index.md" in mkdocs
    assert "Models: models.md" in mkdocs
    assert "API: api.md" in mkdocs
    assert "Compatibility: compatibility.md" in mkdocs

    assert "[Model capability matrix](models.md)" in index_doc
    assert "[Python API reference](api.md)" in index_doc
    assert "[Compatibility guide](compatibility.md)" in index_doc
