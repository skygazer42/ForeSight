from __future__ import annotations

from pathlib import Path


def _read_repo_file(path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).read_text(encoding="utf-8")


def test_adapter_examples_use_public_adapter_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expectations = {
        "examples/adapters_shared_bundle.py": ("to_beta_bundle", "from_beta_bundle"),
        "examples/adapters_sktime.py": ("make_sktime_forecaster_adapter",),
        "examples/adapters_darts.py": ("to_darts_bundle", "from_darts_bundle"),
        "examples/adapters_gluonts.py": ("to_gluonts_bundle", "from_gluonts_bundle"),
    }

    for rel_path, public_names in expectations.items():
        path = repo_root / rel_path
        assert path.is_file()
        source = path.read_text(encoding="utf-8")
        assert "from foresight.adapters import" in source
        assert "from foresight.adapters." not in source
        for public_name in public_names:
            assert public_name in source


def test_adapter_docs_cover_minimal_examples_and_install_paths() -> None:
    adapters_doc = _read_repo_file("docs/adapters.md")

    assert 'pip install "foresight-ts[sktime]"' in adapters_doc
    assert "bundle = to_beta_bundle(long_df)" in adapters_doc
    assert "restored = from_beta_bundle(bundle)" in adapters_doc
    assert "adapter = make_sktime_forecaster_adapter(" in adapters_doc
    assert "bundle = to_darts_bundle(long_df)" in adapters_doc
    assert "restored = from_darts_bundle(bundle)" in adapters_doc
    assert "bundle = to_gluonts_bundle(long_df)" in adapters_doc
    assert "restored = from_gluonts_bundle(bundle)" in adapters_doc
    assert "mapping-based schema keyed by `unique_id`" in adapters_doc
    assert "older single-series beta shape" not in adapters_doc


def test_installation_docs_list_adapter_extras() -> None:
    install_doc = _read_repo_file("docs/getting-started/installation.md")

    assert "| `sktime` | `pip install foresight-ts[sktime]` | sktime | >= 0.30 |" in install_doc
    assert "| `darts` | `pip install foresight-ts[darts]` | u8darts | >= 0.30 |" in install_doc
    assert "| `gluonts` | `pip install foresight-ts[gluonts]` | gluonts | >= 0.15 |" in install_doc
