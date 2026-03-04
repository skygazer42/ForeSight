from __future__ import annotations

import importlib.resources as resources


def test_packaged_rnn_paper_metadata_resource_exists() -> None:
    entry = resources.files("foresight.data") / "rnn_paper_metadata.json"
    assert entry.is_file()
    # A minimal sanity check that it's non-empty and JSON-like.
    raw = entry.read_text(encoding="utf-8")
    assert raw.strip().startswith("{")
    assert raw.strip().endswith("}")
