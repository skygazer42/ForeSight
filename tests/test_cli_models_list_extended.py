from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_models_list_tsv_columns_and_header_for_rnnpaper() -> None:
    proc = _run_cli(
        "models",
        "list",
        "--prefix",
        "torch-rnnpaper-elman-srn",
        "--columns",
        "key,paper_id,paper_year",
        "--header",
    )
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split("\t") == ["key", "paper_id", "paper_year"]
    assert lines[1].split("\t") == ["torch-rnnpaper-elman-srn-direct", "elman-srn", "1990"]


def test_models_list_json_sort_and_limit() -> None:
    proc = _run_cli(
        "models",
        "list",
        "--format",
        "json",
        "--sort",
        "key",
        "--desc",
        "--limit",
        "10",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert isinstance(rows, list)
    assert len(rows) == 10
    keys = [r["key"] for r in rows]
    assert keys == sorted(keys, reverse=True)


def test_models_list_filter_requires_torch() -> None:
    proc = _run_cli("models", "list", "--format", "json", "--requires", "torch", "--limit", "25")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert isinstance(rows, list)
    assert len(rows) == 25
    assert all("torch" in str(r.get("requires", "")) for r in rows if isinstance(r, dict))


def test_models_list_filter_interface_global() -> None:
    proc = _run_cli("models", "list", "--format", "json", "--interface", "global", "--limit", "20")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert isinstance(rows, list)
    assert rows
    assert all(r.get("interface") == "global" for r in rows if isinstance(r, dict))


def test_models_list_json_includes_capabilities() -> None:
    proc = _run_cli("models", "list", "--format", "json", "--prefix", "xgb-step-lag-global", "--limit", "1")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows[0]["key"] == "xgb-step-lag-global"
    assert rows[0]["capabilities"]["supports_x_cols"] is True
    assert rows[0]["capabilities"]["supports_quantiles"] is True
    assert rows[0]["capabilities"]["supports_interval_forecast"] is True
    assert rows[0]["capabilities"]["supports_interval_forecast_with_x_cols"] is True


def test_models_list_filter_interface_multivariate() -> None:
    proc = _run_cli(
        "models",
        "list",
        "--format",
        "json",
        "--interface",
        "multivariate",
        "--limit",
        "5",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows
    assert all(r.get("interface") == "multivariate" for r in rows if isinstance(r, dict))


def test_models_list_can_filter_by_capability() -> None:
    proc = _run_cli(
        "models",
        "list",
        "--format",
        "json",
        "--capability",
        "supports_x_cols=true",
        "--limit",
        "20",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows
    assert all(rows[i]["capabilities"]["supports_x_cols"] is True for i in range(len(rows)))


def test_models_list_can_filter_by_stability() -> None:
    proc = _run_cli(
        "models",
        "list",
        "--format",
        "json",
        "--prefix",
        "torch-rnnpaper-elman-srn",
        "--stability",
        "experimental",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows
    assert rows[0]["key"] == "torch-rnnpaper-elman-srn-direct"
    assert rows[0]["required_extra"] == "torch"
    assert rows[0]["stability"] == "experimental"


def test_models_search_accepts_stability_and_capability_filters() -> None:
    proc = _run_cli(
        "models",
        "search",
        "elman",
        "--format",
        "json",
        "--stability",
        "experimental",
        "--capability",
        "supports_artifact_save=true",
        "--limit",
        "10",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows
    assert rows[0]["key"] == "torch-rnnpaper-elman-srn-direct"


def test_models_search_filter_interface_multivariate() -> None:
    proc = _run_cli(
        "models",
        "search",
        "graph",
        "--format",
        "json",
        "--interface",
        "multivariate",
        "--limit",
        "10",
    )
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows
    assert any("multivariate" in str(row["key"]) for row in rows)
