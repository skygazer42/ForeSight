from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from foresight.datasets import list_packaged_datasets, load_dataset


def _load_run_benchmarks_module(repo_root: Path):
    path = repo_root / "benchmarks" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_benchmarks", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_packaged_benchmark_datasets_are_listed_and_loadable() -> None:
    keys = list_packaged_datasets()

    assert keys == ["catfish", "ice_cream_interest"]

    for key in keys:
        df = load_dataset(key, nrows=5)
        assert len(df) > 0


def test_smoke_benchmark_config_uses_packaged_datasets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_run_benchmarks_module(repo_root)

    config = mod._load_benchmark_config()  # type: ignore[attr-defined]
    smoke = config["smoke"]

    assert [row["key"] for row in smoke["datasets"]] == ["catfish", "ice_cream_interest"]
    assert {row["key"] for row in smoke["datasets"]}.issubset(set(list_packaged_datasets()))
    assert [row["key"] for row in smoke["models"]] == ["naive-last", "mean", "moving-average"]
    assert smoke["conformal_levels"] == [80]


def test_benchmark_smoke_runner_emits_deterministic_csv_summary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "benchmarks/run_benchmarks.py", "--smoke"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    rows = list(reader)

    assert reader.fieldnames == [
        "model",
        "n_datasets",
        "n_points_total",
        "mae_mean",
        "rmse_mean",
        "mape_mean",
        "smape_mean",
        "coverage_80_mean",
        "mean_width_80_mean",
        "interval_score_80_mean",
        "wall_clock_seconds_total",
        "wall_clock_seconds_mean",
    ]
    assert [row["model"] for row in rows] == ["mean", "moving-average", "naive-last"]
    assert {int(row["n_datasets"]) for row in rows} == {2}
    assert all(int(row["n_points_total"]) > 0 for row in rows)
    assert all(float(row["coverage_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["mean_width_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["interval_score_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["wall_clock_seconds_total"]) >= 0.0 for row in rows)
    assert all(float(row["wall_clock_seconds_mean"]) >= 0.0 for row in rows)
