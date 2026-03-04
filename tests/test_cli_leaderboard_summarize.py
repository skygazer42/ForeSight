from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
    )


def test_leaderboard_summarize_from_json_file(tmp_path: Path) -> None:
    rows = [
        {
            "model": "naive-last",
            "dataset": "d1",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "naive-last",
            "dataset": "d2",
            "mae": 2.0,
            "rmse": 2.0,
            "mape": 0.2,
            "smape": 0.4,
            "n_points": 30,
        },
        {
            "model": "mean",
            "dataset": "d1",
            "mae": 0.5,
            "rmse": 0.5,
            "mape": 0.05,
            "smape": 0.1,
            "n_points": 10,
        },
        {
            "model": "mean",
            "dataset": "d2",
            "mae": 1.5,
            "rmse": 1.5,
            "mape": 0.15,
            "smape": 0.3,
            "n_points": 30,
        },
    ]
    inp = tmp_path / "sweep.json"
    inp.write_text(json.dumps(rows), encoding="utf-8")

    proc = _run_cli("leaderboard", "summarize", "--input", str(inp), "--format", "json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert {r["model"] for r in payload} == {"naive-last", "mean"}

    # default sort: mae_mean asc => mean should be first (avg 1.0) then naive-last (avg 1.5)
    assert payload[0]["model"] == "mean"
    assert payload[1]["model"] == "naive-last"

    mean_row = next(r for r in payload if r["model"] == "mean")
    assert mean_row["n_datasets"] == 2
    assert mean_row["n_datasets_total"] == 2
    assert mean_row["dataset_coverage"] == 1.0
    assert mean_row["mae_mean"] == 1.0
    # weighted mean over n_points: (0.5*10 + 1.5*30) / 40 = 1.25
    assert abs(float(mean_row["mae_wmean"]) - 1.25) < 1e-12
    assert mean_row["mae_rank_mean"] == 1.0
    assert mean_row["mae_rank_wmean"] == 1.0
    assert mean_row["mae_rel_mean"] == 1.0
    assert mean_row["mae_rel_wmean"] == 1.0

    naive_row = next(r for r in payload if r["model"] == "naive-last")
    assert naive_row["mae_rank_mean"] == 2.0
    assert abs(float(naive_row["mae_rel_mean"]) - (5.0 / 3.0)) < 1e-12
    assert abs(float(naive_row["mae_rel_wmean"]) - 1.5) < 1e-12


def test_leaderboard_summarize_from_stdin_csv() -> None:
    csv_text = (
        "model,dataset,mae,rmse,mape,smape,n_points\n"
        "naive-last,d1,1.0,1.0,0.1,0.2,10\n"
        "naive-last,d2,2.0,2.0,0.2,0.4,30\n"
    )
    proc = _run_cli(
        "leaderboard",
        "summarize",
        "--input",
        "-",
        "--input-format",
        "csv",
        "--format",
        "csv",
        stdin=csv_text,
    )
    assert proc.returncode == 0
    first_line = proc.stdout.splitlines()[0]
    assert first_line.startswith("model,n_datasets,")
    assert "naive-last" in proc.stdout


def test_leaderboard_summarize_min_datasets_filters_models(tmp_path: Path) -> None:
    rows = [
        {
            "model": "a",
            "dataset": "d1",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "a",
            "dataset": "d2",
            "mae": 2.0,
            "rmse": 2.0,
            "mape": 0.2,
            "smape": 0.4,
            "n_points": 10,
        },
        {
            "model": "b",
            "dataset": "d1",
            "mae": 3.0,
            "rmse": 3.0,
            "mape": 0.3,
            "smape": 0.6,
            "n_points": 10,
        },
    ]
    inp = tmp_path / "sweep.json"
    inp.write_text(json.dumps(rows), encoding="utf-8")

    proc = _run_cli(
        "leaderboard",
        "summarize",
        "--input",
        str(inp),
        "--format",
        "json",
        "--min-datasets",
        "2",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert [r["model"] for r in payload] == ["a"]


def test_leaderboard_summarize_desc_sort_keeps_missing_last(tmp_path: Path) -> None:
    rows = [
        {
            "model": "a",
            "dataset": "d1",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "b",
            "dataset": "d1",
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "n_points": 10,
        },
    ]
    inp = tmp_path / "sweep.json"
    inp.write_text(json.dumps(rows), encoding="utf-8")

    proc = _run_cli(
        "leaderboard",
        "summarize",
        "--input",
        str(inp),
        "--format",
        "json",
        "--sort=-mae_mean",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert [r["model"] for r in payload] == ["a", "b"]
