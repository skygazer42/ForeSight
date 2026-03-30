from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from foresight.cli_leaderboard import (
    _build_leaderboard_metric_contexts,
    _leaderboard_summary_best_by_dataset_metric,
    _leaderboard_summary_rank_by_dataset_metric_model,
    _summarize_leaderboard_rows,
)


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
    assert mean_row["dataset_coverage"] == pytest.approx(1.0)
    assert mean_row["mae_mean"] == pytest.approx(1.0)
    # weighted mean over n_points: (0.5*10 + 1.5*30) / 40 = 1.25
    assert abs(float(mean_row["mae_wmean"]) - 1.25) < 1e-12
    assert mean_row["mae_rank_mean"] == pytest.approx(1.0)
    assert mean_row["mae_rank_wmean"] == pytest.approx(1.0)
    assert mean_row["mae_rel_mean"] == pytest.approx(1.0)
    assert mean_row["mae_rel_wmean"] == pytest.approx(1.0)

    naive_row = next(r for r in payload if r["model"] == "naive-last")
    assert naive_row["mae_rank_mean"] == pytest.approx(2.0)
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


def test_leaderboard_summarize_treats_near_zero_best_metric_as_zero(tmp_path: Path) -> None:
    rows = [
        {
            "model": "best",
            "dataset": "d1",
            "mae": 1e-15,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "other",
            "dataset": "d1",
            "mae": 1e-6,
            "rmse": 2.0,
            "mape": 0.2,
            "smape": 0.4,
            "n_points": 10,
        },
    ]
    inp = tmp_path / "sweep.json"
    inp.write_text(json.dumps(rows), encoding="utf-8")

    proc = _run_cli("leaderboard", "summarize", "--input", str(inp), "--format", "json")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)

    best_row = next(r for r in payload if r["model"] == "best")
    other_row = next(r for r in payload if r["model"] == "other")

    assert abs(float(best_row["mae_rel_mean"]) - 1.0) < 1e-12
    assert abs(float(best_row["mae_rel_wmean"]) - 1.0) < 1e-12
    assert math.isinf(float(other_row["mae_rel_mean"]))
    assert math.isinf(float(other_row["mae_rel_wmean"]))


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


def test_leaderboard_summarize_breaks_primary_sort_ties_with_mae_mean() -> None:
    rows = [
        {
            "model": "b",
            "dataset": "d1",
            "mae": 2.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "a",
            "dataset": "d2",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
    ]

    summary = _summarize_leaderboard_rows(rows, sort="n_datasets", limit=0)

    assert [row["model"] for row in summary] == ["a", "b"]


def test_leaderboard_summarize_separates_task_groups_and_ignores_skips() -> None:
    rows = [
        {
            "model": "shared-model",
            "dataset": "d1",
            "task_group": "point",
            "status": "ok",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "shared-model",
            "dataset": "d2",
            "task_group": "point",
            "status": "ok",
            "mae": 2.0,
            "rmse": 2.0,
            "mape": 0.2,
            "smape": 0.4,
            "n_points": 10,
        },
        {
            "model": "shared-model",
            "dataset": "d1",
            "task_group": "probabilistic",
            "status": "ok",
            "mae": 10.0,
            "rmse": 10.0,
            "mape": 1.0,
            "smape": 1.5,
            "n_points": 10,
        },
        {
            "model": "shared-model",
            "dataset": "d2",
            "task_group": "probabilistic",
            "status": "skip",
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "n_points": 0,
        },
    ]

    summary = _summarize_leaderboard_rows(rows, sort="task_group", limit=0)

    assert [(row["model"], row["task_group"]) for row in summary] == [
        ("shared-model", "point"),
        ("shared-model", "probabilistic"),
    ]
    point_row = next(row for row in summary if row["task_group"] == "point")
    probabilistic_row = next(row for row in summary if row["task_group"] == "probabilistic")

    assert point_row["n_datasets"] == 2
    assert point_row["mae_mean"] == pytest.approx(1.5)
    assert probabilistic_row["n_datasets"] == 1
    assert probabilistic_row["mae_mean"] == pytest.approx(10.0)


def test_leaderboard_summarize_filters_to_requested_task_group(tmp_path: Path) -> None:
    rows = [
        {
            "model": "shared-model",
            "dataset": "d1",
            "task_group": "point",
            "status": "ok",
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 0.1,
            "smape": 0.2,
            "n_points": 10,
        },
        {
            "model": "shared-model",
            "dataset": "d1",
            "task_group": "probabilistic",
            "status": "ok",
            "mae": 10.0,
            "rmse": 10.0,
            "mape": 1.0,
            "smape": 1.5,
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
        "--task-group",
        "probabilistic",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert [(row["model"], row["task_group"]) for row in payload] == [
        ("shared-model", "probabilistic")
    ]


def test_leaderboard_summary_source_avoids_nested_sort_conditionals() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "foresight" / "cli_leaderboard.py").read_text(encoding="utf-8")

    assert "sv = _num(row.get(secondary)) if secondary else None" not in source
    assert "0.0 if sv is None else (float(-sv) if descending else float(sv))" not in source


def test_leaderboard_summary_best_by_dataset_metric_accepts_pre_grouped_rows() -> None:
    cleaned = [
        {"task_group": "point", "dataset": "d1", "model": "a", "mae": 2.0, "rmse": 4.0},
        {"task_group": "point", "dataset": "d1", "model": "b", "mae": 1.0, "rmse": 3.0},
        {"task_group": "point", "dataset": "d2", "model": "a", "mae": 5.0, "rmse": 6.0},
    ]
    rows_by_dataset = {
        ("point", "d1"): cleaned[:2],
        ("point", "d2"): cleaned[2:],
    }

    out = _leaderboard_summary_best_by_dataset_metric(
        cleaned,
        ["mae", "rmse"],
        rows_by_dataset=rows_by_dataset,
    )

    assert out == {
        ("point", "d1", "mae"): 1.0,
        ("point", "d1", "rmse"): 3.0,
        ("point", "d2", "mae"): 5.0,
        ("point", "d2", "rmse"): 6.0,
    }


def test_leaderboard_summary_rank_by_dataset_metric_model_accepts_pre_grouped_rows() -> None:
    cleaned = [
        {"task_group": "point", "dataset": "d1", "model": "a", "mae": 2.0, "rmse": 4.0},
        {"task_group": "point", "dataset": "d1", "model": "b", "mae": 1.0, "rmse": 3.0},
        {"task_group": "point", "dataset": "d2", "model": "a", "mae": 5.0, "rmse": 6.0},
    ]
    rows_by_dataset = {
        ("point", "d1"): cleaned[:2],
        ("point", "d2"): cleaned[2:],
    }

    out = _leaderboard_summary_rank_by_dataset_metric_model(
        cleaned,
        ["mae", "rmse"],
        rows_by_dataset=rows_by_dataset,
    )

    assert out == {
        ("point", "d1", "mae", "b"): 1.0,
        ("point", "d1", "mae", "a"): 2.0,
        ("point", "d1", "rmse", "b"): 1.0,
        ("point", "d1", "rmse", "a"): 2.0,
        ("point", "d2", "mae", "a"): 1.0,
        ("point", "d2", "rmse", "a"): 1.0,
    }


def test_build_leaderboard_metric_contexts_collects_metric_views() -> None:
    items = [
        {"task_group": "point", "dataset": "d1", "model": "a", "mae": 2.0, "rmse": 4.0, "n_points": 10},
        {"task_group": "point", "dataset": "d2", "model": "a", "mae": 1.0, "rmse": 3.0, "n_points": 20},
    ]
    best_by_dataset_metric = {
        ("point", "d1", "mae"): 1.0,
        ("point", "d1", "rmse"): 4.0,
        ("point", "d2", "mae"): 1.0,
        ("point", "d2", "rmse"): 2.0,
    }
    rank_by_dataset_metric_model = {
        ("point", "d1", "mae", "a"): 2.0,
        ("point", "d1", "rmse", "a"): 1.0,
        ("point", "d2", "mae", "a"): 1.0,
        ("point", "d2", "rmse", "a"): 2.0,
    }

    out = _build_leaderboard_metric_contexts(
        items,
        metrics=["mae", "rmse"],
        model="a",
        best_by_dataset_metric=best_by_dataset_metric,
        rank_by_dataset_metric_model=rank_by_dataset_metric_model,
    )

    assert out["mae"]["values"] == [2.0, 1.0]
    assert out["mae"]["weighted_pairs"] == [(2.0, 10), (1.0, 20)]
    assert out["mae"]["relative_values"] == [2.0, 1.0]
    assert out["mae"]["rank_values"] == [2.0, 1.0]
    assert out["rmse"]["values"] == [4.0, 3.0]
    assert out["rmse"]["relative_pairs"] == [(1.0, 10), (1.5, 20)]
