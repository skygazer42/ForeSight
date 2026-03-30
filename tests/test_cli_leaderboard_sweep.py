from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from foresight.cli_leaderboard import (
    _load_leaderboard_sweep_resume_state,
    _merge_leaderboard_sweep_rows,
)


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


def test_leaderboard_sweep_outputs_json_list(tmp_path: Path) -> None:
    out = tmp_path / "leaderboard_sweep.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last,mean",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert {r["dataset"] for r in payload} == {"catfish", "ice_cream_interest"}
    assert {r["model"] for r in payload} == {"naive-last", "mean"}
    assert {r["status"] for r in payload} == {"ok"}
    assert {r["task_group"] for r in payload} == {"point"}
    assert {r["backend_family"] for r in payload} == {"core"}
    assert out.exists()


def test_leaderboard_sweep_outputs_csv(tmp_path: Path) -> None:
    out = tmp_path / "leaderboard_sweep.csv"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last",
        "--format",
        "csv",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    first_line = proc.stdout.splitlines()[0]
    assert first_line.startswith("model,")
    assert "catfish" in proc.stdout
    assert out.exists()


def test_leaderboard_sweep_parallel_process_backend(tmp_path: Path) -> None:
    out = tmp_path / "leaderboard_sweep_parallel.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last",
        "--jobs",
        "2",
        "--backend",
        "process",
        "--progress",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert {r["dataset"] for r in payload} == {"catfish", "ice_cream_interest"}
    assert {r["model"] for r in payload} == {"naive-last"}
    # progress is printed to stderr (do not break JSON parsing)
    assert "DONE" in (proc.stderr or "")
    assert out.exists()


def test_leaderboard_sweep_chunk_size_zero_groups_by_dataset(tmp_path: Path) -> None:
    out = tmp_path / "leaderboard_sweep_chunk0.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last,mean",
        "--chunk-size",
        "0",
        "--jobs",
        "2",
        "--backend",
        "process",
        "--progress",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert {r["dataset"] for r in payload} == {"catfish", "ice_cream_interest"}
    assert {r["model"] for r in payload} == {"naive-last", "mean"}
    assert "DONE" in (proc.stderr or "")
    assert out.exists()


def test_leaderboard_sweep_emits_phase_timing_logs(tmp_path: Path) -> None:
    out = tmp_path / "leaderboard_sweep_logs.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last",
        "--jobs",
        "1",
        "--log-style",
        "plain",
        "--output",
        str(out),
    )
    assert proc.returncode == 0
    stderr = proc.stderr
    assert "RUN start" in stderr
    assert "PHASE params" in stderr
    assert "PHASE resume" in stderr
    assert "PHASE evaluate" in stderr
    assert "PHASE emit" in stderr


def test_leaderboard_sweep_strict_fails_on_unknown_model() -> None:
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "definitely-not-a-model",
        "--strict",
    )
    assert proc.returncode != 0
    assert "definitely-not-a-model" in (proc.stderr or "")


def test_leaderboard_sweep_model_param_passed_through_with_strict() -> None:
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "moving-average",
        "--model-param",
        "window=0",
        "--strict",
    )
    assert proc.returncode != 0
    assert "window must be >= 1" in (proc.stderr or "")


def test_leaderboard_sweep_resume_skips_existing_pairs(tmp_path: Path) -> None:
    resume = tmp_path / "resume.json"
    # A sentinel row that should be carried through unchanged (and skipped).
    resume.write_text(
        json.dumps(
            [
                {
                    "dataset": "catfish",
                    "model": "naive-last",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "n_points": 1,
                    "mae": 123.456,
                    "rmse": 123.456,
                    "mape": 0.0,
                    "smape": 0.0,
                }
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last,mean",
        "--resume",
        str(resume),
        "--progress",
        "--jobs",
        "1",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    pairs = {(r["dataset"], r["model"]) for r in payload}
    assert pairs == {
        ("catfish", "naive-last"),
        ("catfish", "mean"),
        ("ice_cream_interest", "naive-last"),
        ("ice_cream_interest", "mean"),
    }

    carried = next(r for r in payload if r["dataset"] == "catfish" and r["model"] == "naive-last")
    assert abs(float(carried["mae"]) - 123.456) < 1e-12

    # The skipped task should not appear in the DONE progress lines.
    assert "catfish/naive-last" not in (proc.stderr or "")


def test_merge_leaderboard_sweep_rows_deduplicates_pairs_with_last_value() -> None:
    merged = _merge_leaderboard_sweep_rows(
        resume_rows=[
            {"dataset": "catfish", "model": "naive-last", "mae": 10.0},
            {"dataset": "catfish", "model": "naive-last", "mae": 20.0},
            {"dataset": "ice_cream_interest", "model": "mean", "mae": 30.0},
        ],
        rows=[
            {"dataset": "catfish", "model": "mean", "mae": 40.0},
            {"dataset": "catfish", "model": "naive-last", "mae": 50.0},
        ],
    )

    assert [(row["dataset"], row["model"], float(row["mae"])) for row in merged] == [
        ("catfish", "mean", 40.0),
        ("catfish", "naive-last", 50.0),
        ("ice_cream_interest", "mean", 30.0),
    ]


def test_load_leaderboard_sweep_resume_state_deduplicates_matching_pairs(tmp_path: Path) -> None:
    resume = tmp_path / "resume.json"
    resume.write_text(
        json.dumps(
            [
                {
                    "dataset": "catfish",
                    "model": "naive-last",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "data_dir": "",
                    "model_params": {},
                    "mae": 10.0,
                },
                {
                    "dataset": "catfish",
                    "model": "naive-last",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "data_dir": "",
                    "model_params": {},
                    "mae": 20.0,
                },
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    resume_rows, done_pairs = _load_leaderboard_sweep_resume_state(
        resume_path=str(resume),
        datasets=["catfish"],
        keys=["naive-last"],
        y_col="Total",
        horizon=3,
        step=3,
        min_train_size=12,
        max_windows=None,
        data_dir="",
        model_params={},
        progress=False,
    )

    assert done_pairs == {("catfish", "naive-last")}
    assert len(resume_rows) == 1
    assert float(resume_rows[0]["mae"]) == 20.0


def test_leaderboard_sweep_writes_summary_output(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--max-windows",
        "2",
        "--models",
        "naive-last,mean",
        "--summary-output",
        str(summary),
        "--summary-format",
        "json",
        "--jobs",
        "1",
    )
    assert proc.returncode == 0
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert {r["model"] for r in payload} == {"naive-last", "mean"}
    assert all("mae_mean" in r for r in payload)


def test_leaderboard_sweep_writes_failures_output(tmp_path: Path) -> None:
    failures = tmp_path / "failures.txt"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "definitely-not-a-model",
        "--failures-output",
        str(failures),
        "--jobs",
        "1",
    )
    assert proc.returncode == 0
    assert failures.exists()
    text = failures.read_text(encoding="utf-8")
    assert "SKIP catfish/definitely-not-a-model" in text


def test_leaderboard_sweep_outputs_structured_skip_rows_and_summary_ignores_them(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last,definitely-not-a-model",
        "--summary-output",
        str(summary),
        "--summary-format",
        "json",
        "--jobs",
        "1",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert {(row["model"], row["status"]) for row in rows} == {
        ("naive-last", "ok"),
        ("definitely-not-a-model", "skip"),
    }

    skipped = next(row for row in rows if row["status"] == "skip")
    assert skipped["task_group"] == "point"
    assert skipped["backend_family"] == "unknown"
    assert skipped["error_type"] == "KeyError"
    assert "Unknown model key" in skipped["error_message"]
    assert skipped["skip_reason"] == "error"

    summary_rows = json.loads(summary.read_text(encoding="utf-8"))
    assert [(row["model"], row["task_group"]) for row in summary_rows] == [("naive-last", "point")]


def test_leaderboard_sweep_summary_min_datasets_filters_models(tmp_path: Path) -> None:
    resume = tmp_path / "resume.json"
    summary = tmp_path / "summary.json"

    # Full coverage for naive-last, partial for an invalid model key.
    resume.write_text(
        json.dumps(
            [
                {
                    "dataset": "catfish",
                    "model": "naive-last",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "n_points": 1,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mape": 0.1,
                    "smape": 0.2,
                },
                {
                    "dataset": "ice_cream_interest",
                    "model": "naive-last",
                    "y_col": "interest",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "n_points": 1,
                    "mae": 2.0,
                    "rmse": 2.0,
                    "mape": 0.2,
                    "smape": 0.4,
                },
                {
                    "dataset": "catfish",
                    "model": "definitely-not-a-model",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": None,
                    "n_points": 1,
                    "mae": 999.0,
                    "rmse": 999.0,
                    "mape": 9.9,
                    "smape": 9.9,
                },
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        "leaderboard",
        "sweep",
        "--datasets",
        "catfish,ice_cream_interest",
        "--horizon",
        "3",
        "--step",
        "3",
        "--min-train-size",
        "12",
        "--models",
        "naive-last,definitely-not-a-model",
        "--resume",
        str(resume),
        "--summary-output",
        str(summary),
        "--summary-format",
        "json",
        "--summary-min-datasets",
        "2",
        "--jobs",
        "1",
    )
    assert proc.returncode == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert {r["model"] for r in payload} == {"naive-last"}
