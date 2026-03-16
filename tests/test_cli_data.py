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


def test_cli_data_to_long_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "to-long",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert rows[0]["unique_id"] == "series=0"
    assert rows[0]["ds"].startswith("2020-01-01")


def test_cli_data_prepare_long_inserts_missing_ds(tmp_path: Path) -> None:
    csv_path = tmp_path / "long.csv"
    csv_path.write_text(
        "unique_id,ds,y\ns0,2020-01-01,10\ns0,2020-01-03,12\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "prepare-long",
        "--path",
        str(csv_path),
        "--parse-dates",
        "--freq",
        "D",
        "--y-missing",
        "zero",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert [r["ds"][:10] for r in rows] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert [float(r["y"]) for r in rows] == [10.0, 0.0, 12.0]


def test_cli_data_align_long_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "long.csv"
    csv_path.write_text(
        "unique_id,ds,y\ns0,2020-01-01,1\ns0,2020-01-01,2\ns0,2020-01-03,4\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "align-long",
        "--path",
        str(csv_path),
        "--parse-dates",
        "--freq",
        "D",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert [row["ds"][:10] for row in rows] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert float(rows[0]["y"]) == 2.0


def test_cli_data_clip_outliers_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "long.csv"
    csv_path.write_text(
        "unique_id,ds,y\ns0,2020-01-01,1\ns0,2020-01-02,2\ns0,2020-01-03,3\ns0,2020-01-04,100\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "clip-outliers",
        "--path",
        str(csv_path),
        "--parse-dates",
        "--method",
        "iqr",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert float(rows[-1]["y"]) < 100.0


def test_cli_data_calendar_features_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "long.csv"
    csv_path.write_text(
        "unique_id,ds,y\ns0,2020-01-01,1\ns0,2020-01-02,2\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "calendar-features",
        "--path",
        str(csv_path),
        "--parse-dates",
        "--prefix",
        "cal_",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert "cal_time_idx" in rows[0]
    assert "cal_dow_sin" in rows[0]


def test_cli_data_make_supervised_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "long.csv"
    csv_path.write_text(
        "unique_id,ds,y,promo\n"
        "s0,2020-01-01,1,10\n"
        "s0,2020-01-02,2,11\n"
        "s0,2020-01-03,3,12\n"
        "s0,2020-01-04,4,13\n"
        "s0,2020-01-05,5,14\n"
        "s0,2020-01-06,6,15\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "make-supervised",
        "--path",
        str(csv_path),
        "--parse-dates",
        "--lags",
        "3",
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert "feat_y_lag3" in rows[0]
    assert "feat_x_promo" in rows[0]
    assert rows[0]["y_t+1"] == 4.0
    assert rows[0]["y_t+2"] == 5.0


def test_cli_data_infer_freq_daily(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "data",
        "infer-freq",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 1
    assert rows[0]["unique_id"] == "series=0"
    assert rows[0]["freq"] == "D"


def test_cli_data_splits_rolling_origin_indices() -> None:
    proc = _run_cli(
        "data",
        "splits",
        "rolling-origin",
        "--n-obs",
        "10",
        "--horizon",
        "3",
        "--min-train-size",
        "4",
        "--step-size",
        "2",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert rows[0] == {
        "window": 0,
        "train_start": 0,
        "train_end": 4,
        "test_start": 4,
        "test_end": 7,
    }
