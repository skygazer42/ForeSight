from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import foresight.dataset_long_df_cache as cache_mod
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
    production = config["production_v1"]

    assert [row["key"] for row in smoke["datasets"]] == ["catfish", "ice_cream_interest"]
    assert {row["key"] for row in smoke["datasets"]}.issubset(set(list_packaged_datasets()))
    assert [row["key"] for row in smoke["models"]] == ["naive-last", "mean", "moving-average"]
    assert smoke["conformal_levels"] == [80]
    assert smoke["task_group"] == "point"
    assert smoke["workload"] == "panel_cv"
    assert smoke["scale"] == "tiny"
    assert production["task_group"] == "point"
    assert production["profiling"] is True
    assert production["workload"] == "panel_cv"
    assert production["scale"] == "small"
    assert production["budgets"]["cv_seconds_mean_warn"] > 0.0
    assert production["budgets"]["peak_memory_mb_warn"] > 0.0
    assert [row["key"] for row in production["models"]] == [
        "naive-last",
        "mean",
        "moving-average",
        "drift",
    ]


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
        "task_group",
        "backend_family",
        "n_datasets",
        "ok_rows",
        "skip_rows",
        "n_points_total",
        "mae_mean",
        "rmse_mean",
        "mape_mean",
        "smape_mean",
        "coverage_80_mean",
        "mean_width_80_mean",
        "interval_score_80_mean",
        "cv_seconds_total",
        "cv_seconds_mean",
    ]
    assert [row["model"] for row in rows] == ["mean", "moving-average", "naive-last"]
    assert {row["task_group"] for row in rows} == {"point"}
    assert {row["backend_family"] for row in rows} == {"core"}
    assert {int(row["n_datasets"]) for row in rows} == {2}
    assert {int(row["ok_rows"]) for row in rows} == {2}
    assert {int(row["skip_rows"]) for row in rows} == {0}
    assert all(int(row["n_points_total"]) > 0 for row in rows)
    assert all(float(row["coverage_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["mean_width_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["interval_score_80_mean"]) >= 0.0 for row in rows)
    assert all(float(row["cv_seconds_total"]) >= 0.0 for row in rows)
    assert all(float(row["cv_seconds_mean"]) >= 0.0 for row in rows)


def test_benchmark_smoke_runner_can_force_profile_columns() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "benchmarks/run_benchmarks.py", "--smoke", "--profile"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    rows = list(reader)

    assert reader.fieldnames is not None
    assert "load_seconds_total" in reader.fieldnames
    assert "prepare_seconds_total" in reader.fieldnames
    assert "eval_seconds_total" in reader.fieldnames
    assert "peak_memory_mb_max" in reader.fieldnames
    assert "points_per_second_mean" in reader.fieldnames
    assert "windows_per_second_mean" in reader.fieldnames
    assert rows


def test_production_benchmark_suite_reports_stage_profile_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_run_benchmarks_module(repo_root)

    payload = mod.run_benchmark_suite(config_name="production_v1")  # type: ignore[attr-defined]

    assert payload["profiling"] is True
    assert payload["task_group"] == "point"
    assert payload["summary"]
    assert all("load_seconds_total" in row for row in payload["summary"])
    assert all("prepare_seconds_total" in row for row in payload["summary"])
    assert all("eval_seconds_total" in row for row in payload["summary"])
    assert all("peak_memory_mb_max" in row for row in payload["summary"])
    assert all("points_per_second_mean" in row for row in payload["summary"])
    assert all("windows_per_second_mean" in row for row in payload["summary"])
    assert all(float(row["load_seconds_total"]) >= 0.0 for row in payload["summary"])
    assert all(float(row["prepare_seconds_total"]) >= 0.0 for row in payload["summary"])
    assert all(float(row["eval_seconds_total"]) >= 0.0 for row in payload["summary"])
    assert all(float(row["peak_memory_mb_max"]) >= 0.0 for row in payload["summary"])
    assert all(float(row["points_per_second_mean"]) >= 0.0 for row in payload["summary"])
    assert all(float(row["windows_per_second_mean"]) >= 0.0 for row in payload["summary"])
    assert payload["budgets"]["cv_seconds_mean_warn"] > 0.0
    assert payload["budgets"]["peak_memory_mb_warn"] > 0.0
    assert all("load_seconds" in row for row in payload["rows"])
    assert all("prepare_seconds" in row for row in payload["rows"])
    assert all("eval_seconds" in row for row in payload["rows"])
    assert all("peak_memory_mb" in row for row in payload["rows"])
    assert all("points_per_second" in row for row in payload["rows"])
    assert all("windows_per_second" in row for row in payload["rows"])
    assert all("n_windows" in row for row in payload["rows"])


def test_benchmark_summary_context_collects_ok_rows_and_profile_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_run_benchmarks_module(repo_root)

    context = mod._build_benchmark_summary_context(  # type: ignore[attr-defined]
        [
            {
                "dataset": "d1",
                "status": "ok",
                "n_points": 10,
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 0.1,
                "smape": 0.2,
                "cv_seconds": 3.0,
                "load_seconds": 0.5,
                "prepare_seconds": 0.25,
                "eval_seconds": 2.25,
                "peak_memory_mb": 12.0,
                "points_per_second": 5.0,
                "windows_per_second": 2.0,
                "coverage_80": 0.9,
                "mean_width_80": 1.5,
                "interval_score_80": 2.5,
            },
            {
                "dataset": "d2",
                "status": "skip",
                "n_points": 0,
                "cv_seconds": 1.0,
                "load_seconds": 0.0,
                "prepare_seconds": 0.0,
                "eval_seconds": 0.0,
                "peak_memory_mb": 1.0,
                "points_per_second": 0.0,
                "windows_per_second": 0.0,
            },
            {
                "dataset": "d3",
                "status": "ok",
                "n_points": 30,
                "mae": 3.0,
                "rmse": 4.0,
                "mape": 0.3,
                "smape": 0.4,
                "cv_seconds": 5.0,
                "load_seconds": 0.75,
                "prepare_seconds": 0.5,
                "eval_seconds": 3.75,
                "peak_memory_mb": 20.0,
                "points_per_second": 10.0,
                "windows_per_second": 4.0,
                "coverage_80": 0.7,
                "mean_width_80": 2.5,
                "interval_score_80": 3.5,
            },
        ],
        conformal_levels=[80],
        include_profile=True,
    )

    assert context["n_datasets"] == 3
    assert context["ok_rows"] == 2
    assert context["skip_rows"] == 1
    assert context["n_points_total"] == 40
    assert context["mae_mean"] == pytest.approx(2.0)
    assert context["rmse_mean"] == pytest.approx(3.0)
    assert context["mape_mean"] == pytest.approx(0.2)
    assert context["smape_mean"] == pytest.approx(0.3)
    assert context["cv_seconds_total"] == pytest.approx(9.0)
    assert context["cv_seconds_mean"] == pytest.approx(3.0)
    assert context["load_seconds_total"] == pytest.approx(1.25)
    assert context["prepare_seconds_total"] == pytest.approx(0.75)
    assert context["eval_seconds_total"] == pytest.approx(6.0)
    assert context["peak_memory_mb_max"] == pytest.approx(20.0)
    assert context["points_per_second_mean"] == pytest.approx(5.0)
    assert context["windows_per_second_mean"] == pytest.approx(2.0)
    assert context["coverage_80_mean"] == pytest.approx(0.8)
    assert context["mean_width_80_mean"] == pytest.approx(2.0)
    assert context["interval_score_80_mean"] == pytest.approx(3.0)


def test_benchmark_config_validation_rejects_unknown_task_group() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_run_benchmarks_module(repo_root)

    bad_config = {
        "bad": {
            "description": "invalid",
            "task_group": "definitely-not-valid",
            "profiling": False,
            "datasets": [
                {
                    "key": "catfish",
                    "y_col": "Total",
                    "horizon": 3,
                    "step": 3,
                    "min_train_size": 12,
                    "max_windows": 3,
                }
            ],
            "models": [{"key": "naive-last", "params": {}}],
        }
    }

    try:
        mod._validate_benchmark_config("bad", bad_config["bad"])  # type: ignore[attr-defined]
    except ValueError as exc:  # pragma: no branch
        assert "task_group" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid task_group to raise ValueError")


def test_dataset_long_df_cache_reuses_loaded_data_and_respects_covariate_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Spec:
        time_col = "ds"
        group_cols: tuple[str, ...] = ()
        default_y = "target"

    counts = {"load": 0, "prepare": 0}

    def _fake_get_dataset_spec(key: str) -> Any:
        assert key == "demo"
        return _Spec()

    def _fake_load_dataset(key: str, data_dir: str | None = None) -> pd.DataFrame:
        assert key == "demo"
        assert data_dir is None
        counts["load"] += 1
        return pd.DataFrame(
            {
                "ds": [1],
                "target": [1.0],
                "promo": [10.0],
                "stock": [20.0],
                "store_size": [30.0],
            }
        )

    def _fake_to_long(
        df: pd.DataFrame,
        *,
        time_col: str,
        y_col: str,
        id_cols: tuple[str, ...],
        historic_x_cols: tuple[str, ...],
        future_x_cols: tuple[str, ...],
        static_cols: tuple[str, ...],
        dropna: bool,
    ) -> pd.DataFrame:
        counts["prepare"] += 1
        out = pd.DataFrame(
            {
                "unique_id": ["series=0"],
                "ds": df[time_col].tolist(),
                "y": df[y_col].tolist(),
            }
        )
        out.attrs["historic_x_cols"] = tuple(historic_x_cols)
        out.attrs["future_x_cols"] = tuple(future_x_cols)
        out.attrs["static_cols"] = tuple(static_cols)
        out.attrs["dropna"] = bool(dropna)
        out.attrs["prepare_call"] = int(counts["prepare"])
        return out

    raw_cache_before = dict(cache_mod._DATASET_RAW_CACHE)
    prepared_cache_before = dict(cache_mod._DATASET_LONG_DF_CACHE)
    cache_mod._DATASET_RAW_CACHE.clear()
    cache_mod._DATASET_LONG_DF_CACHE.clear()
    monkeypatch.setattr(cache_mod, "get_dataset_spec", _fake_get_dataset_spec)
    monkeypatch.setattr(cache_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(cache_mod, "to_long", _fake_to_long)

    try:
        first = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
        second = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
        third = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={
                "historic_x_cols": ("stock",),
                "future_x_cols": ("promo",),
                "static_cols": ("store_size",),
            },
        )
        fourth = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={
                "historic_x_cols": ("stock",),
                "future_x_cols": ("promo",),
                "static_cols": ("store_size",),
            },
        )
    finally:
        cache_mod._DATASET_RAW_CACHE.clear()
        cache_mod._DATASET_RAW_CACHE.update(raw_cache_before)
        cache_mod._DATASET_LONG_DF_CACHE.clear()
        cache_mod._DATASET_LONG_DF_CACHE.update(prepared_cache_before)

    assert counts == {"load": 1, "prepare": 2}
    assert first["long_df"] is second["long_df"]
    assert third["long_df"] is fourth["long_df"]
    assert first["long_df"] is not third["long_df"]
    assert first["long_df"].attrs["historic_x_cols"] == ()
    assert first["long_df"].attrs["future_x_cols"] == ()
    assert first["long_df"].attrs["static_cols"] == ()
    assert third["long_df"].attrs["historic_x_cols"] == ("stock",)
    assert third["long_df"].attrs["future_x_cols"] == ("promo",)
    assert third["long_df"].attrs["static_cols"] == ("store_size",)
    assert first["load_seconds"] >= 0.0
    assert first["prepare_seconds"] >= 0.0
    assert second["load_seconds"] == 0.0
    assert second["prepare_seconds"] == 0.0
    assert third["load_seconds"] == 0.0
    assert third["prepare_seconds"] >= 0.0


def test_dataset_long_df_cache_reuses_dataset_spec_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Spec:
        time_col = "ds"
        group_cols: tuple[str, ...] = ()
        default_y = "target"

    counts = {"spec": 0, "load": 0, "prepare": 0}

    def _fake_get_dataset_spec(key: str) -> Any:
        assert key == "demo"
        counts["spec"] += 1
        return _Spec()

    def _fake_load_dataset(key: str, data_dir: str | None = None) -> pd.DataFrame:
        assert key == "demo"
        assert data_dir is None
        counts["load"] += 1
        return pd.DataFrame({"ds": [1], "target": [1.0]})

    def _fake_to_long(
        df: pd.DataFrame,
        *,
        time_col: str,
        y_col: str,
        id_cols: tuple[str, ...],
        historic_x_cols: tuple[str, ...],
        future_x_cols: tuple[str, ...],
        static_cols: tuple[str, ...],
        dropna: bool,
    ) -> pd.DataFrame:
        counts["prepare"] += 1
        return pd.DataFrame(
            {
                "unique_id": ["series=0"],
                "ds": df[time_col].tolist(),
                "y": df[y_col].tolist(),
            }
        )

    raw_cache_before = dict(cache_mod._DATASET_RAW_CACHE)
    prepared_cache_before = dict(cache_mod._DATASET_LONG_DF_CACHE)
    spec_cache_before = dict(getattr(cache_mod, "_DATASET_SPEC_CACHE", {}))
    cache_mod._DATASET_RAW_CACHE.clear()
    cache_mod._DATASET_LONG_DF_CACHE.clear()
    if hasattr(cache_mod, "_DATASET_SPEC_CACHE"):
        cache_mod._DATASET_SPEC_CACHE.clear()
    monkeypatch.setattr(cache_mod, "get_dataset_spec", _fake_get_dataset_spec)
    monkeypatch.setattr(cache_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(cache_mod, "to_long", _fake_to_long)

    try:
        cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
        cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
        cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={"future_x_cols": ("promo",)},
        )
    finally:
        cache_mod._DATASET_RAW_CACHE.clear()
        cache_mod._DATASET_RAW_CACHE.update(raw_cache_before)
        cache_mod._DATASET_LONG_DF_CACHE.clear()
        cache_mod._DATASET_LONG_DF_CACHE.update(prepared_cache_before)
        if hasattr(cache_mod, "_DATASET_SPEC_CACHE"):
            cache_mod._DATASET_SPEC_CACHE.clear()
            cache_mod._DATASET_SPEC_CACHE.update(spec_cache_before)

    assert counts["load"] == 1
    assert counts["prepare"] == 2
    assert counts["spec"] == 1


def test_dataset_long_df_cache_prepared_hit_skips_raw_frame_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Spec:
        time_col = "ds"
        group_cols: tuple[str, ...] = ()
        default_y = "target"

    def _fake_get_dataset_spec(key: str) -> Any:
        assert key == "demo"
        return _Spec()

    def _fake_load_dataset(key: str, data_dir: str | None = None) -> pd.DataFrame:
        assert key == "demo"
        assert data_dir is None
        return pd.DataFrame({"ds": [1], "target": [1.0]})

    def _fake_to_long(
        df: pd.DataFrame,
        *,
        time_col: str,
        y_col: str,
        id_cols: tuple[str, ...],
        historic_x_cols: tuple[str, ...],
        future_x_cols: tuple[str, ...],
        static_cols: tuple[str, ...],
        dropna: bool,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": ["series=0"],
                "ds": df[time_col].tolist(),
                "y": df[y_col].tolist(),
            }
        )

    def _forbid_raw_frame_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("prepared cache hit should not call get_or_build_dataset_frame")

    raw_cache_before = dict(cache_mod._DATASET_RAW_CACHE)
    prepared_cache_before = dict(cache_mod._DATASET_LONG_DF_CACHE)
    spec_cache_before = dict(getattr(cache_mod, "_DATASET_SPEC_CACHE", {}))
    cache_mod._DATASET_RAW_CACHE.clear()
    cache_mod._DATASET_LONG_DF_CACHE.clear()
    if hasattr(cache_mod, "_DATASET_SPEC_CACHE"):
        cache_mod._DATASET_SPEC_CACHE.clear()
    monkeypatch.setattr(cache_mod, "get_dataset_spec", _fake_get_dataset_spec)
    monkeypatch.setattr(cache_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(cache_mod, "to_long", _fake_to_long)

    try:
        first = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
        monkeypatch.setattr(cache_mod, "get_or_build_dataset_frame", _forbid_raw_frame_lookup)
        second = cache_mod.get_or_build_dataset_long_df(
            dataset="demo",
            y_col="target",
            data_dir=None,
            model_params={},
        )
    finally:
        cache_mod._DATASET_RAW_CACHE.clear()
        cache_mod._DATASET_RAW_CACHE.update(raw_cache_before)
        cache_mod._DATASET_LONG_DF_CACHE.clear()
        cache_mod._DATASET_LONG_DF_CACHE.update(prepared_cache_before)
        if hasattr(cache_mod, "_DATASET_SPEC_CACHE"):
            cache_mod._DATASET_SPEC_CACHE.clear()
            cache_mod._DATASET_SPEC_CACHE.update(spec_cache_before)

    assert first["long_df"] is second["long_df"]
    assert second["load_seconds"] == 0.0
    assert second["prepare_seconds"] == 0.0
