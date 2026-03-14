import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_forecast_csv_outputs_future_predictions(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert rows[0]["unique_id"] == "series=0"
    assert rows[0]["step"] == 1
    assert rows[0]["model"] == "naive-last"
    assert rows[0]["ds"].startswith("2020-01-06")
    assert rows[1]["ds"].startswith("2020-01-07")
    assert "mae" not in rows[0]


def test_forecast_csv_can_emit_interval_quantile_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--interval-levels",
        "80,90",
        "--interval-min-train-size",
        "3",
        "--interval-samples",
        "64",
        "--interval-seed",
        "0",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert rows[0]["yhat"] == pytest.approx(5.0)
    assert rows[0]["yhat_lo_80"] == pytest.approx(6.0)
    assert rows[0]["yhat_hi_80"] == pytest.approx(6.0)
    assert rows[0]["yhat_lo_90"] == pytest.approx(6.0)
    assert rows[0]["yhat_hi_90"] == pytest.approx(6.0)


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_supports_arima_trend_parameter(tmp_path: Path) -> None:
    csv_path = tmp_path / "trend.csv"
    csv_path.write_text(
        "ds,y\n" + "\n".join([f"2020-01-{i:02d},{i}" for i in range(1, 31)]),
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "arima",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--model-param",
        "order=0,1,0",
        "--model-param",
        "trend=t",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert [row["ds"][:10] for row in rows] == ["2020-01-31", "2020-02-01", "2020-02-02"]
    assert [round(float(row["yhat"]), 3) for row in rows] == [31.0, 32.0, 33.0]


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_supports_local_sarimax_with_future_covariates(tmp_path: Path) -> None:
    csv_path = tmp_path / "sarimax_exog.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p) + 0.1 * float(i - 1)
        rows.append(f"2020-01-{i:02d},{y},{p}")
    rows.extend(
        [
            "2020-01-31,,1",
            "2020-02-01,,0",
            "2020-02-02,,1",
        ]
    )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "sarimax",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--model-param",
        "order=0,0,0",
        "--model-param",
        "seasonal_order=0,0,0,0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert [row["ds"][:10] for row in payload] == ["2020-01-31", "2020-02-01", "2020-02-02"]
    yhat = [float(row["yhat"]) for row in payload]
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0
    assert abs(yhat[0] - yhat[2]) < 1e-6


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_supports_separate_future_csv_for_local_covariates(tmp_path: Path) -> None:
    history_path = tmp_path / "sarimax_hist.csv"
    future_path = tmp_path / "sarimax_future.csv"

    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p) + 0.1 * float(i - 1)
        rows.append(f"2020-01-{i:02d},{y},{p}")
    history_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    future_path.write_text(
        "ds,promo\n2020-01-31,1\n2020-02-01,0\n2020-02-02,1\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "sarimax",
        "--path",
        str(history_path),
        "--future-path",
        str(future_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--model-param",
        "order=0,0,0",
        "--model-param",
        "seasonal_order=0,0,0,0",
        "--model-param",
        "trend=c",
        "--model-param",
        "future_x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert [row["ds"][:10] for row in payload] == ["2020-01-31", "2020-02-01", "2020-02-02"]
    yhat = [float(row["yhat"]) for row in payload]
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_supports_local_auto_arima_with_future_covariates(tmp_path: Path) -> None:
    csv_path = tmp_path / "auto_arima_exog.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p)
        rows.append(f"2020-01-{i:02d},{y},{p}")
    rows.extend(
        [
            "2020-01-31,,1",
            "2020-02-01,,0",
            "2020-02-02,,1",
        ]
    )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "auto-arima",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--model-param",
        "max_p=0",
        "--model-param",
        "max_d=0",
        "--model-param",
        "max_q=0",
        "--model-param",
        "max_P=0",
        "--model-param",
        "max_D=0",
        "--model-param",
        "max_Q=0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert [row["ds"][:10] for row in payload] == ["2020-01-31", "2020-02-01", "2020-02-02"]
    yhat = [float(row["yhat"]) for row in payload]
    assert yhat[0] > yhat[1] + 4.0
    assert yhat[2] > yhat[1] + 4.0
    assert abs(yhat[0] - yhat[2]) < 1e-3


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_supports_local_sarimax_with_future_covariates_and_intervals(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sarimax_exog_intervals.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p) + 0.1 * float(i - 1)
        rows.append(f"2020-01-{i:02d},{y},{p}")
    rows.extend(
        [
            "2020-01-31,,1",
            "2020-02-01,,0",
            "2020-02-02,,1",
        ]
    )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "sarimax",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--interval-levels",
        "80",
        "--model-param",
        "order=0,0,0",
        "--model-param",
        "seasonal_order=0,0,0,0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    payload = json.loads(proc.stdout)
    assert {"yhat_lo_80", "yhat_hi_80"}.issubset(set(payload[0]))
    assert all(
        float(row["yhat_lo_80"]) <= float(row["yhat"]) <= float(row["yhat_hi_80"])
        for row in payload
    )


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_csv_rejects_local_artifact_save_with_future_covariates(tmp_path: Path) -> None:
    csv_path = tmp_path / "sarimax_exog.csv"
    rows = ["ds,y,promo"]
    promo = ([0, 1, 0, 1, 0] * 6)[:30]
    for i, p in enumerate(promo, start=1):
        y = 10.0 + 5.0 * float(p) + 0.1 * float(i - 1)
        rows.append(f"2020-01-{i:02d},{y},{p}")
    rows.extend(
        [
            "2020-01-31,,1",
            "2020-02-01,,0",
            "2020-02-02,,1",
        ]
    )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "sarimax",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "3",
        "--model-param",
        "order=0,0,0",
        "--model-param",
        "seasonal_order=0,0,0,0",
        "--model-param",
        "trend=c",
        "--model-param",
        "x_cols=promo",
        "--save-artifact",
        str(tmp_path / "sarimax.pkl"),
        "--format",
        "json",
    )
    assert proc.returncode == 2
    assert (
        "Saving local forecast artifacts is not yet supported when x_cols are used" in proc.stderr
    )


def test_forecast_cli_can_save_and_reuse_local_artifact(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    artifact_path = tmp_path / "naive-last.pkl"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0
    assert artifact_path.exists()

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(fit_proc.stdout)


def test_forecast_csv_rejects_local_artifact_save_for_multi_series_input(tmp_path: Path) -> None:
    csv_path = tmp_path / "panel.csv"
    artifact_path = tmp_path / "naive-last.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y",
                "s0,2020-01-01,1",
                "s0,2020-01-02,2",
                "s0,2020-01-03,3",
                "s1,2020-01-01,4",
                "s1,2020-01-02,5",
                "s1,2020-01-03,6",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )

    assert proc.returncode == 2
    assert "Saving local forecast artifacts currently requires a single series" in proc.stderr


def test_forecast_artifact_can_emit_interval_quantile_columns_for_local_artifact(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "toy.csv"
    artifact_path = tmp_path / "naive-last.pkl"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--interval-levels",
        "80,90",
        "--interval-min-train-size",
        "3",
        "--interval-samples",
        "64",
        "--interval-seed",
        "0",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0

    rows = json.loads(reuse_proc.stdout)
    assert len(rows) == 2
    assert rows[0]["yhat"] == pytest.approx(5.0)
    assert rows[0]["yhat_lo_80"] == pytest.approx(6.0)
    assert rows[0]["yhat_hi_80"] == pytest.approx(6.0)
    assert rows[0]["yhat_lo_90"] == pytest.approx(6.0)
    assert rows[0]["yhat_hi_90"] == pytest.approx(6.0)


def test_forecast_artifact_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    artifact_path = tmp_path / "naive-last.pkl"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["artifact_schema_version"] = 999
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 2
    assert "Unsupported artifact schema version" in reuse_proc.stderr


def test_forecast_artifact_rejects_forecast_local_artifact_without_ds_context(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "toy.csv"
    artifact_path = tmp_path / "naive-last.pkl"
    csv_path.write_text(
        "ds,y\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n2020-01-04,4\n2020-01-05,5\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "naive-last",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--parse-dates",
        "--horizon",
        "2",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["extra"].pop("ds", None)
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 2
    assert "Local forecast artifact is missing required ds context" in reuse_proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_csv_supports_global_models_with_future_covariates(tmp_path: Path) -> None:
    csv_path = tmp_path / "panel_future.csv"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
    )
    assert proc.returncode == 0

    rows = json.loads(proc.stdout)
    assert len(rows) == 4
    assert rows[0]["unique_id"] == "store=s0"
    assert rows[0]["ds"].startswith("2020-01-19")
    assert rows[1]["ds"].startswith("2020-01-20")
    assert rows[2]["unique_id"] == "store=s1"
    assert all("yhat" in row for row in rows)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_csv_saved_global_artifact_reuses_observed_cutoff_with_future_covariates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "ridge-global.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0
    assert artifact_path.exists()

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(fit_proc.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_artifact_rejects_intervals_for_global_artifact(tmp_path: Path) -> None:
    csv_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "ridge-global.pkl"
    csv_path.write_text(
        "\n".join(
            [
                "store,ds,y,promo",
                "s0,2020-01-01,0.0,0",
                "s0,2020-01-02,0.5,0",
                "s0,2020-01-03,1.0,0",
                "s0,2020-01-04,1.5,0",
                "s0,2020-01-05,2.0,0",
                "s0,2020-01-06,2.5,0",
                "s0,2020-01-07,3.0,1",
                "s0,2020-01-08,3.5,0",
                "s0,2020-01-09,4.0,0",
                "s0,2020-01-10,4.5,0",
                "s0,2020-01-11,5.0,0",
                "s0,2020-01-12,5.5,0",
                "s0,2020-01-13,6.0,0",
                "s0,2020-01-14,6.5,1",
                "s0,2020-01-15,7.0,0",
                "s0,2020-01-16,7.5,0",
                "s0,2020-01-17,8.0,0",
                "s0,2020-01-18,8.5,0",
                "s0,2020-01-19,,1",
                "s0,2020-01-20,,0",
                "s1,2020-01-01,1.0,0",
                "s1,2020-01-02,1.5,0",
                "s1,2020-01-03,2.0,0",
                "s1,2020-01-04,2.5,0",
                "s1,2020-01-05,3.0,0",
                "s1,2020-01-06,3.5,0",
                "s1,2020-01-07,4.0,1",
                "s1,2020-01-08,4.5,0",
                "s1,2020-01-09,5.0,0",
                "s1,2020-01-10,5.5,0",
                "s1,2020-01-11,6.0,0",
                "s1,2020-01-12,6.5,0",
                "s1,2020-01-13,7.0,0",
                "s1,2020-01-14,7.5,1",
                "s1,2020-01-15,8.0,0",
                "s1,2020-01-16,8.5,0",
                "s1,2020-01-17,9.0,0",
                "s1,2020-01-18,9.5,0",
                "s1,2020-01-19,,1",
                "s1,2020-01-20,,0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(csv_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
        "--id-cols",
        "store",
        "--parse-dates",
        "--horizon",
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--model-param",
        "id_feature=ordinal",
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--interval-levels",
        "80",
        "--format",
        "json",
    )

    assert reuse_proc.returncode == 2
    assert (
        "Forecast intervals are not yet supported for artifact model 'ridge-step-lag-global'"
        in (reuse_proc.stderr)
    )
