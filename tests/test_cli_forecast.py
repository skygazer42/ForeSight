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


def test_forecast_artifact_help_mentions_local_and_global_interval_support() -> None:
    proc = _run_cli("forecast", "artifact", "--help")

    assert proc.returncode == 0
    assert "Optional central interval levels for local artifacts" in proc.stdout
    assert "interval-capable global artifacts" in proc.stdout


def test_forecast_artifact_help_mentions_future_override_requirements() -> None:
    proc = _run_cli("forecast", "artifact", "--help")

    assert proc.returncode == 0
    assert "local overrides replace saved" in proc.stdout
    assert "future context, global overrides may include" in proc.stdout
    assert "global overrides may include" in proc.stdout
    assert "or omit ids for" in proc.stdout
    assert "single-series artifacts" in proc.stdout
    assert "used with --future-path (required" in proc.stdout
    assert "when overriding future context)" in proc.stdout


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
def test_forecast_cli_can_save_and_reuse_local_artifact_with_future_covariates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sarimax_exog.csv"
    artifact_path = tmp_path / "sarimax.pkl"
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

    fit_proc = _run_cli(
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
        str(artifact_path),
        "--format",
        "json",
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
    assert json.loads(reuse_proc.stdout) == json.loads(fit_proc.stdout)[:2]


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_artifact_rejects_horizon_beyond_saved_local_xreg_context(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sarimax_exog.csv"
    artifact_path = tmp_path / "sarimax.pkl"
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

    fit_proc = _run_cli(
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
        str(artifact_path),
        "--format",
        "json",
    )
    assert fit_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "4",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 2
    assert "Requested horizon=4 exceeds artifact max_horizon=3" in reuse_proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed"
)
def test_forecast_artifact_local_xreg_can_override_saved_future_context(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sarimax_exog.csv"
    artifact_path = tmp_path / "sarimax.pkl"
    override_future_path = tmp_path / "sarimax_future_override.csv"
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
    override_future_path.write_text(
        "ds,promo\n2020-01-31,0\n2020-02-01,1\n2020-02-02,0\n2020-02-03,1\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
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
        str(artifact_path),
        "--format",
        "json",
    )
    assert fit_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--future-path",
        str(override_future_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--horizon",
        "4",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0

    payload = json.loads(reuse_proc.stdout)
    assert [row["ds"][:10] for row in payload] == [
        "2020-01-31",
        "2020-02-01",
        "2020-02-02",
        "2020-02-03",
    ]
    yhat = [float(row["yhat"]) for row in payload]
    assert yhat[1] > yhat[0] + 4.0
    assert yhat[3] > yhat[2] + 4.0


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
def test_forecast_csv_can_save_global_artifact_from_separate_future_csv(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "panel_history.csv"
    future_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    history_path.write_text(
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
                "",
            ]
        ),
        encoding="utf-8",
    )
    future_path.write_text(
        "\n".join(
            [
                "store,ds,promo",
                "s0,2020-01-19,1",
                "s0,2020-01-20,0",
                "s1,2020-01-19,1",
                "s1,2020-01-20,0",
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
        str(history_path),
        "--future-path",
        str(future_path),
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
def test_forecast_artifact_global_xreg_can_override_saved_future_context(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "panel_full.csv"
    history_path = tmp_path / "panel_history.csv"
    csv_future_path = tmp_path / "panel_future_override.csv"
    artifact_future_path = tmp_path / "panel_future_override_artifact.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    full_rows = ["store,ds,y,promo"]
    history_rows = ["store,ds,y,promo"]
    csv_future_rows = ["store,ds,promo"]
    artifact_future_rows = ["unique_id,ds,promo"]

    for store_offset, store in enumerate(("s0", "s1")):
        for day in range(1, 19):
            promo = int(day % 2 == 0)
            y = float(store_offset) + 2.0 * float(day - 1) + 20.0 * float(promo)
            row = f"{store},2020-01-{day:02d},{y:.1f},{promo}"
            full_rows.append(row)
            history_rows.append(row)
        for day in (19, 20):
            full_rows.append(f"{store},2020-01-{day:02d},,0")
            csv_future_rows.append(f"{store},2020-01-{day:02d},1")
            artifact_future_rows.append(f"store={store},2020-01-{day:02d},1")

    full_path.write_text("\n".join([*full_rows, ""]) + "\n", encoding="utf-8")
    history_path.write_text("\n".join([*history_rows, ""]) + "\n", encoding="utf-8")
    csv_future_path.write_text("\n".join([*csv_future_rows, ""]) + "\n", encoding="utf-8")
    artifact_future_path.write_text(
        "\n".join([*artifact_future_rows, ""]) + "\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(full_path),
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

    expected_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(history_path),
        "--future-path",
        str(csv_future_path),
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
    assert expected_proc.returncode == 0

    expected_rows = json.loads(expected_proc.stdout)
    baseline_rows = json.loads(fit_proc.stdout)
    assert expected_rows != baseline_rows

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--future-path",
        str(artifact_future_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == expected_rows


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_artifact_global_xreg_accepts_saved_raw_id_columns_in_future_override(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "panel_full.csv"
    history_path = tmp_path / "panel_history.csv"
    future_path = tmp_path / "panel_future_override.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    full_rows = ["store,ds,y,promo"]
    history_rows = ["store,ds,y,promo"]
    future_rows = ["store,ds,promo"]

    for store_offset, store in enumerate(("s0", "s1")):
        for day in range(1, 19):
            promo = int(day % 2 == 0)
            y = float(store_offset) + 2.0 * float(day - 1) + 20.0 * float(promo)
            row = f"{store},2020-01-{day:02d},{y:.1f},{promo}"
            full_rows.append(row)
            history_rows.append(row)
        for day in (19, 20):
            full_rows.append(f"{store},2020-01-{day:02d},,0")
            future_rows.append(f"{store},2020-01-{day:02d},1")

    full_path.write_text("\n".join([*full_rows, ""]) + "\n", encoding="utf-8")
    history_path.write_text("\n".join([*history_rows, ""]) + "\n", encoding="utf-8")
    future_path.write_text("\n".join([*future_rows, ""]) + "\n", encoding="utf-8")

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(full_path),
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

    expected_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(history_path),
        "--future-path",
        str(future_path),
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
    assert expected_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--future-path",
        str(future_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(expected_proc.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_artifact_single_series_global_xreg_accepts_future_override_without_ids(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "single_full.csv"
    history_path = tmp_path / "single_history.csv"
    future_path = tmp_path / "single_future_override.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    full_rows = ["ds,y,promo"]
    history_rows = ["ds,y,promo"]
    future_rows = ["ds,promo"]

    for day in range(1, 19):
        promo = int(day % 2 == 0)
        y = 2.0 * float(day - 1) + 20.0 * float(promo)
        row = f"2020-01-{day:02d},{y:.1f},{promo}"
        full_rows.append(row)
        history_rows.append(row)
    for day in (19, 20):
        full_rows.append(f"2020-01-{day:02d},,0")
        future_rows.append(f"2020-01-{day:02d},1")

    full_path.write_text("\n".join([*full_rows, ""]) + "\n", encoding="utf-8")
    history_path.write_text("\n".join([*history_rows, ""]) + "\n", encoding="utf-8")
    future_path.write_text("\n".join([*future_rows, ""]) + "\n", encoding="utf-8")

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(full_path),
        "--time-col",
        "ds",
        "--y-col",
        "y",
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
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    expected_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
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
        "2",
        "--model-param",
        "lags=5",
        "--model-param",
        "alpha=0.5",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "add_time_features=true",
        "--format",
        "json",
    )
    assert expected_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--future-path",
        str(future_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(expected_proc.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_artifact_single_series_legacy_global_xreg_accepts_future_override_without_ids(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "single_panel_full.csv"
    history_path = tmp_path / "single_panel_history.csv"
    future_with_id_path = tmp_path / "single_panel_future_with_id.csv"
    future_without_id_path = tmp_path / "single_panel_future_without_id.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    full_rows = ["store,ds,y,promo"]
    history_rows = ["store,ds,y,promo"]
    future_with_id_rows = ["store,ds,promo"]
    future_without_id_rows = ["ds,promo"]

    for day in range(1, 19):
        promo = int(day % 2 == 0)
        y = 2.0 * float(day - 1) + 20.0 * float(promo)
        row = f"s0,2020-01-{day:02d},{y:.1f},{promo}"
        full_rows.append(row)
        history_rows.append(row)
    for day in (19, 20):
        full_rows.append(f"s0,2020-01-{day:02d},,0")
        future_with_id_rows.append(f"s0,2020-01-{day:02d},1")
        future_without_id_rows.append(f"2020-01-{day:02d},1")

    full_path.write_text("\n".join([*full_rows, ""]) + "\n", encoding="utf-8")
    history_path.write_text("\n".join([*history_rows, ""]) + "\n", encoding="utf-8")
    future_with_id_path.write_text("\n".join([*future_with_id_rows, ""]) + "\n", encoding="utf-8")
    future_without_id_path.write_text(
        "\n".join([*future_without_id_rows, ""]) + "\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(full_path),
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
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["extra"] = dict(payload.get("extra", {}))
    payload["extra"].pop("id_cols", None)
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    expected_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(history_path),
        "--future-path",
        str(future_with_id_path),
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
        "--format",
        "json",
    )
    assert expected_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--future-path",
        str(future_without_id_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(expected_proc.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed"
)
def test_forecast_artifact_single_series_global_xreg_with_saved_id_cols_accepts_future_override_without_ids(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "single_panel_full.csv"
    history_path = tmp_path / "single_panel_history.csv"
    future_with_id_path = tmp_path / "single_panel_future_with_id.csv"
    future_without_id_path = tmp_path / "single_panel_future_without_id.csv"
    artifact_path = tmp_path / "ridge-global.pkl"

    full_rows = ["store,ds,y,promo"]
    history_rows = ["store,ds,y,promo"]
    future_with_id_rows = ["store,ds,promo"]
    future_without_id_rows = ["ds,promo"]

    for day in range(1, 19):
        promo = int(day % 2 == 0)
        y = 2.0 * float(day - 1) + 20.0 * float(promo)
        row = f"s0,2020-01-{day:02d},{y:.1f},{promo}"
        full_rows.append(row)
        history_rows.append(row)
    for day in (19, 20):
        full_rows.append(f"s0,2020-01-{day:02d},,0")
        future_with_id_rows.append(f"s0,2020-01-{day:02d},1")
        future_without_id_rows.append(f"2020-01-{day:02d},1")

    full_path.write_text("\n".join([*full_rows, ""]) + "\n", encoding="utf-8")
    history_path.write_text("\n".join([*history_rows, ""]) + "\n", encoding="utf-8")
    future_with_id_path.write_text("\n".join([*future_with_id_rows, ""]) + "\n", encoding="utf-8")
    future_without_id_path.write_text(
        "\n".join([*future_without_id_rows, ""]) + "\n",
        encoding="utf-8",
    )

    fit_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(full_path),
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
        "--format",
        "json",
        "--save-artifact",
        str(artifact_path),
    )
    assert fit_proc.returncode == 0

    expected_proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "ridge-step-lag-global",
        "--path",
        str(history_path),
        "--future-path",
        str(future_with_id_path),
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
        "--format",
        "json",
    )
    assert expected_proc.returncode == 0

    reuse_proc = _run_cli(
        "forecast",
        "artifact",
        "--artifact",
        str(artifact_path),
        "--horizon",
        "2",
        "--future-path",
        str(future_without_id_path),
        "--time-col",
        "ds",
        "--parse-dates",
        "--format",
        "json",
    )
    assert reuse_proc.returncode == 0
    assert json.loads(reuse_proc.stdout) == json.loads(expected_proc.stdout)


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


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")
def test_forecast_artifact_emits_interval_columns_for_quantile_global_artifact(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "panel_future.csv"
    artifact_path = tmp_path / "xgb-global.pkl"
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
        "xgb-step-lag-global",
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
        "n_estimators=25",
        "--model-param",
        "learning_rate=0.1",
        "--model-param",
        "max_depth=3",
        "--model-param",
        "subsample=0.9",
        "--model-param",
        "colsample_bytree=0.9",
        "--model-param",
        "x_cols=promo",
        "--model-param",
        "quantiles=0.1,0.5,0.9",
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
    assert reuse_proc.returncode == 0

    payload = json.loads(reuse_proc.stdout)
    assert payload
    assert {"yhat_p10", "yhat_p50", "yhat_p90", "yhat_lo_80", "yhat_hi_80"}.issubset(
        set(payload[0])
    )
    assert all(abs(float(row["yhat"]) - float(row["yhat_p50"])) < 1e-9 for row in payload)
    assert all(abs(float(row["yhat_lo_80"]) - float(row["yhat_p10"])) < 1e-9 for row in payload)
    assert all(abs(float(row["yhat_hi_80"]) - float(row["yhat_p90"])) < 1e-9 for row in payload)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_forecast_csv_torch_training_logs_to_stderr_without_polluting_stdout(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "torch_train.csv"
    csv_path.write_text(
        "ds,y\n"
        "2020-01-01,1\n"
        "2020-01-02,2\n"
        "2020-01-03,3\n"
        "2020-01-04,4\n"
        "2020-01-05,5\n"
        "2020-01-06,6\n"
        "2020-01-07,7\n"
        "2020-01-08,8\n"
        "2020-01-09,9\n"
        "2020-01-10,10\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "torch-mlp-direct",
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
        "--model-param",
        "lags=4",
        "--model-param",
        "epochs=2",
        "--model-param",
        "hidden_sizes=8",
        "--model-param",
        "batch_size=4",
        "--model-param",
        "patience=1",
        "--model-param",
        "min_epochs=1",
    )

    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert "RUN start" in proc.stderr
    assert "TRAIN start" in proc.stderr
    assert "EPOCH 1/" in proc.stderr
    assert "TRAIN done" in proc.stderr
    assert "RUN done" in proc.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_forecast_csv_no_progress_suppresses_epoch_logs(tmp_path: Path) -> None:
    csv_path = tmp_path / "torch_train.csv"
    csv_path.write_text(
        "ds,y\n"
        "2020-01-01,1\n"
        "2020-01-02,2\n"
        "2020-01-03,3\n"
        "2020-01-04,4\n"
        "2020-01-05,5\n"
        "2020-01-06,6\n"
        "2020-01-07,7\n"
        "2020-01-08,8\n"
        "2020-01-09,9\n"
        "2020-01-10,10\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "torch-mlp-direct",
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
        "--no-progress",
        "--model-param",
        "lags=4",
        "--model-param",
        "epochs=2",
        "--model-param",
        "hidden_sizes=8",
        "--model-param",
        "batch_size=4",
        "--model-param",
        "patience=1",
        "--model-param",
        "min_epochs=1",
    )

    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    assert "RUN start" in proc.stderr
    assert "TRAIN start" in proc.stderr
    assert "EPOCH " not in proc.stderr
    assert "RUN done" in proc.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_forecast_csv_log_file_captures_structured_torch_training_metrics(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "torch_train.csv"
    log_path = tmp_path / "torch-train.jsonl"
    csv_path.write_text(
        "ds,y\n"
        "2020-01-01,1\n"
        "2020-01-02,2\n"
        "2020-01-03,3\n"
        "2020-01-04,4\n"
        "2020-01-05,5\n"
        "2020-01-06,6\n"
        "2020-01-07,7\n"
        "2020-01-08,8\n"
        "2020-01-09,9\n"
        "2020-01-10,10\n",
        encoding="utf-8",
    )

    proc = _run_cli(
        "forecast",
        "csv",
        "--model",
        "torch-mlp-direct",
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
        "--log-style",
        "quiet",
        "--log-file",
        str(log_path),
        "--model-param",
        "lags=4",
        "--model-param",
        "epochs=2",
        "--model-param",
        "hidden_sizes=8",
        "--model-param",
        "batch_size=4",
        "--model-param",
        "patience=2",
        "--model-param",
        "min_epochs=1",
    )

    assert proc.returncode == 0
    assert "RUN start" not in proc.stderr
    assert "TRAIN start" not in proc.stderr
    assert "EPOCH " not in proc.stderr
    assert "TRAIN done" not in proc.stderr
    rows = json.loads(proc.stdout)
    assert len(rows) == 2

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = {record.get("run_id") for record in records}
    assert len(run_ids) == 1
    assert next(iter(run_ids))
    events = {record["event"]: record for record in records}
    assert "run_started" in events
    assert "train_started" in events
    assert "train_completed" in events
    assert "run_completed" in events

    epoch_records = [record for record in records if record["event"] == "train_epoch_completed"]
    assert len(epoch_records) == 2

    train_started = events["train_started"]["payload"]
    assert train_started["effective_batch_size"] == 4
    assert train_started["train_samples"] > 0
    assert train_started["trainable_parameters"] > 0
    assert train_started["total_parameters"] >= train_started["trainable_parameters"]
    assert train_started["optimizer"] == "adam"
    assert train_started["scheduler"] == "none"
    assert train_started["amp"] is False
    assert train_started["device_type"] == "cpu"
    assert isinstance(train_started["cuda_available"], bool)

    first_epoch = epoch_records[0]["payload"]
    assert first_epoch["epoch"] == 1
    assert first_epoch["total_epochs"] == 2
    assert first_epoch["lr"] > 0.0
    assert first_epoch["avg_grad_norm"] > 0.0
    assert first_epoch["epoch_seconds"] >= 0.0
    assert first_epoch["samples_per_second"] > 0.0
    assert first_epoch["batches_per_second"] > 0.0
    assert "best_improved" in first_epoch

    train_completed = events["train_completed"]["payload"]
    assert train_completed["stop_reason"] == "completed"
    assert train_completed["total_seconds"] >= 0.0
    assert train_completed["final_lr"] > 0.0
