from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import foresight.optional_deps as optional_deps


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


def test_doctor_json_reports_environment_and_dependency_status() -> None:
    proc = _run_cli("doctor")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["package"]["version"]
    assert payload["package"]["module_path"]
    assert payload["python"]["version"]
    assert payload["python"]["executable"]
    assert payload["dependencies"]["torch"]["available"] is optional_deps.get_dependency_status(
        "torch"
    ).available
    assert payload["dependencies"]["torch"]["recommended_extra"] == "torch"
    assert payload["dependencies"]["torch"]["package_install_command"] == 'pip install "foresight-ts[torch]"'
    assert payload["datasets"]["store_sales"]["available"] in {True, False}
    assert payload["datasets"]["store_sales"]["packaged"] is False
    assert payload["datasets"]["catfish"]["source"] in {"package", "repo", "env", "data_dir"}
    assert isinstance(payload["findings"], list)
    assert all("severity" in item and "scope" in item and "message" in item for item in payload["findings"])
    assert payload["summary"]["status"] in {"ok", "warn", "error"}
    assert isinstance(payload["summary"]["warning_count"], int)
    assert isinstance(payload["summary"]["error_count"], int)


def test_doctor_text_reports_sections() -> None:
    proc = _run_cli("doctor", "--format", "text")

    assert proc.returncode == 0
    assert "ForeSight Doctor" in proc.stdout
    assert "Package" in proc.stdout
    assert "Python" in proc.stdout
    assert "Dependencies" in proc.stdout
    assert "Datasets" in proc.stdout
    assert "Status:" in proc.stdout


def test_doctor_uses_explicit_data_dir_when_provided(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    catfish_dir = data_root / "statistics time series"
    catfish_dir.mkdir(parents=True)
    csv_path = catfish_dir / "catfish.csv"
    csv_path.write_text("Date,Total\n2020-01-01,1\n", encoding="utf-8")

    proc = _run_cli("--data-dir", str(data_root), "doctor")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["datasets"]["catfish"]["source"] == "data_dir"
    assert payload["datasets"]["catfish"]["path"] == str(csv_path.resolve())
    assert payload["datasets"]["store_sales"]["source"] == "data_dir"
    assert payload["datasets"]["store_sales"]["available"] is False


def test_doctor_strict_returns_one_when_warnings_present(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    catfish_dir = data_root / "statistics time series"
    catfish_dir.mkdir(parents=True)
    (catfish_dir / "catfish.csv").write_text("Date,Total\n2020-01-01,1\n", encoding="utf-8")

    proc = _run_cli("--data-dir", str(data_root), "doctor", "--format", "text", "--strict")

    assert proc.returncode == 1
    assert "Status: WARN" in proc.stdout
    assert "Warnings" in proc.stdout
    assert "store_sales" in proc.stdout
