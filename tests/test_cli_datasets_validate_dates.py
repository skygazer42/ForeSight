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


def test_datasets_validate_fails_on_unparseable_dates(tmp_path: Path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "data" / "cashflow_data.csv").write_text(
        "date,cashflow_category,cashflow_subcategory,cashflow,branch_id\n"
        "not-a-date,cash_in,sales,1.0,001\n",
        encoding="utf-8",
    )

    proc = _run_cli("--data-dir", str(root), "datasets", "validate", "--dataset", "cashflow_data", "--nrows", "5")
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "FAIL cashflow_data" in out

