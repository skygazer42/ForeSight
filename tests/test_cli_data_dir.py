import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "foresight", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_data_dir_overrides_env(tmp_path: Path):
    root_env = tmp_path / "env_root"
    (root_env / "data").mkdir(parents=True)
    shutil.copyfile("data/store_sales.csv", root_env / "data" / "store_sales.csv")

    root_flag = tmp_path / "flag_root"
    (root_flag / "data").mkdir(parents=True)
    shutil.copyfile("data/store_sales.csv", root_flag / "data" / "store_sales.csv")

    # Make env root invalid by removing the file after setting it.
    env = {"FORESIGHT_DATA_DIR": str(root_env)}
    (root_env / "data" / "store_sales.csv").unlink()

    proc = _run_cli(
        "--data-dir",
        str(root_flag),
        "datasets",
        "preview",
        "store_sales",
        "--nrows",
        "3",
        env_extra=env,
    )
    assert proc.returncode == 0
    assert "sales" in (proc.stdout + proc.stderr)

