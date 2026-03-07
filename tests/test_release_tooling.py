from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_release_check_plan_mentions_docs_and_benchmark_steps() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "tools/release_check.py", "--plan"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "python tools/check_capability_docs.py" in proc.stdout
    assert "python tools/generate_model_capability_docs.py --check" in proc.stdout
    assert "python benchmarks/run_benchmarks.py --smoke" in proc.stdout
    assert "mkdocs build --strict" in proc.stdout


def test_docs_workflow_builds_and_deploys_mkdocs_site() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert "name: Docs" in workflow
    assert "actions/configure-pages" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow
    assert "python tools/generate_model_capability_docs.py" in workflow
    assert "python tools/generate_rnn_docs.py" in workflow
    assert "mkdocs build --strict" in workflow


def test_release_docs_cover_docs_site_and_benchmark_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_doc = (repo_root / "docs" / "RELEASE.md").read_text(encoding="utf-8")

    assert "python tools/generate_model_capability_docs.py" in release_doc
    assert "python tools/generate_rnn_docs.py" in release_doc
    assert "python benchmarks/run_benchmarks.py --smoke" in release_doc
    assert "mkdocs build --strict" in release_doc
    assert ".github/workflows/docs.yml" in release_doc
