#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SONAR_TEST_TARGETS = [
    "tests/test_fetch_rnn_paper_metadata.py",
    "tests/test_forecasting_internals.py",
    "tests/test_models_adida.py",
    "tests/test_models_global_regression_validation.py",
    "tests/test_models_intermittent.py",
    "tests/test_models_intermittent_more.py",
    "tests/test_models_optional_deps_torch.py::test_torch_global_models_smoke_when_installed",
    "tests/test_models_torch_crossformer_pyraformer_smoke.py::test_torch_crossformer_and_pyraformer_global_smoke",
    "tests/test_models_torch_xformer_seq2seq_smoke.py::test_torch_xformer_and_rnn_global_smoke",
    "tests/test_release_tooling.py",
    "tests/test_model_validation_messages.py",
    "tests/test_models_registry.py",
    "tests/test_models_theta.py",
    "tests/test_models_theta_auto.py",
    "tests/test_no_mergeable_nested_ifs.py",
    "tests/test_no_nested_conditionals.py",
    "tests/test_no_float_literal_comparisons.py",
    "tests/test_root_import.py",
    "tests/test_sonar_coverage_recent_fixes.py",
    "tests/test_sonar_torch_rename_coverage_smoke.py",
    "tests/test_features_lag.py",
    "tests/test_features_tabular.py",
    "tests/test_features_time.py",
    "tests/test_no_sonar_low_hanging_smells.py",
    "tests/test_torch_global_validation_messages.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sonar_pytest_command(coverage_path: str) -> list[str]:
    return [
        "python",
        "-m",
        "pytest",
        "-q",
        *SONAR_TEST_TARGETS,
        "--cov=foresight",
        f"--cov-report=xml:{coverage_path}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the repository's Sonar-focused pytest suite and emit coverage.xml."
    )
    parser.add_argument(
        "--coverage-path",
        default="coverage.xml",
        help="Coverage XML output path (default: coverage.xml).",
    )
    args = parser.parse_args(argv)

    coverage_path = str(args.coverage_path).strip() or "coverage.xml"
    cmd = _sonar_pytest_command(coverage_path)
    cmd[0] = sys.executable
    subprocess.run(cmd, cwd=str(_repo_root()), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
