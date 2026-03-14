from __future__ import annotations

from pathlib import Path


def _read_repo_file(path: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / path).read_text(encoding="utf-8")


def test_intermittent_models_avoid_direct_zero_float_comparisons() -> None:
    source = _read_repo_file("src/foresight/models/intermittent.py")

    assert "np.all(x == 0.0)" not in source
    assert "if y != 0.0" not in source


def test_theta_helpers_avoid_direct_zero_float_comparisons() -> None:
    source = _read_repo_file("src/foresight/models/theta.py")

    assert "if denom == 0.0" not in source


def test_global_regression_avoids_direct_power_float_comparison() -> None:
    source = _read_repo_file("src/foresight/models/global_regression.py")

    assert "power_f == 1.0" not in source
