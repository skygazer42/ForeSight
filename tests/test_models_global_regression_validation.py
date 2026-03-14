from __future__ import annotations

import numpy as np
import pytest

from foresight.models.global_regression import (
    _is_effectively_one,
    _validate_tweedie_targets,
)


def test_is_effectively_one_uses_tight_tolerance() -> None:
    assert _is_effectively_one(1.0)
    assert _is_effectively_one(1.0 + 1e-13)
    assert not _is_effectively_one(1.0 + 1e-6)


def test_validate_tweedie_targets_rejects_negative_values_for_power_one() -> None:
    with pytest.raises(ValueError, match="power=1 requires non-negative"):
        _validate_tweedie_targets(power=1.0, y_train=np.array([-1.0, 0.5], dtype=float))


def test_validate_tweedie_targets_rejects_zero_values_for_power_above_one() -> None:
    with pytest.raises(ValueError, match="power>1 requires strictly positive"):
        _validate_tweedie_targets(power=1.5, y_train=np.array([0.0, 1.0], dtype=float))


def test_validate_tweedie_targets_accepts_positive_targets_for_power_above_one() -> None:
    _validate_tweedie_targets(power=1.5, y_train=np.array([0.5, 1.0], dtype=float))
