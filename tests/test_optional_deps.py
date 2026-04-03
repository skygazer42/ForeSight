from __future__ import annotations

import builtins
import types

import pytest

import foresight.optional_deps as optional_deps
from foresight.models import (
    global_regression,
    hf_time_series,
    multivariate,
    regression,
    statsmodels_wrap,
    torch_nn,
)


def test_torch_namespace_stub_is_reported_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_stub = types.ModuleType("torch")

    monkeypatch.setattr(
        optional_deps, "_find_spec", lambda name: object() if name == "torch" else None
    )
    monkeypatch.setattr(
        optional_deps,
        "_import_module",
        lambda name: torch_stub if name == "torch" else pytest.fail(f"unexpected import: {name}"),
    )

    status = optional_deps.get_dependency_status("torch")

    assert status.name == "torch"
    assert status.spec_found is True
    assert status.available is False
    assert status.version is None
    assert "missing required attributes" in str(status.reason)


def test_require_torch_rejects_namespace_stub_without_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_stub = types.ModuleType("torch")

    monkeypatch.setattr(
        optional_deps, "_find_spec", lambda name: object() if name == "torch" else None
    )
    monkeypatch.setattr(
        optional_deps,
        "_import_module",
        lambda name: torch_stub if name == "torch" else pytest.fail(f"unexpected import: {name}"),
    )

    with pytest.raises(
        ImportError,
        match='Torch models require PyTorch\\. Install with: pip install "foresight-ts\\[torch\\]" or pip install -e "\\.\\[torch\\]"',
    ):
        torch_nn._require_torch()


def test_dependency_status_exposes_install_commands() -> None:
    status = optional_deps.get_dependency_status("ml").as_dict()

    assert status["recommended_extra"] == "ml"
    assert status["package_install_command"] == 'pip install "foresight-ts[ml]"'
    assert status["editable_install_command"] == 'pip install -e ".[ml]"'


def test_extra_status_exposes_install_commands() -> None:
    status = optional_deps.get_extra_status("torch").as_dict()

    assert status["package_install_command"] == 'pip install "foresight-ts[torch]"'
    assert status["editable_install_command"] == 'pip install -e ".[torch]"'


def test_sktime_extra_status_exposes_install_commands() -> None:
    status = optional_deps.get_extra_status("sktime").as_dict()

    assert status["package_install_command"] == 'pip install "foresight-ts[sktime]"'
    assert status["editable_install_command"] == 'pip install -e ".[sktime]"'


def test_missing_dependency_message_includes_package_and_editable_commands() -> None:
    msg = optional_deps.missing_dependency_message("ml", subject="ridge_lag_forecast")

    assert msg == (
        "ridge_lag_forecast requires scikit-learn. "
        'Install with: pip install "foresight-ts[ml]" or pip install -e ".[ml]"'
    )


def _patch_import_error(
    monkeypatch: pytest.MonkeyPatch,
    *,
    blocked_roots: set[str],
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if root in blocked_roots:
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_ridge_lag_forecast_missing_sklearn_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='ridge_lag_forecast requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        regression.ridge_lag_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_ridge_step_lag_global_missing_sklearn_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='ridge-step-lag-global requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        global_regression.ridge_step_lag_global_forecaster(lags=2)


def test_var_forecast_missing_statsmodels_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"statsmodels"})

    with pytest.raises(
        ImportError,
        match='var_forecast requires statsmodels\\. Install with: pip install "foresight-ts\\[stats\\]" or pip install -e "\\.\\[stats\\]"',
    ):
        multivariate.var_forecast([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], 1)


def test_sarimax_forecast_missing_statsmodels_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"statsmodels"})

    with pytest.raises(
        ImportError,
        match='sarimax_forecast requires statsmodels\\. Install with: pip install "foresight-ts\\[stats\\]" or pip install -e "\\.\\[stats\\]"',
    ):
        statsmodels_wrap.sarimax_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 1)


def test_hf_timeseries_transformer_missing_transformers_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_time_series, "_require_torch", lambda: object())
    _patch_import_error(monkeypatch, blocked_roots={"transformers"})

    with pytest.raises(
        ImportError,
        match='hf-timeseries-transformer-direct requires transformers\\. Install with: pip install "foresight-ts\\[transformers\\]" or pip install -e "\\.\\[transformers\\]"',
    ):
        hf_time_series.hf_timeseries_transformer_direct_forecast([1.0] * 10, 1)


def test_xgb_step_lag_global_missing_xgboost_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"xgboost"})

    with pytest.raises(
        ImportError,
        match='xgb-step-lag-global requires xgboost\\. Install with: pip install "foresight-ts\\[xgb\\]" or pip install -e "\\.\\[xgb\\]"',
    ):
        global_regression.xgb_step_lag_global_forecaster(lags=2)


def test_lightgbm_global_missing_lightgbm_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"lightgbm"})

    with pytest.raises(
        ImportError,
        match='lgbm-step-lag-global requires lightgbm\\. Install with: pip install "foresight-ts\\[lgbm\\]" or pip install -e "\\.\\[lgbm\\]"',
    ):
        global_regression.lgbm_step_lag_global_forecaster(lags=2)


def test_catboost_global_missing_catboost_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"catboost"})

    with pytest.raises(
        ImportError,
        match='catboost-step-lag-global requires catboost\\. Install with: pip install "foresight-ts\\[catboost\\]" or pip install -e "\\.\\[catboost\\]"',
    ):
        global_regression.catboost_step_lag_global_forecaster(lags=2)


def test_rf_lag_direct_missing_sklearn_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='rf_lag_direct_forecast requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        regression.rf_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_bayesian_ridge_lag_direct_missing_sklearn_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='bayesian_ridge_lag_direct_forecast requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        regression.bayesian_ridge_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_passive_aggressive_lag_direct_missing_sklearn_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='passive_aggressive_lag_direct_forecast requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        regression.passive_aggressive_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_decision_tree_lag_direct_missing_sklearn_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"sklearn"})

    with pytest.raises(
        ImportError,
        match='decision_tree_lag_direct_forecast requires scikit-learn\\. Install with: pip install "foresight-ts\\[ml\\]" or pip install -e "\\.\\[ml\\]"',
    ):
        regression.decision_tree_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_xgb_lag_direct_missing_xgboost_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"xgboost"})

    with pytest.raises(
        ImportError,
        match='xgboost lag models requires xgboost\\. Install with: pip install "foresight-ts\\[xgb\\]" or pip install -e "\\.\\[xgb\\]"',
    ):
        regression.xgb_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_lgbm_lag_direct_missing_lightgbm_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"lightgbm"})

    with pytest.raises(
        ImportError,
        match='lightgbm lag models requires lightgbm\\. Install with: pip install "foresight-ts\\[lgbm\\]" or pip install -e "\\.\\[lgbm\\]"',
    ):
        regression.lgbm_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)


def test_catboost_lag_direct_missing_catboost_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_error(monkeypatch, blocked_roots={"catboost"})

    with pytest.raises(
        ImportError,
        match='catboost lag models requires catboost\\. Install with: pip install "foresight-ts\\[catboost\\]" or pip install -e "\\.\\[catboost\\]"',
    ):
        regression.catboost_lag_direct_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 2, lags=2)
