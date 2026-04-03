from __future__ import annotations

import foresight.models.specs as specs_mod


def test_base_model_capabilities_marks_local_xreg_models_without_artifact_save() -> None:
    capabilities = specs_mod._base_model_capabilities(  # type: ignore[attr-defined]
        interface="local",
        param_help={"x_cols": "Future covariates"},
    )

    assert capabilities["supports_panel"] is True
    assert capabilities["supports_univariate"] is True
    assert capabilities["supports_multivariate"] is False
    assert capabilities["supports_x_cols"] is True
    assert capabilities["supports_future_covariates"] is True
    assert capabilities["supports_historic_covariates"] is True
    assert capabilities["supports_interval_forecast"] is True
    assert capabilities["supports_interval_forecast_with_x_cols"] is False
    assert capabilities["supports_artifact_save"] is False


def test_base_model_capabilities_marks_multivariate_models_without_panel_support() -> None:
    capabilities = specs_mod._base_model_capabilities(  # type: ignore[attr-defined]
        interface="multivariate",
        param_help={},
    )

    assert capabilities["supports_panel"] is False
    assert capabilities["supports_univariate"] is False
    assert capabilities["supports_multivariate"] is True
    assert capabilities["supports_probabilistic"] is False
    assert capabilities["supports_conformal_eval"] is False
    assert capabilities["supports_artifact_save"] is False
