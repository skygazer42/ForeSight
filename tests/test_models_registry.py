from pathlib import Path

from foresight.models.registry import get_model_spec, list_models


def test_list_models_contains_expected_keys():
    keys = set(list_models())
    assert "naive-last" in keys
    assert "seasonal-naive" in keys


def test_model_spec_has_description():
    spec = get_model_spec("naive-last")
    assert isinstance(spec.description, str)
    assert spec.description


def test_model_spec_exposes_normalized_capabilities():
    spec = get_model_spec("naive-last")

    assert isinstance(spec.capabilities, dict)
    assert spec.capabilities["supports_x_cols"] is False
    assert spec.capabilities["supports_quantiles"] is False
    assert spec.capabilities["supports_interval_forecast"] is True
    assert spec.capabilities["supports_interval_forecast_with_x_cols"] is False
    assert spec.capabilities["supports_artifact_save"] is True
    assert spec.capabilities["requires_future_covariates"] is False


def test_model_spec_capabilities_reflect_model_family_support():
    sarimax = get_model_spec("sarimax")
    assert sarimax.capabilities["supports_x_cols"] is True
    assert sarimax.capabilities["supports_interval_forecast"] is True
    assert sarimax.capabilities["supports_interval_forecast_with_x_cols"] is True

    auto_arima = get_model_spec("auto-arima")
    assert auto_arima.capabilities["supports_x_cols"] is True
    assert auto_arima.capabilities["supports_interval_forecast"] is True
    assert auto_arima.capabilities["supports_interval_forecast_with_x_cols"] is True

    global_xgb = get_model_spec("xgb-step-lag-global")
    assert global_xgb.capabilities["supports_x_cols"] is True
    assert global_xgb.capabilities["supports_quantiles"] is True
    assert global_xgb.capabilities["supports_interval_forecast"] is True
    assert global_xgb.capabilities["supports_interval_forecast_with_x_cols"] is True
    assert global_xgb.capabilities["supports_artifact_save"] is True

    var = get_model_spec("var")
    assert var.interface == "multivariate"
    assert var.capabilities["supports_artifact_save"] is False


def test_readme_documents_all_model_capability_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    documented = sorted(
        {key for model_key in list_models() for key in get_model_spec(model_key).capabilities}
    )
    assert documented == [
        "requires_future_covariates",
        "supports_artifact_save",
        "supports_interval_forecast",
        "supports_interval_forecast_with_x_cols",
        "supports_quantiles",
        "supports_x_cols",
    ]
    for key in documented:
        assert f"`{key}`" in readme


def test_new_torch_global_models_are_registered():
    keys = set(list_models())
    assert "torch-nlinear-global" in keys
    assert "torch-dlinear-global" in keys
    assert "torch-deepar-global" in keys
    assert "torch-fedformer-global" in keys
    assert "torch-nonstationary-transformer-global" in keys


def test_new_torch_mamba_rwkv_models_are_registered():
    keys = set(list_models())
    assert "torch-mamba-direct" in keys
    assert "torch-rwkv-direct" in keys
    assert "torch-mamba-global" in keys
    assert "torch-rwkv-global" in keys


def test_new_torch_hyena_models_are_registered():
    keys = set(list_models())
    assert "torch-hyena-direct" in keys
    assert "torch-hyena-global" in keys


def test_new_torch_dilated_rnn_and_kan_models_are_registered():
    keys = set(list_models())
    assert "torch-dilated-rnn-direct" in keys
    assert "torch-dilated-rnn-global" in keys
    assert "torch-kan-direct" in keys
    assert "torch-kan-global" in keys


def test_new_torch_scinet_and_etsformer_models_are_registered():
    keys = set(list_models())
    assert "torch-scinet-direct" in keys
    assert "torch-scinet-global" in keys
    assert "torch-etsformer-direct" in keys
    assert "torch-etsformer-global" in keys


def test_new_torch_esrnn_models_are_registered():
    keys = set(list_models())
    assert "torch-esrnn-direct" in keys
    assert "torch-esrnn-global" in keys


def test_new_torch_crossformer_models_are_registered():
    keys = set(list_models())
    assert "torch-crossformer-direct" in keys
    assert "torch-crossformer-global" in keys


def test_new_torch_pyraformer_models_are_registered():
    keys = set(list_models())
    assert "torch-pyraformer-direct" in keys
    assert "torch-pyraformer-global" in keys


def test_new_torch_xformer_attention_variants_are_registered():
    keys = set(list_models())
    assert "torch-xformer-probsparse-ln-gelu-direct" in keys
    assert "torch-xformer-autocorr-ln-gelu-direct" in keys
    assert "torch-xformer-reformer-ln-gelu-direct" in keys
    assert "torch-xformer-logsparse-ln-gelu-direct" in keys
    assert "torch-xformer-longformer-ln-gelu-direct" in keys
    assert "torch-xformer-bigbird-ln-gelu-direct" in keys
    assert "torch-xformer-probsparse-global" in keys
    assert "torch-xformer-autocorr-global" in keys
    assert "torch-xformer-reformer-global" in keys
    assert "torch-xformer-logsparse-global" in keys
    assert "torch-xformer-longformer-global" in keys
    assert "torch-xformer-bigbird-global" in keys
