from pathlib import Path

from foresight.base import (
    RegistryForecaster as BaseRegistryForecaster,
    RegistryGlobalForecaster as BaseRegistryGlobalForecaster,
)
from foresight.models.registry import ModelSpec, get_model_spec, list_models
from foresight.models.specs import (
    LocalForecasterFn as RuntimeLocalForecasterFn,
    ModelFactory as RuntimeModelFactory,
)
from foresight.models.specs import ModelSpec as RuntimeModelSpec

WAVE1A_TORCH_LOCAL_KEYS = (
    "torch-informer-direct",
    "torch-autoformer-direct",
    "torch-nonstationary-transformer-direct",
    "torch-fedformer-direct",
    "torch-itransformer-direct",
    "torch-timesnet-direct",
    "torch-tft-direct",
    "torch-timemixer-direct",
    "torch-sparsetsf-direct",
)

LIGHTWEIGHT_TORCH_LOCAL_KEYS = (
    "torch-lightts-direct",
    "torch-frets-direct",
)

DECOMP_TORCH_LOCAL_KEYS = (
    "torch-film-direct",
    "torch-micn-direct",
)

WAVE1B_TORCH_LOCAL_KEYS = (
    "torch-koopa-direct",
    "torch-samformer-direct",
)

RETENTION_TORCH_LOCAL_KEYS = (
    "torch-retnet-direct",
    "torch-retnet-recursive",
)
RETENTION_TORCH_GLOBAL_KEYS = ("torch-retnet-global",)

TIME_XER_TORCH_KEYS = (
    "torch-timexer-direct",
    "torch-timexer-global",
)

STATE_SPACE_TORCH_LOCAL_KEYS = (
    "torch-lmu-direct",
    "torch-s4d-direct",
)

CT_RNN_TORCH_LOCAL_KEYS = (
    "torch-ltc-direct",
    "torch-cfc-direct",
)

REVIVAL_TORCH_LOCAL_KEYS = (
    "torch-xlstm-direct",
    "torch-mamba2-direct",
)

SSM_TORCH_LOCAL_KEYS = (
    "torch-s4-direct",
    "torch-s5-direct",
)

RECURRENT_REVIVAL_TORCH_LOCAL_KEYS = (
    "torch-griffin-direct",
    "torch-hawk-direct",
)

TORCH_MULTIVARIATE_KEYS = (
    "torch-stid-multivariate",
    "torch-stgcn-multivariate",
    "torch-graphwavenet-multivariate",
)

TRANSFORMERS_LOCAL_KEYS = ("hf-timeseries-transformer-direct",)


def test_list_models_contains_expected_keys():
    keys = set(list_models())
    assert "naive-last" in keys
    assert "seasonal-naive" in keys
    assert "weighted-moving-average" in keys
    assert "moving-median" in keys
    assert "seasonal-drift" in keys


def test_wave1a_torch_local_models_are_registered():
    keys = set(list_models())
    for key in WAVE1A_TORCH_LOCAL_KEYS:
        assert key in keys


def test_lightweight_torch_local_models_are_registered():
    keys = set(list_models())
    for key in LIGHTWEIGHT_TORCH_LOCAL_KEYS:
        assert key in keys


def test_decomposition_torch_local_models_are_registered():
    keys = set(list_models())
    for key in DECOMP_TORCH_LOCAL_KEYS:
        assert key in keys


def test_wave1b_torch_local_models_are_registered():
    keys = set(list_models())
    for key in WAVE1B_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RETENTION_TORCH_LOCAL_KEYS:
        assert key in keys


def test_retnet_torch_global_models_are_registered() -> None:
    keys = set(list_models())
    for key in RETENTION_TORCH_GLOBAL_KEYS:
        assert key in keys


def test_timexer_torch_models_are_registered() -> None:
    keys = set(list_models())
    for key in TIME_XER_TORCH_KEYS:
        assert key in keys


def test_state_space_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in STATE_SPACE_TORCH_LOCAL_KEYS:
        assert key in keys


def test_continuous_time_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in CT_RNN_TORCH_LOCAL_KEYS:
        assert key in keys


def test_revival_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_ssm_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in SSM_TORCH_LOCAL_KEYS:
        assert key in keys


def test_recurrent_revival_torch_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in RECURRENT_REVIVAL_TORCH_LOCAL_KEYS:
        assert key in keys


def test_torch_multivariate_models_are_registered() -> None:
    keys = set(list_models())
    for key in TORCH_MULTIVARIATE_KEYS:
        assert key in keys


def test_transformers_local_models_are_registered() -> None:
    keys = set(list_models())
    for key in TRANSFORMERS_LOCAL_KEYS:
        assert key in keys


def test_catalog_shards_preserve_cross_family_lookup() -> None:
    keys = ["naive-last", "ridge-lag", "arima", "torch-dlinear-direct", "var"]

    for key in keys:
        assert get_model_spec(key).key == key


def test_model_spec_has_description():
    spec = get_model_spec("naive-last")
    assert isinstance(spec.description, str)
    assert spec.description


def test_registry_returns_modelspec_instances() -> None:
    spec = get_model_spec("naive-last")

    assert isinstance(spec, RuntimeModelSpec)
    assert spec.interface == "local"


def test_registry_supports_historical_compatibility_imports() -> None:
    namespace: dict[str, object] = {}

    exec(
        "from foresight.models.registry import "
        "RegistryForecaster, RegistryGlobalForecaster, LocalForecasterFn, ModelFactory",
        namespace,
    )

    assert namespace["RegistryForecaster"] is BaseRegistryForecaster
    assert namespace["RegistryGlobalForecaster"] is BaseRegistryGlobalForecaster
    assert namespace["LocalForecasterFn"] is RuntimeLocalForecasterFn
    assert namespace["ModelFactory"] is RuntimeModelFactory


def test_model_spec_exposes_normalized_capabilities():
    spec = get_model_spec("naive-last")

    assert isinstance(spec.capabilities, dict)
    assert spec.capabilities["supports_x_cols"] is False
    assert spec.capabilities["supports_quantiles"] is False
    assert spec.capabilities["supports_interval_forecast"] is True
    assert spec.capabilities["supports_interval_forecast_with_x_cols"] is False
    assert spec.capabilities["supports_artifact_save"] is True
    assert spec.capabilities["requires_future_covariates"] is False


def test_model_spec_capability_overrides_can_require_future_covariates() -> None:
    spec = ModelSpec(
        key="__test__",
        description="test",
        factory=lambda **_params: None,
        param_help={"x_cols": "Required future covariates"},
        capability_overrides={"requires_future_covariates": True},
    )

    assert spec.capabilities["supports_x_cols"] is True
    assert spec.capabilities["requires_future_covariates"] is True


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

    timexer_local = get_model_spec("torch-timexer-direct")
    assert timexer_local.capabilities["supports_x_cols"] is True
    assert timexer_local.capabilities["requires_future_covariates"] is True

    timexer_global = get_model_spec("torch-timexer-global")
    assert timexer_global.capabilities["supports_x_cols"] is True
    assert timexer_global.capabilities["requires_future_covariates"] is True


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
