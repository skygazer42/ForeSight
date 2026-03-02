from foresight.models.registry import get_model_spec, list_models


def test_list_models_contains_expected_keys():
    keys = set(list_models())
    assert "naive-last" in keys
    assert "seasonal-naive" in keys


def test_model_spec_has_description():
    spec = get_model_spec("naive-last")
    assert isinstance(spec.description, str)
    assert spec.description


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


def test_new_torch_xformer_attention_variants_are_registered():
    keys = set(list_models())
    assert "torch-xformer-probsparse-ln-gelu-direct" in keys
    assert "torch-xformer-autocorr-ln-gelu-direct" in keys
    assert "torch-xformer-reformer-ln-gelu-direct" in keys
    assert "torch-xformer-probsparse-global" in keys
    assert "torch-xformer-autocorr-global" in keys
    assert "torch-xformer-reformer-global" in keys
