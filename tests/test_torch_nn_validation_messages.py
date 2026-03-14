from foresight.models import torch_nn


def test_torch_nn_exposes_shared_validation_messages() -> None:
    assert torch_nn._HIDDEN_SIZE_MIN_MSG == "hidden_size must be >= 1"
    assert torch_nn._NUM_LAYERS_MIN_MSG == "num_layers must be >= 1"
    assert torch_nn._DROPOUT_RANGE_MSG == "dropout must be in [0, 1)"
    assert torch_nn._HORIZON_MIN_MSG == "horizon must be >= 1"
    assert torch_nn._LAGS_MIN_MSG == "lags must be >= 1"
    assert torch_nn._NUM_BLOCKS_MIN_MSG == "num_blocks must be >= 1"
    assert torch_nn._D_MODEL_MIN_MSG == "d_model must be >= 1"
    assert torch_nn._NHEAD_MIN_MSG == "nhead must be >= 1"
    assert torch_nn._D_MODEL_HEAD_DIVISIBILITY_MSG == "d_model must be divisible by nhead"
    assert torch_nn._DIM_FEEDFORWARD_MIN_MSG == "dim_feedforward must be >= 1"
    assert torch_nn._PATCH_LEN_MIN_MSG == "patch_len must be >= 1"
    assert torch_nn._SEGMENT_LEN_MIN_MSG == "segment_len must be >= 1"
    assert torch_nn._SEGMENT_LEN_MAX_LAGS_MSG == "segment_len must be <= lags"


def test_manual_recurrent_builders_use_shared_validation_messages() -> None:
    try:
        torch_nn._make_manual_gru_cell(input_size=1, hidden_size=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._HIDDEN_SIZE_MIN_MSG
    else:
        raise AssertionError("expected hidden_size validation error")

    try:
        torch_nn._make_manual_gru(input_size=1, hidden_size=1, num_layers=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._NUM_LAYERS_MIN_MSG
    else:
        raise AssertionError("expected num_layers validation error")

    try:
        torch_nn._make_manual_lstm(input_size=1, hidden_size=1, num_layers=1, dropout=1.0)
    except ValueError as exc:
        assert str(exc) == torch_nn._DROPOUT_RANGE_MSG
    else:
        raise AssertionError("expected dropout validation error")


def test_lagged_xy_builder_uses_shared_horizon_validation_message() -> None:
    try:
        torch_nn._make_lagged_xy_multi(torch_nn.np.arange(10.0), lags=3, horizon=0)
    except ValueError as exc:
        assert str(exc) == torch_nn._HORIZON_MIN_MSG
    else:
        raise AssertionError("expected horizon validation error")

    try:
        torch_nn._make_lagged_xy_multi(torch_nn.np.arange(10.0), lags=0, horizon=1)
    except ValueError as exc:
        assert str(exc) == torch_nn._LAGS_MIN_MSG
    else:
        raise AssertionError("expected lags validation error")


def test_torch_nn_direct_forecasters_use_shared_structural_validation_messages() -> None:
    train = torch_nn.np.arange(20.0)

    cases = [
        (
            torch_nn.torch_nbeats_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "num_blocks": 0},
            "_NUM_BLOCKS_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "d_model": 0},
            "_D_MODEL_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "nhead": 0},
            "_NHEAD_MIN_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "d_model": 5, "nhead": 2},
            "_D_MODEL_HEAD_DIVISIBILITY_MSG",
        ),
        (
            torch_nn.torch_transformer_direct_forecast,
            {
                "train": train,
                "horizon": 1,
                "lags": 4,
                "d_model": 4,
                "nhead": 2,
                "dim_feedforward": 0,
            },
            "_DIM_FEEDFORWARD_MIN_MSG",
        ),
        (
            torch_nn.torch_patchtst_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "patch_len": 0},
            "_PATCH_LEN_MIN_MSG",
        ),
        (
            torch_nn.torch_crossformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "segment_len": 0},
            "_SEGMENT_LEN_MIN_MSG",
        ),
        (
            torch_nn.torch_crossformer_direct_forecast,
            {"train": train, "horizon": 1, "lags": 4, "segment_len": 5},
            "_SEGMENT_LEN_MAX_LAGS_MSG",
        ),
    ]

    for fn, kwargs, expected_attr in cases:
        try:
            fn(**kwargs)
        except ValueError as exc:
            assert str(exc) == getattr(torch_nn, expected_attr)
        else:
            raise AssertionError(f"expected validation error for {fn.__name__}")
