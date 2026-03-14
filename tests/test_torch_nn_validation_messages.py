from foresight.models import torch_nn


def test_torch_nn_exposes_shared_validation_messages() -> None:
    assert torch_nn._HIDDEN_SIZE_MIN_MSG == "hidden_size must be >= 1"
    assert torch_nn._NUM_LAYERS_MIN_MSG == "num_layers must be >= 1"
    assert torch_nn._DROPOUT_RANGE_MSG == "dropout must be in [0, 1)"
    assert torch_nn._HORIZON_MIN_MSG == "horizon must be >= 1"


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
