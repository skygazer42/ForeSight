from __future__ import annotations

import importlib


def test_adapters_namespace_exports_beta_bridge_symbols() -> None:
    adapters = importlib.import_module("foresight.adapters")

    assert sorted(adapters.__all__) == [
        "SktimeForecasterAdapter",
        "from_darts_timeseries",
        "make_sktime_forecaster_adapter",
        "to_darts_timeseries",
        "to_gluonts_list_dataset",
    ]


def test_root_package_does_not_promote_adapter_symbols_into_stable_surface() -> None:
    root = importlib.import_module("foresight")

    assert "make_sktime_forecaster_adapter" not in root.__all__
    assert "to_darts_timeseries" not in root.__all__
    assert "to_gluonts_list_dataset" not in root.__all__
