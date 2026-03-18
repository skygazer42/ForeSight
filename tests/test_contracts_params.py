from __future__ import annotations

import pytest

from foresight.contracts.capabilities import require_x_cols_if_needed
from foresight.contracts.params import (
    normalize_covariate_roles,
    normalize_static_cols,
    normalize_x_cols,
    parse_interval_levels,
    required_quantiles_for_interval_levels,
)


def test_normalize_covariate_roles_merges_legacy_x_cols() -> None:
    historic, future = normalize_covariate_roles(
        {"x_cols": "promo,price", "historic_x_cols": ("stock",), "static_cols": ("store_size",)}
    )

    assert historic == ("stock",)
    assert future == ("promo", "price")


def test_parse_interval_levels_accepts_percent_inputs() -> None:
    assert parse_interval_levels("80,90") == (0.8, 0.9)


def test_normalize_x_cols_prefers_canonical_future_x_cols_on_model_params_dict() -> None:
    assert normalize_x_cols({"future_x_cols": "promo,price"}) == ("promo", "price")


def test_normalize_static_cols_supports_model_params_dict_and_raw_values() -> None:
    assert normalize_static_cols({"static_cols": "store_size,region_code"}) == (
        "store_size",
        "region_code",
    )
    assert normalize_static_cols(("store_size", "region_code")) == ("store_size", "region_code")


def test_required_quantiles_for_interval_levels_normalizes_percentage_inputs() -> None:
    assert required_quantiles_for_interval_levels((80, 90)) == (0.05, 0.1, 0.5, 0.9, 0.95)


def test_require_x_cols_if_needed_rejects_missing_required_covariates() -> None:
    with pytest.raises(
        ValueError,
        match=r"Model 'demo' requires future covariates via x_cols in forecast_model_long_df",
    ):
        require_x_cols_if_needed(
            model="demo",
            capabilities={"requires_future_covariates": True},
            x_cols=(),
            context="forecast_model_long_df",
        )
