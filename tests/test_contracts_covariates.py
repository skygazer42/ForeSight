from __future__ import annotations

from foresight.contracts.covariates import (
    CovariateSpec,
    resolve_covariate_roles,
    resolve_model_param_covariates,
)


def test_resolve_model_param_covariates_merges_legacy_x_cols() -> None:
    spec = resolve_model_param_covariates(
        {"x_cols": "promo,price", "historic_x_cols": ("stock",)}
    )

    assert isinstance(spec, CovariateSpec)
    assert spec.historic_x_cols == ("stock",)
    assert spec.future_x_cols == ("promo", "price")
    assert spec.all_x_cols == ("stock", "promo", "price")


def test_resolve_covariate_roles_deduplicates_merged_future_columns() -> None:
    spec = resolve_covariate_roles(
        x_cols=("promo", "price"),
        historic_x_cols=("stock", "promo"),
        future_x_cols=("price", "temp"),
    )

    assert isinstance(spec, CovariateSpec)
    assert spec.historic_x_cols == ("stock", "promo")
    assert spec.future_x_cols == ("price", "temp", "promo")
    assert spec.all_x_cols == ("stock", "promo", "price", "temp")
