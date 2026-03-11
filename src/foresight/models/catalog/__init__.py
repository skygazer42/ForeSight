from __future__ import annotations

from typing import Any

from .classical import build_classical_catalog
from .foundation import build_foundation_catalog
from .ml import build_ml_catalog
from .multivariate import build_multivariate_catalog
from .stats import build_stats_catalog
from .torch_global import build_torch_global_catalog
from .torch_local import build_torch_local_catalog

_SHARD_BUILDERS = (
    ("classical", build_classical_catalog),
    ("ml", build_ml_catalog),
    ("stats", build_stats_catalog),
    ("torch_local", build_torch_local_catalog),
    ("torch_global", build_torch_global_catalog),
    ("multivariate", build_multivariate_catalog),
    ("foundation", build_foundation_catalog),
)


def build_catalog(context: Any) -> dict[str, Any]:
    catalog: dict[str, Any] = {}
    for shard_name, builder in _SHARD_BUILDERS:
        shard = builder(context)
        clash = set(shard).intersection(catalog)
        if clash:
            raise RuntimeError(
                f"Internal error: model key collision(s) while composing {shard_name}: {sorted(clash)}"
            )
        catalog.update(shard)
    return catalog


__all__ = ["build_catalog"]
