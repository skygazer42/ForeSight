from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _normalize_hierarchy(hierarchy: Mapping[str, Sequence[str]] | dict[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(hierarchy, Mapping):
        raise TypeError("hierarchy must be a mapping of parent -> children")

    out: dict[str, tuple[str, ...]] = {}
    for parent, children in hierarchy.items():
        parent_s = str(parent).strip()
        if not parent_s:
            raise ValueError("hierarchy contains an empty parent label")
        child_items = tuple(str(child).strip() for child in children if str(child).strip())
        if not child_items:
            raise ValueError(f"hierarchy parent {parent_s!r} must have at least one child")
        if len(set(child_items)) != len(child_items):
            raise ValueError(f"hierarchy parent {parent_s!r} contains duplicate children")
        out[parent_s] = child_items

    if not out:
        raise ValueError("hierarchy must be non-empty")

    parents = set(out)
    children = {child for items in out.values() for child in items}
    roots = parents.difference(children)
    if not roots:
        raise ValueError("hierarchy must contain at least one root node")

    def _visit(node: str, stack: tuple[str, ...]) -> None:
        if node in stack:
            cycle = " -> ".join([*stack, node])
            raise ValueError(f"hierarchy contains a cycle: {cycle}")
        for child in out.get(node, ()):
            _visit(child, (*stack, node))

    for root in roots:
        _visit(root, ())

    return out


def _require_tidy_df(df: Any, *, value_col: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("forecast_df must be a pandas DataFrame")
    required = {"unique_id", "ds", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df.duplicated(subset=["unique_id", "ds"]).any():
        raise ValueError("DataFrame contains duplicate (unique_id, ds) rows")
    return df.copy()


def _all_nodes(hierarchy: dict[str, tuple[str, ...]]) -> set[str]:
    return set(hierarchy).union({child for children in hierarchy.values() for child in children})


def _roots(hierarchy: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    parents = set(hierarchy)
    children = {child for items in hierarchy.values() for child in items}
    return tuple(sorted(parents.difference(children)))


def _leaf_nodes(hierarchy: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(sorted(_all_nodes(hierarchy).difference(hierarchy)))


def _node_order(hierarchy: dict[str, tuple[str, ...]]) -> list[str]:
    order: list[str] = []

    def _walk(node: str) -> None:
        if node in order:
            return
        order.append(node)
        for child in hierarchy.get(node, ()):
            _walk(child)

    for root in _roots(hierarchy):
        _walk(root)

    return order


def _pivot_values(df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    out = df.pivot(index="ds", columns="unique_id", values=value_col)
    out = out.sort_index(axis=0).sort_index(axis=1)
    return out


def _historical_totals(history_df: pd.DataFrame, hierarchy: dict[str, tuple[str, ...]]) -> dict[str, float]:
    pivot = _pivot_values(history_df, value_col="y")
    totals: dict[str, float] = {}

    def _total(node: str) -> float:
        if node in totals:
            return totals[node]
        if node in hierarchy:
            value = float(sum(_total(child) for child in hierarchy[node]))
        else:
            if node not in pivot.columns:
                raise ValueError(f"history_df missing required leaf node: {node!r}")
            value = float(np.nansum(pivot[node].to_numpy(dtype=float, copy=False)))
        totals[node] = value
        return value

    for root in _roots(hierarchy):
        _total(root)

    return totals


def _normalize_exog_agg(
    exog_agg: Mapping[str, str] | dict[str, str] | None,
    *,
    columns: Sequence[str],
    reserved: Sequence[str],
) -> dict[str, str]:
    if exog_agg is None:
        return {}
    if not isinstance(exog_agg, Mapping):
        raise TypeError("exog_agg must be a mapping of column -> aggregation")

    allowed = {"sum", "mean", "min", "max"}
    reserved_set = {str(col) for col in reserved}
    available = {str(col) for col in columns}
    out: dict[str, str] = {}
    for col, agg in exog_agg.items():
        col_s = str(col).strip()
        agg_s = str(agg).strip().lower()
        if not col_s:
            raise ValueError("exog_agg contains an empty column name")
        if col_s in reserved_set:
            raise ValueError(f"exog_agg cannot target reserved column: {col_s!r}")
        if col_s not in available:
            raise KeyError(f"forecast_df missing exogenous column required by exog_agg: {col_s!r}")
        if agg_s not in allowed:
            raise ValueError(
                f"exog_agg contains unknown aggregation for column {col_s!r}: {agg_s!r}. "
                f"Allowed: {sorted(allowed)}"
            )
        out[col_s] = agg_s
    return out


def _aggregate_series(parts: Sequence[pd.Series], *, agg: str) -> pd.Series:
    if not parts:
        raise ValueError("aggregate_series requires at least one child series")
    if len(parts) == 1:
        return parts[0].astype(float)

    if agg == "sum":
        total = parts[0].astype(float).copy()
        for part in parts[1:]:
            total = total.add(part.astype(float), fill_value=0.0)
        return total.astype(float)

    frame = pd.concat([part.astype(float) for part in parts], axis=1)
    if agg == "mean":
        return frame.mean(axis=1).astype(float)
    if agg == "min":
        return frame.min(axis=1).astype(float)
    if agg == "max":
        return frame.max(axis=1).astype(float)
    raise ValueError(f"Unknown aggregation: {agg!r}")


def _reconcile_exog_bottom_up(
    *,
    forecast_df: pd.DataFrame,
    hierarchy: dict[str, tuple[str, ...]],
    exog_agg: dict[str, str],
    node_order: Sequence[str],
) -> pd.DataFrame:
    if not exog_agg:
        return forecast_df.loc[:, ["unique_id", "ds"]].drop_duplicates().reset_index(drop=True)

    leaves = _leaf_nodes(hierarchy)
    nodes = _all_nodes(hierarchy)
    merged: pd.DataFrame | None = None

    for col, agg in exog_agg.items():
        pivot = _pivot_values(forecast_df, value_col=col)
        missing_leaves = [node for node in leaves if node not in pivot.columns]
        if missing_leaves:
            raise ValueError(f"bottom_up exog aggregation for {col!r} requires leaf forecasts for: {missing_leaves}")

        cache: dict[str, pd.Series] = {}

        def _series(node: str) -> pd.Series:
            if node in cache:
                return cache[node]
            if node in hierarchy:
                cache[node] = _aggregate_series([_series(child) for child in hierarchy[node]], agg=agg)
            else:
                cache[node] = pivot[node].astype(float)
            return cache[node]

        rows: list[dict[str, Any]] = []
        for node in node_order:
            for ds, value in _series(node).items():
                rows.append({"unique_id": node, "ds": ds, col: float(value)})

        extras = [str(node) for node in pivot.columns if str(node) not in nodes]
        for node in extras:
            for ds, value in pivot[node].items():
                rows.append({"unique_id": node, "ds": ds, col: float(value)})

        col_df = pd.DataFrame(rows)
        if merged is None:
            merged = col_df
        else:
            merged = merged.merge(col_df, on=["unique_id", "ds"], how="outer")

    if merged is None:
        return forecast_df.loc[:, ["unique_id", "ds"]].drop_duplicates().reset_index(drop=True)
    return merged.sort_values(["ds", "unique_id"], kind="mergesort").reset_index(drop=True)


def reconcile_hierarchical_forecasts(
    *,
    forecast_df: Any,
    hierarchy: Mapping[str, Sequence[str]] | dict[str, Sequence[str]],
    method: str,
    history_df: Any = None,
    yhat_col: str = "yhat",
    exog_agg: Mapping[str, str] | dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Reconcile a tidy forecast table to satisfy a hierarchy.

    Supported methods:
      - `bottom_up`: keep leaf forecasts and aggregate upward
      - `top_down`: split parent forecasts to descendants using historical proportions
    """
    hierarchy_norm = _normalize_hierarchy(hierarchy)
    fc = _require_tidy_df(forecast_df, value_col=str(yhat_col))
    value_col = str(yhat_col)
    method_key = str(method).strip().lower()
    if method_key not in {"bottom_up", "top_down"}:
        raise ValueError("method must be 'bottom_up' or 'top_down'")
    exog_agg_norm = _normalize_exog_agg(
        exog_agg,
        columns=fc.columns,
        reserved=("unique_id", "ds", value_col, "y"),
    )
    if exog_agg_norm and method_key != "bottom_up":
        raise ValueError("exog_agg is currently only supported for bottom_up reconciliation")

    pivot = _pivot_values(fc, value_col=value_col)
    node_order = _node_order(hierarchy_norm)
    nodes = _all_nodes(hierarchy_norm)

    if method_key == "bottom_up":
        leaves = _leaf_nodes(hierarchy_norm)
        missing_leaves = [node for node in leaves if node not in pivot.columns]
        if missing_leaves:
            raise ValueError(f"bottom_up requires leaf forecasts for: {missing_leaves}")

        cache: dict[str, pd.Series] = {}

        def _series(node: str) -> pd.Series:
            if node in cache:
                return cache[node]
            if node in hierarchy_norm:
                parts = [_series(child) for child in hierarchy_norm[node]]
                total = parts[0].copy()
                for part in parts[1:]:
                    total = total.add(part, fill_value=0.0)
                out = total
            else:
                out = pivot[node].astype(float)
            cache[node] = out
            return out

        rows: list[dict[str, Any]] = []
        for node in node_order:
            series = _series(node)
            for ds, value in series.items():
                rows.append({"unique_id": node, "ds": ds, value_col: float(value)})

        extras = [col for col in pivot.columns if col not in nodes]
        for node in extras:
            for ds, value in pivot[node].items():
                rows.append({"unique_id": str(node), "ds": ds, value_col: float(value)})

        out = pd.DataFrame(rows).sort_values(["ds", "unique_id"], kind="mergesort").reset_index(drop=True)
        if exog_agg_norm:
            exog_df = _reconcile_exog_bottom_up(
                forecast_df=fc,
                hierarchy=hierarchy_norm,
                exog_agg=exog_agg_norm,
                node_order=node_order,
            )
            out = out.merge(exog_df, on=["unique_id", "ds"], how="left")
        return out

    hist = _require_tidy_df(history_df, value_col="y")
    totals = _historical_totals(hist, hierarchy_norm)

    provided_nodes = [str(col) for col in pivot.columns.tolist()]
    missing_roots = [root for root in _roots(hierarchy_norm) if root not in pivot.columns]
    if missing_roots:
        raise ValueError(f"top_down requires root forecasts for: {missing_roots}")

    results: dict[str, pd.Series] = {}

    def _allocate(node: str, values: pd.Series) -> None:
        results[node] = values.astype(float)
        children = hierarchy_norm.get(node)
        if not children:
            return

        denom = float(sum(totals[child] for child in children))
        if abs(denom) < 1e-12:
            raise ValueError(f"Cannot compute top_down proportions for parent {node!r}: zero history")

        for child in children:
            share = float(totals[child]) / denom
            _allocate(child, values * share)

    for root in _roots(hierarchy_norm):
        _allocate(root, pivot[root].astype(float))

    rows = []
    for node in node_order:
        series = results[node]
        for ds, value in series.items():
            rows.append({"unique_id": node, "ds": ds, value_col: float(value)})

    extras = [node for node in provided_nodes if node not in nodes]
    for node in extras:
        for ds, value in pivot[node].items():
            rows.append({"unique_id": node, "ds": ds, value_col: float(value)})

    return pd.DataFrame(rows).sort_values(["ds", "unique_id"], kind="mergesort").reset_index(drop=True)


def check_hierarchical_consistency(
    forecast_df: Any,
    *,
    hierarchy: Mapping[str, Sequence[str]] | dict[str, Sequence[str]],
    yhat_col: str = "yhat",
    atol: float = 1e-8,
) -> dict[str, Any]:
    """
    Check whether parent forecasts equal the sum of their direct children at each timestamp.
    """
    hierarchy_norm = _normalize_hierarchy(hierarchy)
    fc = _require_tidy_df(forecast_df, value_col=str(yhat_col))
    pivot = _pivot_values(fc, value_col=str(yhat_col))
    value_col = str(yhat_col)

    inconsistencies: list[dict[str, Any]] = []
    for parent, children in hierarchy_norm.items():
        if parent not in pivot.columns:
            inconsistencies.append(
                {"unique_id": parent, "ds": None, "reason": "missing_parent", value_col: None}
            )
            continue
        missing_children = [child for child in children if child not in pivot.columns]
        if missing_children:
            for child in missing_children:
                inconsistencies.append(
                    {"unique_id": parent, "child": child, "ds": None, "reason": "missing_child"}
                )
            continue

        child_sum = pivot.loc[:, list(children)].sum(axis=1)
        diff = pivot[parent] - child_sum
        bad_mask = np.abs(diff.to_numpy(dtype=float, copy=False)) > float(atol)
        for ds, parent_value, child_value, delta in zip(
            pivot.index[bad_mask],
            pivot[parent][bad_mask],
            child_sum[bad_mask],
            diff[bad_mask],
            strict=True,
        ):
            inconsistencies.append(
                {
                    "unique_id": parent,
                    "ds": ds,
                    "reason": "sum_mismatch",
                    "parent_value": float(parent_value),
                    "children_sum": float(child_value),
                    "diff": float(delta),
                }
            )

    return {
        "is_consistent": len(inconsistencies) == 0,
        "n_inconsistencies": int(len(inconsistencies)),
        "inconsistencies": inconsistencies,
    }
