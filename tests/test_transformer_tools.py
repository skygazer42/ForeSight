from __future__ import annotations

import ast
from pathlib import Path


def _load_adjustment_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    path = root / "transformer time series" / "Time-Series" / "utils" / "tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keep = {"_starts_detected_anomaly_run", "_mark_anomaly_run", "adjustment"}
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in keep
    ]
    module = ast.Module(body=body, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_adjustment_marks_entire_detected_anomaly_run() -> None:
    namespace = _load_adjustment_namespace()
    adjustment = namespace["adjustment"]

    gt = [0, 1, 1, 1, 0]
    pred = [0, 0, 1, 0, 0]

    _, adjusted = adjustment(gt, pred)

    assert adjusted == [0, 1, 1, 1, 0]


def test_adjustment_leaves_undetected_anomaly_run_unchanged() -> None:
    namespace = _load_adjustment_namespace()
    adjustment = namespace["adjustment"]

    gt = [0, 1, 1, 0]
    pred = [0, 0, 0, 0]

    _, adjusted = adjustment(gt, pred)

    assert adjusted == [0, 0, 0, 0]
