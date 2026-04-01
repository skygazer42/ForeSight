from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


def _load_batch_execution_module(repo_root: Path):
    path = repo_root / "src" / "foresight" / "batch_execution.py"
    assert path.exists()
    spec = importlib.util.spec_from_file_location("foresight.batch_execution", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_batch_tasks_sequentially_returns_rows_in_completion_order() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    calls: list[tuple[str, tuple[str, ...], str]] = []

    def _worker(dataset: str, model_keys: tuple[str, ...], suffix: str) -> tuple[list[dict[str, Any]], list[str]]:
        calls.append((dataset, model_keys, suffix))
        return ([{"label": f"{dataset}:{','.join(model_keys)}:{suffix}"}], [])

    tasks = [
        mod.BatchTask(label="catfish/[2 models]", task_args=("catfish", ("naive-last", "mean"))),
        mod.BatchTask(label="ice_cream_interest/naive-last", task_args=("ice_cream_interest", ("naive-last",))),
    ]

    rows, failures = mod.run_batch_tasks(
        tasks,
        jobs=1,
        backend="thread",
        progress=False,
        strict=False,
        worker=_worker,
        worker_args=("suffix",),
    )

    assert calls == [
        ("catfish", ("naive-last", "mean"), "suffix"),
        ("ice_cream_interest", ("naive-last",), "suffix"),
    ]
    assert failures == 0
    assert [row["label"] for row in rows] == [
        "catfish:naive-last,mean:suffix",
        "ice_cream_interest:naive-last:suffix",
    ]


def test_run_batch_tasks_non_strict_collects_worker_exceptions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    errors_out: list[str] = []

    def _worker(dataset: str, model_keys: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[str]]:
        if dataset == "broken":
            raise ValueError("boom")
        return ([{"dataset": dataset, "models": list(model_keys)}], [f"WARN {dataset}"])

    tasks = [
        mod.BatchTask(label="ok/naive-last", task_args=("ok", ("naive-last",))),
        mod.BatchTask(label="broken/mean", task_args=("broken", ("mean",))),
    ]

    rows, failures = mod.run_batch_tasks(
        tasks,
        jobs=1,
        backend="thread",
        progress=False,
        strict=False,
        worker=_worker,
        errors_out=errors_out,
    )

    assert rows == [{"dataset": "ok", "models": ["naive-last"]}]
    assert failures == 2
    assert errors_out == [
        "WARN ok",
        "SKIP broken/mean: ValueError: boom",
    ]


def test_run_batch_tasks_strict_raises_labeled_runtime_error() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)

    def _worker(dataset: str, model_keys: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[str]]:
        raise KeyError(f"Unknown dataset {dataset} for {model_keys!r}")

    tasks = [
        mod.BatchTask(label="catfish/naive-last", task_args=("catfish", ("naive-last",))),
    ]

    with pytest.raises(RuntimeError, match="catfish/naive-last: KeyError"):
        mod.run_batch_tasks(
            tasks,
            jobs=1,
            backend="thread",
            progress=False,
            strict=True,
            worker=_worker,
        )


def test_run_batch_tasks_can_collect_task_stats() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    stats_out: list[object] = []

    def _worker(dataset: str, model_keys: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[str]]:
        return (
            [{"dataset": dataset, "model": key} for key in model_keys],
            [f"WARN {dataset}"] if dataset == "catfish" else [],
        )

    rows, failures = mod.run_batch_tasks(
        [
            mod.BatchTask(
                label="catfish/[2 models]",
                task_args=("catfish", ("naive-last", "mean")),
                task_scope="leaderboard_sweep",
                dataset="catfish",
                model_count=2,
                requested_chunk_size="auto",
                resolved_chunk_size=2,
            ),
            mod.BatchTask(
                label="ice_cream_interest/naive-last",
                task_args=("ice_cream_interest", ("naive-last",)),
                task_scope="leaderboard_sweep",
                dataset="ice_cream_interest",
                model_count=1,
                requested_chunk_size="auto",
                resolved_chunk_size=2,
            ),
        ],
        jobs=1,
        backend="thread",
        progress=False,
        strict=False,
        worker=_worker,
        stats_out=stats_out,
    )

    assert failures == 1
    assert len(rows) == 3
    assert [stat.label for stat in stats_out] == [
        "catfish/[2 models]",
        "ice_cream_interest/naive-last",
    ]
    assert [stat.task_scope for stat in stats_out] == [
        "leaderboard_sweep",
        "leaderboard_sweep",
    ]
    assert [stat.dataset for stat in stats_out] == [
        "catfish",
        "ice_cream_interest",
    ]
    assert [stat.model_count for stat in stats_out] == [2, 1]
    assert [stat.requested_chunk_size for stat in stats_out] == ["auto", "auto"]
    assert [stat.resolved_chunk_size for stat in stats_out] == [2, 2]
    assert [stat.row_count for stat in stats_out] == [2, 1]
    assert [stat.failure_count for stat in stats_out] == [1, 0]
    assert all(float(stat.elapsed_seconds) >= 0.0 for stat in stats_out)


def test_task_report_rows_add_command_level_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)

    rows = mod.task_report_rows(
        [
            mod.BatchTaskStat(
                label="catfish/[2 models]",
                elapsed_seconds=0.25,
                row_count=2,
                failure_count=0,
                task_scope="leaderboard_sweep",
                dataset="catfish",
                model_count=2,
                requested_chunk_size="auto",
                resolved_chunk_size=2,
            )
        ],
        backend="thread",
        jobs=4,
    )

    assert rows == [
        {
            "task_scope": "leaderboard_sweep",
            "dataset": "catfish",
            "model_count": 2,
            "requested_chunk_size": "auto",
            "resolved_chunk_size": 2,
            "backend": "thread",
            "jobs": 4,
            "chunk_size": 2,
            "label": "catfish/[2 models]",
            "elapsed_seconds": 0.25,
            "row_count": 2,
            "failure_count": 0,
        }
    ]
