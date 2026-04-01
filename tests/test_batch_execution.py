from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def _load_batch_execution_module(repo_root: Path):
    path = repo_root / "src" / "foresight" / "batch_execution.py"
    assert path.exists()
    return importlib.import_module("foresight.batch_execution")


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


def test_make_task_stat_copies_batch_task_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)

    task = mod.BatchTask(
        label="catfish/[2 models]",
        task_args=("catfish", ("naive-last", "mean")),
        task_scope="leaderboard_sweep",
        dataset="catfish",
        model_count=2,
        requested_chunk_size="auto",
        resolved_chunk_size=2,
    )

    stat = mod._make_task_stat(  # type: ignore[attr-defined]
        task,
        elapsed_seconds=0.25,
        row_count=2,
        failure_count=1,
    )

    assert stat == mod.BatchTaskStat(
        label="catfish/[2 models]",
        elapsed_seconds=0.25,
        row_count=2,
        failure_count=1,
        task_scope="leaderboard_sweep",
        dataset="catfish",
        model_count=2,
        requested_chunk_size="auto",
        resolved_chunk_size=2,
    )


def test_task_report_rows_add_command_level_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)

    rows = mod.task_report_rows(
        [
            mod.BatchTaskStat(
                label="ice_cream_interest/[2 models]",
                elapsed_seconds=0.5,
                row_count=2,
                failure_count=0,
                task_scope="leaderboard_sweep",
                dataset="ice_cream_interest",
                model_count=2,
                requested_chunk_size="auto",
                resolved_chunk_size=2,
            ),
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
            ),
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
        },
        {
            "task_scope": "leaderboard_sweep",
            "dataset": "ice_cream_interest",
            "model_count": 2,
            "requested_chunk_size": "auto",
            "resolved_chunk_size": 2,
            "backend": "thread",
            "jobs": 4,
            "chunk_size": 2,
            "label": "ice_cream_interest/[2 models]",
            "elapsed_seconds": 0.5,
            "row_count": 2,
            "failure_count": 0,
        }
    ]


def test_format_task_reports_supports_json_csv_and_markdown() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    rows = [
        {
            "task_scope": "benchmark",
            "dataset": "catfish",
            "model_count": 3,
            "requested_chunk_size": "auto",
            "resolved_chunk_size": 0,
            "backend": "thread",
            "jobs": 2,
            "chunk_size": 0,
            "label": "catfish/[3 models]",
            "elapsed_seconds": 0.5,
            "row_count": 3,
            "failure_count": 0,
        }
    ]

    json_text = mod.format_task_reports(rows, fmt="json")
    csv_text = mod.format_task_reports(rows, fmt="csv")
    md_text = mod.format_task_reports(rows, fmt="md")

    assert '"task_scope": "benchmark"' in json_text
    assert csv_text.splitlines()[0].startswith("task_scope,dataset,model_count,")
    assert "| task_scope | dataset | model_count |" in md_text


def test_format_task_reports_uses_shared_cli_formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    calls: list[tuple[list[dict[str, Any]], list[str], str]] = []

    class _FakeCliShared:
        @staticmethod
        def _format_table(rows: list[dict[str, Any]], *, columns: list[str], fmt: str) -> str:
            calls.append((rows, columns, fmt))
            return "formatted"

    monkeypatch.setattr(mod, "_get_cli_shared_module", lambda: _FakeCliShared())

    text = mod.format_task_reports(
        [
            {
                "task_scope": "benchmark",
                "dataset": "catfish",
                "model_count": 3,
                "requested_chunk_size": "auto",
                "resolved_chunk_size": 0,
                "backend": "thread",
                "jobs": 2,
                "chunk_size": 0,
                "label": "catfish/[3 models]",
                "elapsed_seconds": 0.5,
                "row_count": 3,
                "failure_count": 0,
            }
        ],
        fmt="md",
    )

    assert text == "formatted"
    assert calls == [
        (
            [
                {
                    "task_scope": "benchmark",
                    "dataset": "catfish",
                    "model_count": 3,
                    "requested_chunk_size": "auto",
                    "resolved_chunk_size": 0,
                    "backend": "thread",
                    "jobs": 2,
                    "chunk_size": 0,
                    "label": "catfish/[3 models]",
                    "elapsed_seconds": 0.5,
                    "row_count": 3,
                    "failure_count": 0,
                }
            ],
            mod.task_report_columns(),
            "md",
        )
    ]


def test_write_task_reports_formats_and_writes_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    rows = [
        {
            "task_scope": "benchmark",
            "dataset": "catfish",
            "model_count": 3,
            "requested_chunk_size": "auto",
            "resolved_chunk_size": 0,
            "backend": "thread",
            "jobs": 2,
            "chunk_size": 0,
            "label": "catfish/[3 models]",
            "elapsed_seconds": 0.5,
            "row_count": 3,
            "failure_count": 0,
        }
    ]
    out = tmp_path / "task-reports.md"

    text = mod.write_task_reports(rows, fmt="md", output=str(out))

    assert "| task_scope | dataset | model_count |" in text
    assert out.read_text(encoding="utf-8") == text + "\n"


def test_write_task_reports_uses_shared_cli_output_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    writes: list[tuple[str, str]] = []

    class _FakeCliShared:
        @staticmethod
        def _format_table(rows: list[dict[str, Any]], *, columns: list[str], fmt: str) -> str:
            return "formatted-json"

        @staticmethod
        def _write_table(rows: list[dict[str, Any]], *, columns: list[str], output: str, fmt: str) -> str:
            writes.append((output, "formatted-json"))
            return "formatted-json"

    monkeypatch.setattr(mod, "_get_cli_shared_module", lambda: _FakeCliShared())

    text = mod.write_task_reports(
        [
            {
                "task_scope": "benchmark",
                "dataset": "catfish",
                "model_count": 3,
                "requested_chunk_size": "auto",
                "resolved_chunk_size": 0,
                "backend": "thread",
                "jobs": 2,
                "chunk_size": 0,
                "label": "catfish/[3 models]",
                "elapsed_seconds": 0.5,
                "row_count": 3,
                "failure_count": 0,
            }
        ],
        fmt="json",
        output="/tmp/fake-task-report.json",
    )

    assert text == "formatted-json"
    assert writes == [("/tmp/fake-task-report.json", "formatted-json")]


def test_emit_task_reports_builds_rows_and_writes_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)
    out = tmp_path / "task-reports.json"

    text = mod.emit_task_reports(
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
        fmt="json",
        output=str(out),
    )

    assert '"task_scope": "leaderboard_sweep"' in text
    assert out.read_text(encoding="utf-8") == text + "\n"


def test_resolve_auto_chunk_size_balances_jobs_and_datasets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_batch_execution_module(repo_root)

    assert mod.resolve_auto_chunk_size("0", dataset_count=2, model_count=4, jobs=2) == 0
    assert mod.resolve_auto_chunk_size("3", dataset_count=2, model_count=5, jobs=4) == 3
    assert mod.resolve_auto_chunk_size("auto", dataset_count=2, model_count=4, jobs=2) == 0
    assert mod.resolve_auto_chunk_size("auto", dataset_count=1, model_count=5, jobs=4) == 2
    assert mod.resolve_auto_chunk_size("auto", dataset_count=1, model_count=1, jobs=4) == 1
