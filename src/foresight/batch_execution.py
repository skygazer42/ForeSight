from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, TextIO

_cli_shared: Any | None = None


def _get_cli_shared_module() -> Any:
    from .module_cache import get_cached_module

    return get_cached_module(globals(), "_cli_shared", ".cli_shared", __package__)


@dataclass(frozen=True)
class BatchTask:
    label: str
    task_args: tuple[Any, ...]
    task_scope: str = ""
    dataset: str = ""
    model_count: int = 0
    requested_chunk_size: str = ""
    resolved_chunk_size: int = 0


@dataclass(frozen=True)
class BatchTaskStat:
    label: str
    elapsed_seconds: float
    row_count: int
    failure_count: int
    task_scope: str = ""
    dataset: str = ""
    model_count: int = 0
    requested_chunk_size: str = ""
    resolved_chunk_size: int = 0


def _coerce_timed_task_result(
    result: Any,
) -> tuple[list[dict[str, Any]], list[str], float]:
    if isinstance(result, tuple) and len(result) == 3:
        rows, errors, elapsed_seconds = result
        return list(rows), list(errors), float(elapsed_seconds)
    if isinstance(result, tuple) and len(result) == 2:
        rows, errors = result
        return list(rows), list(errors), 0.0
    raise TypeError("task result must be (rows, errors) or (rows, errors, elapsed_seconds)")


def resolve_auto_chunk_size(
    raw_chunk_size: object,
    *,
    dataset_count: int,
    model_count: int,
    jobs: int,
) -> int:
    text = str(raw_chunk_size).strip().lower()
    if text != "auto":
        chunk_size = int(raw_chunk_size)
        if chunk_size < 0:
            raise ValueError("--chunk-size must be >= 0")
        return chunk_size

    if model_count <= 1:
        return 1
    if jobs <= 1 or dataset_count >= jobs:
        return 0

    target_tasks_per_dataset = max(1, math.ceil(float(jobs) / float(max(1, dataset_count))))
    chunk_size = max(1, math.ceil(float(model_count) / float(target_tasks_per_dataset)))
    return 0 if chunk_size >= model_count else int(chunk_size)


def task_report_columns() -> list[str]:
    return [
        "task_scope",
        "dataset",
        "model_count",
        "requested_chunk_size",
        "resolved_chunk_size",
        "backend",
        "jobs",
        "chunk_size",
        "label",
        "elapsed_seconds",
        "row_count",
        "failure_count",
    ]


def task_report_rows(
    stats: list[BatchTaskStat],
    *,
    backend: str,
    jobs: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            **asdict(stat),
            "backend": str(backend),
            "jobs": int(jobs),
            "chunk_size": int(stat.resolved_chunk_size),
        }
        for stat in stats
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("task_scope", "")),
            str(row.get("dataset", "")),
            str(row.get("label", "")),
        )
    )
    return rows


def format_task_reports(rows: list[dict[str, Any]], *, fmt: str) -> str:
    return _get_cli_shared_module()._format_table(
        rows,
        columns=task_report_columns(),
        fmt=fmt,
    )


def write_task_reports(rows: list[dict[str, Any]], *, fmt: str, output: str) -> str:
    text = format_task_reports(rows, fmt=fmt)
    out_s = str(output).strip()
    if out_s:
        _get_cli_shared_module()._write_table(
            rows,
            columns=task_report_columns(),
            output=out_s,
            fmt=fmt,
        )
    return text


def emit_task_reports(
    stats: list[BatchTaskStat],
    *,
    backend: str,
    jobs: int,
    fmt: str,
    output: str,
) -> str:
    rows = task_report_rows(
        stats,
        backend=backend,
        jobs=jobs,
    )
    return write_task_reports(rows, fmt=fmt, output=output)


def record_task_errors(
    errors: list[str],
    *,
    errors_out: list[str] | None,
) -> int:
    failures = 0
    for err in errors:
        failures += 1
        print(err, file=sys.stderr)
        if errors_out is not None:
            errors_out.append(str(err))
    return failures


def build_task_executor(*, backend_s: str, max_workers: int) -> Any:
    import concurrent.futures
    import multiprocessing

    if backend_s not in {"thread", "process"}:
        raise ValueError("--backend must be one of: thread, process")

    if backend_s == "process":
        ctx = multiprocessing.get_context("spawn")
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        )

    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


def resolve_task_result(
    fut: Any,
    *,
    label: str,
    strict: bool,
    sibling_futures: Any,
    errors_out: list[str] | None,
) -> tuple[list[dict[str, Any]], int, float]:
    try:
        rows, errors, elapsed_seconds = _coerce_timed_task_result(fut.result())
        return rows, record_task_errors(errors, errors_out=errors_out), float(elapsed_seconds)
    except Exception as e:  # noqa: BLE001
        if strict:
            for other in sibling_futures:
                if other is not fut:
                    other.cancel()
            raise RuntimeError(f"{label}: {type(e).__name__}: {e}") from e

        line = f"SKIP {label}: {type(e).__name__}: {e}"
        print(line, file=sys.stderr)
        if errors_out is not None:
            errors_out.append(line)
        return [], 1, 0.0


def _run_timed_task(
    worker: Any,
    task_args: tuple[Any, ...],
    worker_args: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[str], float]:
    started = time.perf_counter()
    rows, errors = worker(*task_args, *worker_args)
    return rows, errors, float(time.perf_counter() - started)


def run_batch_tasks_sequential(
    tasks: list[BatchTask],
    *,
    progress: bool,
    strict: bool,
    worker: Any,
    worker_args: tuple[Any, ...],
    errors_out: list[str] | None,
    stats_out: list[BatchTaskStat] | None = None,
    progress_stream: TextIO | None = None,
) -> tuple[list[dict[str, Any]], int]:
    stream = sys.stderr if progress_stream is None else progress_stream
    out: list[dict[str, Any]] = []
    failures = 0
    n = len(tasks)
    for i, task in enumerate(tasks, start=1):
        started = time.perf_counter()
        try:
            rows, errors = worker(*task.task_args, *worker_args)
            elapsed_seconds = float(time.perf_counter() - started)
            task_failures = record_task_errors(errors, errors_out=errors_out)
            out.extend(rows)
            failures += task_failures
            if stats_out is not None:
                stats_out.append(
                    BatchTaskStat(
                        label=task.label,
                        elapsed_seconds=elapsed_seconds,
                        row_count=len(rows),
                        failure_count=task_failures,
                        task_scope=str(task.task_scope),
                        dataset=str(task.dataset),
                        model_count=int(task.model_count),
                        requested_chunk_size=str(task.requested_chunk_size),
                        resolved_chunk_size=int(task.resolved_chunk_size),
                    )
                )
        except Exception as e:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"{task.label}: {type(e).__name__}: {e}") from e
            elapsed_seconds = float(time.perf_counter() - started)
            failures += 1
            line = f"SKIP {task.label}: {type(e).__name__}: {e}"
            print(line, file=sys.stderr)
            if errors_out is not None:
                errors_out.append(line)
            if stats_out is not None:
                stats_out.append(
                    BatchTaskStat(
                        label=task.label,
                        elapsed_seconds=elapsed_seconds,
                        row_count=0,
                        failure_count=1,
                        task_scope=str(task.task_scope),
                        dataset=str(task.dataset),
                        model_count=int(task.model_count),
                        requested_chunk_size=str(task.requested_chunk_size),
                        resolved_chunk_size=int(task.resolved_chunk_size),
                    )
                )
        if progress:
            print(f"DONE {i}/{n} {task.label}", file=stream)
    return out, failures


def run_batch_tasks(
    tasks: list[BatchTask],
    *,
    jobs: int,
    backend: str,
    progress: bool,
    strict: bool,
    worker: Any,
    worker_args: tuple[Any, ...] = (),
    errors_out: list[str] | None = None,
    stats_out: list[BatchTaskStat] | None = None,
    progress_stream: TextIO | None = None,
) -> tuple[list[dict[str, Any]], int]:
    n = len(tasks)
    if n <= 0:
        return [], 0

    if jobs <= 1:
        return run_batch_tasks_sequential(
            tasks,
            progress=progress,
            strict=strict,
            worker=worker,
            worker_args=worker_args,
            errors_out=errors_out,
            stats_out=stats_out,
            progress_stream=progress_stream,
        )

    stream = sys.stderr if progress_stream is None else progress_stream
    max_workers = min(int(jobs), max(1, n))
    backend_s = str(backend).strip().lower()
    executor = build_task_executor(backend_s=backend_s, max_workers=max_workers)

    import concurrent.futures

    done = 0
    failures = 0
    out: list[dict[str, Any]] = []
    try:
        fut_to_task = {
            executor.submit(_run_timed_task, worker, task.task_args, worker_args): task for task in tasks
        }
        for fut in concurrent.futures.as_completed(fut_to_task):
            task = fut_to_task[fut]
            rows, task_failures, elapsed_seconds = resolve_task_result(
                fut,
                label=task.label,
                strict=strict,
                sibling_futures=fut_to_task.keys(),
                errors_out=errors_out,
            )
            out.extend(rows)
            failures += task_failures
            if stats_out is not None:
                stats_out.append(
                    BatchTaskStat(
                        label=task.label,
                        elapsed_seconds=elapsed_seconds,
                        row_count=len(rows),
                        failure_count=task_failures,
                        task_scope=str(task.task_scope),
                        dataset=str(task.dataset),
                        model_count=int(task.model_count),
                        requested_chunk_size=str(task.requested_chunk_size),
                        resolved_chunk_size=int(task.resolved_chunk_size),
                    )
                )
            done += 1
            if progress:
                print(f"DONE {done}/{n} {task.label}", file=stream)
    finally:
        executor.shutdown(cancel_futures=True)

    return out, failures


__all__ = [
    "BatchTask",
    "BatchTaskStat",
    "build_task_executor",
    "format_task_reports",
    "emit_task_reports",
    "record_task_errors",
    "resolve_task_result",
    "resolve_auto_chunk_size",
    "task_report_columns",
    "task_report_rows",
    "write_task_reports",
    "run_batch_tasks",
    "run_batch_tasks_sequential",
]
