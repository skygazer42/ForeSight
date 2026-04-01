from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class BatchTask:
    label: str
    task_args: tuple[Any, ...]
    task_scope: str = ""
    dataset: str = ""
    model_count: int = 0


@dataclass(frozen=True)
class BatchTaskStat:
    label: str
    elapsed_seconds: float
    row_count: int
    failure_count: int
    task_scope: str = ""
    dataset: str = ""
    model_count: int = 0


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
    "record_task_errors",
    "resolve_task_result",
    "run_batch_tasks",
    "run_batch_tasks_sequential",
]
