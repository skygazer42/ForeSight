from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import os
import sys
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.text import Text
except Exception:  # noqa: BLE001
    Console = None
    Text = None

LOG_STYLE_CHOICES = ("auto", "rich", "plain", "quiet")
LOG_LEVEL_CHOICES = ("info", "debug")

_LEVEL_RANK = {
    "debug": 10,
    "info": 20,
    "error": 30,
}
_LOGGER_VAR: contextvars.ContextVar[_CliRuntimeLogger | None] = contextvars.ContextVar(
    "foresight_cli_runtime_logger",
    default=None,
)


@dataclass(frozen=True)
class CliLogConfig:
    style: str = "auto"
    level: str = "info"
    log_file: str = ""
    no_progress: bool = False


def register_runtime_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-style",
        choices=list(LOG_STYLE_CHOICES),
        default="auto",
        help="CLI log renderer (default: auto; writes logs to stderr only).",
    )
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVEL_CHOICES),
        default="info",
        help="CLI log verbosity (default: info).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional JSONL file to append structured CLI log events.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress-style logs while keeping lifecycle summaries.",
    )


def compact_log_payload(
    payload: Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    if payload is not None:
        for key, value in payload.items():
            items[str(key)] = value
    for key, value in kwargs.items():
        items[str(key)] = value
    return {
        key: value
        for key, value in items.items()
        if value is not None and value != "" and value != () and value != [] and value != {}
    }


def _normalize_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_log_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_log_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_log_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Index):
        return [_normalize_log_value(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _format_log_value(value: Any) -> str:
    normalized = _normalize_log_value(value)
    if isinstance(normalized, float):
        return f"{normalized:.6g}"
    if isinstance(normalized, list):
        return "[" + ",".join(_format_log_value(item) for item in normalized) + "]"
    if isinstance(normalized, dict):
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return str(normalized)


@dataclass(frozen=True)
class _CliRuntimeRecord:
    timestamp: str
    clock: str
    run_id: str
    command: str
    event: str
    level: str
    message: str
    payload: dict[str, Any]
    progress: bool


class _CliRuntimeLogger:
    def __init__(self, *, command: str, config: CliLogConfig) -> None:
        self._command = str(command)
        self._run_id = uuid.uuid4().hex
        self._config = CliLogConfig(
            style=str(config.style).lower().strip() or "auto",
            level=str(config.level).lower().strip() or "info",
            log_file=str(config.log_file).strip(),
            no_progress=bool(config.no_progress),
        )
        self._stderr_enabled = self._config.style != "quiet"
        self._renderer = self._resolve_renderer()
        self._log_path = None
        if self._config.log_file:
            self._log_path = Path(self._config.log_file).expanduser().resolve(strict=False)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_renderer(self) -> str:
        style = self._config.style
        if style == "quiet":
            return "plain"
        if style == "plain":
            return "plain"
        if Console is not None and Text is not None:
            return "rich"
        return "plain"

    def emit(
        self,
        *,
        event: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
        level: str = "info",
        progress: bool = False,
    ) -> None:
        level_name = str(level).lower().strip() or "info"
        if _LEVEL_RANK.get(level_name, _LEVEL_RANK["info"]) < _LEVEL_RANK[self._config.level]:
            return
        if bool(progress) and self._config.no_progress:
            return

        normalized_payload = {
            str(key): _normalize_log_value(value) for key, value in compact_log_payload(payload).items()
        }
        now = datetime.now().astimezone()
        record = _CliRuntimeRecord(
            timestamp=now.isoformat(timespec="seconds"),
            clock=now.strftime("%H:%M:%S"),
            run_id=self._run_id,
            command=self._command,
            event=str(event),
            level=level_name,
            message=str(message),
            payload=normalized_payload,
            progress=bool(progress),
        )

        if self._stderr_enabled:
            if self._renderer == "rich":
                self._render_rich(record)
            else:
                self._render_plain(record)
        if self._log_path is not None:
            self._append_jsonl(record)

    def _render_plain(self, record: _CliRuntimeRecord) -> None:
        payload_text = self._payload_text(record.payload)
        line = f"[{record.clock}] {record.message}"
        if payload_text:
            line = f"{line} {payload_text}"
        print(line, file=sys.stderr)

    def _render_rich(self, record: _CliRuntimeRecord) -> None:
        if Console is None or Text is None:
            self._render_plain(record)
            return
        styles = {
            "debug": "magenta",
            "info": "cyan",
            "error": "bold red",
        }
        text = Text()
        text.append(f"[{record.clock}] ", style="dim")
        text.append(record.message, style=styles.get(record.level, "cyan"))
        payload_text = self._payload_text(record.payload)
        if payload_text:
            text.append(" ")
            text.append(payload_text, style="bright_black")
        Console(stderr=True, highlight=False, soft_wrap=True).print(text)

    def _append_jsonl(self, record: _CliRuntimeRecord) -> None:
        if self._log_path is None:
            return
        payload = {
            "timestamp": record.timestamp,
            "run_id": record.run_id,
            "command": record.command,
            "event": record.event,
            "level": record.level,
            "message": record.message,
            "progress": record.progress,
            "payload": record.payload,
        }
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _payload_text(payload: Mapping[str, Any]) -> str:
        return " ".join(f"{key}={_format_log_value(value)}" for key, value in payload.items())


def _config_from_args(args: Any) -> CliLogConfig:
    return CliLogConfig(
        style=str(getattr(args, "log_style", "auto")),
        level=str(getattr(args, "log_level", "info")),
        log_file=str(getattr(args, "log_file", "")),
        no_progress=bool(getattr(args, "no_progress", False)),
    )


@contextlib.contextmanager
def activate_cli_logging(*, command: str, args: Any) -> Iterator[None]:
    logger = _CliRuntimeLogger(command=str(command), config=_config_from_args(args))
    token = _LOGGER_VAR.set(logger)
    try:
        yield
    finally:
        _LOGGER_VAR.reset(token)


@contextlib.contextmanager
def command_scope(
    args: Any,
    *,
    command: str,
    payload: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    base_payload = compact_log_payload(
        {
            "command": str(command),
            "pid": int(os.getpid()),
            "python_version": str(sys.version.split()[0]),
        },
        **compact_log_payload(payload),
    )
    with activate_cli_logging(command=str(command), args=args):
        emit_cli_event(
            "RUN start",
            event="run_started",
            payload=base_payload,
        )
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            emit_cli_event(
                "RUN failed",
                event="run_failed",
                level="error",
                payload=compact_log_payload(
                    base_payload,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        else:
            emit_cli_event(
                "RUN done",
                event="run_completed",
                payload=base_payload,
            )


def emit_cli_event(
    message: str,
    *,
    event: str,
    payload: Mapping[str, Any] | None = None,
    level: str = "info",
    progress: bool = False,
) -> None:
    logger = _LOGGER_VAR.get()
    if logger is None:
        return
    logger.emit(
        event=str(event),
        message=str(message),
        payload=payload,
        level=str(level),
        progress=bool(progress),
    )
