from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Callable, Optional

ProgressCallback = Callable[..., None]


class LogLevel(IntEnum):
    DEFAULT = 0
    VERBOSE = 1
    DEBUG = 2


_CURRENT_LEVEL = LogLevel.DEFAULT


def set_log_level(level: LogLevel | str) -> None:
    global _CURRENT_LEVEL
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    _CURRENT_LEVEL = level


def get_log_level() -> LogLevel:
    return _CURRENT_LEVEL


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, *, level: LogLevel = LogLevel.DEFAULT) -> None:
    if level > _CURRENT_LEVEL:
        return
    print(f"[{_ts()}] {message}", flush=True)


def emit_progress(
    progress_callback: Optional[ProgressCallback],
    message: str,
    *,
    level: LogLevel = LogLevel.DEFAULT,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, level=level)


def workflow_bracket_label(prefix: str, smiles: Optional[str] = None) -> str:
    if smiles:
        return f"[{prefix}] {smiles}"
    return f"[{prefix}]"
