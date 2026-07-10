from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Optional

ProgressCallback = Callable[..., None]


class LogLevel(IntEnum):
    DEFAULT = 0
    VERBOSE = 1
    DEBUG = 2


_CURRENT_LEVEL = LogLevel.DEFAULT
_CRASH_ON_WARNING = False
_USER_WARNINGS: list["UserWarningRecord"] = []


@dataclass(frozen=True)
class UserWarningRecord:
    message: str
    context: Optional[str] = None


def set_log_level(level: LogLevel | str) -> None:
    global _CURRENT_LEVEL
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    _CURRENT_LEVEL = level


def get_log_level() -> LogLevel:
    return _CURRENT_LEVEL


def set_crash_on_warning(enabled: bool) -> None:
    global _CRASH_ON_WARNING
    _CRASH_ON_WARNING = bool(enabled)


def clear_user_warnings() -> None:
    _USER_WARNINGS.clear()


def get_user_warnings() -> list[UserWarningRecord]:
    return list(_USER_WARNINGS)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, *, level: LogLevel = LogLevel.DEFAULT) -> None:
    if level > _CURRENT_LEVEL:
        return
    print(f"[{_ts()}] {message}", flush=True)


def record_user_warning(message: str, *, context: Optional[str] = None) -> None:
    record = UserWarningRecord(message=message, context=context)
    _USER_WARNINGS.append(record)
    context_prefix = f"{context}: " if context else ""
    log(f"USER WARNING: {context_prefix}{message}")
    if _CRASH_ON_WARNING:
        raise SystemExit(f"Crash-on-warning: {context_prefix}{message}")


def log_user_warning_summary() -> None:
    if not _USER_WARNINGS:
        return
    log(f"Aggregate user warnings: {len(_USER_WARNINGS)} warning(s) recorded")
    for idx, warning in enumerate(_USER_WARNINGS, start=1):
        context_prefix = f"{warning.context}: " if warning.context else ""
        log(f"  Warning {idx}: {context_prefix}{warning.message}")


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
