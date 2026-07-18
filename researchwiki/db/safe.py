"""Debug-visible wrapper for the try/except-around-`get_connection()` pattern.

The read paths across `tasks/status.py`, `tasks/insights.py`, and
`tasks/lint/db_checks.py` all follow the same shape: open a connection,
run a query, close, and downgrade any failure to an empty result so a
fresh clone (or a corrupt DB) doesn't crash the CLI. That "swallow
silently" behavior is deliberate, but it also means a real corruption
hides behind an empty status page. `safe_read` keeps the graceful
downgrade and adds a `RESEARCHWIKI_DEBUG=1` breadcrumb so `status`,
`insights`, and `lint` all trace to stderr with a consistent label.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, TypeVar

from .connection import get_connection

T = TypeVar("T")


def _debug_enabled() -> bool:
    # Case-insensitive so `RESEARCHWIKI_DEBUG=False` (or FALSE/No/off) all
    # correctly disable the breadcrumb. Only `1`, `true`, `yes`, `on` enable.
    val = os.environ.get("RESEARCHWIKI_DEBUG", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _emit(label: str, phase: str, exc: BaseException) -> None:
    print(f"safe_read [{label}]: {phase} — {type(exc).__name__}: {exc}",
          file=sys.stderr)


def safe_read(fn: Callable[[object], T], *, default: T, label: str) -> T:
    """Run `fn(conn)` with a fresh connection; return `default` on any error.

    `label` is what identifies the call site in the debug breadcrumb — pass
    a stable `<module>.<function>` string (e.g. `"status.claim_grading"`).
    Behavior in default mode is unchanged from the previous per-site
    `try/except Exception: return default`. Under `RESEARCHWIKI_DEBUG=1`
    each failure prints one stderr line naming the label + phase.
    """
    try:
        conn = get_connection()
    except Exception as e:
        if _debug_enabled():
            _emit(label, "get_connection", e)
        return default
    try:
        return fn(conn)
    except Exception as e:
        if _debug_enabled():
            _emit(label, "query", e)
        return default
    finally:
        try:
            conn.close()
        except Exception:
            pass
