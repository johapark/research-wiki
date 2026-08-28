"""Deprecated alias for researchwiki scout citations.

The command remains available because CLI names and ``--json`` output are
published compatibility surfaces. New callers should use ``scout``.
"""

from __future__ import annotations

import sys

from ..scouting import citations


def main(argv: list[str]) -> int:
    print(
        "note: `researchwiki audit` is now `researchwiki scout citations`.",
        file=sys.stderr,
    )
    return citations.main(
        argv,
        prog="researchwiki audit",
        log_tag="audit",
        report_title="# Semantic Scholar Audit Report",
    )
