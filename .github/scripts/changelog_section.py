#!/usr/bin/env python3
"""Print one release's CHANGELOG.md section — the body published as release notes.

Lives here rather than inline in `release.yml` so it can be unit-tested: the first
version of this logic terminated a section only at the next `## [` heading, which
meant the *oldest* section ran on to end-of-file and swallowed the link-reference
block ("[0.1.0]: https://…") into the published notes. A release-time bug in a
workflow is the worst place to discover anything, so the rule is pinned by
`tests/test_version.py` instead.

Usage:  changelog_section.py <version-or-tag> [changelog-path]
Exits 1 with a message when the section is missing or empty.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `## [0.2.0] - 2026-08-07`. `## [Unreleased]` is intentionally unmatched: it is
# the staging area, never a release.
_HEADING = r"^## \[{version}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$"

# A section ends at the next release heading OR at the link-reference block, which
# in Keep a Changelog sits at the bottom of the file after the last section.
_SECTION_END = re.compile(r"^(?:## \[|\[[^\]]+\]:\s*https?://)", re.MULTILINE)


def section_body(text: str, version: str) -> str:
    """Return the body of `version`'s section. Raises ValueError if absent/empty."""
    version = version.lstrip("v")
    start = re.search(_HEADING.format(version=re.escape(version)), text, re.MULTILINE)
    if not start:
        raise ValueError(f"CHANGELOG.md has no released section for {version}")
    rest = text[start.end():]
    end = _SECTION_END.search(rest)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise ValueError(f"CHANGELOG.md section for {version} is empty")
    return body


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    path = Path(argv[1]) if len(argv) > 1 else Path("CHANGELOG.md")
    try:
        print(section_body(path.read_text(encoding="utf-8"), argv[0]))
    except (ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
