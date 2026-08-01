"""Logging helpers.

- `log(msg, tag)`: stderr status output matching the existing `[ingest] …` format.
- `append_log_md(kind, headline, details)`: chronological `log.md` entries at the
  repo root, using Karpathy's parseable-prefix convention (one `## [YYYY-MM-DD]
  kind | headline` per entry).
"""

from __future__ import annotations

import sys
from datetime import date



def log(msg: str, tag: str = "ingest") -> None:
    print(f"[{tag}] {msg}", file=sys.stderr)


def append_log_md(kind: str, headline: str, details: str = "") -> None:
    """Append one entry to `wiki/log.md`.

    Format:
        ## [YYYY-MM-DD] {kind} | {headline}
        {details}

    `kind` is one of `ingest`, `query`, `lint`, `synthesize`. Creates the file
    on first use. `wiki/log.md` is gitignored by default (per-user history)
    and lives inside wiki/ so an Obsidian vault on wiki/ can browse it.
    """
    from .paths import log_path
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    entry = f"## [{today}] {kind} | {headline}\n"
    if details:
        entry += f"{details.rstrip()}\n"
    entry += "\n"

    if not path.exists():
        header = (
            "# log.md\n\n"
            "Chronological record of wiki operations (ingests, queries, lints, "
            "synthesizes). Auto-appended by `researchwiki` commands. "
            "Gitignored by default — this is per-user research history.\n\n"
        )
        path.write_text(header + entry, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(entry)
