"""DB drift detection — does the DB still reflect the canonical markdown?

`db verify` walks `wiki/` and reports four kinds of drift between the file
state and the indexed state:

  - missing  : page exists in wiki/ but no DB row
  - extra    : DB row exists for a stem that no longer has a wiki page
  - stale    : page mtime > indexed_at (DB needs rebuild for this stem)
  - moved    : same stem, different category directory

Markdown is canonical; on drift, run `db rebuild`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import wiki_dir
from .connection import get_connection
from .rebuild import _is_meta_page, _parse_frontmatter


@dataclass
class VerifyReport:
    pages_scanned: int = 0
    papers_in_db: int = 0
    missing: list[str] = field(default_factory=list)         # in wiki, not in db
    extra: list[str] = field(default_factory=list)           # in db, not in wiki
    stale: list[tuple[str, int, int]] = field(default_factory=list)   # (stem, page_mtime, indexed_at)
    moved: list[tuple[str, str, str]] = field(default_factory=list)   # (stem, db_category, fs_category)

    invalid: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.missing or self.extra or self.stale or self.moved or self.invalid)


def _has_frontmatter(md: Path) -> bool:
    """Cheap probe — does the file open with a YAML frontmatter fence?"""
    try:
        # Rebuild's text-mode parser accepts both newline conventions. Mirror
        # that behavior here while keeping this probe encoding-independent:
        # Windows writers persist the same logical fence as ``---\r\n``.
        with md.open("rb") as f:
            head = f.read(5)
        return head.startswith((b"---\n", b"---\r\n"))
    except OSError:
        return False


def complete_markdown_scan(root: Path) -> list[Path]:
    """Every `.md` under `root`, or raise — never a silently partial list.

    `Path.rglob` skips a directory it cannot read, so a page there looks
    deleted. Callers use absence as deletion evidence (`lint --fix`, journal
    reconciliation), and a missing `wiki/` (or a symlink into an offline synced
    folder) is not an empty corpus. Both cases raise instead.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"wiki directory not found: {root}")

    def scan_error(error):
        raise error
    return sorted(Path(folder) / name
                  for folder, _, names in os.walk(root, onerror=scan_error)
                  for name in names if name.endswith(".md"))


def verify() -> VerifyReport:
    report = VerifyReport()
    conn = get_connection()

    db_rows = {r["stem"]: r for r in conn.execute(
        "SELECT stem, category, page_type, page_path, page_mtime, indexed_at "
        "FROM papers"
    )}
    report.papers_in_db = len(db_rows)

    fs_stems: set[str] = set()
    root = wiki_dir()
    try:
        markdown = complete_markdown_scan(root)
    except OSError:
        conn.close()
        raise
    for md in markdown:
        # Root bookkeeping files (`index.md`, `log.md`,
        # `pdfs-failed-parsing.md`) are skipped by `db rebuild`
        # (`_is_meta_page`), so they never get a `papers` row. Skip them
        # here too — otherwise, once they carry `type: meta` frontmatter,
        # they'd pass the frontmatter probe below and be reported as
        # spurious `missing` drift. Single source of truth: reuse
        # rebuild's exclusion so verify and rebuild can't disagree.
        if _is_meta_page(md, root):
            continue
        # Other files without YAML frontmatter (scratch notes) are
        # likewise not indexed — skip rather than report as drift.
        if not _has_frontmatter(md) and md.stem not in db_rows:
            continue
        stem = md.stem
        fs_stems.add(stem)
        report.pages_scanned += 1
        row = db_rows.get(stem)
        if _parse_frontmatter(md) is None:
            report.invalid.append(stem)
            continue
        if row is None:
            report.missing.append(stem)
            continue
        # Meta pages (log.md, index.md) are append-only bookkeeping;
        # dashboard pages (views.md) hold Dataview queries the DB never
        # consumes — Obsidian renders them client-side at read time. In
        # both cases the DB row is a fixed schema, unrelated to body
        # content, so mtime > indexed_at doesn't signal real drift.
        # (Paper/synthesis/idea/concept/reference pages DO signal drift
        # when edited — those need `db rebuild` to reflect the change.)
        if row["page_type"] not in ("meta", "dashboard"):
            page_mtime = int(md.stat().st_mtime)
            if page_mtime > row["indexed_at"]:
                report.stale.append((stem, page_mtime, row["indexed_at"]))
        fs_category = md.parent.name
        if fs_category != row["category"]:
            report.moved.append((stem, row["category"], fs_category))

    for stem in db_rows.keys() - fs_stems:
        report.extra.append(stem)

    conn.close()
    return report
