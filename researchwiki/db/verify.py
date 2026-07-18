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

from dataclasses import dataclass, field
from pathlib import Path

from ..paths import wiki_dir
from .connection import get_connection


@dataclass
class VerifyReport:
    pages_scanned: int = 0
    papers_in_db: int = 0
    missing: list[str] = field(default_factory=list)         # in wiki, not in db
    extra: list[str] = field(default_factory=list)           # in db, not in wiki
    stale: list[tuple[str, int, int]] = field(default_factory=list)   # (stem, page_mtime, indexed_at)
    moved: list[tuple[str, str, str]] = field(default_factory=list)   # (stem, db_category, fs_category)

    @property
    def is_clean(self) -> bool:
        return not (self.missing or self.extra or self.stale or self.moved)


def _has_frontmatter(md: Path) -> bool:
    """Cheap probe — does the file open with a YAML frontmatter fence?"""
    try:
        with md.open("rb") as f:
            head = f.read(4)
        return head == b"---\n"
    except OSError:
        return False


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
    if root.exists():
        for md in sorted(root.rglob("*.md")):
            # Files without YAML frontmatter (`index.md`, `log.md`, scratch
            # notes) are intentionally not indexed — skip rather than
            # report them as drift.
            if not _has_frontmatter(md):
                continue
            stem = md.stem
            fs_stems.add(stem)
            report.pages_scanned += 1
            row = db_rows.get(stem)
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
