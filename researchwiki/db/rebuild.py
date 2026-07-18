"""Rebuild the structured DB from the canonical wiki/ + papers/ state.

The DB is a derived index. Every row is reproducible from the markdown files
+ caches; nothing here is authoritative. `rebuild` walks every page under
`wiki/`, parses YAML frontmatter (real PyYAML, not the line-based parser
from wiki.py — frontmatter has lists for tags and category), parses claims
for paper-type pages, and upserts into the `papers` and `claims` tables.

Deletion detection: stems present in the DB but not in the current walk are
removed. CASCADE on the FK takes their claims out with them.

Idempotent — running twice in a row is a no-op (mtime-aware on the second
pass; we still rewrite rows so the reproducibility property holds even if
mtime is unchanged).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..paths import papers_dir, wiki_dir
from ..claim_graph.slug import (
    SLUG_SCHEME_VERSION,
    compute_claim_slug,
    disambiguate_slug,
)
from ..grade.parser import parse_claims as _parse_claims
from ..wiki import Page
from .connection import get_connection

# Bookkeeping files that live directly under wiki/ (not a category subdir)
# and carry no YAML frontmatter — `index.md`, `log.md`, `pdfs-failed-parsing.md`
# per CLAUDE.md's page-type catalogue. They aren't pages, so they shouldn't
# count as parse errors on every rebuild (that previously forced exit code 2
# unconditionally and skipped the post-rebuild claim-graph reconcile step).
_META_FILENAMES = frozenset({"index.md", "log.md", "pdfs-failed-parsing.md"})


def _is_meta_page(md: Path, root: Path) -> bool:
    """True for a known bookkeeping file sitting directly under `root`."""
    return md.parent == root and md.name in _META_FILENAMES


@dataclass
class RebuildStats:
    pages_scanned: int = 0
    papers_upserted: int = 0
    claims_upserted: int = 0
    papers_deleted: int = 0
    parse_errors: list[str] = None

    def __post_init__(self):
        if self.parse_errors is None:
            self.parse_errors = []


def _parse_frontmatter(md: Path) -> tuple[dict, str] | None:
    """Parse YAML frontmatter + return (frontmatter, body). None if malformed."""
    text = md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    try:
        fm = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None
    body = text[end + 5:]
    return fm, body


def _coerce_int(v) -> int | None:
    """YAML ints come through as int; strings parsed via int(). None on failure."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


def _coerce_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _upsert_paper(
    conn: sqlite3.Connection,
    page: Page,
    fm: dict,
    now_ts: int,
) -> None:
    """Write or update a single papers row."""
    pdf_path_raw = fm.get("pdf_path")
    pdf_mtime: int | None = None
    if pdf_path_raw:
        p = Path(str(pdf_path_raw))
        if p.exists():
            pdf_mtime = int(p.stat().st_mtime)

    page_mtime = int(page.path.stat().st_mtime)

    # Directory is canonical (schema comment: "subdir under wiki/"). YAML
    # `category:` is denormalized bookkeeping — when the two diverge (older
    # synthesis pages still carry `category: [overviews]`), the directory
    # placement reflects the user's intent. Reading YAML here would put the
    # DB into permanent drift on every rebuild.
    category = page.category

    row = {
        "stem": page.stem,
        "category": category,
        "page_type": str(fm.get("type", "paper")),
        "title": _coerce_str(fm.get("title")),
        "year": _coerce_int(fm.get("year")),
        "doi": _coerce_str(fm.get("doi")) or None,
        "venue": _coerce_str(fm.get("venue")) or None,
        "publication_status": _coerce_str(fm.get("publication_status")) or None,
        "authors": _coerce_str(fm.get("authors")) or None,
        "senior_authors": _coerce_str(fm.get("senior_authors")) or None,
        "tags": json.dumps(fm.get("tags") or []),
        "pdf_path": _coerce_str(pdf_path_raw) or None,
        "page_path": str(page.path),
        "page_mtime": page_mtime,
        "pdf_mtime": pdf_mtime,
        "raw_frontmatter": json.dumps(fm, default=str, ensure_ascii=False),
        "indexed_at": now_ts,
    }

    conn.execute(
        """
        INSERT INTO papers (
            stem, category, page_type, title, year, doi, venue, publication_status,
            authors, senior_authors, tags, pdf_path, page_path, page_mtime, pdf_mtime,
            raw_frontmatter, indexed_at
        ) VALUES (
            :stem, :category, :page_type, :title, :year, :doi, :venue, :publication_status,
            :authors, :senior_authors, :tags, :pdf_path, :page_path, :page_mtime, :pdf_mtime,
            :raw_frontmatter, :indexed_at
        )
        ON CONFLICT(stem) DO UPDATE SET
            category=excluded.category,
            page_type=excluded.page_type,
            title=excluded.title,
            year=excluded.year,
            doi=excluded.doi,
            venue=excluded.venue,
            publication_status=excluded.publication_status,
            authors=excluded.authors,
            senior_authors=excluded.senior_authors,
            tags=excluded.tags,
            pdf_path=excluded.pdf_path,
            page_path=excluded.page_path,
            page_mtime=excluded.page_mtime,
            pdf_mtime=excluded.pdf_mtime,
            raw_frontmatter=excluded.raw_frontmatter,
            indexed_at=excluded.indexed_at
        """,
        row,
    )


def _resolve_slug(
    conn: sqlite3.Connection, page_stem: str, section: str, position: int, text: str,
    already_assigned: set[str],
) -> str:
    """Compute a claim_slug, disambiguating collisions by position (§3.1).

    A collision here means two claims in the SAME paper normalize to identical
    text (e.g. "Method A improves accuracy." at two positions). Rare in
    practice; the position suffix breaks the tie deterministically. We track
    `already_assigned` within this paper's upsert batch so two fresh claims
    with matching text also disambiguate cleanly on the first write.
    """
    base = compute_claim_slug(section, text)
    if base not in already_assigned:
        return base
    return disambiguate_slug(base, position)


def _upsert_claims(conn: sqlite3.Connection, page: Page) -> int:
    """Reconcile claims rows for this paper against a fresh parse of the page.

    Grader-output columns (bm25_top1, semantic_score, last_graded_at, …) are
    preserved when the claim text at a given (section, position) is unchanged.
    Changed text → grader columns reset to NULL; new claims start NULL; rows
    whose (section, position) no longer exist are deleted (CASCADE-safe).

    `claim_slug` is computed deterministically from (section, normalized text).
    Refreshed on every upsert — text unchanged ⇒ slug unchanged (foundation of
    edge durability); text changed ⇒ new slug ⇒ any cached edges pointing at
    the old slug become stale on the next `claim-graph reconcile`.

    This is the load-bearing invariant for incremental writes: rebuild can run
    arbitrarily often without wiping the grader's work.
    """
    fresh = list(_parse_claims(page))
    fresh_keys = {(c.section, c.position) for c in fresh}

    existing = {
        (r["section"], r["position"]): r["text"]
        for r in conn.execute(
            "SELECT section, position, text FROM claims WHERE paper_stem = ?",
            (page.stem,),
        )
    }

    # Drop rows that no longer correspond to a parsed claim.
    for (section, position) in existing.keys() - fresh_keys:
        conn.execute(
            "DELETE FROM claims WHERE paper_stem = ? AND section = ? AND position = ?",
            (page.stem, section, position),
        )

    # Slugs are content-addressed on (section, text), so two claims that swap
    # text between positions (pos1: A→B, pos2: B→A) would transiently put the
    # same slug on two rows during the row-by-row reassignment below and trip
    # UNIQUE(paper_stem, claim_slug) — aborting the whole rebuild transaction.
    # Clear every slug for this paper first: NULLs don't collide under UNIQUE,
    # and the loop re-sets each to its resolved value. Only claim_slug is
    # touched — grader-output columns are preserved, and the committed slug
    # value for unchanged text is identical, so edge durability holds.
    conn.execute(
        "UPDATE claims SET claim_slug = NULL WHERE paper_stem = ?", (page.stem,)
    )

    n = 0
    slugs_seen_in_batch: set[str] = set()
    for c in fresh:
        slug = _resolve_slug(
            conn, page.stem, c.section, c.position, c.text, slugs_seen_in_batch,
        )
        slugs_seen_in_batch.add(slug)
        prior_text = existing.get((c.section, c.position))
        if prior_text is None:
            conn.execute(
                """
                INSERT INTO claims (paper_stem, section, position, text, claim_slug, is_cross_ref)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (page.stem, c.section, c.position, c.text, slug,
                 1 if c.is_cross_ref else 0),
            )
        elif prior_text != c.text:
            # Text changed at the same (section, position): reset grader output.
            conn.execute(
                """
                UPDATE claims
                   SET text = ?,
                       claim_slug = ?,
                       is_cross_ref = ?,
                       bm25_top1 = NULL,
                       bm25_top3_mean = NULL,
                       bm25_top1_chunk_id = NULL,
                       supporting_text = NULL,
                       semantic_score = NULL,
                       embed_model = NULL,
                       negation_mismatch = NULL,
                       numeric_tokens = NULL,
                       numeric_unmatched = NULL,
                       last_graded_at = NULL
                 WHERE paper_stem = ? AND section = ? AND position = ?
                """,
                (c.text, slug, 1 if c.is_cross_ref else 0,
                 page.stem, c.section, c.position),
            )
        else:
            # Identical text at the same position — is_cross_ref derives from
            # text alone, so nothing to refresh; preserve all grader columns.
            # BUT: backfill claim_slug when it's NULL on an existing row (e.g.,
            # after the ADD COLUMN migration ran on a pre-existing DB).
            conn.execute(
                "UPDATE claims SET claim_slug = ? "
                " WHERE paper_stem = ? AND section = ? AND position = ? "
                "   AND (claim_slug IS NULL OR claim_slug != ?)",
                (slug, page.stem, c.section, c.position, slug),
            )
        n += 1
    return n


def _upsert_one(conn: sqlite3.Connection, md: Path, now_ts: int) -> tuple[Page | None, int]:
    """Parse and upsert one markdown page. Returns (page, n_claims) or (None, 0).

    `n_claims` is 0 for non-paper page types — their claims are pruned.
    Returns (None, 0) if the file's frontmatter is unparseable.
    """
    parsed = _parse_frontmatter(md)
    if parsed is None:
        return None, 0
    fm, body = parsed
    page = Page(
        path=md,
        stem=md.stem,
        category=md.parent.name,
        fm=fm if isinstance(fm, dict) else {},
        body=body,
    )
    _upsert_paper(conn, page, fm, now_ts)

    page_type = str(fm.get("type", "paper"))
    if page_type == "paper":
        n_claims = _upsert_claims(conn, page)
    else:
        # Non-paper pages (synthesis / guidance / whitepaper / ...) don't
        # have gradable claims today — prune any leftovers but keep the
        # `papers` row for joins and queries.
        conn.execute("DELETE FROM claims WHERE paper_stem = ?", (page.stem,))
        n_claims = 0
    return page, n_claims


def upsert_page(md: Path) -> tuple[Page | None, int]:
    """Public single-page upsert — for callers that just wrote one md file.

    Opens its own connection and commits before returning. Use this from
    `wiki.commit_page` and `lint --fix`'s drift-fix path so DB state never
    lags markdown by more than one page write.
    """
    conn = get_connection()
    now_ts = int(time.time())
    try:
        with conn:
            page, n = _upsert_one(conn, md, now_ts)
        return page, n
    finally:
        conn.close()


def find_by_doi(doi: str) -> dict | None:
    """Look up a single paper row by DOI. Case-insensitive.

    Returns {stem, category, page_type, title, year, indexed_at, page_path}
    or None when nothing matches. Use as a cheap "have we seen this paper?"
    check before kicking off ingest work — avoids re-running reconcile +
    LLM author + grader for a duplicate that promote.py would catch later.
    """
    if not doi:
        return None
    doi_lower = doi.lower().strip()
    if not doi_lower:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT stem, category, page_type, title, year, indexed_at, page_path
              FROM papers
             WHERE LOWER(doi) = ?
             LIMIT 1
            """,
            (doi_lower,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "stem": row["stem"],
        "category": row["category"],
        "page_type": row["page_type"],
        "title": row["title"],
        "year": row["year"],
        "indexed_at": row["indexed_at"],
        "page_path": row["page_path"],
    }


def delete_page(stem: str) -> bool:
    """Public single-page delete. CASCADE removes the stem's claims."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("DELETE FROM papers WHERE stem = ?", (stem,))
            return cur.rowcount > 0
    finally:
        conn.close()


def rebuild(verbose: bool = False) -> RebuildStats:
    """Walk wiki/ + papers/ and rewrite the DB index. Returns stats."""
    stats = RebuildStats()
    now_ts = int(time.time())
    seen: set[str] = set()

    conn = get_connection()
    root = wiki_dir()
    if not root.exists():
        return stats

    with conn:
        # Persist the slug scheme version. Rebuild is the right place: it's
        # the atomic point at which every claim's slug is recomputed under
        # the current scheme. `claim-graph reconcile` reads this to decide
        # whether cached edges should be marked stale.
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("slug_scheme_version", str(SLUG_SCHEME_VERSION)),
        )

        for md in sorted(root.rglob("*.md")):
            if _is_meta_page(md, root):
                continue
            stats.pages_scanned += 1
            page, n_claims = _upsert_one(conn, md, now_ts)
            if page is None:
                stats.parse_errors.append(str(md))
                continue
            seen.add(page.stem)
            stats.papers_upserted += 1
            stats.claims_upserted += n_claims
            if verbose:
                page_type = page.fm.get("type", "paper") if isinstance(page.fm, dict) else "paper"
                tail = f"({n_claims} claims)" if page_type == "paper" else f"(page_type={page_type})"
                print(f"  + {page.category}/{page.stem}  {tail}")

        # Deletion detection: drop papers (and their claims via CASCADE) that
        # exist in the DB but not in the current walk.
        existing = {r["stem"] for r in conn.execute("SELECT stem FROM papers")}
        gone = existing - seen
        if gone:
            conn.executemany("DELETE FROM papers WHERE stem = ?", [(s,) for s in gone])
            stats.papers_deleted = len(gone)
            if verbose:
                for s in sorted(gone):
                    print(f"  - {s}")

    conn.close()
    return stats
