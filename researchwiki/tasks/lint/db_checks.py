"""Sqlite-backed checks: `claims` table grading coverage + structured-DB
drift against `wiki/`.

Both checks silently no-op (return empty results) when the DB is
unavailable so a fresh clone running `lint` with no DB state never
blocks. Existing surfaces (`status`, `db verify`) are the loud path
for total DB absence.
"""

from __future__ import annotations

from ...log import log
from ...paths import wiki_dir


def find_ungraded_papers() -> list[dict]:
    """Find paper-type pages whose claims have not all been graded.

    Returns a list of {stem, n_claims, n_ungraded} for every paper that
    has at least one claim row with `last_graded_at IS NULL`. Excludes
    pages with zero claims (those don't need grading).
    """
    from ...db.safe import safe_read

    def _query(conn):
        return conn.execute(
            """
            SELECT p.stem,
                   COUNT(c.id) AS n_claims,
                   SUM(CASE WHEN c.last_graded_at IS NULL THEN 1 ELSE 0 END) AS n_ungraded
              FROM papers p
              JOIN claims c ON c.paper_stem = p.stem
             WHERE p.page_type = 'paper'
             GROUP BY p.stem
            HAVING n_ungraded > 0
             ORDER BY p.stem
            """
        ).fetchall()

    rows = safe_read(_query, default=[], label="lint.find_ungraded_papers")
    return [
        {"stem": r["stem"], "n_claims": r["n_claims"], "n_ungraded": r["n_ungraded"]}
        for r in rows
    ]


def find_zero_claim_papers() -> list[dict]:
    """Paper-type pages that produced NO claim rows at all.

    The complement of `find_ungraded_papers`, which `JOIN`s `claims` and so
    cannot see a paper with none — a gap it documents but doesn't cover. A page
    with zero claims is inert as evidence: `claims` returns nothing for it, no
    `[[stem#slug]]` anchor exists, and `grade synthesis` has nothing to verify a
    citation against. Yet `backfill hook` succeeds on it and every other check
    stays quiet, so without this it looks fully migrated.

    Near-always caused by non-canonical H2 headings: claim extraction reads only
    the exact names in `grade.parser.SECTION_KEYS`, so a page whose findings sit
    under `## Findings` yields nothing. The fix is renaming the heading, then
    `db rebuild` — see `prompts/migration-backfill.md`.

    Scoped to `page_path` under `wiki/` so a row left behind by a test run
    against a tmp dir can't register as a corpus defect.
    """
    from ...db.safe import safe_read

    def _query(conn):
        return conn.execute(
            """
            SELECT p.stem, p.category
              FROM papers p
             WHERE p.page_type = 'paper'
               AND p.page_path LIKE ?
               AND NOT EXISTS (
                     SELECT 1 FROM claims c WHERE c.paper_stem = p.stem
                   )
             ORDER BY p.stem
            """,
            (f"%{wiki_dir().name}/%",),
        ).fetchall()

    rows = safe_read(_query, default=[], label="lint.find_zero_claim_papers")
    return [{"stem": r["stem"], "category": r["category"]} for r in rows]


def find_stems_missing_claim_overlap() -> list[str]:
    """Paper pages with claims that `claim-overlap` has not covered.

    Claim-overlap is opt-in at ingest (`--claim-overlap`) because it spends an
    LLM judge call per candidate pair to confirm a link on roughly one paper in
    ten — batching it is cheaper than paying per ingest. The cost of batching is
    that coverage decays silently, so it needs a check: without one, a stem that
    was never examined is indistinguishable from one examined and correctly found
    to have no real overlap (the common outcome).

    Advisory, not a defect. A pending stem means an opportunity was not taken,
    not that anything on disk is wrong — drain with
    `researchwiki claim-overlap --backlog`.
    """
    from ...db.safe import safe_read

    def _query(conn):
        from ..claim_overlap import find_backlog
        return find_backlog(conn)

    return safe_read(_query, default=[], label="lint.find_stems_missing_claim_overlap")


def db_drift_check_and_fix(
    apply_fix: bool,
) -> tuple[dict[str, list], dict[str, int]]:
    """Detect drift between wiki/ and the structured DB; optionally reconcile.

    Returns (drift, fixed) where:
      drift = {"missing": [...], "extra": [...], "stale": [...], "moved": [...]}
              (entries are stems for missing/extra,
              {stem, page_mtime, indexed_at} for stale,
              {stem, db_category, fs_category} for moved)
      fixed = {"upserted": N, "deleted": N}  -- empty when apply_fix is False
              or no drift, or when the DB is unreachable.
    """
    try:
        from ...db.verify import verify as _verify
        report = _verify()
    except Exception:
        return {}, {}

    drift = {
        "missing": list(report.missing),
        "extra": list(report.extra),
        "stale": [
            {"stem": s, "page_mtime": pm, "indexed_at": ia}
            for s, pm, ia in report.stale
        ],
        "moved": [
            {"stem": s, "db_category": dbc, "fs_category": fsc}
            for s, dbc, fsc in report.moved
        ],
    }

    fixed: dict[str, int] = {}
    if not apply_fix or report.is_clean:
        return drift, fixed

    from ...db import upsert_page, delete_page
    upserted = 0
    upsert_stems: set[str] = set()
    upsert_stems.update(report.missing)
    upsert_stems.update(s for s, _, _ in report.stale)
    upsert_stems.update(s for s, _, _ in report.moved)
    for stem in sorted(upsert_stems):
        for md in wiki_dir().rglob(f"{stem}.md"):
            try:
                upsert_page(md)
                upserted += 1
            except Exception as e:
                log(f"db-fix: upsert failed for {stem} ({type(e).__name__}: {e})",
                    tag="lint")
            break
    deleted = 0
    for stem in report.extra:
        try:
            if delete_page(stem):
                deleted += 1
        except Exception as e:
            log(f"db-fix: delete failed for {stem} ({type(e).__name__}: {e})",
                tag="lint")
    if upserted:
        fixed["upserted"] = upserted
    if deleted:
        fixed["deleted"] = deleted
    return drift, fixed
