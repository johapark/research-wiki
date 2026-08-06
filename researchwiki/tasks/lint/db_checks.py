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


# --- Near-duplicate claim sets -------------------------------------------
# Tuned on the live 389-paper / 11,717-claim corpus (2026-08). At 0.25 the
# check returns 6 pairs; 0.20 returns 16 (past reviewable), 0.30 returns 3 and
# drops the known-bad li-2026/gold-2026 pair. `min_claims` keeps short pages
# out: with 4 claims one shared neighbour is already a 0.25 share.
_DUP_CLAIM_THRESHOLD = 0.25
_DUP_CLAIM_MIN_CLAIMS = 8
# Below this fraction of claims present in the embedding cache the top-1
# ranking is computed against a corpus with holes in it, so the shares stop
# meaning anything and the check declines to report.
_DUP_CLAIM_MIN_COVERAGE = 0.5


def _duplicate_claim_pairs(
    stems: list[str],
    vecs,
    *,
    threshold: float = _DUP_CLAIM_THRESHOLD,
    min_claims: int = _DUP_CLAIM_MIN_CLAIMS,
) -> list[dict]:
    """Reciprocal top-1 concentration between every pair of pages.

    `stems[i]` is the owning page of claim vector `vecs[i]` (rows L2-normalized).
    For each claim, find its single nearest claim on *another* page; a page pair
    scores `min(share_a→b, share_b→a)` where `share_a→b` is the fraction of a's
    claims whose nearest off-page claim lives in b. Pure function over
    (stems, vecs) so it's testable without a DB or an embedding model.
    """
    import numpy as np

    by_stem: dict[str, list[int]] = {}
    for i, s in enumerate(stems):
        by_stem.setdefault(s, []).append(i)
    counts = {s: len(idxs) for s, idxs in by_stem.items()}

    # One (n_claims_on_page × n_claims) product per page: pages hold tens of
    # claims, so this stays a few hundred small matmuls instead of one
    # n×n matrix, and masking a page's own columns is a single slice.
    top1: dict[tuple[str, str], int] = {}
    for stem, idxs in by_stem.items():
        sim = vecs[idxs] @ vecs.T
        sim[:, idxs] = -2.0
        for j in sim.argmax(axis=1):
            key = (stem, stems[int(j)])
            top1[key] = top1.get(key, 0) + 1

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for a, b in top1:
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        if counts[a] < min_claims or counts[b] < min_claims:
            continue
        share_a = top1.get((a, b), 0) / counts[a]
        share_b = top1.get((b, a), 0) / counts[b]
        score = min(share_a, share_b)
        if score < threshold:
            continue
        first, second = key
        s1, s2 = (share_a, share_b) if first == a else (share_b, share_a)
        out.append({
            "score": round(score, 3),
            "pages": [
                {"stem": first, "n_claims": counts[first],
                 "top1_share": round(s1, 3),
                 "n_top1_in_other": top1.get((first, second), 0)},
                {"stem": second, "n_claims": counts[second],
                 "top1_share": round(s2, 3),
                 "n_top1_in_other": top1.get((second, first), 0)},
            ],
        })
    out.sort(key=lambda d: (-d["score"], d["pages"][0]["stem"]))
    return out


def find_duplicate_claim_sets() -> list[dict] | None:
    """Page pairs whose claim sets substantially duplicate each other.

    Catches the misattribution failure where a *commentary* on a paper (Nature
    Genetics "Research highlight", a News & Views, an editorial) is ingested as
    `type: paper`. Claim extraction then credits the original authors' work to
    the commentary, and neither fidelity gate objects: the claims genuinely are
    in the commentary's PDF, and the gates check faithfulness to the cited
    source, not entitlement to the claim. What *is* observable is structural —
    two pages end up asserting the same body of work.

    Metric: **reciprocal top-1 concentration.** For every claim, find its single
    nearest claim on another page (cached bi-encoder cosine); a page pair scores
    `min(share_a→b, share_b→a)` over those top-1 destinations. Reported at
    ≥ 0.25.

    Why rank-based and not a similarity threshold — measured on this corpus:

      - Lexical token Jaccard per claim pair, the cheap option, cannot see this
        at all. A highlight paraphrases *upward*: "Introduces SIGnature, a
        framework combining explainable AI with single-cell foundation model
        attributions" against the source's "Introduces SIGnature, a
        gradient-based attribution framework (IG, DeepLIFT, IxG via Captum…)"
        scores Jaccard 0.19. No claim pair in the known-bad li-2026/gold-2026
        pair cleared 0.4, while unrelated pages did.
      - Absolute cosine is no better: every page pair in this corpus sits near a
        ~0.70 floor of mean-of-max claim similarity, and the known-bad pair
        ranked 1,152nd of 75,466 on it.
      - Rank-based aggregation is scale-free, which is exactly what survives the
        abstraction gap: the highlight's claims have nowhere better to point.
        The same metric puts the known-bad pair 6th of 6,521.

    Lexical would have been cheaper and is what the cost of this check buys off:
    it reads `.semantic-cache/claims.npy` directly, so a warm cache costs one
    18 MB read plus a few hundred small matmuls (<1s) and never loads the
    bi-encoder or rewrites the cache. A cold or thin (<50% coverage) cache
    returns **None** — "check skipped", the same convention `lint` already uses
    for `invalid_frontmatter` when PyYAML is missing — rather than paying a
    corpus embed inside a check that's meant to be instant. Warm it with any
    `researchwiki claim-overlap` run.

    The threshold is corpus-relative, which is the metric's one real weakness:
    a share is only meaningful against the number of other pages a claim could
    have pointed at, so on a 20-page wiki 0.25 is reachable by chance and on
    this 389-page one it isn't. Re-tune (and re-read the hit list) if the corpus
    changes size by an order of magnitude.

    **Advisory, not a defect.** Legitimate near-duplication is common and the
    top of this list proves it: two clinical trials of the same base-editing
    therapy, two reviews of one disease, a paper and its own preprint, two
    surveys of one subfield. The check reports the *pair* with both directional
    shares and claim counts; deciding which page (if either) isn't entitled to
    the claims is the reviewer's call — the shorter page, or one whose venue
    says "News & Views", is the usual suspect.
    """
    from ...db.safe import safe_read

    def _query(conn):
        return conn.execute(
            """
            SELECT c.paper_stem, c.section, c.position, c.text
              FROM claims c
              JOIN papers p ON p.stem = c.paper_stem
             WHERE p.page_type = 'paper'
               AND p.page_path LIKE ?
               AND c.is_cross_ref = 0
             ORDER BY c.paper_stem, c.section, c.position
            """,
            (f"%{wiki_dir().name}/%",),
        ).fetchall()

    rows = safe_read(_query, default=None, label="lint.find_duplicate_claim_sets")
    if not rows:
        return None

    claims = [
        {"paper_stem": r["paper_stem"], "section": r["section"],
         "position": r["position"], "text": r["text"]}
        for r in rows
    ]
    try:
        from ...index.claim_embeddings import load_cached_claim_embeddings
        cached = load_cached_claim_embeddings(claims)
    except Exception:
        return None
    if cached is None:
        return None
    vecs, row_idx = cached
    if len(row_idx) < _DUP_CLAIM_MIN_COVERAGE * len(claims):
        return None

    return _duplicate_claim_pairs([claims[i]["paper_stem"] for i in row_idx], vecs)


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
