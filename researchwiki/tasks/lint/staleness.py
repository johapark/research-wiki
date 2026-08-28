"""Staleness checks: pages that drifted out of sync with the surrounding corpus.

  - stale_synthesis: a synthesis page's `generated_at` predates the ingest date
    of a paper it references (mtime is deliberately NOT used — see
    `_source_change_date`)
  - stale_by_content: page's `topic_seed` top-hits include unreferenced papers
  - stale_by_audit_count: a cached `wiki_papers_at_audit:` value drifted
  - stale_evolution_proposals: proposal directories ≥ 7 days old
"""

from __future__ import annotations

import time
from pathlib import Path

from ...paths import ingest_dir
from .walk import extract_links, page_key


def _source_change_date(md: Path, fm: dict, db_dates: dict | None = None):
    """Date a referenced page's *substance* last changed.

    Prefers YAML `ingested_at` over filesystem mtime. A paper page's mtime moves
    for any edit at all, and the overwhelmingly common edit is a Related Papers
    bullet — `lint --fix` back-links, claim-overlap cross-links, concept-hub
    attachment, gloss rewrites. None of those change what the paper says, so
    none of them should age a synthesis page that cites it.

    Measured 2026-08-05 after a maintenance run that rewrote back-link bullets
    across ~250 paper pages: of 277 papers reported as `newer_references`, 231
    had mtimes bumped that day and only 4 had actually been (re)ingested. The
    remaining 227 were pure false positives, and permanent ones — mtimes never
    move back, so every affected synthesis page would stay flagged until its
    `generated_at` was bumped past the maintenance date, which would falsely
    claim a review that never happened.

    `ingested_at` moves on (re)ingest, which is the substantive event this check
    documents caring about ("re-examined when sources change"). The trade is
    recall: a hand-edited paper page that doesn't touch `ingested_at` no longer
    ages its citing pages. That is the right trade at these ratios.

    Three tiers, and **no mtime tier**. Falling back to mtime for pages without
    the field was the original design and it kept the exact bug the YAML
    preference was introduced to fix, just narrowed to fewer pages: on
    2026-08-06 a one-line YAML key removal rewrote 402 files and took
    `stale_synthesis` from 11 to 24, the third such self-inflicted spike in two
    days. A mechanical rewrite is not a content change, and mtime cannot tell the
    difference.

      1. YAML `ingested_at` — 373 of 443 pages.
      2. `ingest_iterations.created_at` (via `db_dates`) — a *recorded ingest
         event* rather than a filesystem artifact, covering 6 more.
      3. Otherwise **None**, meaning "unknown, so don't judge". Callers already
         skip `None`.

    Tier 3 exempts 23 legacy paper pages (`eddy-2011`, `mariani-2013`,
    `steinegger-2017`, …). That costs almost nothing real: they were ingested
    long before any current synthesis was generated, so they cannot be newer
    than one, and the only way they *appeared* newer was the artifact this
    removes. "Not checked" is the honest state for a page whose ingest date
    nothing recorded — better than a date that reports every maintenance pass as
    a finding, because those false positives are permanent (mtimes never move
    back) and clearing one means bumping `generated_at`, which falsely claims a
    review that never happened.
    """
    from datetime import datetime
    raw = fm.get("ingested_at")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).strip().strip("\"'")).date()
        except (ValueError, TypeError):
            pass
    if db_dates:
        return db_dates.get(md.stem)
    return None


def _db_ingest_dates() -> dict:
    """`{stem: date}` of each paper's earliest recorded ingest attempt.

    Tier 2 of `_source_change_date`. One batched query rather than one per page,
    and `safe_read` so a missing or locked DB degrades to `{}` — which pushes
    those pages to tier 3 ("unknown, don't judge") instead of failing a lint run
    that is otherwise pure-filesystem.

    `MIN(created_at)` is deliberate: a re-ingest adds rows, and the *first*
    attempt is when the paper's substance entered the wiki. Using MAX would make
    every retry look like a fresh source change.
    """
    from datetime import datetime
    from ...db.safe import safe_read

    def _query(conn):
        return conn.execute(
            """
            SELECT paper_stem, MIN(created_at) AS first_seen
              FROM ingest_iterations
             WHERE paper_stem IS NOT NULL AND paper_stem != ''
             GROUP BY paper_stem
            """
        ).fetchall()

    rows = safe_read(_query, default=[], label="lint.staleness_db_dates")
    out: dict = {}
    for r in rows:
        try:
            out[r["paper_stem"]] = datetime.fromtimestamp(r["first_seen"]).date()
        except (TypeError, ValueError, OSError):
            continue
    return out


def find_stale_synthesis(
    pages: list[Path],
    pages_fm: dict[Path, dict],
    known: set[str],
) -> list[tuple[Path, list[str]]]:
    """Pages a referenced paper has outrun: the paper's ingest date is newer than
    the page's `generated_at`. Covers synthesis/, ideas/ and concepts/ — all three
    depend on referenced papers and all three should be re-examined when sources
    change. Reported under the `stale_synthesis` JSON key for backward
    compatibility.

    "Ingest date" is `_source_change_date`, which never consults mtime; a paper
    whose date is unknown is skipped rather than guessed at.

    Comparison is at date resolution: `generated_at` is a YYYY-MM-DD field, so
    a paper modified on the same calendar day does NOT trigger staleness —
    only modifications strictly after the page's generated_at day. Otherwise
    bumping generated_at after a same-day re-ingest never clears the flag.
    """
    from datetime import datetime
    stale: list[tuple[Path, list[str]]] = []
    db_dates = _db_ingest_dates()
    source_dates = {
        page_key(p): _source_change_date(p, pages_fm.get(p, {}), db_dates)
        for p in pages
    }
    for md in pages:
        if md.parent.name not in ("synthesis", "ideas", "concepts"):
            continue
        fm = pages_fm.get(md, {})
        gen = fm.get("generated_at")
        if not gen:
            continue
        try:
            # `gen` is a string under read_page's line parser but a
            # datetime.date once frontmatter is parsed as real YAML; str()
            # normalizes both, and TypeError guards the non-string case.
            gen_date = datetime.strptime(str(gen), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        text = md.read_text(encoding="utf-8")
        links = extract_links(text, known)
        newer = [
            lk for lk in links
            if (d := source_dates.get(lk)) is not None and d > gen_date
        ]
        if newer:
            stale.append((md, newer))
    return stale


def unreferenced_top_hits(
    backend,
    md: Path,
    seed: str,
    known: set[str],
    top_n: int = 10,
) -> list[dict]:
    """Top-N `more_like_text(seed)` paper hits this page does NOT cite.

    Returns [{stem, key, score, title}, ...]. Empty when the seed retrieves
    nothing or every hit is already linked. Used by both `find_stale_by_content`
    (batch) and `check-coverage` (single-page CLI), which is why the backend is
    passed in rather than constructed here — check-coverage needs to surface
    "backend unavailable" as a clear exit-2 error, while batch lint silently
    skips. Each caller decides its own backend-availability semantics.
    """
    text = md.read_text(encoding="utf-8")
    linked = extract_links(text, known)
    self_key = page_key(md)
    try:
        hits = backend.more_like_text(seed, limit=top_n, page_type="paper")
    except Exception:
        return []
    return [
        {"key": h.key, "stem": h.stem, "score": round(h.score, 2), "title": h.title}
        for h in hits
        if h.key not in linked and h.key != self_key
    ]


def find_stale_by_content(
    pages: list[Path],
    pages_fm: dict[Path, dict],
    known: set[str],
    top_n: int = 10,
) -> list[tuple[Path, list[dict]]]:
    """For each page with a `topic_seed` YAML field, surface papers that rank
    in the top-N of `more_like_text(seed)` but aren't linked from the page.

    Returns a list of (page_path, [{stem, key, score}, ...]). Empty list if
    the search index isn't built (callers should ignore the absence; `lint`
    surfaces "can't check" rather than "nothing stale").
    """
    seeded = [(md, pages_fm[md].get("topic_seed", "").strip()) for md in pages]
    seeded = [(md, s) for md, s in seeded if s]
    if not seeded:
        return []

    try:
        from ...search import get_default_backend, SearchBackendUnavailable
        backend = get_default_backend()
        try:
            backend.query("__probe__", limit=1)
        except SearchBackendUnavailable:
            return []
    except Exception:
        return []

    out: list[tuple[Path, list[dict]]] = []
    for md, seed in seeded:
        unreferenced = unreferenced_top_hits(backend, md, seed, known, top_n=top_n)
        if unreferenced:
            out.append((md, unreferenced))
    return out


def find_stale_by_audit_count(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[Path, int, int]]:
    """Pages with a cached `wiki_papers_at_audit:` value that's drifted.

    Threshold: max(5 papers, 20% of cached) delta. Signals "time to re-run
    `researchwiki scout` and re-merge the suggested-additions output."
    """
    paper_count = sum(1 for md in pages if md.parent.name not in ("synthesis", "references", "concepts"))
    out: list[tuple[Path, int, int]] = []
    for md in pages:
        cached_raw = str(pages_fm[md].get("wiki_papers_at_audit", "")).strip()
        if not cached_raw.isdigit():
            continue
        cached_n = int(cached_raw)
        delta = paper_count - cached_n
        threshold = max(5, int(cached_n * 0.2))
        if delta >= threshold:
            out.append((md, cached_n, paper_count))
    return out


def find_stale_evolution_proposals() -> list[tuple[str, int, int]]:
    """`.ingest/*-evolution-proposals/` directories ≥ 7 days old.

    Returns [(stem, n_proposal_files, age_days)] sorted oldest-first.
    Either the proposals should be applied, or the directory should be
    deleted; lint surfaces them so they don't accumulate forever.
    """
    base = ingest_dir()
    if not base.exists():
        return []
    now = time.time()
    out: list[tuple[str, int, int]] = []
    for d in base.glob("*-evolution-proposals"):
        if not d.is_dir():
            continue
        age_days = int((now - d.stat().st_mtime) / 86400)
        if age_days < 7:
            continue
        n_files = sum(1 for f in d.glob("*.md"))
        stem = d.name.replace("-evolution-proposals", "")
        out.append((stem, n_files, age_days))
    out.sort(key=lambda x: -x[2])
    return out
