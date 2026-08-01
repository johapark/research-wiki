"""Staleness checks: pages that drifted out of sync with the surrounding corpus.

  - stale_synthesis: a synthesis page's mtime is older than referenced papers'
  - stale_by_content: page's `topic_seed` top-hits include unreferenced papers
  - stale_by_audit_count: a cached `wiki_papers_at_audit:` value drifted
  - stale_evolution_proposals: proposal directories ≥ 7 days old
"""

from __future__ import annotations

import time
from pathlib import Path

from ...paths import ingest_dir
from .walk import extract_links, page_key


def find_stale_synthesis(
    pages: list[Path],
    pages_fm: dict[Path, dict],
    known: set[str],
) -> list[tuple[Path, list[str]]]:
    """Pages whose referenced papers' mtimes are newer than the page's
    `generated_at` field. Covers both synthesis/ and ideas/ — both depend on
    referenced_papers and both should be re-examined when sources change.
    Reported under the `stale_synthesis` JSON key for backward compatibility.

    Comparison is at date resolution: `generated_at` is a YYYY-MM-DD field, so
    a paper modified on the same calendar day does NOT trigger staleness —
    only modifications strictly after the page's generated_at day. Otherwise
    bumping generated_at after a same-day re-ingest never clears the flag.
    """
    from datetime import datetime
    stale: list[tuple[Path, list[str]]] = []
    all_mtime_dates = {
        page_key(p): datetime.fromtimestamp(p.stat().st_mtime).date()
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
            if (d := all_mtime_dates.get(lk)) is not None and d > gen_date
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
    `researchwiki audit` and re-merge the suggested-additions output."
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
