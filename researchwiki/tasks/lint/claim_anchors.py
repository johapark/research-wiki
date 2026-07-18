"""Dangling `[[stem#slug]]` claim-anchor lint.

Scans every wiki page for claim anchors and flags any whose (stem, slug)
pair no longer resolves against `claims(paper_stem, claim_slug)` in
state.db. This is the claim-level analogue of `broken_wikilinks` —
anchors are content-addressed, so a "broken" one means either the target
paper was removed OR the target claim's text has changed (regenerating
its slug).

Emits per-page hit lists; the JSON key `dangling_claim_anchors` lets CI
gate on zero-drift.
"""

from __future__ import annotations

from pathlib import Path

from ...grade.grounding import (
    ClaimDBUnavailable,
    extract_claim_anchors,
    _resolve_claim_anchors,
)


def find_dangling_claim_anchors(
    pages_body: dict[Path, str],
) -> list[dict]:
    """Return [{page, stem, slug, dangling}] for every unresolved anchor.

    Batch-resolves against state.db in one query rather than per-anchor.
    Empty list on any DB failure — we don't want the lint to gate on an
    unavailable DB.
    """
    # Collect all (page, stem, slug) references across the corpus.
    per_page: dict[Path, list[tuple[str, str]]] = {}
    all_pairs: set[tuple[str, str]] = set()
    for md, body in pages_body.items():
        pairs = extract_claim_anchors(body)
        if pairs:
            per_page[md] = pairs
            all_pairs.update(pairs)

    if not all_pairs:
        return []

    # _resolve_claim_anchors now raises ClaimDBUnavailable on a DB failure
    # (instead of returning an empty set indistinguishable from genuine drift),
    # so we can lean permissive cleanly: skip the check when the DB is down.
    try:
        resolved = _resolve_claim_anchors(all_pairs)
    except ClaimDBUnavailable:
        return []

    dangling: list[dict] = []
    for md, pairs in per_page.items():
        for stem, slug in pairs:
            if (stem, slug) not in resolved:
                dangling.append({
                    "page": md,
                    "stem": stem,
                    "slug": slug,
                })
    return dangling
