"""Durable citation form for a claim hit.

One helper so every renderer agrees on `[[stem#slug]]` — CLAUDE.md's rule
that `claim_id:NNN` never gets written into a page is easier to hold when
there's a single formatter and callers can't accidentally roll their own.
"""

from __future__ import annotations


def format_claim_ref(hit: dict) -> str:
    """Return `[[stem#slug]]` when the hit carries a slug, else `[[stem]]`.

    Falls back to a bare stem wikilink for rows that predate the slug
    migration (identical-text branch of `_upsert_claims` backfills NULL
    slugs, but a partially rebuilt DB may still surface them).
    """
    slug = hit.get("claim_slug")
    stem = hit["paper_stem"]
    return f"[[{stem}#{slug}]]" if slug else f"[[{stem}]]"
