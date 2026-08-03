"""Permanent suppression list for concept-hub candidates that failed the
concept-vs-glossary thesis test (see docs/concept-vs-glossary.md).

The thesis gate in `scaffold.run()` stops a *bad* hub from being written,
but detection in `candidates.py` is stateless — it re-derives candidates
fresh from `keywords:`/`tags:`/claims every call, so a term you looked at
and rejected (e.g. it collapses into a definition already covered by the
term's own paper page) resurfaces on the next `status` or
`candidates concepts` run with no memory of that decision.

This is a manual, explicit denylist — not a decay stamp like
`categories.other_saturation_warning`. A decay stamp re-nags after a
window because the underlying condition (an overgrown `other/` bucket)
can change; a declined concept term doesn't age back into being a concept
just because time passed, so suppression here is permanent until a human
removes the entry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..fsatomic import write_json_atomic
from .candidates import _canonical_key, _term_slug

DECLINES_FILENAME = ".concept-declines.json"


def _declines_path() -> Path:
    from ..paths import wiki_root
    return wiki_root() / DECLINES_FILENAME


def load_declines() -> dict[str, dict]:
    """slug -> {term, reason, declined_at}. Empty dict if the file is
    absent or unreadable — never raises."""
    p = _declines_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def declined_slugs() -> set[str]:
    return set(load_declines().keys())


def declined_canon() -> set[str]:
    """Canonical keys of every declined *term* — so declining one morphological
    form ("foundation models") also suppresses its near-dupes ("foundation
    model"). Canonicalizes the stored term, not the slug key. Paired with the
    exact-slug check (union, not replacement) so every legacy decline still
    matches even if a merged candidate's representative slug shifts."""
    return {_canonical_key(e.get("term", ""))
            for e in load_declines().values() if e.get("term")}


def _write_declines(declines: dict[str, dict]) -> None:
    """Persist the whole list atomically.

    Atomicity matters more here than for the derived caches: this file is
    irreplaceable human judgment (or reviewed LLM triage), and `load_declines`
    treats a corrupt file as an empty one — so a truncated write would silently
    un-suppress every past decline with no error, and the only symptom would be
    declined noise quietly reappearing in `candidates concepts` / `status`.
    """
    write_json_atomic(_declines_path(), declines)


def add_decline(term: str, reason: str, *, source: str = "manual") -> str:
    """Record `term` as permanently declined. Returns its slug.

    Overwrites any existing entry for the same slug (re-declining updates
    the reason/timestamp rather than erroring).

    `source` records provenance: `"manual"` (a human ran `--decline`) or
    `"llm-triage"` (batch classifier auto-declined it). Entries written
    before this field existed are read back as `"manual"` via `.get`.

    For more than one term at a time use `add_declines` — it collapses the
    read-modify-write into a single atomic write.
    """
    return add_declines([(term, reason)], source=source)[0]


def add_declines(pairs: list[tuple[str, str]], *,
                 source: str = "manual") -> list[str]:
    """Record several `(term, reason)` declines in ONE read-modify-write.

    `apply_triage` can decline hundreds of terms in a run; calling
    `add_decline` per term would re-read, re-serialize and re-write the whole
    (growing) file each time — quadratic, and a much wider window in which an
    interrupted run leaves the list mangled. Returns the slugs in input order.
    """
    if not pairs:
        return []
    declines = load_declines()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    slugs: list[str] = []
    for term, reason in pairs:
        slug = _term_slug(term)
        declines[slug] = {
            "term": term,
            "reason": reason,
            "declined_at": stamp,
            "source": source,
        }
        slugs.append(slug)
    _write_declines(declines)
    return slugs


def remove_decline(term_or_slug: str) -> bool:
    """Undo a decline. Returns True if an entry was removed."""
    slug = _term_slug(term_or_slug)
    declines = load_declines()
    if slug not in declines:
        return False
    del declines[slug]
    _write_declines(declines)
    return True
