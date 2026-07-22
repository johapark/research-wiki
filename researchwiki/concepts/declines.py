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

from .candidates import _term_slug

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
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def declined_slugs() -> set[str]:
    return set(load_declines().keys())


def add_decline(term: str, reason: str, *, source: str = "manual") -> str:
    """Record `term` as permanently declined. Returns its slug.

    Overwrites any existing entry for the same slug (re-declining updates
    the reason/timestamp rather than erroring).

    `source` records provenance: `"manual"` (a human ran `--decline`) or
    `"llm-triage"` (batch classifier auto-declined it). Entries written
    before this field existed are read back as `"manual"` via `.get`.
    """
    slug = _term_slug(term)
    declines = load_declines()
    declines[slug] = {
        "term": term,
        "reason": reason,
        "declined_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
    }
    _declines_path().write_text(json.dumps(declines, ensure_ascii=False, indent=2) + "\n")
    return slug


def remove_decline(term_or_slug: str) -> bool:
    """Undo a decline. Returns True if an entry was removed."""
    slug = _term_slug(term_or_slug)
    declines = load_declines()
    if slug not in declines:
        return False
    del declines[slug]
    _declines_path().write_text(json.dumps(declines, ensure_ascii=False, indent=2) + "\n")
    return True
