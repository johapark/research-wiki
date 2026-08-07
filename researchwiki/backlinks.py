"""Shared back-link helper.

A "back-link" is a bullet on page B's `## Related Papers` section recording
that B stands in some relationship to paper A. Forward links — what A says
about B — are editorial and live on A's page. Back-links are structural and
live on B's page.

The relationship is directional, and the bullet states it from the reading
page's point of view ("this paper" = the page the bullet sits on), so the
same edge is worded differently on each end. `invert_relationship_note`
below is what keeps the two ends consistent.

Callers:
  - `agents/promote.py` at ingest commit, for every verified citation-graph
    or topical candidate the new paper has.
  - `tasks/lint/link_checks.py` `--fix` for the symmetric
    `missing_backlinks` finding.
  - `tasks/claim_overlap.py` and `concepts/scaffold.py`, which pass their
    own markers.

All pass through the same `append_related_paper` to keep the on-disk
convention consistent: same bullet shape, same `(auto-added; refine)` marker,
same idempotency rule (skip if `[[source_key]]` already appears anywhere in
the target body).
"""

from __future__ import annotations

import re
from pathlib import Path

from .fsatomic import update_locked
from .wiki import commit_page


_RELATED_HEADING_RE = re.compile(r"^##\s+Related Papers\s*$", re.MULTILINE | re.IGNORECASE)
_NEXT_HEADING_RE = re.compile(r"^##?\s+", re.MULTILINE)

# Canonical relationship phrasings for auto-added bullets. `lint --fix` parses
# these back off the page to recover an edge's direction, so `promote.py`
# imports them instead of keeping its own literals — a divergent copy in
# either place would silently degrade every mirrored note to TOPICAL_NOTE.
CITES_NOTE = "cites this paper (auto-added; refine)"
CITED_BY_NOTE = "cited by this paper (auto-added; refine)"
TOPICAL_NOTE = "topically related (auto-added; refine)"

# Recency phrasings, for edges where no citation was ever established but the
# two pages' `year:` fields are known. These assert strictly less than a
# citation and strictly more than TOPICAL_NOTE: "there is newer work on this"
# is what a reader of an older page most wants from a related-papers list, and
# it needs no evidence beyond two YAML fields. Deliberately free of citation
# language so `invert_relationship_note` and the repair pass's claim probe both
# read them as non-directional — which also makes rewriting them idempotent.
MORE_RECENT_NOTE = "more recent work on this topic (auto-added; refine)"
EARLIER_NOTE = "earlier work on this topic (auto-added; refine)"

# The claim each canonical note makes, minus the marker. Probed longest-first
# so "cited by this paper" is never mistaken for "cites this paper".
_CITED_BY_CLAIM = "cited by this paper"
_CITES_CLAIM = "cites this paper"


def invert_relationship_note(source_note: str) -> str:
    """Note for the bullet mirroring one that reads `source_note`.

    A bullet on page S reading `[[T]] — cites this paper` asserts that T
    cites S, because "this paper" resolves to the page the bullet is on.
    The mirror bullet on T therefore has to assert the converse —
    `[[S]] — cited by this paper` — so the direction inverts.

    Anything that is not a canonical citation phrasing (LLM-authored prose,
    the claim-overlap and concept-link markers, an absent note) degrades to
    `TOPICAL_NOTE`. Understating a real citation is harmless; asserting an
    unverified one is the fabrication CLAUDE.md's cross-link corollary
    forbids, and that asymmetry is why the fallback is the weakest claim
    rather than the most likely one.
    """
    text = (source_note or "").lower()
    if _CITED_BY_CLAIM in text:
        return CITES_NOTE
    if _CITES_CLAIM in text:
        return CITED_BY_NOTE
    return TOPICAL_NOTE


# A "no related papers" placeholder written by the page author when it had
# nothing to link: `(none)`, `(none — no overlapping wiki papers)`, `(None yet.)`.
#
# Must be **parenthesised**, or the bare word alone on its line. The first
# version of this pattern was `^\s*\(?\s*none\b.*$` — optional paren, then
# anything — which also matched an ordinary sentence that happens to open with
# the word, so `_drop_none_placeholder` deleted
# "None of the three replicates agreed, so this link is tentative." outright.
# That path runs on every `append_related_paper` call (every ingest, every
# `lint --fix`, every claim-overlap link), which makes a false positive here
# silent prose loss in the framework's hottest write path.
#
# Requiring the parentheses is what separates a placeholder from a sentence:
# every placeholder observed across the 62 pages cleaned on 2026-08-06 carried
# them, and no prose sentence does.
#
# `m` is load-bearing even though both current callers `.match()` a single
# already-split line: this is a module constant that crosses a package boundary,
# and `^…$` anchors invite a `.search(page_text)` over a whole page. Without
# MULTILINE that call would silently match only at offset 0.
NONE_PLACEHOLDER_RE = re.compile(
    r"""(?imx)
    ^\s*
    (?:
        \( \s* none\b [^)]* \)     # (none) / (none — no overlapping wiki papers)
      | none [\s.]*                 # or the bare word alone on the line
    )
    \s*$
    """
)


def _drop_none_placeholder(section_body: str) -> str:
    """Remove a `(none…)` placeholder from a Related Papers section body.

    Called when a bullet is about to be inserted, because the placeholder and a
    bullet are contradictory and the insertion is what makes it stale. Without
    this the placeholder simply stays and the bullet lands underneath it: 62
    pages in the corpus carried a literal `(none)` line sitting above real
    Related Papers bullets, every one of them written by exactly this path.

    Drops only a parenthesised placeholder or a bare `none` line — see
    `NONE_PLACEHOLDER_RE` for why that tightening matters. Prose that merely
    *begins* with the word ("None of the three replicates agreed…") is kept, as
    is any bullet containing it.
    """
    kept = [ln for ln in section_body.split("\n") if not NONE_PLACEHOLDER_RE.match(ln)]
    out = "\n".join(kept).strip()
    return (out + "\n") if out else ""


def append_related_paper(
    target_path: Path, source_key: str,
    note: str = TOPICAL_NOTE,
) -> bool:
    """Append a back-link bullet to `target_path`'s Related Papers section.

    `source_key` is the `category/stem` of the paper the bullet points at.
    `note` is the trailing explanation; it defaults to the weakest claim so
    that a caller which omits it understates the relationship rather than
    fabricating a citation. Callers that know the direction pass one of
    `CITES_NOTE` / `CITED_BY_NOTE` (or the result of
    `invert_relationship_note`); claim-overlap and concepts pass their own
    markers.

    Idempotent: returns False (no write) if the target's body already links
    the source in ANY wikilink form — `[[category/stem]]`, bare `[[stem]]`
    (the form CLAUDE.md mandates in tables), an aliased `[[stem|…]]`, or a
    claim anchor `[[stem#slug]]`. Creates the section if missing. Otherwise
    inserts the bullet at the section's tail (before the next heading).

    Returns True iff the file was modified.
    """
    if not target_path.exists():
        return False

    source_link = f"[[{source_key}]]"
    bullet = f"- {source_link} — {note}"
    # Any wikilink whose target resolves to the source paper: optional
    # `category/` prefix, then the bare stem, terminated by `]`, `|`, or `#`.
    stem = source_key.rsplit("/", 1)[-1]
    already_linked_re = re.compile(
        r"\[\[(?:[^\]\|#]*/)?" + re.escape(stem) + r"[\]\|#]"
    )

    def _splice(text: str) -> str:
        # Idempotent: returning `text` unchanged signals update_locked to skip
        # the write. Concurrent ingests editing the same target page are
        # serialized by the flock, so this read-modify-write can't be clobbered.
        if already_linked_re.search(text):
            return text
        m = _RELATED_HEADING_RE.search(text)
        if not m:
            sep = "\n\n" if not text.endswith("\n") else ("\n" if not text.endswith("\n\n") else "")
            return text + f"{sep}## Related Papers\n\n{bullet}\n"
        section_start = m.end()
        next_m = _NEXT_HEADING_RE.search(text, section_start + 1)
        section_end = next_m.start() if next_m else len(text)
        section_body = text[section_start:section_end].rstrip() + "\n"
        section_body = _drop_none_placeholder(section_body)
        new_body = section_body + bullet + "\n\n"
        return text[:section_start] + "\n" + new_body + text[section_end:]

    changed = update_locked(target_path, _splice, missing_ok=False)
    if changed:
        commit_page(target_path)
    return changed
